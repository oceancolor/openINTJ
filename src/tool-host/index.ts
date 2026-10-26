/**
 * P-04 工具宿主（Utility Process）
 *
 * 职责（《系统设计》§3.1.1.1）：承载 node-pty / child_process / Git CLI / 预览子进程，
 * 并且是**裁决权的唯一持有者**。
 *
 * 首版取舍已解除（第七轮）：命令执行仍用 `child_process.spawn`（argv 数组、不经 shell），
 * 但内嵌终端（US-15）已接上 node-pty —— 两者服务不同场景，不互相替代：
 *   child_process.spawn = Agent 发起的「跑一条命令」（一次性、无交互、有超时与截断）
 *   node-pty            = 用户终端（持久会话、TTY 语义、可中断、可缩放）
 * 终端里的每条命令仍走本进程的裁决链（见 terminal.ts），不是旁路。
 *
 * 安全铁律（按验收报告 D10 修正后）：
 *   1. 风险分级在本进程内完成，调用方无法「声明」风险等级
 *   2. 裁决由本进程查真值表得出，调用方无法「声明」已获批准
 *   3. 逐次审批必须持主进程签发的 HMAC 票据，且票据绑定 runId + 命令哈希，单次使用
 *   4. 未获批准**不产生任何进程**（US-08 AC-3）
 *   5. 命令一律以 argv 数组执行，不拼 shell 字符串（防注入）
 */

import { spawn } from 'node:child_process'
import { isAbsolute, join, relative, resolve } from 'node:path'
import { classify, type ClassifyResult } from './classifier'
import { AuditLog } from './audit-log'
import {
  DEFAULT_TIER,
  decideOutcome,
  type ApprovalTier,
  type DecisionBasis,
  type RiskClass,
  type RiskDecisionContext
} from '../shared/approval'
import { verifyTicket, type ApprovalTicket } from '../shared/approval-token'
import { mcpCommandString } from '../shared/mcp'
import { McpHub } from './mcp-client'
import { TerminalHost } from './terminal'
import { WorkspaceWatcher } from './file-watcher'
import type { TerminalCreateParams } from '../shared/terminal'
import { FILE_CHANGE_CHANNELS } from '../shared/session'
import {
  TOOL_METHODS,
  type McpCallParams,
  type McpCallOutcome,
  type McpDenialParams,
  type McpPrecheckParams,
  type PrecheckParams,
  type PrecheckResult,
  type RecordDenialParams,
  type ToolExecParams,
  type ToolExecResult,
  type ToolInitParams,
  type WireEnvelope
} from '../shared/ipc'

interface ParentPortLike {
  on(event: 'message', cb: (e: { data: unknown; ports: MessagePortLike[] }) => void): void
}
interface MessagePortLike {
  on(event: 'message', cb: (e: { data: unknown }) => void): void
  postMessage(msg: unknown): void
  start(): void
}

const parentPort = (process as unknown as { parentPort: ParentPortLike }).parentPort

/* ------------------------------------------------------------------ *
 * 权威状态 —— 只能由主进程经 tool.init / tool.tier.set 写入
 * ------------------------------------------------------------------ */

/** 票据密钥。未初始化前一律拒绝执行（fail-closed） */
let ticketSecret: string | null = null
/** 工作区根目录：写入判定的边界 */
let workspaceRoot: string | null = null
/** 当前审批档位 —— 主进程推送，不接受调用方在请求里自带 */
let currentTier: ApprovalTier = DEFAULT_TIER

/** 会话级授权：runId → 已获授权的风险等级集合（任务结束即失效） */
const sessionGrants = new Map<string, Set<string>>()
/** 已使用的票据，防重放 */
const consumedTickets = new Map<string, number>()

/**
 * 审计日志。惰性打开：init 时才知道数据目录，且 init 之前的任何执行都会被拒。
 * 打开失败即视为执行通道不可用 —— 没有审计就不跑。
 */
let auditLog: AuditLog | null = null

function openAuditLog(dataDir: string): void {
  auditLog?.close()
  auditLog = new AuditLog(join(dataDir, 'openintj.db'))
}

/* ------------------------------------------------------------------ *
 * 事件出口
 * ------------------------------------------------------------------ */

/** 绑定的端口。tool-host 以前只做「请求-响应」，终端需要主动推流，所以留住它 */
let outPort: MessagePortLike | null = null

/**
 * 向主进程推事件。
 *
 * 与 agent-host 的 emitEvent 不同，这里**不经过脱敏管线**：
 * 终端输出是用户自己屏幕上的回显，遮掉它只会让人以为文件是空的。
 * 代价与边界写在 shared/terminal.ts 的 TERMINAL_OUTPUT_IS_NOT_REDACTED 注释里 ——
 * 一旦要把它落盘或喂给模型，必须先补脱敏。
 */
function emitEvent(channel: string, payload: unknown): void {
  if (!outPort) return
  try {
    outPort.postMessage({ kind: 'evt', channel, payload } satisfies WireEnvelope)
  } catch (err) {
    // 推流失败不能影响执行：终端输出丢了只是少看几行，命令该跑还得跑
    console.warn(`[tool-host] 事件推送失败（${channel}）：${(err as Error).message}`)
  }
}

/* ------------------------------------------------------------------ *
 * 终端（US-15）
 * ------------------------------------------------------------------ */

let terminalHost: TerminalHost | null = null

/**
 * 惰性创建终端宿主。
 *
 * 依赖从本进程的模块状态注入，而不是让 terminal.ts 自己去读全局 ——
 * 那样会出现「两份 currentTier」这种最典型的漂移。
 */
function getTerminalHost(): TerminalHost {
  if (terminalHost) return terminalHost
  terminalHost = new TerminalHost({
    workspaceRoot: () => requireRoot(),
    currentTier: () => currentTier,
    classify,
    // 直接复用 shell 那条路径的裁决函数：终端不是在裁决之外另立一套
    decide: (req, risk) => decide(req, risk),
    audit: (runId, ctx, executed) => {
      if (!auditLog) throw new Error('审计库未就绪，拒绝执行（没有审计就不执行）')
      auditLog.record(runId, ctx, executed)
    },
    isInsideRoot: (target) => isInside(requireRoot(), target),
    emit: emitEvent
  })
  return terminalHost
}

/* ------------------------------------------------------------------ *
 * MCP（D6）
 * ------------------------------------------------------------------ */

/** MCP 连接池。惰性创建：init 时才知道配置文件路径 */
let mcpHub: McpHub | null = null
let mcpConfigPath: string | null = null
/** 当前这一次 reload 的完成信号；与 `mcpLoading` 配对使用 */
let mcpReady: Promise<void> | null = null
let mcpLoading = false

/**
 * 加载 MCP 配置并启动连接。
 *
 * `init` 里**不 await** —— 一个 npx 现装包的 server 可能要 20s 才起来，
 * 把 init 卡住就等于把「票据密钥下发」卡住，而那个失败的后果是全链路 fail-closed
 * （什么都跑不动）。少几个工具是降级，跑不动是事故，降级优先。
 */
function loadMcp(path: string): Promise<void> {
  mcpConfigPath = path
  mcpHub = new McpHub(path)
  mcpLoading = true
  mcpReady = mcpHub
    .reload()
    .catch((err: Error) => {
      console.warn(`[tool-host] MCP 加载异常：${err.message}`)
    })
    .finally(() => {
      mcpLoading = false
    })
  return mcpReady
}

/**
 * MCP 调用的风险分级。
 *
 * 为什么不复用 `classifier.ts`：那套规则是给 shell 命令字符串用的（正则匹配 `rm -rf`），
 * 而 MCP 工具的参数是结构化 JSON，「它到底会不会写盘」静态判断不了 ——
 * 能力由 server 决定，不由参数决定。
 *
 * 所以只按配置里的 `trust` 分两档：
 *   readonly → L0，自动放行（用户明确声明那个 server 只读）
 *   confirm  → L2，逐次审批（默认值，最谨慎）
 * 宁可让用户多点一次「允许」，也不替用户猜一个外部进程能干什么。
 */
const MCP_RULE_READONLY = 'R6-mcp-readonly'
const MCP_RULE_CALL = 'R6-mcp-call'

function classifyMcp(trust: 'readonly' | 'confirm'): {
  riskClass: RiskClass
  matchedRuleId: string
  reason: string
} {
  return trust === 'readonly'
    ? {
        riskClass: 'L0',
        matchedRuleId: MCP_RULE_READONLY,
        reason: 'MCP 工具（server 声明为只读，自动放行）'
      }
    : {
        riskClass: 'L2',
        matchedRuleId: MCP_RULE_CALL,
        reason: 'MCP 工具调用（外部进程能力不可静态判断，按工作区外写入级别处理）'
      }
}

function hasSessionGrant(runId: string, riskClass: string): boolean {
  return sessionGrants.get(runId)?.has(riskClass) ?? false
}

function grantSession(runId: string, riskClass: string): void {
  const set = sessionGrants.get(runId) ?? new Set<string>()
  set.add(riskClass)
  sessionGrants.set(runId, set)
}

function consumeTicket(ticketId: string, expiresAt: number): void {
  const now = Date.now()
  // 顺手清理过期记录，避免长期运行内存增长
  for (const [id, exp] of consumedTickets) if (exp <= now) consumedTickets.delete(id)
  consumedTickets.set(ticketId, expiresAt)
}

/* ------------------------------------------------------------------ *
 * 命令执行
 * ------------------------------------------------------------------ */

interface ParsedCommand {
  file: string
  args: string[]
}

/**
 * 极简命令解析：按空白切分，支持单双引号包裹。
 * 不做变量展开、不做管道 —— 需要管道时由 Agent 拆成多次调用，
 * 这样每一次调用都能被独立裁决与留痕。
 */
function parseCommand(command: string): ParsedCommand {
  const tokens: string[] = []
  let cur = ''
  let quote: '"' | "'" | null = null

  for (let i = 0; i < command.length; i++) {
    const ch = command[i]!
    if (quote) {
      if (ch === quote) quote = null
      else cur += ch
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      continue
    }
    if (/\s/.test(ch)) {
      if (cur) {
        tokens.push(cur)
        cur = ''
      }
      continue
    }
    cur += ch
  }
  if (cur) tokens.push(cur)

  if (tokens.length === 0) throw new Error('空命令')
  return { file: tokens[0]!, args: tokens.slice(1) }
}

const MAX_OUTPUT_BYTES = 512 * 1024

function execCommand(params: ToolExecParams): Promise<Omit<ToolExecResult, 'risk'>> {
  const started = Date.now()
  const { file, args } = parseCommand(params.command)

  return new Promise<Omit<ToolExecResult, 'risk'>>((resolve, reject) => {
    const child = spawn(file, args, {
      cwd: params.cwd,
      shell: false, // 关键：不经 shell，杜绝 `;` `|` 注入
      windowsHide: true,
      env: {
        // 最小环境：不透传用户全部环境变量，避免凭据随子进程泄露
        PATH: process.env['PATH'] ?? '',
        HOME: process.env['HOME'] ?? process.env['USERPROFILE'] ?? '',
        LANG: process.env['LANG'] ?? 'en_US.UTF-8',
        // 明确禁止交互式提权
        GIT_TERMINAL_PROMPT: '0'
      }
    })

    let stdout = ''
    let stderr = ''
    let truncated = false
    let killedByTimeout = false

    const append = (target: 'out' | 'err', chunk: Buffer): void => {
      const s = chunk.toString('utf8')
      if (target === 'out') {
        if (stdout.length + s.length > MAX_OUTPUT_BYTES) {
          stdout += s.slice(0, MAX_OUTPUT_BYTES - stdout.length)
          truncated = true
        } else stdout += s
      } else {
        if (stderr.length + s.length > MAX_OUTPUT_BYTES) {
          stderr += s.slice(0, MAX_OUTPUT_BYTES - stderr.length)
          truncated = true
        } else stderr += s
      }
    }

    child.stdout?.on('data', (c: Buffer) => append('out', c))
    child.stderr?.on('data', (c: Buffer) => append('err', c))

    const timeoutMs = params.timeoutMs ?? 120_000
    const timer = setTimeout(() => {
      killedByTimeout = true
      child.kill('SIGKILL')
    }, timeoutMs)

    child.on('error', (err) => {
      clearTimeout(timer)
      reject(err)
    })

    child.on('close', (code) => {
      clearTimeout(timer)
      resolve({
        exitCode: killedByTimeout ? null : code,
        stdout,
        stderr: killedByTimeout ? `${stderr}\n[超时 ${timeoutMs}ms，进程已终止]` : stderr,
        durationMs: Date.now() - started,
        truncated
      })
    })
  })
}

/* ------------------------------------------------------------------ *
 * 裁决
 * ------------------------------------------------------------------ */

interface Verdict {
  allow: boolean
  outcome: RiskDecisionContext['outcome']
  basis: DecisionBasis
  reason?: string
}

/**
 * 裁决所需的输入 —— 故意不含 `risk`。
 *
 * shell 命令与 MCP 调用共用这个形状：一方带来的是命令行，另一方带来的是
 * `mcp:server:tool {...}` 摘要串。票据绑定的是这个字符串的哈希，
 * 所以两条路径能走同一套验票逻辑，而不是各写一套（各写两套，迟早只有一套是对的）。
 */
interface SandboxRequest {
  runId: string
  command: string
  approvalTicket?: ApprovalTicket
}

/**
 * 裁决 —— 本进程独立完成，不看调用方脸色。
 *
 * AUTO_APPROVE      → 放行
 * SESSION_GRANT     → 本任务该风险等级已有授权则放行；否则需会话范围票据，放行并记账
 * PER_CALL_PROMPT   → 需单次票据，放行即作废
 * DENY              → 拒绝
 */
function decide(params: SandboxRequest, risk: ClassifyResult): Verdict {
  const outcome = decideOutcome(currentTier, risk.riskClass)
  const ticket = params.approvalTicket

  if (outcome === 'DENY') {
    return { allow: false, outcome, basis: 'policy-deny', reason: `策略拒绝（风险 ${risk.riskClass}）` }
  }

  if (outcome === 'AUTO_APPROVE') {
    return { allow: true, outcome, basis: 'risk-table' }
  }

  if (outcome === 'SESSION_GRANT') {
    if (hasSessionGrant(params.runId, risk.riskClass)) {
      return { allow: true, outcome, basis: 'user-grant' }
    }
    if (!ticket) {
      return {
        allow: false,
        outcome,
        basis: 'risk-table',
        reason: `首次执行风险 ${risk.riskClass} 需用户授权本任务内放行`
      }
    }
    const v = checkTicket(ticket, params, risk)
    if (!v.ok) return { allow: false, outcome, basis: 'policy-deny', reason: v.reason }
    if (ticket.scope === 'session') grantSession(params.runId, risk.riskClass)
    return { allow: true, outcome, basis: 'user-grant' }
  }

  // PER_CALL_PROMPT
  if (!ticket) {
    return { allow: false, outcome, basis: 'risk-table', reason: `风险 ${risk.riskClass} 需逐次审批` }
  }
  const v = checkTicket(ticket, params, risk)
  if (!v.ok) return { allow: false, outcome, basis: 'policy-deny', reason: v.reason }
  return { allow: true, outcome, basis: 'user-grant' }
}

function checkTicket(
  ticket: ApprovalTicket,
  params: SandboxRequest,
  risk: ClassifyResult
): { ok: boolean; reason?: string } {
  if (!ticketSecret) return { ok: false, reason: '票据密钥未就绪，拒绝执行（fail-closed）' }
  if (consumedTickets.has(ticket.ticketId)) return { ok: false, reason: '票据已被使用（防重放）' }

  const v = verifyTicket(ticketSecret, ticket, {
    runId: params.runId,
    command: params.command,
    riskClass: risk.riskClass
  })
  if (!v.ok) return v

  consumeTicket(ticket.ticketId, ticket.expiresAt)
  return { ok: true }
}

/* ------------------------------------------------------------------ *
 * 工作区文件变化监听
 * ------------------------------------------------------------------ */

/**
 * 监听器实例。放在模块级而不是 init 里 new ——
 * 一个进程只该有一个监听者，否则同一份改动会被上报两次。
 */
let watcher: WorkspaceWatcher | null = null

/**
 * 启动监听。
 *
 * 只上报**工作区内**的变化：`relative()` 之后仍以 `..` 开头的路径直接丢掉。
 * 这不是洁癖 —— 「变更」页的意义是「这次任务动了你的哪些文件」，
 * 越界的路径既不该出现在列表里，也不该让人误以为任务碰了它。
 */
function startWatching(root: string): { ok: boolean; recursive: boolean; reason?: string } {
  stopWatching()
  watcher = new WorkspaceWatcher(root, (notice) => {
    emitEvent(FILE_CHANGE_CHANNELS.changed, notice)
  })
  const result = watcher.start()
  if (!result.ok) {
    console.warn(`[tool-host] 文件变化监听未启动：${result.reason ?? '未知原因'}`)
    return result
  }
  console.log(
    `[tool-host] 文件变化监听已启动（root=${root}，基线 ${result.files} 个文件，递归=${result.recursive}）`
  )
  return result
}

function stopWatching(): void {
  watcher?.stop()
  watcher = null
}

/* ------------------------------------------------------------------ *
 * 请求处理
 * ------------------------------------------------------------------ */

async function handleRequest(method: string, params: unknown): Promise<unknown> {
  switch (method) {
    case TOOL_METHODS.ping:
      return `tool-host online (pid=${process.pid}, tier=${currentTier})`

    case TOOL_METHODS.init: {
      const p = params as ToolInitParams
      ticketSecret = p.ticketSecret
      workspaceRoot = p.workspaceRoot
      // 审计库打开失败要让 init 直接失败（进而让主进程的 init 重试并最终告警），
      // 而不是等到第一条命令才发现「原来审计一直没写进去」
      openAuditLog(p.dataDir)
      // MCP 连接不阻塞 init（见 loadMcp 注释）
      if (p.mcpConfigPath) void loadMcp(p.mcpConfigPath)
      // 文件变化监听同样不阻塞 init：它是「变更」页的输入，起不来也不该挡住执行通道
      const watchResult = startWatching(p.workspaceRoot)
      return { ok: true, workspaceRoot, watching: watchResult.ok, watchReason: watchResult.reason }
    }

    /**
     * 开始 / 重启监听（幂等）。
     *
     * 与 exec 一样要求先初始化：没有工作区根就没有边界，
     * 而监听工作区外的内容等于给界面开了一扇不该有的窗。
     */
    case TOOL_METHODS.fsWatch: {
      return startWatching(requireRoot())
    }

    case TOOL_METHODS.fsWatchStop: {
      stopWatching()
      return { ok: true }
    }

    case TOOL_METHODS.setTier: {
      const { tier } = params as { tier: ApprovalTier }
      const prev = currentTier
      currentTier = tier
      // 收紧档位时清掉全部会话级授权，避免旧授权把新档位架空
      const tightened = (tier === 'strict' && prev !== 'strict') || (tier === 'standard' && prev === 'autonomous')
      if (tightened) sessionGrants.clear()
      return { ok: true, prev, cleared: tightened ? sessionGrants.size : 0 }
    }

    case TOOL_METHODS.classify: {
      const p = params as { command: string; cwd: string }
      return classify({ command: p.command, cwd: p.cwd, workspaceRoot: requireRoot() })
    }

    /**
     * 预判 —— 只说结论，不执行、不落审计。
     *
     * 审计只在真正发起执行时写：预判可能被调用方用来「先看看要不要弹窗」，
     * 若每次都落审计，事件表会被预判噪音灌满，真正的执行记录反而难查。
     */
    case TOOL_METHODS.precheck: {
      const p = params as PrecheckParams
      const root = requireRoot()
      const risk = classify({ command: p.command, cwd: p.cwd, workspaceRoot: root })
      const effective: ClassifyResult = isInside(root, p.cwd)
        ? risk
        : { riskClass: 'L2', matchedRuleId: 'R3-02', reason: `工作目录越出工作区根：${p.cwd}` }

      const outcome = decideOutcome(currentTier, effective.riskClass)
      const grant = hasSessionGrant(p.runId, effective.riskClass)
      const needsTicket = outcome === 'PER_CALL_PROMPT' || (outcome === 'SESSION_GRANT' && !grant)

      return {
        riskClass: effective.riskClass,
        matchedRuleId: effective.matchedRuleId,
        reason: effective.reason,
        outcome,
        needsTicket,
        hasSessionGrant: grant
      } satisfies PrecheckResult
    }

    case TOOL_METHODS.recordDenial: {
      const p = params as RecordDenialParams
      const root = requireRoot()
      if (!auditLog) throw new Error('审计库未就绪，拒绝记录（没有审计就不留痕）')

      // 重新分级：不接受调用方送来的等级（和 exec 同一套判定，只是不执行）
      const risk = classify({ command: p.command, cwd: p.cwd, workspaceRoot: root })
      const effective: ClassifyResult = isInside(root, p.cwd)
        ? risk
        : { riskClass: 'L2', matchedRuleId: 'R3-02', reason: `工作目录越出工作区根：${p.cwd}` }

      auditLog.record(
        p.runId,
        {
          command: p.command,
          riskClass: effective.riskClass,
          tier: currentTier,
          outcome: decideOutcome(currentTier, effective.riskClass),
          basis: p.basis,
          matchedRuleId: effective.matchedRuleId
        },
        false
      )
      return { ok: true, riskClass: effective.riskClass }
    }

    case TOOL_METHODS.revokeSessionGrants: {
      const { runId } = params as { runId?: string }
      if (runId) sessionGrants.delete(runId)
      else sessionGrants.clear()
      return { ok: true, remaining: sessionGrants.size }
    }

    /* ---------------- MCP（D6） ---------------- */

    /**
     * 重新加载 mcp.json。
     *
     * 与 init 里的首次加载相反，这里**必须 await**：用户点了「重新加载」，
     * 界面上就该直接看到新的连接结果，而不是看到一个还在转圈的 'connecting'。
     */
    case TOOL_METHODS.mcpReload: {
      const p = (params ?? {}) as { mcpConfigPath?: string }
      const path = p.mcpConfigPath ?? mcpConfigPath
      if (!path) throw new Error('未配置 MCP：init 时未下发 mcpConfigPath')
      await loadMcp(path)
      return {
        ok: true,
        configPath: path,
        configError: mcpHub?.getConfigError() ?? null,
        servers: mcpHub?.status() ?? []
      }
    }

    /**
     * 连接状态。**不等待**加载完成 —— 状态查询本身不能变成 20 秒的阻塞操作，
     * 否则设置页会白屏。正在连的 server 状态是 'connecting'，界面照实显示即可。
     */
    case TOOL_METHODS.mcpStatus: {
      return {
        enabled: mcpHub !== null,
        loading: mcpLoading,
        configPath: mcpConfigPath,
        configError: mcpHub?.getConfigError() ?? null,
        servers: mcpHub?.status() ?? []
      }
    }

    /** 工具清单。这里允许等一次加载：它只在循环开头取一次，不在每步调用 */
    case TOOL_METHODS.mcpList: {
      if (!mcpHub) return []
      if (mcpLoading && mcpReady) {
        await Promise.race([mcpReady, new Promise((r) => setTimeout(r, 30_000))])
      }
      return mcpHub.listTools()
    }

    /**
     * MCP 调用的预判。与 shell 预判同形（PrecheckResult），但分级依据不同：
     * MCP 只能按 server 声明的 trust 分，静态判断不了结构化参数会做什么。
     *
     * 找不到工具时按 `confirm` 处理 —— 一个连工具名都对不上的调用更该被拦。
     */
    case TOOL_METHODS.mcpPrecheck: {
      const p = params as McpPrecheckParams
      const info = mcpHub?.findTool(p.server, p.tool) ?? null
      const risk = classifyMcp(info?.trust ?? 'confirm')
      const outcome = decideOutcome(currentTier, risk.riskClass)
      const grant = hasSessionGrant(p.runId, risk.riskClass)
      const needsTicket = outcome === 'PER_CALL_PROMPT' || (outcome === 'SESSION_GRANT' && !grant)

      return {
        riskClass: risk.riskClass,
        matchedRuleId: risk.matchedRuleId,
        reason: info ? risk.reason : `${risk.reason}；但未找到工具 ${p.server}/${p.tool}`,
        outcome,
        needsTicket,
        hasSessionGrant: grant
      } satisfies PrecheckResult
    }

    /**
     * 执行 MCP 工具调用 —— 与 shell 执行**同一条**裁决与审计链。
     *
     * 顺序同样刻意：「先留痕再执行」。被拦下的调用也入库（executed=0），
     * 否则「模型试过调这个工具但被拒了」这件事在事后无从查证。
     */
    case TOOL_METHODS.mcpCall: {
      const p = params as McpCallParams

      if (!ticketSecret || !workspaceRoot) {
        throw new Error('拒绝执行：tool-host 未完成初始化（缺少票据密钥或工作区根目录）')
      }
      const hub = mcpHub
      if (!hub) throw new Error('拒绝执行：MCP 未启用（未配置 mcp.json）')

      const info = hub.findTool(p.server, p.tool)
      const risk = classifyMcp(info?.trust ?? 'confirm')
      // 票据绑定这个字符串的哈希 —— 必须与 agent-host 算出的逐字节一致
      const command = mcpCommandString(p.server, p.tool, p.args)
      const verdict = decide({ runId: p.runId, command, approvalTicket: p.approvalTicket }, risk)

      const audit: RiskDecisionContext = {
        command,
        riskClass: risk.riskClass,
        tier: currentTier,
        outcome: verdict.outcome,
        basis: verdict.basis,
        matchedRuleId: risk.matchedRuleId
      }

      if (!auditLog) throw new Error('拒绝执行：审计库未就绪（没有审计就不执行）')
      auditLog.record(p.runId, audit, verdict.allow)

      if (!verdict.allow) {
        throw new Error(
          `拒绝执行：${verdict.reason ?? '未获批准'}（风险 ${audit.riskClass}，档位 ${audit.tier}，规则 ${audit.matchedRuleId}），未发起任何 MCP 调用`
        )
      }

      const started = Date.now()
      const r = await hub.callTool(p.server, p.tool, p.args)
      return {
        ok: r.ok,
        text: r.text,
        isError: r.isError,
        durationMs: Date.now() - started,
        risk: audit
      } satisfies McpCallOutcome
    }

    /**
     * 记录一次「未发起」的 MCP 调用（用户拒绝 / 超时）。
     *
     * 分级在这里重算，不采信调用方送来的等级；票据不存在也不影响 —— 这本来就没执行。
     */
    case TOOL_METHODS.mcpDenial: {
      const p = params as McpDenialParams
      if (!auditLog) throw new Error('审计库未就绪，拒绝记录（没有审计就不留痕）')

      const info = mcpHub?.findTool(p.server, p.tool) ?? null
      const risk = classifyMcp(info?.trust ?? 'confirm')
      auditLog.record(
        p.runId,
        {
          command: mcpCommandString(p.server, p.tool, p.args),
          riskClass: risk.riskClass,
          tier: currentTier,
          outcome: decideOutcome(currentTier, risk.riskClass),
          basis: p.basis,
          matchedRuleId: risk.matchedRuleId
        },
        false
      )
      return { ok: true, riskClass: risk.riskClass }
    }

    case TOOL_METHODS.auditRecent: {
      const { runId, limit } = params as { runId: string; limit?: number }
      if (!auditLog) throw new Error('审计库未就绪')
      return auditLog.recent(runId, limit ?? 20)
    }

    case TOOL_METHODS.exec: {
      const p = params as ToolExecParams

      // 未初始化即拒绝：宁可跑不动，也不要在没有密钥与边界的情况下执行命令
      if (!ticketSecret || !workspaceRoot) {
        throw new Error('拒绝执行：tool-host 未完成初始化（缺少票据密钥或工作区根目录）')
      }

      const risk = classify({ command: p.command, cwd: p.cwd, workspaceRoot })

      // cwd 越界本身也按「工作区外」对待，避免用相对路径消解边界
      const cwdOutside = !isInside(workspaceRoot, p.cwd)
      const effective: ClassifyResult = cwdOutside
        ? { riskClass: 'L2', matchedRuleId: 'R3-02', reason: `工作目录越出工作区根：${p.cwd}` }
        : risk

      const verdict = decide(p, effective)

      const audit: RiskDecisionContext = {
        command: p.command,
        riskClass: effective.riskClass,
        tier: currentTier,
        outcome: verdict.outcome,
        basis: verdict.basis,
        matchedRuleId: effective.matchedRuleId
      }

      // 先留痕再执行：拦下的也要记，否则「试过但被拒」这件事在事后无从查证
      if (!auditLog) {
        throw new Error('拒绝执行：审计库未就绪（没有审计就不执行）')
      }
      auditLog.record(p.runId, audit, verdict.allow)

      if (!verdict.allow) {
        throw new Error(
          `拒绝执行：${verdict.reason ?? '未获批准'}（风险 ${audit.riskClass}，档位 ${audit.tier}，规则 ${audit.matchedRuleId}），未获批准不产生进程`
        )
      }

      const result = await execCommand(p)
      return { ...result, risk: audit } satisfies ToolExecResult
    }

    /* ---------------- 终端（US-15） ---------------- */

    /**
     * 创建 PTY 会话。
     *
     * 与 exec 的准入条件一致：未初始化一律拒绝 —— 没有工作区根就没有边界，
     * 没有审计库就没有留痕，这两样缺任何一样都不该拉起一个能跑命令的 shell。
     */
    case TOOL_METHODS.terminalCreate: {
      if (!workspaceRoot) throw new Error('拒绝创建终端：tool-host 未完成初始化（缺少工作区根目录）')
      if (!auditLog) throw new Error('拒绝创建终端：审计库未就绪（没有审计就不执行）')
      const p = params as TerminalCreateParams
      return getTerminalHost().create({ cwd: p.cwd, cols: p.cols, rows: p.rows })
    }

    /**
     * 提交一条命令 —— 终端路径的唯一入口。
     *
     * `approvalTicket` 只可能来自主进程（票据密钥只在 tool-host，agent-host 与渲染层都签不出来），
     * 所以这里甚至可以接收它而不担心伪造：伪造的票过不了 verifyTicket。
     */
    case TOOL_METHODS.terminalInput: {
      if (!workspaceRoot) throw new Error('拒绝执行：tool-host 未完成初始化')
      if (!auditLog) throw new Error('拒绝执行：审计库未就绪（没有审计就不执行）')
      const p = params as { sessionId: string; command: string; approvalTicket?: ApprovalTicket }
      return getTerminalHost().submit(p.sessionId, p.command, p.approvalTicket)
    }

    case TOOL_METHODS.terminalInterrupt: {
      const p = params as { sessionId: string }
      return getTerminalHost().interrupt(p.sessionId)
    }

    case TOOL_METHODS.terminalResize: {
      const p = params as { sessionId: string; cols: number; rows: number }
      return getTerminalHost().resize(p.sessionId, p.cols, p.rows)
    }

    case TOOL_METHODS.terminalClose: {
      const p = params as { sessionId: string }
      return getTerminalHost().close(p.sessionId)
    }

    case TOOL_METHODS.terminalList: {
      return terminalHost?.list() ?? []
    }

    default:
      throw new Error(`未知方法：${method}`)
  }
}

function requireRoot(): string {
  if (!workspaceRoot) throw new Error('tool-host 未完成初始化（缺少工作区根目录）')
  return workspaceRoot
}

function isInside(root: string, target: string): boolean {
  const rel = relative(resolve(root), resolve(target))
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel))
}

/* ------------------------------------------------------------------ *
 * 端口接入
 * ------------------------------------------------------------------ */

function bindPort(port: MessagePortLike): void {
  // 留住端口：终端要主动推输出流，纯请求-响应模型做不到
  outPort = port
  port.on('message', (e) => {
    const env = e.data as WireEnvelope
    if (env.kind !== 'req') return

    handleRequest(env.method, env.params).then(
      (result) => {
        port.postMessage({ kind: 'res', requestId: env.requestId, ok: true, result } satisfies WireEnvelope)
      },
      (err: Error) => {
        port.postMessage({
          kind: 'res',
          requestId: env.requestId,
          ok: false,
          error: { code: 'TOOL_HOST_ERROR', message: err.message }
        } satisfies WireEnvelope)
      }
    )
  })
  port.start()
}

if (!parentPort) {
  console.error('[tool-host] 必须以 Electron Utility Process 方式启动')
  process.exit(1)
}

/**
 * 回收 PTY 子进程。
 *
 * 不做这件事的后果不是「报错」，是**留下孤儿 shell**：用户关了应用，
 * 机器上还挂着一个 powershell 在跑 —— 这类问题极难归因到「那个终端」。
 * Utility Process 被杀时 exit 不保证触发，所以能挂的钩子都挂上。
 */
function disposeTerminals(): void {
  try {
    terminalHost?.disposeAll()
  } catch {
    /* 清理阶段不该再抛 */
  }
}
process.on('exit', disposeTerminals)
process.on('SIGTERM', () => {
  disposeTerminals()
  process.exit(0)
})
process.on('SIGINT', () => {
  disposeTerminals()
  process.exit(0)
})

parentPort.on('message', (e) => {
  const data = e.data as { kind?: string }
  if (data?.kind === 'port-init' && e.ports.length > 0) {
    bindPort(e.ports[0]!)
  }
})
