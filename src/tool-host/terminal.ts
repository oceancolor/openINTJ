/**
 * 终端宿主（P-04 tool-host 内）
 *
 * 为什么终端必须住在 tool-host：它是「会产生进程」的东西。
 * 裁决权在 tool-host（D10 之后调用方连 risk 字段都传不进来），
 * 若把 pty 放到主进程或渲染层，就等于给终端开了一条不经过裁决的通道 ——
 * 那正是 US-15 的 AC-2 明令禁止的。
 *
 * 形态：持久 PTY 会话（pty 进程在整个会话期内不退出，`cd` / `export` 会保留），
 *      但**命令提交**由本文件掌管：整行分级 → 放行才把 `命令 + \r` 写进 pty。
 *      详细取舍见 shared/terminal.ts 的文件头。
 */

import { resolve } from 'node:path'
import type * as NodePty from 'node-pty'
import type { ApprovalTicket } from '../shared/approval-token'
import type { ApprovalTier, RiskDecisionContext } from '../shared/approval'
import type { ClassifyResult } from './classifier'
import type { TerminalSessionInfo, TerminalSubmitResult } from '../shared/terminal'
import { TERMINAL_CHANNELS, terminalRunId } from '../shared/terminal'

/** 裁决所需输入 —— 与 index.ts 的 SandboxRequest 同形（故意不含 risk） */
export interface SandboxRequestLike {
  runId: string
  command: string
  approvalTicket?: ApprovalTicket
}

export interface VerdictLike {
  allow: boolean
  outcome: RiskDecisionContext['outcome']
  basis: RiskDecisionContext['basis']
  reason?: string
}

export interface TerminalDeps {
  workspaceRoot: () => string
  currentTier: () => ApprovalTier
  classify: (o: { command: string; cwd: string; workspaceRoot: string }) => ClassifyResult
  decide: (req: SandboxRequestLike, risk: ClassifyResult) => VerdictLike
  /** 写审计：与 shell / MCP 走同一张表，先留痕再执行 */
  audit: (runId: string, ctx: RiskDecisionContext, executed: boolean) => void
  /** 是否在工作区根内 —— cwd 越界按 L2 处理（与 exec 一致） */
  isInsideRoot: (target: string) => boolean
  /** 向主进程推事件（输出流 / 会话退出） */
  emit: (channel: string, payload: unknown) => void
}

interface Session {
  id: string
  pty: NodePty.IPty
  shell: string
  cwd: string
  cols: number
  rows: number
  startedAt: number
  policyProfileVersion: number
  /** 最近一次输出时间 —— 界面用它判断「还在跑 / 空闲」 */
  lastOutputAt: number
  disposed: boolean
}

/** 会话上限。终端是重资源（每个会话一个 shell 进程），不能让人开一百个 */
const MAX_SESSIONS = 3

/** 输出事件节流：pty 会一个字符一个字符地吐，逐字 postMessage 会把端口打爆 */
const OUTPUT_FLUSH_MS = 12
/** 单次刷出的最大字符数，避免 4MB 输出卡住事件循环 */
const OUTPUT_FLUSH_MAX = 64 * 1024

export class TerminalHost {
  private sessions = new Map<string, Session>()
  private seq = 0

  constructor(private readonly deps: TerminalDeps) {}

  /**
   * 拉起一个会话。
   *
   * 这里**不做命令裁决** —— 创建会话本身不执行任何用户命令（只拉起一个 shell），
   * 风险来自「往它里面写什么」，那一关在 submit() 里。
   * 但 cwd 必须限在工作区根内：让 shell 从工作区外启动，后续每条命令都带越界属性，
   * 属于把边界一开始就让掉。
   */
  create(params: { cwd?: string; cols: number; rows: number }): TerminalSessionInfo {
    if (this.sessions.size >= MAX_SESSIONS) {
      throw new Error(`终端会话已达上限（${MAX_SESSIONS} 个），请先关闭不用的会话`)
    }

    const root = this.deps.workspaceRoot()
    const cwd = resolve(params.cwd ?? root)
    if (!this.deps.isInsideRoot(cwd)) {
      throw new Error(`拒绝创建会话：工作目录越出工作区根（${cwd}）`)
    }

    // 延迟 require：node-pty 是原生模块，加载失败时要给出能看懂的错，
    // 而不是在模块顶层抛一个「Cannot find module」把整个 tool-host 带崩。
    let pty: typeof NodePty
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      pty = require('node-pty') as typeof NodePty
    } catch (err) {
      throw new Error(
        `终端不可用：node-pty 原生模块加载失败（${(err as Error).message}）。` +
          `常见原因是原生模块与 Electron ABI 不匹配`
      )
    }

    const shell = pickShell()
    const cols = clampDim(params.cols, 120)
    const rows = clampDim(params.rows, 30)

    const proc = pty.spawn(shell.file, shell.args, {
      name: 'xterm-256color',
      cols,
      rows,
      cwd,
      // 最小环境：不继承宿主的一堆变量（与 exec 的「最小环境」同一条纪律）
      env: {
        ...baseEnv(),
        TERM: 'xterm-256color',
        // 让 shell 知道自己在被程序驱动，别去做花哨的交互初始化
        TERM_PROGRAM: 'openINTJ'
      }
    })

    const id = `term-${Date.now().toString(36)}-${(this.seq += 1)}`
    const session: Session = {
      id,
      pty: proc,
      shell: shell.file,
      cwd,
      cols,
      rows,
      startedAt: Date.now(),
      policyProfileVersion: tierVersion(this.deps.currentTier()),
      lastOutputAt: Date.now(),
      disposed: false
    }
    this.sessions.set(id, session)
    this.wireOutput(session)

    proc.onExit(({ exitCode, signal }) => {
      if (session.disposed) return
      session.disposed = true
      this.flushBuffer(session, true)
      this.deps.emit(TERMINAL_CHANNELS.exit, { sessionId: id, exitCode, signal })
      this.sessions.delete(id)
    })

    return {
      sessionId: id,
      shell: shell.file,
      cwd,
      cols,
      rows,
      startedAt: session.startedAt,
      policyProfileVersion: session.policyProfileVersion
    }
  }

  /**
   * 输出转发。
   *
   * 节流不是优化，是**正确性**：`yes` 这类命令每秒几十万字符，
   * 逐字 postMessage 会让主进程的消息队列成为瓶颈，界面反而更卡。
   * 攒一小段再发，视觉上没有差别。
   */
  private buffers = new Map<string, string>()

  private wireOutput(session: Session): void {
    session.pty.onData((chunk) => {
      session.lastOutputAt = Date.now()
      const prev = this.buffers.get(session.id) ?? ''
      const next = prev + chunk
      if (next.length >= OUTPUT_FLUSH_MAX) {
        this.buffers.set(session.id, '')
        this.deps.emit(TERMINAL_CHANNELS.data, { sessionId: session.id, data: next })
        return
      }
      this.buffers.set(session.id, next)
      if (!this.flushTimers.has(session.id)) {
        const timer = setTimeout(() => {
          this.flushTimers.delete(session.id)
          this.flushBuffer(session, false)
        }, OUTPUT_FLUSH_MS)
        // 定时器不该把进程钉住
        timer.unref?.()
        this.flushTimers.set(session.id, timer)
      }
    })
  }

  private flushTimers = new Map<string, NodeJS.Timeout>()

  private flushBuffer(session: Session, force: boolean): void {
    const buf = this.buffers.get(session.id)
    if (!buf) return
    if (!force && buf.length === 0) return
    this.buffers.set(session.id, '')
    this.deps.emit(TERMINAL_CHANNELS.data, { sessionId: session.id, data: buf })
  }

  /**
   * 提交一条命令 —— 本文件的核心。
   *
   * 顺序刻意与 exec 一致：**分级 → 裁决 → 先留痕 → 再写 pty**。
   * 任何一步反了都会出问题：
   *   - 先写后裁决 → 命令已经跑了，裁决只是事后追认；
   *   - 先执行后留痕 → 崩在中间的进程不会留下「它尝试过什么」。
   */
  submit(
    sessionId: string,
    rawCommand: string,
    approvalTicket?: ApprovalTicket
  ): TerminalSubmitResult {
    const session = this.sessions.get(sessionId)
    if (!session || session.disposed) return { kind: 'no-session' }

    const command = rawCommand.trim()
    if (!command) return { kind: 'empty' }

    const root = this.deps.workspaceRoot()
    const risk = this.deps.classify({ command, cwd: session.cwd, workspaceRoot: root })
    const effective: ClassifyResult = this.deps.isInsideRoot(session.cwd)
      ? risk
      : { riskClass: 'L2', matchedRuleId: 'R3-02', reason: `工作目录越出工作区根：${session.cwd}` }

    const runId = terminalRunId(sessionId)
    const verdict = this.deps.decide({ runId, command, approvalTicket }, effective)

    // 需要授权时**先不写审计**：这一刻还没决定（用户可能还会批），
    // 审计写在这里会让「等待裁决」留下一条 executed=false 的假拒绝记录。
    // 真正的两种结局各有入口：批准后 submit 再来一次（带票据）、被拒走 recordDenial。
    const awaitingApproval = !verdict.allow && verdict.basis === 'risk-table'
    if (awaitingApproval) {
      return {
        kind: 'need-approval',
        command,
        riskClass: effective.riskClass,
        matchedRuleId: effective.matchedRuleId,
        reason: verdict.reason
      }
    }

    const audit: RiskDecisionContext = {
      command,
      riskClass: effective.riskClass,
      tier: this.deps.currentTier(),
      outcome: verdict.outcome,
      basis: verdict.basis,
      matchedRuleId: effective.matchedRuleId
    }

    // 先留痕：被拒的也记（否则「试过但被拒」事后无从查证）
    this.deps.audit(runId, audit, verdict.allow)

    if (!verdict.allow) {
      return {
        kind: 'denied',
        command,
        riskClass: effective.riskClass,
        matchedRuleId: effective.matchedRuleId,
        reason: `${verdict.reason ?? '未获批准'}（风险 ${effective.riskClass}，档位 ${audit.tier}，规则 ${effective.matchedRuleId}）`
      }
    }

    // 写 pty。pty 自己会回显这行（终端驱动 echo），所以界面不需要本地回显 ——
    // 这也是「不做键盘直通」之后反而少了一个双重回显问题的原因。
    session.pty.write(`${command}\r`)

    return {
      kind: 'ran',
      command,
      riskClass: effective.riskClass,
      matchedRuleId: effective.matchedRuleId,
      basis: verdict.basis
    }
  }

  /**
   * 中断（Ctrl+C）。
   *
   * 不做裁决：中断不是「新命令」，它是「把正在跑的东西停下」。
   * 拦中断既没有安全收益（它只会减少副作用），还会让用户觉得终端是坏的。
   */
  interrupt(sessionId: string): { ok: boolean; reason?: string } {
    const session = this.sessions.get(sessionId)
    if (!session || session.disposed) return { ok: false, reason: '会话不存在' }
    session.pty.write('\x03')
    return { ok: true }
  }

  resize(sessionId: string, cols: number, rows: number): { ok: boolean; cols: number; rows: number } {
    const session = this.sessions.get(sessionId)
    if (!session || session.disposed) return { ok: false, cols: 0, rows: 0 }
    const c = clampDim(cols, session.cols)
    const r = clampDim(rows, session.rows)
    if (c === session.cols && r === session.rows) return { ok: true, cols: c, rows: r }
    // 尺寸一起变：只改一个会让 shell 按旧宽高重新布局，输出错位（AC-5 要的就是这个不出错）
    session.pty.resize(c, r)
    session.cols = c
    session.rows = r
    return { ok: true, cols: c, rows: r }
  }

  close(sessionId: string): { ok: boolean } {
    const session = this.sessions.get(sessionId)
    if (!session) return { ok: false }
    this.kill(session)
    this.sessions.delete(sessionId)
    return { ok: true }
  }

  /** 进程退出时统一回收 —— 留下孤儿 shell 是比崩溃更难发现的问题 */
  disposeAll(): void {
    for (const [id, session] of this.sessions) {
      this.kill(session)
      this.sessions.delete(id)
    }
  }

  list(): TerminalSessionInfo[] {
    return [...this.sessions.values()].map((s) => ({
      sessionId: s.id,
      shell: s.shell,
      cwd: s.cwd,
      cols: s.cols,
      rows: s.rows,
      startedAt: s.startedAt,
      policyProfileVersion: s.policyProfileVersion
    }))
  }

  private kill(session: Session): void {
    if (session.disposed) return
    session.disposed = true
    try {
      // Windows 上 node-pty 的 kill 走 ConPTY 的 ClosePseudoConsole，
      // exitCode=0 不代表「正常结束」，只代表「我们要求它结束」
      session.pty.kill()
    } catch {
      /* 已经死了就算了 —— 清理阶段不该再抛 */
    }
    const timer = this.flushTimers.get(session.id)
    if (timer) {
      clearTimeout(timer)
      this.flushTimers.delete(session.id)
    }
    this.buffers.delete(session.id)
  }
}

/* ------------------------------------------------------------------ *
 * 平台细节
 * ------------------------------------------------------------------ */

function clampDim(v: number, fallback: number): number {
  if (!Number.isFinite(v)) return fallback
  const n = Math.floor(v)
  if (n < 2) return 2
  if (n > 500) return 500
  return n
}

/** 档位 → 策略版本号。会话记下创建时的版本，审计里能对上「当时是哪一档」 */
function tierVersion(tier: ApprovalTier): number {
  return tier === 'strict' ? 3 : tier === 'standard' ? 2 : 1
}

function pickShell(): { file: string; args: string[] } {
  if (process.platform === 'win32') {
    // 用 powershell 而不是 cmd：cmd 的控制台代码页在 ConPTY 下容易出现中文乱码，
    // 而乱码会让人误判「终端坏了」。`-NoLogo` 去掉启动横幅（否则每次开面板先看到版权声明）。
    return { file: 'powershell.exe', args: ['-NoLogo', '-NoProfile'] }
  }
  const shell = process.env['SHELL'] || '/bin/bash'
  return { file: shell, args: ['-i'] }
}

/**
 * 最小环境。
 *
 * 与 exec 同一条纪律：不整包继承 `process.env`（宿主可能带着一堆 token）。
 * 但终端比单条命令多一样需求 —— 用户得能用 `git` / `node`：
 * 所以保留 PATH 与少数几个定位用的变量，其余一律不带。
 */
function baseEnv(): Record<string, string> {
  const keep = [
    'PATH',
    'Path',
    'PATHEXT',
    'SystemRoot',
    'SYSTEMROOT',
    'windir',
    'TEMP',
    'TMP',
    'HOME',
    'USERPROFILE',
    'HOMEDRIVE',
    'HOMEPATH',
    'APPDATA',
    'LOCALAPPDATA',
    'COMSPEC',
    'LANG',
    'LC_ALL',
    'USER',
    'LOGNAME',
    'SHELL',
    'PWD'
  ]
  const out: Record<string, string> = {}
  for (const k of keep) {
    const v = process.env[k]
    if (typeof v === 'string') out[k] = v
  }
  return out
}
