/**
 * 跨进程通信契约
 *
 * 架构约束（《系统设计》§3.1、C8）：
 *   P-01 主进程 ↔ P-03 agent-host / P-04 tool-host 一律走 MessagePort 结构化克隆，
 *   **不开本地 HTTP 端口**。渲染进程通过 preload 的 contextBridge 暴露的窄接口访问，
 *   contextIsolation = true / nodeIntegration = false。
 */

import type { EventEnvelope, Checkpoint } from './events'
import type { GoalContract } from './contract'
import type { LlmProtocol, LlmStatus, TokenUsage } from './llm'
import type {
  ApprovalRequest,
  ApprovalTier,
  DecisionBasis,
  DecisionOutcome,
  RiskClass,
  RiskDecisionContext
} from './approval'
import type { ApprovalTicket } from './approval-token'
import type { McpServerStatus } from './mcp'
import type {
  TerminalDataPayload,
  TerminalExitPayload,
  TerminalSessionInfo,
  TerminalSubmitResult
} from './terminal'
import type {
  IdeLayoutRect,
  IdeStartParams,
  IdeStartResult,
  IdeStatus,
  IdeStopResult
} from './ide'

/* ------------------------------------------------------------------ *
 * 通用信封
 * ------------------------------------------------------------------ */

export type WireEnvelope =
  | { kind: 'req'; requestId: string; method: string; params?: unknown }
  | { kind: 'res'; requestId: string; ok: true; result?: unknown }
  | { kind: 'res'; requestId: string; ok: false; error: { code: string; message: string } }
  | { kind: 'evt'; channel: string; payload: unknown }

/* ------------------------------------------------------------------ *
 * 渲染进程 → 主进程（preload 暴露的窄接口）
 * ------------------------------------------------------------------ */

export interface RendererApi {
  /** 应用与运行时信息 */
  getBootInfo(): Promise<BootInfo>
  /** 订阅执行流事件（Action / Observation） */
  onEvent(handler: (e: EventEnvelope) => void): () => void
  /** 订阅审批请求 */
  onApprovalRequest(handler: (r: ApprovalRequest) => void): () => void
  /** 订阅审批撤回（run 收尾时清除已失效的卡片） */
  onApprovalCancel(handler: (p: { runId: string; requestIds: string[]; reason: string }) => void): () => void
  /** 订阅档位变更（启动时从库里恢复会走这条，界面要跟着变） */
  onTierChanged(handler: (tier: ApprovalTier) => void): () => void
  /** 订阅子进程状态变化（启动 / 在线 / 崩溃） */
  onProcessStatus(handler: (s: { service: string; status: string; reason?: string }) => void): () => void
  /** 应答审批 */
  respondApproval(requestId: string, approved: boolean, remember: boolean): Promise<void>
  /** 读取当前审批档位 */
  getApprovalTier(): Promise<ApprovalTier>
  /** 切换审批档位 */
  setApprovalTier(tier: ApprovalTier): Promise<{ ok: boolean; reason?: string }>
  /** 检查点列表 */
  listCheckpoints(runId: string): Promise<Checkpoint[]>
  /** 凭据保险箱可用性 —— 不可用时 policy = 'fail-stop'，拒绝写入明文 */
  vaultStatus(): Promise<{ available: boolean; backend: string | null; policy: string }>
  /** 凭据中心状态：当前生效来源、掩码后的密钥（绝不含明文） */
  credentialStatus(): Promise<CredentialStatus>
  /** 保存凭据到保险箱；apiKey 留空表示沿用保险箱里已有的那一把 */
  credentialSave(input: CredentialInput): Promise<CredentialSaveResult>
  /** 清除保险箱里的凭据（环境变量若仍在，会回落为 env 来源） */
  credentialClear(): Promise<{ ok: boolean; reason?: string }>
  /** 用当前生效凭据发一次最小请求，验证端点与密钥能用 */
  llmTest(): Promise<LlmTestResult>
  /** MCP 连接状态（含每个 server 的失败原因 —— 界面必须显示） */
  mcpStatus(): Promise<McpStatusSnapshot>
  /** 重读 mcp.json 并重连；返回重连后的状态 */
  mcpReload(): Promise<McpStatusSnapshot>
  /** 启动一次任务运行（硬闸门：先出契约，确认后才执行） */
  startRun(goal: string): Promise<StartRunResult>
  /** 契约确认后，让模型自己驱动「想一步、做一步」的循环 */
  startAgentLoop(runId: string): Promise<RunLoopResult>
  /** 暂停：软中断，在当前步结束后停下，不掐断正在跑的命令 */
  pauseRun(runId: string): Promise<{ ok: boolean; reason?: string }>
  /** 发起一次工具调用（与模型驱动同一条裁决链，手动入口不是旁路） */
  invokeTool(input: { runId: string; command: string }): Promise<ToolInvokeOutcome>

  /* ---- 终端（US-15） ---- */

  /** 打开一个 PTY 会话（cwd 缺省为工作区根） */
  terminalOpen(input: { cols: number; rows: number; cwd?: string }): Promise<TerminalSessionInfo>
  /**
   * 提交一条命令。返回 `need-approval` 时主进程会自动弹审批卡，
   * 批准后由主进程带着票据再提交一次 —— 渲染层不必自己管票据。
   */
  terminalSubmit(sessionId: string, command: string): Promise<TerminalSubmitResult>
  /** 中断（Ctrl+C），不裁决 */
  terminalInterrupt(sessionId: string): Promise<{ ok: boolean; reason?: string }>
  terminalResize(sessionId: string, cols: number, rows: number): Promise<{ ok: boolean }>
  terminalClose(sessionId: string): Promise<{ ok: boolean }>
  /** 订阅终端输出流 */
  onTerminalData(handler: (p: TerminalDataPayload) => void): () => void
  /** 订阅终端会话退出 */
  onTerminalExit(handler: (p: TerminalExitPayload) => void): () => void

  /* ---- 内嵌 IDE（F27 / D5） ---- */

  /** IDE 状态：未安装 / 已装未启 / 启动中 / 可用 / 失败 / 已停止 */
  ideStatus(): Promise<IdeStatus>
  /** 拉起内嵌 IDE。返回最新状态，失败时带可显示的原因 */
  ideStart(input?: IdeStartParams): Promise<IdeStartResult>
  /** 关闭内嵌 IDE 并杀干净进程树 */
  ideStop(): Promise<IdeStopResult>
  /**
   * 告诉主进程「工作台该显示在哪」。
   *
   * 为什么不由渲染层自己嵌：iframe 第三方工作台要放松 CSP，而放松 CSP 是对
   * **整个**渲染进程生效的 —— 为了一个面板把主界面的安全基线降掉，不划算。
   * 由主进程用一个独立 session 的 WebContentsView 覆盖上去，两边互不干扰。
   */
  ideLayout(rect: IdeLayoutRect | null): Promise<{ ok: boolean }>
  /** 订阅 IDE 状态变化（启动中 → 可用 / 失败） */
  onIdeStatus(handler: (s: IdeStatus) => void): () => void

  /** 健康探针：验证三进程链路连通 */
  ping(): Promise<{ main: string; agentHost: string; toolHost: string }>
}

/* ------------------------------------------------------------------ *
 * 凭据中心（IC-06）
 * ------------------------------------------------------------------ */

/** 凭据来源：保险箱里的显式配置优先，环境变量作为兜底 */
export type CredentialSource = 'vault' | 'env' | 'none'

export interface CredentialStatus {
  vaultAvailable: boolean
  vaultBackend: string | null
  /** 保险箱不可用时的策略，恒为 'fail-stop' */
  vaultPolicy: 'accept' | 'fail-stop'
  /** 当前实际生效的是哪一份 */
  source: CredentialSource
  /**
   * 保险箱里是否真有一份凭据。
   *
   * 必须与 source 分开：env 来源时 maskedKey 也有值（那是环境变量里那把 key 的掩码），
   * 若只按 maskedKey 判断，界面会误报「已保存」并让「清除凭据」可点 ——
   * 而它清的是保险箱，清不掉环境变量。
   */
  vaultHasCredential: boolean
  /** 掩码后的密钥（如 sk-…4tEbM）—— 明文永不出主进程 */
  maskedKey: string | null
  protocol: LlmProtocol | null
  baseUrl: string | null
  model: string | null
  disableThinking: boolean
  /** 保险箱文件存在但解不开时的原因；正常为 null */
  vaultError: string | null
  /** 环境变量缺哪几项（用于提示「不配保险箱也能用」） */
  envMissing: string[]
  savedAt: number | null
}

export interface CredentialInput {
  protocol: LlmProtocol
  baseUrl: string
  model: string
  /** 留空表示「沿用保险箱里已有的密钥」，改模型名时不必重打 40 位 key */
  apiKey?: string
  disableThinking?: boolean
}

export interface CredentialSaveResult {
  ok: boolean
  reason?: string
  /** 保存后的状态，省一次往返 */
  status?: CredentialStatus
}

export interface LlmTestResult {
  ok: boolean
  latencyMs: number
  model?: string
  protocol?: LlmProtocol
  /** 模型实际回了什么（截断），用于确认「它真的通了」而不是只连上了 */
  sample?: string
  reason?: string
}

/**
 * agent-host → P-01 的反向调用：拉当前生效凭据。
 *
 * 明文密钥走 MessagePort（进程间共享内存），不落盘、不进事件流、不回渲染层。
 * 之所以「按需拉」而不是「启动即推送」：设置改完立刻生效，也不需要失效缓存。
 */
export const CREDENTIAL_METHODS = {
  get: 'credential.get'
} as const

/** 启动一次 run 的返回：含契约草稿与模型接入状态 */export interface StartRunResult {
  runId: string
  state: string
  /** 模型起草成功的契约；未配置/解析失败时为空，需用户手写或先配模型 */
  contract?: GoalContract
  llm: LlmStatus
}

/** 一次工具调用的最终结果（含被拒的情况） */
export interface ToolInvokeOutcome {  ok: boolean
  /** 被拒或出错时的原因；ok=true 时为空 */
  reason?: string
  /** tool-host 给出的审计上下文（六要素），被拒时同样有值 */
  risk?: RiskDecisionContext
  result?: Omit<ToolExecResult, 'risk'>
}

/**
 * Agent 循环的返回。
 *
 * outcome 只有四种，没有「大概算成功」这种第五种：
 *   delivered —— 模型调了 finish 并给出交付说明
 *   failed    —— 模型调用失败 / 没调 finish 就停 / 单步异常
 *   max-steps —— 撞到步数上限
 *   paused    —— 用户中途暂停
 */
export interface RunLoopResult {
  outcome: 'delivered' | 'failed' | 'paused' | 'max-steps'
  steps: number
  summary?: string
  satisfied?: { criterion: string; evidence: string }[]
  unmet?: { criterion: string; reason: string }[]
  reason?: string
  /** 全循环累计 token 用量 —— 成本核算与预算告警的依据 */
  usage: TokenUsage
}

export interface BootInfo {
  version: string
  platform: NodeJS.Platform
  /** 沙箱能力：Windows 侧 isolated = false（C1，不得承诺等价 macOS） */
  sandbox: {
    isolated: boolean
    primitive: 'seatbelt' | 'landlock+seccomp' | 'windows-degraded'
    /** 降级提示文案 —— UI 必须原样展示，不可关闭 */
    notice: string | null
  }
  approval: {
    tier: ApprovalTier
    /** Windows 上恒为 true：宽松档禁用 */
    autonomousBlocked: boolean
  }
  processes: { agentHost: 'online' | 'offline'; toolHost: 'online' | 'offline' }
}

/* ------------------------------------------------------------------ *
 * 主进程 → P-03 agent-host
 * ------------------------------------------------------------------ */

export const AGENT_METHODS = {
  ping: 'agent.ping',
  startRun: 'agent.run.start',
  /** 确认或否决目标契约 —— IC-07 硬闸门，未确认不得执行工具 */
  confirmContract: 'agent.contract.confirm',
  /** 当前 LLM 接入状态（不含密钥） */
  llmStatus: 'agent.llm.status',
  /** 用当前生效凭据发一次最小请求，验端点与密钥是否真能用 */
  llmTest: 'agent.llm.test',
  pauseRun: 'agent.run.pause',
  resumeRun: 'agent.run.resume',
  /**
   * 启动 Agent 主循环：模型自己想、自己请求执行，但每次执行都要过 tool-host 的裁决。
   * 前置条件是契约已确认（IC-07），否则直接拒。
   */
  runLoop: 'agent.run.loop',
  /**
   * 本次循环会发给模型的工具清单。
   *
   * 存在的理由不是调试：模型「能看见哪些工具」是用户该知道的事 ——
   * 配了一个 MCP server 却因为没连上而模型根本看不到，用户只会觉得模型变笨了。
   */
  toolCatalog: 'agent.tool.catalog',
  listCheckpoints: 'agent.checkpoint.list',
  getTier: 'agent.tier.get',
  setTier: 'agent.tier.set',
  approvalDecision: 'agent.approval.decide',
  /**
   * 应用设置的读写（D9）。
   *
   * 为什么走 agent-host 而不是主进程自己开一个 SQLite 连接：
   * 事件库已经由 agent-host 独占持有，再加一个写连接只会给 WAL 添锁。
   * 主进程是「决定设置值」的地方，agent-host 是「把值存住」的地方，分工清晰。
   */
  settingGet: 'agent.setting.get',
  settingSet: 'agent.setting.set',
  /**
   * 发起一次工具调用。
   * LLM 接入后由编排循环调用；在此之前由控制台「手动发起」按钮驱动 ——
   * 两条路径走的是同一条裁决链，手动入口不是旁路。
   */
  invokeTool: 'agent.tool.invoke'
} as const

export type AgentMethod = (typeof AGENT_METHODS)[keyof typeof AGENT_METHODS]

/* ------------------------------------------------------------------ *
 * 主进程 → P-04 tool-host
 * ------------------------------------------------------------------ */

export const TOOL_METHODS = {
  ping: 'tool.ping',
  /**
   * 启动握手：主进程下发**仅 tool-host 可见**的票据密钥与工作区根目录。
   * 密钥绝不经 agent-host 中转 —— 否则 agent-host 可自行签票，闸门即失效。
   */
  init: 'tool.init',
  /** 审批档位由主进程推送（权威来源），不接受调用方在请求里自带 */
  setTier: 'tool.tier.set',
  /**
   * 预判：只给结论，不执行、不落审计。
   * 调用方据此决定「要不要去要授权」，但**要不到授权就跑不了** ——
   * 预判结果不是凭据（exec 会重新算一遍并自行裁决）。
   */
  precheck: 'tool.precheck',
  /** 经策略层裁决后执行命令 */
  exec: 'tool.exec',
  /** 只做风险分级，不执行（供策略层预判） */
  classify: 'tool.classify',
  /**
   * 记录一次「未执行」的裁决（用户拒绝 / 审批超时）。
   *
   * 为什么需要单独一条：被拒时根本不会调用 exec，而审计六要素对拒绝同样成立 ——
   * 「用户拒绝过什么」是审计里最该留的一类记录，不能只留在事件流里。
   * 由 tool-host 自己重新分级后落库，调用方无法指定风险等级。
   */
  recordDenial: 'tool.audit.deny',
  /** 释放某任务的会话级授权 */
  revokeSessionGrants: 'tool.grants.revoke',

  /* ---- MCP（D6） ---- */
  /** 重读 mcp.json 并重连所有 server */
  mcpReload: 'tool.mcp.reload',
  /** 连接状态（含失败原因 —— 界面必须显示） */
  mcpStatus: 'tool.mcp.status',
  /** 可用工具清单（含 inputSchema，供 Agent 组装 tool spec） */
  mcpList: 'tool.mcp.list',
  /** MCP 调用的预判（分级 + 是否需要票据），不执行、不落审计 */
  mcpPrecheck: 'tool.mcp.precheck',
  /** 执行 MCP 工具调用。与 shell 调用同一条裁决与审计链 */
  mcpCall: 'tool.mcp.call',
  /**
   * 记录一次「未发起」的 MCP 裁决（用户拒绝 / 审批超时）。
   *
   * 不能复用 shell 的 recordDenial：那条路径会用命令字符串跑 shell 分类器，
   * 而 `mcp:server:tool {...}` 这串东西对 shell 分类器毫无意义，
   * 算出来的等级是错的 —— 审计里写一个错误的风险等级，比不写更糟。
   */
  mcpDenial: 'tool.mcp.deny',
  /**
   * 读某任务最近的审计记录。
   *
   * 说明给将来的自己：这不是调试口子，是「审计可被核对」的必需能力 ——
   * 读不出来的审计等于没有审计。
   */
  auditRecent: 'tool.audit.recent',

  /* ---- 终端（US-15 / F29） ---- */
  /**
   * 创建 PTY 会话。
   *
   * 注意这里**不做命令裁决** —— 拉起一个空 shell 不执行用户命令，
   * 风险来自「往里写什么」，那一关在 terminalInput 里。
   * 但 cwd 仍限在工作区根内：让 shell 在工作区外启动等于一开始就让掉边界。
   */
  terminalCreate: 'tool.term.create',
  /**
   * 提交一条**完整命令**并裁决。
   *
   * 参数是整行而不是按键流 —— 这是「终端不绕过策略层」（US-15 AC-2）的前提：
   * 只有掌握命令边界，才可能做到「执行前」而不是「执行后」拦截。
   */
  terminalInput: 'tool.term.input',
  /** 中断（Ctrl+C）。不裁决：中断只减少副作用，不产生新命令 */
  terminalInterrupt: 'tool.term.interrupt',
  terminalResize: 'tool.term.resize',
  terminalClose: 'tool.term.close',
  terminalList: 'tool.term.list'
} as const

export interface RecordDenialParams {
  runId: string
  command: string
  cwd: string
  /** 拒绝原因分类：用户拒绝 / 超时 / 其它策略性拒绝 */
  basis: DecisionBasis
}

export type ToolMethod = (typeof TOOL_METHODS)[keyof typeof TOOL_METHODS]

export interface PrecheckParams {
  runId: string
  command: string
  cwd: string
}

export interface PrecheckResult {
  riskClass: RiskClass
  matchedRuleId: string
  reason: string
  /** 按当前档位应有的处置（只读结论，不可作为执行凭据） */
  outcome: DecisionOutcome
  /** 是否必须先拿到票据才能执行 */
  needsTicket: boolean
  /** 该任务是否已有此风险等级的会话授权 */
  hasSessionGrant: boolean
}

export interface ToolInitParams {
  /** HMAC 密钥，仅主进程与 tool-host 各持一份 */
  ticketSecret: string
  /** 工作区根目录 —— 所有写入判定的边界 */
  workspaceRoot: string
  /** 数据目录 —— tool-host 在此写审批审计（不由调用方代写） */
  dataDir: string
  /** MCP 配置文件路径；不传则不启用 MCP */
  mcpConfigPath?: string
}

/**
 * 执行请求。
 *
 * 注意这里**没有** `risk` 字段：风险分级与裁决都在 tool-host 内部完成。
 * 调用方只能「请求执行某条命令」，不能「声明这条命令已获批准」。
 */
export interface ToolExecParams {
  runId: string
  agentId: string
  command: string
  cwd: string
  /**
   * 逐次审批（PER_CALL_PROMPT）路径下，由主进程签发的一次性票据；
   * AUTO_APPROVE 不需要，SESSION_GRANT 首次需要、之后凭会话授权放行。
   */
  approvalTicket?: ApprovalTicket
  timeoutMs?: number
}

export interface ToolExecResult {
  exitCode: number | null
  stdout: string
  stderr: string
  durationMs: number
  truncated: boolean
  /** tool-host 自己得出的审计六要素 —— 调用方只读，落库时原样记录 */
  risk: RiskDecisionContext
}

/* ---- MCP（D6） ---- */

export interface McpPrecheckParams {
  runId: string
  server: string
  tool: string
}

export interface McpCallParams {
  runId: string
  agentId: string
  server: string
  tool: string
  args: unknown
  approvalTicket?: ApprovalTicket
}

export interface McpDenialParams {
  runId: string
  server: string
  tool: string
  args: unknown
  basis: DecisionBasis
}

export interface McpCallOutcome {
  ok: boolean
  /** 给模型看的结果文本 */
  text: string
  isError: boolean
  durationMs: number
  risk: RiskDecisionContext
}

/**
 * 一次循环实际会发给模型的工具清单。
 *
 * `specs` 就是原样交给 LLM 的那份（内置 + MCP），`mcp` 额外标注来源，
 * 便于界面按 server 分组显示。两者由同一个函数产出，不存在「界面显示有、
 * 模型却看不到」这种对不上的情况。
 */
export interface ToolCatalog {
  builtin: string[]
  mcp: { name: string; server: string; trust: string }[]
  total: number
}

/**
 * MCP 状态快照 —— 设置页与探针共用。
 *
 * `configError` 与「没有配置」必须分开：前者是用户的配置写坏了，需要告诉他；
 * 后者是正常初始态（没装过 MCP server），提示一句「去写个 mcp.json」即可。
 */
export interface McpStatusSnapshot {
  enabled: boolean
  /** 首次加载 / 重载进行中 —— 界面应显示「连接中」而不是「0 个工具」 */
  loading: boolean
  configPath: string | null
  configError: string | null
  servers: McpServerStatus[]
}

/* ------------------------------------------------------------------ *
 * 事件通道名
 * ------------------------------------------------------------------ */

export const EVENT_CHANNELS = {
  agentEvent: 'agent.event',
  approvalRequest: 'approval.request',
  /**
   * 撤回审批请求。
   *
   * 循环收尾时如果还挂着未应答的请求，那些卡片已经没有任何人在等 ——
   * 留在界面上只会让用户点了没反应，还以为自己被忽略了。
   * 显示出去的东西必须是活的，死了要收回去。
   */
  approvalCancel: 'approval.cancel',
  /**
   * 档位被**非界面操作**改动时通知渲染层（目前只有「启动时从库里恢复」一条）。
   *
   * 没有这条通道的话，恢复会比界面首屏慢半步，用户看到的是默认档位 ——
   * 于是他会以为「我上次设的严格档没保存」，然后重新点一遍。
   */
  tierChanged: 'tier.changed',
  processStatus: 'process.status'
} as const
