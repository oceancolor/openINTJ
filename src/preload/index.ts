/**
 * preload —— 渲染进程的唯一特权入口
 *
 * contextIsolation = true / nodeIntegration = false，
 * 这里只暴露**窄接口**，不暴露 ipcRenderer 本体、不暴露 fs / child_process。
 */

import { contextBridge, ipcRenderer } from 'electron'
import type { ApprovalRequest, ApprovalTier } from '../shared/approval'
import type { EventEnvelope, Checkpoint } from '../shared/events'
import type { GoalContract } from '../shared/contract'
import type { LlmStatus } from '../shared/llm'
import type {
  BootInfo,
  CredentialInput,
  CredentialSaveResult,
  CredentialStatus,
  LlmTestResult,
  McpStatusSnapshot,
  RunLoopResult,
  StartRunResult,
  ToolInvokeOutcome
} from '../shared/ipc'
import type {
  TerminalDataPayload,
  TerminalExitPayload,
  TerminalSessionInfo,
  TerminalSubmitResult
} from '../shared/terminal'
import type {
  IdeLayoutRect,
  IdeStartParams,
  IdeStartResult,
  IdeStatus,
  IdeStopResult
} from '../shared/ide'
import type {
  AgentSession,
  ChatMessage,
  FileChange,
  PreviewTarget,
  SessionSendResult,
  TaskItem
} from '../shared/session'

const api = {
  getBootInfo: (): Promise<BootInfo> => ipcRenderer.invoke('api:bootInfo'),

  ping: (): Promise<{ main: string; agentHost: string; toolHost: string }> =>
    ipcRenderer.invoke('api:ping'),

  onEvent: (handler: (e: EventEnvelope) => void): (() => void) => {
    const listener = (_: unknown, payload: EventEnvelope): void => handler(payload)
    ipcRenderer.on('evt:agent', listener)
    return () => ipcRenderer.removeListener('evt:agent', listener)
  },

  onApprovalRequest: (handler: (r: ApprovalRequest) => void): (() => void) => {
    const listener = (_: unknown, payload: ApprovalRequest): void => handler(payload)
    ipcRenderer.on('evt:approval', listener)
    return () => ipcRenderer.removeListener('evt:approval', listener)
  },

  /** 撤回已失效的审批请求（run 已收尾）—— 卡片必须跟着消失，不能留僵尸 */
  onApprovalCancel: (handler: (p: { runId: string; requestIds: string[]; reason: string }) => void): (() => void) => {
    const listener = (_: unknown, payload: { runId: string; requestIds: string[]; reason: string }): void =>
      handler(payload)
    ipcRenderer.on('evt:approvalCancel', listener)
    return () => ipcRenderer.removeListener('evt:approvalCancel', listener)
  },

  onProcessStatus: (
    handler: (s: { service: string; status: string; reason?: string }) => void
  ): (() => void) => {
    const listener = (_: unknown, payload: { service: string; status: string; reason?: string }): void =>
      handler(payload)
    ipcRenderer.on('evt:status', listener)
    return () => ipcRenderer.removeListener('evt:status', listener)
  },

  /** 档位被非界面操作改动（启动恢复）时通知界面 */
  onTierChanged: (handler: (tier: ApprovalTier) => void): (() => void) => {
    const listener = (_: unknown, payload: ApprovalTier): void => handler(payload)
    ipcRenderer.on('evt:tier', listener)
    return () => ipcRenderer.removeListener('evt:tier', listener)
  },

  respondApproval: (requestId: string, approved: boolean, remember: boolean): Promise<void> =>
    ipcRenderer.invoke('api:approvalRespond', requestId, approved, remember),

  getApprovalTier: (): Promise<ApprovalTier> => ipcRenderer.invoke('api:getTier'),

  setApprovalTier: (tier: ApprovalTier): Promise<{ ok: boolean; reason?: string }> =>
    ipcRenderer.invoke('api:setTier', tier),

  listCheckpoints: (runId: string): Promise<Checkpoint[]> =>
    ipcRenderer.invoke('api:listCheckpoints', runId),

  vaultStatus: (): Promise<{ available: boolean; backend: string | null; policy: string }> =>
    ipcRenderer.invoke('api:vaultStatus'),

  /** 凭据中心状态（只含掩码，拿不到明文） */
  credentialStatus: (): Promise<CredentialStatus> => ipcRenderer.invoke('api:credentialStatus'),

  /** 保存凭据到保险箱；apiKey 留空表示沿用已有密钥 */
  credentialSave: (input: CredentialInput): Promise<CredentialSaveResult> =>
    ipcRenderer.invoke('api:credentialSave', input),

  /** 清除保险箱凭据 */
  credentialClear: (): Promise<{ ok: boolean; reason?: string; status?: CredentialStatus }> =>
    ipcRenderer.invoke('api:credentialClear'),

  /** 用当前生效凭据发一次最小请求 */
  llmTest: (): Promise<LlmTestResult> => ipcRenderer.invoke('api:llmTest'),

  /** MCP 连接状态（含每个 server 的失败原因） */
  mcpStatus: (): Promise<McpStatusSnapshot> => ipcRenderer.invoke('api:mcpStatus'),

  /** 重读 mcp.json 并重连 */
  mcpReload: (): Promise<McpStatusSnapshot> => ipcRenderer.invoke('api:mcpReload'),

  /** 启动一次 run（含模型起草的目标契约与接入状态） */
  startRun: (goal: string, opts?: { draft?: boolean }): Promise<StartRunResult> =>
    ipcRenderer.invoke('api:startRun', goal, opts),

  /** 发起一次工具调用（与模型驱动同一条裁决链，手动入口不是旁路） */
  invokeTool: (input: { runId: string; command: string }): Promise<ToolInvokeOutcome> =>
    ipcRenderer.invoke('api:invokeTool', input),

  /** 契约确认后，让模型自己驱动循环（模型也只能请求执行，不能自带批准） */
  startAgentLoop: (runId: string): Promise<RunLoopResult> =>
    ipcRenderer.invoke('api:runLoop', runId),

  /** 暂停：软中断，在步边界停下 */
  pauseRun: (runId: string): Promise<{ ok: boolean; reason?: string }> =>
    ipcRenderer.invoke('api:pauseRun', runId),

  /** 确认 / 否决目标契约（IC-07 硬闸门） */
  confirmContract: (input: {
    runId: string
    accepted: boolean
    contract?: GoalContract
  }): Promise<{ ok: boolean; state?: string; reason?: string }> =>
    ipcRenderer.invoke('api:confirmContract', input),

  /** LLM 接入状态（不含密钥） */
  llmStatus: (): Promise<LlmStatus> => ipcRenderer.invoke('api:llmStatus'),

  /* ---- 终端（US-15） ---- */

  /** 打开 PTY 会话（cwd 缺省为工作区根） */
  terminalOpen: (input: { cols: number; rows: number; cwd?: string }): Promise<TerminalSessionInfo> =>
    ipcRenderer.invoke('api:terminalOpen', input),

  /**
   * 提交一条完整命令。
   *
   * 注意参数是**整行**而非按键流：这是「执行前裁决」的前提。
   * 需要审批时主进程会自己弹卡、签票、重新提交，渲染层只是等一个结果。
   */
  terminalSubmit: (sessionId: string, command: string): Promise<TerminalSubmitResult> =>
    ipcRenderer.invoke('api:terminalSubmit', sessionId, command),

  terminalInterrupt: (sessionId: string): Promise<{ ok: boolean; reason?: string }> =>
    ipcRenderer.invoke('api:terminalInterrupt', sessionId),

  terminalResize: (sessionId: string, cols: number, rows: number): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('api:terminalResize', sessionId, cols, rows),

  terminalClose: (sessionId: string): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('api:terminalClose', sessionId),

  /** 输出流（高频；只推给当前屏幕，不落事件库） */
  onTerminalData: (handler: (p: TerminalDataPayload) => void): (() => void) => {
    const listener = (_: unknown, payload: TerminalDataPayload): void => handler(payload)
    ipcRenderer.on('evt:termData', listener)
    return () => ipcRenderer.removeListener('evt:termData', listener)
  },

  onTerminalExit: (handler: (p: TerminalExitPayload) => void): (() => void) => {
    const listener = (_: unknown, payload: TerminalExitPayload): void => handler(payload)
    ipcRenderer.on('evt:termExit', listener)
    return () => ipcRenderer.removeListener('evt:termExit', listener)
  },

  /* ---- 内嵌 IDE（F27 / D5） ---- */

  /** IDE 状态（未安装 / 已装未启 / 启动中 / 可用 / 失败 / 已停止） */
  ideStatus: (): Promise<IdeStatus> => ipcRenderer.invoke('api:ideStatus'),

  /** 拉起内嵌 IDE。失败时返回值里带可显示的原因，不抛异常 */
  ideStart: (input?: IdeStartParams): Promise<IdeStartResult> =>
    ipcRenderer.invoke('api:ideStart', input ?? {}),

  /** 关闭内嵌 IDE 并杀干净进程树 */
  ideStop: (): Promise<IdeStopResult> => ipcRenderer.invoke('api:ideStop'),

  /**
   * 上报工作台显示区域。传 null 表示切走了标签页，主进程会把 view 移出。
   * 尺寸变化时也走这条（窗口缩放 / 侧栏折叠）。
   */
  ideLayout: (rect: IdeLayoutRect | null): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('api:ideLayout', rect),

  onIdeStatus: (handler: (s: IdeStatus) => void): (() => void) => {
    const listener = (_: unknown, payload: IdeStatus): void => handler(payload)
    ipcRenderer.on('evt:ideStatus', listener)
    return () => ipcRenderer.removeListener('evt:ideStatus', listener)
  },

  /* ---- 会话（IDE 右栏：对话 / 任务 / 变更 / 预览） ---- */

  /** 会话列表 */
  sessionList: (): Promise<AgentSession[]> => ipcRenderer.invoke('api:sessionList'),

  /** 新建会话 */
  sessionCreate: (title?: string): Promise<AgentSession> => ipcRenderer.invoke('api:sessionCreate', title),

  /** 某会话的消息 */
  sessionMessages: (sessionId: string): Promise<ChatMessage[]> =>
    ipcRenderer.invoke('api:sessionMessages', sessionId),

  /**
   * 发一句话。
   *
   * 窄接口只收「会话 id + 正文」：什么算一轮、契约怎么起、循环怎么跑，
   * 全在 agent-host 里定。渲染层不持有这些规则，也就不可能与后端判断不一致。
   */
  sessionSend: (input: { sessionId?: string; text: string }): Promise<SessionSendResult> =>
    ipcRenderer.invoke('api:sessionSend', input),

  /** 确认 / 否决契约（IC-07 硬闸门） */
  sessionConfirmContract: (input: {
    sessionId: string
    accepted: boolean
    contract?: GoalContract
  }): Promise<{ ok: boolean; session?: AgentSession; reason?: string }> =>
    ipcRenderer.invoke('api:sessionConfirmContract', input),

  sessionTasks: (sessionId: string): Promise<TaskItem[]> =>
    ipcRenderer.invoke('api:sessionTasks', sessionId),

  sessionChanges: (sessionId: string): Promise<FileChange[]> =>
    ipcRenderer.invoke('api:sessionChanges', sessionId),

  sessionConfirmChange: (changeId: string): Promise<{ ok: boolean }> =>
    ipcRenderer.invoke('api:sessionConfirmChange', changeId),

  sessionPreview: (sessionId: string): Promise<PreviewTarget[]> =>
    ipcRenderer.invoke('api:sessionPreview', sessionId),

  /** 手动登记一个预览地址；只接受回环地址，非法地址由后端拒掉 */
  sessionAddPreview: (
    input: { sessionId: string; url: string; title?: string }
  ): Promise<{ ok: boolean; reason?: string; preview?: PreviewTarget }> =>
    ipcRenderer.invoke('api:sessionPreviewAdd', input),

  /** 订阅工作区文件变化 */
  onFileChanged: (handler: (c: FileChange) => void): (() => void) => {
    const listener = (_: unknown, payload: FileChange): void => handler(payload)
    ipcRenderer.on('evt:fileChanged', listener)
    return () => ipcRenderer.removeListener('evt:fileChanged', listener)
  },

  /** 用系统浏览器打开外部地址 */
  openExternal: (url: string): Promise<void> => ipcRenderer.invoke('api:openExternal', url)
}

contextBridge.exposeInMainWorld('openintj', api)

export type OpenIntjApi = typeof api
