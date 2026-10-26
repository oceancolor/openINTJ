/**
 * Agent 会话契约（IDE 右栏：对话 / 任务 / 变更 / 预览）
 *
 * ── 为什么要有「会话」这一层 ──────────────────────────────────────
 * 原来控制台的交互是「一次性」的：输入框里下一句话 → startRun → 确认契约 → 跑循环。
 * 跑完就散了，没有「上一轮说到哪」这回事。
 *
 * 但 IDE 右栏要的是**对话**：用户的每一句都留在上下文里，模型能引用前面说过的话，
 * 任务列表与文件变化也按会话归堆。所以会话是「多轮的载体」，run 是会话里的一次执行：
 *
 *   会话（session） 1 ──── n 轮消息（message）
 *        │
 *        └── 0 或 1 个 run（契约确认后才真正起执行）
 *
 * ── 状态为什么分六种 ──────────────────────────────────────────────
 * `empty` 是「还没说第一句话」，`contract-pending` 是「契约在等你点头」。
 * 这两个不是 TaskRunState 里的状态，但界面必须区分：
 * 前者该显示空白引导，后者该显示待确认的契约卡。
 * 少了这两个，界面只能靠「有没有 contract」去猜，而猜错的表现是按钮点了没反应。
 */

/** 会话状态 —— 在 run 状态机之外多了「空会话」与「待确认」两个界面态 */
export type SessionState =
  /** 新建但还没发过消息 */
  | 'empty'
  /** 契约已起草，等用户确认（IC-07 硬闸门） */
  | 'contract-pending'
  /** 循环执行中 */
  | 'running'
  /** 已暂停（软中断） */
  | 'paused'
  /** 模型调了 finish 并给出交付说明 */
  | 'delivered'
  /** 失败 / 撞上限 */
  | 'failed'

export interface AgentSession {
  sessionId: string
  /** 会话标题：默认取第一句话的前 24 个字符 */
  title: string
  createdAt: number
  updatedAt: number
  /**
   * 绑定的 run。第一轮对话之前为 null —— 因为起 run 意味着调模型起草契约，
   * 空会话不该白烧一次 token。
   */
  runId: string | null
  state: SessionState
}

export type ChatRole = 'user' | 'agent' | 'system'

/**
 * 消息种类。
 *
 * 为什么要分 kind 而不是只留 text：对话里有三种「不是一句话」的东西 ——
 * 契约草稿（要能点确认）、审批请求（要能点允许）、执行结果（要能看证据）。
 * 把它们压成 text 的结果是界面只能**显示**，而这三样东西天生是要**被操作**的。
 */
export type ChatKind =
  | 'text'
  /** 契约草稿（payload: { contract, llm }）—— 待用户确认 */
  | 'contract-draft'
  /** 契约确认结果（payload: { accepted }） */
  | 'contract-result'
  /** 审批请求（payload: ApprovalRequest）—— 卡片是否消失由 ApprovalDecided 决定 */
  | 'approval'
  /** 执行结果（payload: RunLoopResult 摘要） */
  | 'delivery'
  /** 出错或被拒（payload: { message }） */
  | 'error'
  /** 过程中的进展（payload: { stage, text }） */
  | 'progress'

export interface ChatMessage {
  messageId: string
  sessionId: string
  role: ChatRole
  kind: ChatKind
  /** 正文（落库前已过脱敏管线） */
  text: string
  at: number
  /** 关联事件位点（契约 / 审批 / 交付才有），用于「这条消息对应哪一步」 */
  atSeq: number | null
  /** kind !== 'text' 时的结构化负载 */
  payload: unknown
}

/** 任务列表里的一行 —— 一次 run 一行，不是一步一行（界面不该被 12 步淹没） */
export interface TaskItem {
  taskId: string
  runId: string
  title: string
  /** TaskRunState 字符串，界面按它着色 */
  state: string
  steps: number
  at: number
}

/** 工作区里的文件变化 —— 由 tool-host 的监听器上报 */
export interface FileChange {
  changeId: string
  runId: string
  /** 相对工作区根的路径，分隔符统一为 `/` */
  path: string
  kind: 'added' | 'modified' | 'deleted'
  at: number
  /**
   * 用户是否已确认过这条变化。
   *
   * 确认是**签收**不是**回滚**：它表示「我看过这个改动了」，
   * 不改变任何文件内容，也不免除这次执行已经落下的审计记录。
   */
  confirmed: boolean
}

/** 预览目标 —— 没有就不亮「预览」签 */
export interface PreviewTarget {
  previewId: string
  runId: string
  url: string
  title: string
  at: number
}

/**
 * 会话相关方法名。
 *
 * 与 AGENT_METHODS 分开成一张表，是因为它们服务的对象不同：
 * AGENT_METHODS 里的是「跑任务」，这里的是「跟任务说话」。
 * 分开之后，将来要按会话做权限/审计，不用在一张 30 个方法的表里翻。
 */
export const SESSION_METHODS = {
  list: 'session.list',
  create: 'session.create',
  messages: 'session.messages',
  /** 发一句话。首轮起草契约，之后是追加指令 */
  send: 'session.send',
  /** 确认 / 否决当前会话的契约（IC-07 硬闸门） */
  confirmContract: 'session.contract.confirm',
  /** 任务列表（该会话的 run） */
  tasks: 'session.tasks',
  /** 文件变化 */
  changes: 'session.changes',
  /**
   * 登记一条文件变化。
   *
   * 由主进程转交：监听器在 tool-host（那里才有 fs），而 changelog 归 agent-host
   * 持有（会话状态都在那边）。主进程只做搬运，不自己记 —— 记在两处迟早对不上。
   */
  noteChange: 'session.change.note',
  /** 用户确认某条文件变化 */
  confirmChange: 'session.change.confirm',
  /** 预览目标 */
  preview: 'session.preview',
  /** 手动登记一个预览地址（只接受回环地址） */
  previewAdd: 'session.preview.add'
} as const

export interface SessionSendResult {
  session: AgentSession
  messages: ChatMessage[]
  /** 这一轮停在哪：等确认 / 跑完了 / 没配模型 / 出错 */
  phase: 'contract-pending' | 'finished' | 'awaiting-model' | 'error'
}

/**
 * 文件变化事件通道（tool-host → 主进程 → 渲染层）。
 *
 * 与终端输出同源：**不进事件库**。原因是内容不可控（用户可能正在改 .env），
 * 而这条路径没有过脱敏。只上报「哪个文件动了、怎么动的」——
 * 判断这个改动是不是自己想要的，是人该做的事。
 */
export const FILE_CHANGE_CHANNELS = {
  changed: 'file.changed'
} as const
