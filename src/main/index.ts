/**
 * P-01 主进程入口
 *
 * 职责（《系统设计》§3.1.1.1）：
 *   - 窗口生命周期、IPC 路由、safeStorage 凭据保险箱、子进程监护
 *   - 不承载 Agent 主循环（那在 P-03），不直接执行命令（那在 P-04）
 *
 * 安全基线：contextIsolation = true / nodeIntegration = false / 无 remote / CSP 由渲染层注入
 */

import { app, BrowserWindow, desktopCapturer, ipcMain, safeStorage, shell, WebContentsView } from 'electron'
import { randomBytes, randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { WorkerHost } from './process-manager'
import {
  clearVault,
  credentialFilePath,
  maskKey,
  normalizeCredential,
  readVault,
  toLlmConfig,
  writeVault
} from './credential-store'
import {
  AGENT_METHODS,
  CREDENTIAL_METHODS,
  EVENT_CHANNELS,
  TOOL_METHODS,
  type BootInfo,
  type CredentialInput,
  type CredentialSource,
  type CredentialStatus,
  type LlmTestResult,
  type McpCallOutcome,
  type McpStatusSnapshot,
  type ToolExecResult,
  type ToolInitParams
} from '../shared/ipc'
import {
  isMcpToolName,
  mcpCommandString,
  parseMcpToolName,
  type McpServerStatus,
  type McpToolInfo
} from '../shared/mcp'
import { parseProtocol, readLlmConfig, type LlmConfig } from '../shared/llm'
import { applyRedaction } from '../shared/redaction'
import { homedir, networkInterfaces } from 'node:os'
import { DEFAULT_TIER, isTierAllowedOnPlatform, type ApprovalTier } from '../shared/approval'
import { SETTING_KEYS } from '../shared/schema'
import { issueTicket, type ApprovalTicket } from '../shared/approval-token'
import { TERMINAL_CHANNELS, terminalRunId, type TerminalSubmitResult } from '../shared/terminal'
import { VscodiumHost } from './vscodium'
import { FORBIDDEN_TRADEMARKS } from '../shared/ide'
import type { IdeLayoutRect, IdeStartParams, IdeStartResult, IdeStatus, IdeStopResult } from '../shared/ide'
import {
  FILE_CHANGE_CHANNELS,
  SESSION_METHODS,
  type AgentSession,
  type ChatMessage,
  type FileChange,
  type PreviewTarget,
  type SessionSendResult,
  type TaskItem
} from '../shared/session'
import type { Checkpoint } from '../shared/events'
import type { ApprovalRequest } from '../shared/approval'

let mainWindow: BrowserWindow | null = null
let agentHost: WorkerHost | null = null
let toolHost: WorkerHost | null = null

/** 当前审批档位。持久化落 t_app_setting，此为首版内存态，重启回落默认 */
let currentTier: ApprovalTier = DEFAULT_TIER

/**
 * 最近一次起过的 run。
 *
 * tool-host 的监听器不知道「当前在跑哪个 run」（那是编排层的概念），
 * 所以文件变化上报到主进程后，由主进程补上 runId —— 补在这里是因为
 * 主进程是唯一同时认识「执行」与「界面」两层的地方。
 */
let lastRunId: string | null = null

/**
 * 票据密钥 —— 全应用唯一副本的「签发侧」。
 *
 * 只经 tool-host 专用端口下发（见 initToolHost）。**绝不经 agent-host 转发**，
 * 否则 agent-host 能自行签票，裁决权收回就白做了（验收报告 D10）。
 */
const TICKET_SECRET = randomBytes(32).toString('hex')

/**
 * 工作区根目录：所有写入判定的边界，启动即固定，运行期不可被调用方改写。
 *
 * 打包态不能沿用 cwd：快捷方式启动时 cwd 是安装目录（C:\Program Files\openINTJ），
 * 那里既不该被当成工作区（只读、且混着应用自身文件），越界判定也会跟着失真。
 * 所以打包后默认落到用户目录下，并顺手建出来 —— 空的也比指向安装目录好。
 */
function resolveWorkspaceRoot(): string {
  const fromEnv = process.env['OPENINTJ_WORKSPACE']
  if (fromEnv) return fromEnv
  if (!app.isPackaged) return process.cwd()
  const dir = join(homedir(), 'openINTJ')
  try {
    mkdirSync(dir, { recursive: true })
  } catch {
    /* 建不出来也照原样返回：启动不该因为目录失败而中断 */
  }
  return dir
}

const WORKSPACE_ROOT = resolveWorkspaceRoot()

/** 待审批请求 → 供签发票据时取命令原文与风险等级 */
const pendingApprovals = new Map<string, ApprovalRequest>()

/**
 * 终端审批的等待者。
 *
 * 为什么不复用 `pendingApprovals`：那张表的应答会转给 agent-host（那边有人在 await），
 * 而终端的发起方是主进程自己。混在一起的话，终端点「允许」会去唤醒一个根本不存在的
 * agent-host 等待者，结果两边都不动 —— 而且不报错。
 *
 * 复用同一张审批卡（同一个事件通道），但**各留一份等待者**，是这个取舍的核心。
 */
const pendingTerminalApprovals = new Map<
  string,
  { req: ApprovalRequest; settle: (d: { approved: boolean; remember: boolean }) => void }
>()

/** 审批等待上限，与 N-04b 拍板的 120s 一致：超时按拒绝处理，绝不「超时视为同意」 */
const TERMINAL_APPROVAL_TIMEOUT_MS = 120_000

/* ------------------------------------------------------------------ *
 * 内嵌 IDE（F27 / D5）
 * ------------------------------------------------------------------ */

/**
 * IDE 宿主。惰性创建 —— 没点「启动」就不该有一个 node 进程在后台跑。
 */
let ideHost: VscodiumHost | null = null

/**
 * 承载工作台的 WebContentsView。
 *
 * 为什么是独立 view 而不是渲染层里的 iframe：iframe 第三方工作台要放松 CSP，
 * 而 CSP 是**整个渲染进程**一份的 —— 为一个面板放松整条基线不划算。
 * 独立 view 还自带一份独立 session（见 ensureIdeView 的 partition），
 * 工作台的 cookie / localStorage 与我们主界面天然隔离。
 */
let ideView: WebContentsView | null = null

/** 最后一次上报的显示区；view 重建或窗口缩放后要能照原样贴回去 */
let ideRect: IdeLayoutRect | null = null

function getIdeHost(): VscodiumHost {
  if (!ideHost) {
    ideHost = new VscodiumHost((s) => {
      mainWindow?.webContents.send('evt:ideStatus', s)
    })
  }
  return ideHost
}

/** 状态推给渲染层。用的是 IDE 自己的通道，不与终端/审批混用 */
function pushIdeStatus(s: IdeStatus): void {
  mainWindow?.webContents.send('evt:ideStatus', s)
}

/**
 * 建（或复用）工作台 view。
 *
 * partition 独立是刻意的：工作台是第三方代码，它的存储不该和主界面共享，
 * 反过来也一样 —— 我们的凭据相关的东西不该落到它能读到的地方。
 */
function ensureIdeView(): WebContentsView | null {
  if (!mainWindow) return null
  if (ideView) return ideView

  const view = new WebContentsView({
    webPreferences: {
      // 第三方工作台：不给 node 能力，且独立分区
      partition: 'persist:ide',
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webviewTag: false
    }
  })

  // 工作台里的外链一律走系统浏览器 —— 在应用内打开会绕过我们的边界
  view.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })

  ideView = view
  return view
}

/**
 * 把 view 贴到指定矩形；传 null（或尺寸退化为 0）就隐藏它。
 *
 * 用 setVisible 而不是反复 add/removeChildView：反复摘挂会让工作台的
 * WebContents 反复失焦/获焦，正在编辑的文件可能被触发一次多余的保存提示。
 * 隐藏态的 view 不接收输入事件，所以不会出现「在控制台敲命令、
 * 按键被背后的工作台吃掉」这种情况。
 */
function applyIdeLayout(rect: IdeLayoutRect | null): void {
  ideRect = rect
  if (!mainWindow || !ideView) return

  if (!mainWindow.contentView.children.includes(ideView)) {
    mainWindow.contentView.addChildView(ideView)
  }

  const visible = Boolean(rect) && (rect!.width >= 1 || rect!.height >= 1)
  ideView.setVisible(visible)
  if (!visible || !rect) return
  ideView.setBounds({
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    width: Math.round(rect.width),
    height: Math.round(rect.height)
  })
}

/** 关闭工作台时隐藏 view —— 停止 IDE 与切走标签页共用这一条 */
function detachIdeView(): void {
  ideView?.setVisible(false)
}

/**
 * 夹取显示区数值。
 *
 * 渲染层是我们的代码，但它终究是 Web 内容 —— 一次 layout 抖动算出 NaN
 * 或负值，setBounds 会直接抛错。夹取而不是拒绝，是为了让「尺寸算差了一点」
 * 不至于变成「面板整个黑掉」。
 */
function sanitizeRect(rect: IdeLayoutRect | null): IdeLayoutRect | null {
  if (!rect || typeof rect !== 'object') return null
  const clamp = (v: unknown, max: number): number => {
    const n = Number(v)
    return Number.isFinite(n) ? Math.max(0, Math.min(max, n)) : 0
  }
  return {
    x: clamp(rect.x, 100_000),
    y: clamp(rect.y, 100_000),
    width: clamp(rect.width, 100_000),
    height: clamp(rect.height, 100_000)
  }
}

/* ------------------------------------------------------------------ *
 * 平台沙箱能力（C1：Windows 不得承诺等价 macOS）
 * ------------------------------------------------------------------ */

function detectSandbox(): BootInfo['sandbox'] {
  switch (process.platform) {
    case 'darwin':
      return { isolated: true, primitive: 'seatbelt', notice: null }
    case 'linux':
      return { isolated: true, primitive: 'landlock+seccomp', notice: null }
    case 'win32':
      return {
        isolated: false,
        primitive: 'windows-degraded',
        // C1 必显文案：四处（徽标 / 首次启用浮层 / 关于页 / 安装说明）共用，且不提供关闭开关
        notice:
          'Windows 侧采用「审批在环 + 命令白名单 + 工作区根目录约束」的组合替代 OS 级沙箱。' +
          '隔离强度弱于 macOS，不等价。world-writable 目录下的写入无法被阻止。'
      }
    default:
      return {
        isolated: false,
        primitive: 'windows-degraded',
        notice: '当前平台无已验证的 OS 级隔离原语，已按降级模式运行。'
      }
  }
}

/* ------------------------------------------------------------------ *
 * 窗口
 * ------------------------------------------------------------------ */

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 760,
    show: false,
    backgroundColor: '#030014', // 深空紫黑底，避免白闪
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false, // preload 需要 node 能力（仅暴露窄接口）
      webviewTag: false
    }
  })

  mainWindow.once('ready-to-show', () => mainWindow?.show())

  // 外链一律走系统浏览器，不在应用内打开
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })

  if (process.env['ELECTRON_RENDERER_URL']) {
    void mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    void mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })

  // 渲染层异常必须可见，不静默失败
  mainWindow.webContents.on('render-process-gone', (_e, details) => {
    console.error(`[renderer] 渲染进程退出：${details.reason}（exitCode=${details.exitCode}）`)
  })
  mainWindow.webContents.on('did-fail-load', (_e, code, desc) => {
    console.error(`[renderer] 页面加载失败：${code} ${desc}`)
  })
}

/* ------------------------------------------------------------------ *
 * 凭据解析（IC-06）
 * ------------------------------------------------------------------ */

interface ResolvedCredential {
  source: CredentialSource
  config: LlmConfig | null
  /** 保险箱里是否真有一份 —— 与 source 分开，见 CredentialStatus 里的说明 */
  vaultHasCredential: boolean
  maskedKey: string | null
  savedAt: number | null
  /** 保险箱文件在但解不开时的原因；界面必须显示，不能当「没配过」 */
  vaultError: string | null
  envMissing: string[]
}

/**
 * 决定「此刻到底用哪份凭据」—— 全进程只在这里判断一次。
 *
 * 优先级：保险箱 > 环境变量。理由：保险箱里的是用户在界面里显式填的，
 * 是更近的意图；环境变量更像「运维下发的默认值」。
 *
 * 保险箱解不开时的处置值得单说：如果环境变量可用，就**回落**到 env，
 * 但把 vaultError 一并带出去让界面告警。不回落会让一份坏掉的密文
 * 把本来能用的环境变量也拖死；而静默不告警则会让用户以为配置丢了。
 */
function resolveCredential(): ResolvedCredential {
  const dataDir = app.getPath('userData')
  const vault = readVault(dataDir)
  const env = readLlmConfig(process.env)
  const envMissing = env.configured ? [] : env.missing

  if (vault.kind === 'ok') {
    return {
      source: 'vault',
      config: toLlmConfig(vault.credential),
      vaultHasCredential: true,
      maskedKey: maskKey(vault.credential.apiKey),
      savedAt: vault.credential.savedAt,
      vaultError: null,
      envMissing
    }
  }

  const vaultError = vault.kind === 'unreadable' ? vault.reason : null

  if (env.configured) {
    return {
      source: 'env',
      config: env.config,
      vaultHasCredential: false,
      maskedKey: maskKey(env.config.apiKey),
      savedAt: null,
      vaultError,
      envMissing: []
    }
  }

  return {
    source: 'none',
    config: null,
    vaultHasCredential: false,
    maskedKey: null,
    savedAt: null,
    vaultError,
    envMissing
  }
}

function buildCredentialStatus(): CredentialStatus {
  const r = resolveCredential()
  const available = safeStorage.isEncryptionAvailable()
  return {
    vaultAvailable: available,
    vaultBackend: available ? (safeStorage.getSelectedStorageBackend?.() ?? 'os-native') : null,
    vaultPolicy: available ? 'accept' : 'fail-stop',
    source: r.source,
    vaultHasCredential: r.vaultHasCredential,
    maskedKey: r.maskedKey,
    protocol: r.config?.protocol ?? null,
    baseUrl: r.config?.baseUrl ?? null,
    model: r.config?.model ?? null,
    disableThinking: r.config?.disableThinking ?? true,
    vaultError: r.vaultError,
    envMissing: r.envMissing,
    savedAt: r.savedAt
  }
}

/* ------------------------------------------------------------------ *
 * 子进程启动与事件桥接
 * ------------------------------------------------------------------ */

function startWorkers(): void {
  // 数据目录由主进程统一下发，子进程不自己猜路径
  const dataDir = app.getPath('userData')

  agentHost = new WorkerHost({
    modulePath: join(__dirname, 'agent-host.js'),
    serviceName: 'openintj-agent-host',
    requestTimeoutMs: 30_000,
    env: { OPENINTJ_DATA_DIR: dataDir }
  })

  toolHost = new WorkerHost({
    modulePath: join(__dirname, 'tool-host.js'),
    serviceName: 'openintj-tool-host',
    requestTimeoutMs: 120_000 // 命令执行允许更长
  })

  // agent-host 的事件推到渲染进程（执行流）
  agentHost.onEvent((channel, payload) => {
    if (channel === EVENT_CHANNELS.agentEvent) {
      mainWindow?.webContents.send('evt:agent', payload)
    } else if (channel === EVENT_CHANNELS.approvalRequest) {
      // 记下来：用户在界面上点「允许」时，主进程据此签发绑定命令原文的票据
      const req = payload as ApprovalRequest
      if (req?.requestId) pendingApprovals.set(req.requestId, req)
      mainWindow?.webContents.send('evt:approval', payload)
    } else if (channel === EVENT_CHANNELS.approvalCancel) {
      // 循环收尾时收回僵尸卡片：主进程的镜像也要清，否则票据还能被签出来
      const { requestIds } = payload as { requestIds?: string[] }
      for (const id of requestIds ?? []) pendingApprovals.delete(id)
      mainWindow?.webContents.send('evt:approvalCancel', payload)
    }
  })

  agentHost.onStatus((status, reason) => {
    mainWindow?.webContents.send('evt:status', { service: 'agentHost', status, reason })
  })
  toolHost.onStatus((status, reason) => {
    mainWindow?.webContents.send('evt:status', { service: 'toolHost', status, reason })
  })
  /**
   * tool-host 的主动推送（目前只有终端输出）。
   *
   * 输出流**不进** `pendingApprovals` / 事件库，直接透传给渲染层：
   * 它是高频、短生命周期、只对当前屏幕有意义的数据，落盘只会让事件表变成日志。
   */
  toolHost.onEvent((channel, payload) => {
    if (channel === TERMINAL_CHANNELS.data) {
      mainWindow?.webContents.send('evt:termData', payload)
    } else if (channel === TERMINAL_CHANNELS.exit) {
      mainWindow?.webContents.send('evt:termExit', payload)
    } else if (channel === FILE_CHANGE_CHANNELS.changed) {
      // 文件变化：先转交 agent-host 登记（会话状态都在那边），再通知界面刷新。
      // 顺序不能反 —— 反了的话界面先拉，拉到的是登记前的旧列表。
      const notice = payload as { changeId: string; path: string; kind: string; at: number }
      void agentHost
        ?.request(SESSION_METHODS.noteChange, {
          runId: lastRunId ?? '-',
          path: notice.path,
          kind: notice.kind
        })
        .catch((err: Error) => {
          console.warn(`[main] 文件变化登记失败：${err.message}`)
        })
      mainWindow?.webContents.send('evt:fileChanged', payload)
    }
  })

  /**
   * P-03 反向调用转发 —— 不开 P-03 ↔ P-04 直连通道。
   *
   * 好处是「谁调用了谁」在主进程只有一个入口，便于审计与限流；
   * 代价是多一跳，但都在进程内 MessagePort 上，可忽略。
   */
  agentHost.handle(TOOL_METHODS.precheck, async (params) => {
    if (!toolHost) throw new Error('tool-host 未就绪')
    return await toolHost.request(TOOL_METHODS.precheck, params, 15_000)
  })
  agentHost.handle(TOOL_METHODS.exec, async (params) => {
    if (!toolHost) throw new Error('tool-host 未就绪')
    // 命令执行可能很久，超时给足；真正的执行超时由 tool-host 自己管
    return await toolHost.request(TOOL_METHODS.exec, params, 180_000)
  })
  agentHost.handle(TOOL_METHODS.recordDenial, async (params) => {
    if (!toolHost) throw new Error('tool-host 未就绪')
    return await toolHost.request(TOOL_METHODS.recordDenial, params, 15_000)
  })

  /* ---- MCP（D6）：与 shell 调用同一条链，只是换了载荷形状 ---- */
  agentHost.handle(TOOL_METHODS.mcpPrecheck, async (params) => {
    if (!toolHost) throw new Error('tool-host 未就绪')
    return await toolHost.request(TOOL_METHODS.mcpPrecheck, params, 15_000)
  })
  agentHost.handle(TOOL_METHODS.mcpCall, async (params) => {
    if (!toolHost) throw new Error('tool-host 未就绪')
    // MCP 工具自带 60s 上限，链路超时给足即可
    return await toolHost.request(TOOL_METHODS.mcpCall, params, 120_000)
  })
  agentHost.handle(TOOL_METHODS.mcpList, async (params) => {
    if (!toolHost) throw new Error('tool-host 未就绪')
    // 首次加载可能还在连（npx 现装包很慢），给 35s 覆盖 tool-host 内部的 30s 等待
    return await toolHost.request(TOOL_METHODS.mcpList, params, 35_000)
  })

  /**
   * agent-host 反向拉凭据。
   *
   * 这是**唯一**把明文密钥送进子进程的通道，走 MessagePort（共享内存），
   * 不落盘、不进事件流、不回渲染层。之所以让子进程按需来拉而不是启动时推一次：
   * 设置一改立刻生效，也不需要任何缓存失效逻辑。
   */
  agentHost.handle(CREDENTIAL_METHODS.get, async () => {
    const r = resolveCredential()
    if (!r.config) {
      return {
        configured: false,
        missing: r.vaultError ? [`保险箱凭据无法解密：${r.vaultError}`] : r.envMissing
      }
    }
    return { configured: true, config: r.config }
  })

  agentHost.start()
  toolHost.start()
}

/* ------------------------------------------------------------------ *
 * IPC 路由（渲染进程 ↔ 主进程 ↔ 子进程）
 * ------------------------------------------------------------------ */

/**
 * 跑一条终端命令 —— 含「等用户审批」的完整往返。
 *
 * 这条函数是整个终端设计里唯一允许拿票据的地方。顺序刻意写成
 * 「先问 tool-host 要不要票 → 要票才弹卡 → 拿到票再提交一次」，
 * 而不是「先弹卡再执行」：分级必须由 tool-host 做，主进程只管问人与签票。
 * 反过来（主进程自己判断风险）就等于把裁决权搬回主进程，D10 白做了。
 */
async function runTerminalCommand(sessionId: string, command: string): Promise<TerminalSubmitResult> {
  if (!toolHost) throw new Error('tool-host 未就绪，终端不可用')

  const first = await toolHost.request<TerminalSubmitResult>(
    TOOL_METHODS.terminalInput,
    { sessionId, command },
    30_000
  )
  if (first.kind !== 'need-approval') return first

  const requestId = `term-${randomUUID()}`
  const req: ApprovalRequest = {
    requestId,
    runId: terminalRunId(sessionId),
    // agentId 在终端场景没有意义，但不能留空（界面按它分组泳道）
    agentId: 'terminal',
    parallelLaneId: 'terminal',
    command: first.command,
    cwd: WORKSPACE_ROOT,
    riskClass: first.riskClass,
    allowAsyncPrompt: false,
    at: Date.now()
  }

  const decision = await new Promise<{ approved: boolean; remember: boolean }>((resolve) => {
    const timer = setTimeout(() => {
      // 超时按拒绝处理。绝不「超时视为同意」—— 那是把等待变成默认许可。
      pendingTerminalApprovals.delete(requestId)
      mainWindow?.webContents.send('evt:approvalCancel', {
        runId: req.runId,
        requestIds: [requestId],
        reason: '审批超时（120s），按拒绝处理'
      })
      resolve({ approved: false, remember: false })
    }, TERMINAL_APPROVAL_TIMEOUT_MS)

    pendingTerminalApprovals.set(requestId, {
      req,
      settle: (d) => {
        clearTimeout(timer)
        resolve(d)
      }
    })
    mainWindow?.webContents.send('evt:approval', req)
  })

  if (!decision.approved) {
    // 被拒也要留痕：终端里「我试过但被拦了」是最该能查的一类记录
    await toolHost
      .request(
        TOOL_METHODS.recordDenial,
        {
          runId: req.runId,
          command: first.command,
          cwd: WORKSPACE_ROOT,
          basis: 'user-deny'
        },
        15_000
      )
      .catch((e: Error) => console.warn(`[main] 终端拒绝记录失败：${e.message}`))

    return {
      kind: 'denied',
      command: first.command,
      riskClass: first.riskClass,
      matchedRuleId: first.matchedRuleId,
      reason: '用户拒绝执行'
    }
  }

  // 票据由主进程签，绑定的是「终端会话 + 命令原文 + 风险等级」；
  // tool-host 会重新分级并验票，两边的命令字符串必须逐字节一致。
  const ticket: ApprovalTicket = issueTicket(TICKET_SECRET, {
    runId: req.runId,
    command: first.command,
    riskClass: first.riskClass,
    scope: decision.remember ? 'session' : 'once'
  })

  return await toolHost.request<TerminalSubmitResult>(
    TOOL_METHODS.terminalInput,
    { sessionId, command: first.command, approvalTicket: ticket },
    30_000
  )
}

function registerIpc(): void {
  ipcMain.handle('api:bootInfo', (): BootInfo => {
    return {
      version: app.getVersion(),
      platform: process.platform,
      sandbox: detectSandbox(),
      approval: {
        tier: currentTier,
        autonomousBlocked: !isTierAllowedOnPlatform('autonomous', process.platform)
      },
      processes: {
        agentHost: agentHost?.status === 'online' ? 'online' : 'offline',
        toolHost: toolHost?.status === 'online' ? 'online' : 'offline'
      }
    }
  })

  ipcMain.handle('api:ping', async () => {
    const [agentHostId, toolHostId] = await Promise.all([
      agentHost?.request<string>(AGENT_METHODS.ping).catch((e: Error) => `error: ${e.message}`),
      toolHost?.request<string>(TOOL_METHODS.ping).catch((e: Error) => `error: ${e.message}`)
    ])
    return {
      main: 'online',
      agentHost: agentHostId ?? 'offline',
      toolHost: toolHostId ?? 'offline'
    }
  })

  ipcMain.handle('api:getTier', () => currentTier)

  ipcMain.handle('api:setTier', async (_e, tier: ApprovalTier) => {
    if (!isTierAllowedOnPlatform(tier, process.platform)) {
      return {
        ok: false,
        reason:
          'Windows 侧「宽松」档默认禁用：该平台无 OS 级隔离兜底，审批是唯一防线。' +
          '如需启用须显式风险确认并二次确认。'
      }
    }
    const prev = currentTier
    currentTier = tier
    // 收紧档位要求撤销全部会话级授权（正在执行的命令不中断）
    const tightened = tierWeight(tier) < tierWeight(prev)
    if (tightened && toolHost) {
      await toolHost.request(TOOL_METHODS.revokeSessionGrants, { scope: 'all' }).catch(() => {
        /* 撤销失败不阻断档位切换，但需留痕（首版仅日志） */
        console.warn('[main] 撤销会话级授权失败')
      })
    }
    // 档位是裁决输入，必须同步给裁决方（tool-host 不接受请求里自带的档位）
    await toolHost?.request(TOOL_METHODS.setTier, { tier }).catch(() => {
      console.warn('[main] 向 tool-host 推送档位失败，其裁决将沿用上一档位')
    })
    // 通知 agent-host 档位已切换（落 PolicyTierSwitched 审计事件）
    await agentHost?.request(AGENT_METHODS.setTier, { tier, prev }).catch(() => undefined)

    // 落库放在最后：先把运行时改到位再持久化。
    // 反过来的话，落库成功而推送失败，用户会以为「设了严格档」，
    // 实际裁决还在用旧档位 —— 这种「以为设了」比「提示设置失败」危险得多。
    await persistTier(tier)
    return { ok: true }
  })

  ipcMain.handle('api:approvalRespond', async (_e, requestId: string, approved: boolean, remember: boolean) => {
    /**
     * 终端审批先分流。
     *
     * 终端的发起方是主进程自己（runTerminalCommand 在 await），
     * 若走下面那条 agent-host 分支，会去唤醒一个不存在的等待者：
     * 卡片消失、没有任何反应、也不报错 —— 最难查的那种坏法。
     */
    const term = pendingTerminalApprovals.get(requestId)
    if (term) {
      pendingTerminalApprovals.delete(requestId)
      term.settle({ approved, remember })
      return { ok: true, ticketIssued: approved }
    }

    const req = pendingApprovals.get(requestId)
    pendingApprovals.delete(requestId)

    // 「已批准」不是一句话，是一张绑定 runId + 命令哈希的票据；tool-host 只认票据
    let ticket: ApprovalTicket | undefined
    if (approved && req) {
      ticket = issueTicket(TICKET_SECRET, {
        runId: req.runId,
        command: req.command,
        riskClass: req.riskClass,
        scope: remember ? 'session' : 'once'
      })
    } else if (approved && !req) {
      console.warn(`[main] 审批请求 ${requestId} 已不在待办中，不签发票据（拒绝执行）`)
    }

    await agentHost?.request(AGENT_METHODS.approvalDecision, {
      requestId,
      approved,
      remember,
      ticket
    })
    return { ok: true, ticketIssued: Boolean(ticket) }
  })

  ipcMain.handle('api:listCheckpoints', async (_e, runId: string): Promise<Checkpoint[]> => {
    const res = await agentHost?.request<Checkpoint[]>(AGENT_METHODS.listCheckpoints, { runId })
    return res ?? []
  })

  // 启动一次 run —— 首版只推进到「契约草稿已生成（待模型）」，
  // 不伪造模型输出（N2：不接 Mock）
  ipcMain.handle('api:startRun', async (_e, goal: string, opts?: { draft?: boolean }) => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    // 模型起草可能要一两分钟，超时给足；未配置模型时是秒回
    const res = await agentHost.request<{ runId: string }>(
      AGENT_METHODS.startRun,
      { goal, draft: opts?.draft },
      180_000
    )
    if (res?.runId) lastRunId = res.runId
    return res
  })

  // 发起一次工具调用：整条链可能包含「等用户点允许」，超时给足
  ipcMain.handle('api:invokeTool', async (_e, input: { runId: string; command: string }) => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    return await agentHost.request(AGENT_METHODS.invokeTool, input, 300_000)
  })

  // Agent 主循环：最多 12 步，每步可能等审批（120s），所以整体超时按上限给足
  ipcMain.handle('api:runLoop', async (_e, runId: string) => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    return await agentHost.request(AGENT_METHODS.runLoop, { runId }, 1_800_000)
  })

  // 暂停：软中断，别在命令执行到一半时掐断
  ipcMain.handle('api:pauseRun', async (_e, runId: string) => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    return await agentHost.request(AGENT_METHODS.pauseRun, { runId }, 10_000)
  })

  // 确认目标契约（IC-07 硬闸门）；模型起草可能很慢，超时给足
  ipcMain.handle('api:confirmContract', async (_e, input: { runId: string; accepted: boolean; contract?: unknown }) => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    return await agentHost.request(AGENT_METHODS.confirmContract, input, 30_000)
  })

  /* ---- 会话（IDE 右栏：对话 / 任务 / 变更 / 预览） ---- */

  /** 会话列表 */
  ipcMain.handle('api:sessionList', async (): Promise<AgentSession[]> => {
    if (!agentHost) return []
    return (await agentHost.request<AgentSession[]>(SESSION_METHODS.list, undefined, 10_000)) ?? []
  })

  /** 新建会话 */
  ipcMain.handle('api:sessionCreate', async (_e, title?: string): Promise<AgentSession> => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    return await agentHost.request<AgentSession>(SESSION_METHODS.create, { title }, 10_000)
  })

  /** 某会话的消息 */
  ipcMain.handle('api:sessionMessages', async (_e, sessionId: string): Promise<ChatMessage[]> => {
    if (!agentHost) return []
    return (await agentHost.request<ChatMessage[]>(SESSION_METHODS.messages, { sessionId }, 10_000)) ?? []
  })

  /**
   * 发一句话。
   *
   * 超时给足 1800s：一轮循环最多 12 步，每步可能等审批 120s。
   * 这不是「放任它跑」—— 界面上有中断入口（暂停），
   * 而静默超时返回会让用户以为「说完了」，实际还在跑。
   */
  ipcMain.handle('api:sessionSend', async (_e, input: { sessionId?: string; text: string }) => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    return await agentHost.request<SessionSendResult>(SESSION_METHODS.send, input, 1_800_000)
  })

  /** 确认 / 否决契约 —— 走的就是 IC-07 那条闸门，不另开一条 */
  ipcMain.handle('api:sessionConfirmContract', async (_e, input: { sessionId: string; accepted: boolean; contract?: unknown }) => {
    if (!agentHost) throw new Error('agent-host 尚未就绪')
    return await agentHost.request(SESSION_METHODS.confirmContract, input, 30_000)
  })

  /** 任务列表 */
  ipcMain.handle('api:sessionTasks', async (_e, sessionId: string): Promise<TaskItem[]> => {
    if (!agentHost) return []
    return (await agentHost.request<TaskItem[]>(SESSION_METHODS.tasks, { sessionId }, 10_000)) ?? []
  })

  /** 文件变化 */
  ipcMain.handle('api:sessionChanges', async (_e, sessionId: string): Promise<FileChange[]> => {
    if (!agentHost) return []
    return (await agentHost.request<FileChange[]>(SESSION_METHODS.changes, { sessionId }, 10_000)) ?? []
  })

  /** 用户确认某条文件变化 */
  ipcMain.handle('api:sessionConfirmChange', async (_e, changeId: string) => {
    if (!agentHost) return { ok: false }
    return await agentHost.request<{ ok: boolean }>(SESSION_METHODS.confirmChange, { changeId }, 10_000)
  })

  /** 预览目标 */
  ipcMain.handle('api:sessionPreview', async (_e, sessionId: string): Promise<PreviewTarget[]> => {
    if (!agentHost) return []
    return (await agentHost.request<PreviewTarget[]>(SESSION_METHODS.preview, { sessionId }, 10_000)) ?? []
  })

  /** 手动登记预览地址 */
  ipcMain.handle(
    'api:sessionPreviewAdd',
    async (_e, input: { sessionId: string; url: string; title?: string }) => {
      if (!agentHost) return { ok: false, reason: 'agent-host 尚未就绪' }
      return await agentHost.request(SESSION_METHODS.previewAdd, input, 10_000)
    }
  )

  /**
   * 外链一律走系统浏览器。
   *
   * 应用内不承载外部内容：预览页面里跑什么脚本我们控制不了，
   * 让它进主界面等于把 CSP 撕一个口子。
   */
  ipcMain.handle('api:openExternal', async (_e, url: string): Promise<void> => {
    await shell.openExternal(url)
  })

  // LLM 接入状态（不含密钥，只报 host 与模型名）
  ipcMain.handle('api:llmStatus', async () => {
    if (!agentHost) {
      return { configured: false, protocol: null, model: null, baseUrlHost: null, missing: ['agent-host 未就绪'] }
    }
    return await agentHost.request(AGENT_METHODS.llmStatus, undefined, 10_000)
  })

  // 凭据中心：读状态（只给掩码，明文永不出主进程）
  ipcMain.handle('api:credentialStatus', (): CredentialStatus => buildCredentialStatus())

  // 凭据中心：保存进保险箱
  ipcMain.handle('api:credentialSave', async (_e, input: CredentialInput) => {
    // fail-stop：没有系统密钥链就不写，绝不降级成明文
    if (!safeStorage.isEncryptionAvailable()) {
      return { ok: false, reason: '系统密钥链不可用，拒绝保存（不会降级成明文存储）' }
    }

    const dataDir = app.getPath('userData')
    const existing = readVault(dataDir)

    // 密钥留空 = 沿用已有那一把，这样改模型名不必重打 40 位 key
    const prevKey = existing.kind === 'ok' ? existing.credential.apiKey : ''
    const apiKey = (input.apiKey ?? '').trim() || prevKey
    if (!apiKey) return { ok: false, reason: '首次保存必须填写 API Key' }

    const credential = normalizeCredential({ ...input, apiKey, savedAt: Date.now() })
    if (!credential) return { ok: false, reason: '配置不完整，或 Base URL 不是 http(s) 地址' }

    const written = writeVault(dataDir, credential)
    if (!written.ok) return { ok: false, reason: written.reason }

    // 配置变了，让渲染层重新拉一次模型状态（顶栏显示的就是它）
    return { ok: true, status: buildCredentialStatus() }
  })

  // 凭据中心：清除保险箱凭据（环境变量若仍在，会回落为 env 来源）
  ipcMain.handle('api:credentialClear', () => {
    const result = clearVault(app.getPath('userData'))
    if (!result.ok) return result
    return { ok: true, status: buildCredentialStatus() }
  })

  // 测试连接：交给 agent-host 用真实协议实现发一次最小请求，不在 main 里重写一遍协议
  ipcMain.handle('api:llmTest', async (): Promise<LlmTestResult> => {
    if (!agentHost) return { ok: false, latencyMs: 0, reason: 'agent-host 未就绪' }
    return await agentHost.request<LlmTestResult>(AGENT_METHODS.llmTest, undefined, 45_000)
  })

  /* ---- 终端（US-15） ---- */

  ipcMain.handle('api:terminalOpen', async (_e, input: { cols: number; rows: number; cwd?: string }) => {
    if (!toolHost) throw new Error('tool-host 未就绪，终端不可用')
    return await toolHost.request<{ sessionId: string }>(TOOL_METHODS.terminalCreate, input, 20_000)
  })

  /**
   * 终端命令提交 —— 会在这里等「用户点允许」。
   *
   * 超时给足 240s：一次审批最多等 120s，加上命令本身可能要跑一会儿。
   * 但**不能无限等**：无限等会让界面看起来像卡死，而用户并不知道自己在等什么。
   */
  ipcMain.handle('api:terminalSubmit', async (_e, sessionId: string, command: string) => {
    if (!toolHost) throw new Error('tool-host 未就绪，终端不可用')
    return await runTerminalCommand(sessionId, command)
  })

  ipcMain.handle('api:terminalInterrupt', async (_e, sessionId: string) => {
    if (!toolHost) return { ok: false, reason: 'tool-host 未就绪' }
    return await toolHost.request<{ ok: boolean; reason?: string }>(
      TOOL_METHODS.terminalInterrupt,
      { sessionId },
      10_000
    )
  })

  ipcMain.handle('api:terminalResize', async (_e, sessionId: string, cols: number, rows: number) => {
    if (!toolHost) return { ok: false }
    return await toolHost.request<{ ok: boolean }>(
      TOOL_METHODS.terminalResize,
      { sessionId, cols, rows },
      10_000
    )
  })

  ipcMain.handle('api:terminalClose', async (_e, sessionId: string) => {
    if (!toolHost) return { ok: false }
    return await toolHost.request<{ ok: boolean }>(TOOL_METHODS.terminalClose, { sessionId }, 10_000)
  })

  /* ---- 内嵌 IDE（F27 / D5） ---- */

  ipcMain.handle('api:ideStatus', (): IdeStatus => getIdeHost().status())

  /**
   * 拉起 IDE 并加载工作台。
   *
   * 失败不抛异常 —— 「IDE 起不来」是要显示在面板上的状态，不是要炸掉
   * 整个界面的错误。抛异常会让渲染层只能 catch 到一个 message，
   * 连失败原因都拿不全。
   */
  ipcMain.handle('api:ideStart', async (_e, input: IdeStartParams = {}): Promise<IdeStartResult> => {
    const host = getIdeHost()
    const folder = typeof input?.workspaceFolder === 'string' && input.workspaceFolder
      ? input.workspaceFolder
      : WORKSPACE_ROOT

    const status = await host.start({ workspaceFolder: folder })
    if (status.phase === 'ready') {
      const view = ensureIdeView()
      const url = host.urlWithToken()
      if (view && url) {
        // 加载带令牌的地址：服务端会 302 并把 vscode-tkn 种进这个 view 的 session
        await view.webContents.loadURL(url).catch((e: Error) => {
          console.error(`[main] 工作台加载失败：${e.message}`)
        })
        applyIdeLayout(ideRect)
      }
    }
    pushIdeStatus(status)
    return { ok: status.phase === 'ready', status }
  })

  ipcMain.handle('api:ideStop', async (): Promise<IdeStopResult> => {
    const status = await getIdeHost().stop()
    detachIdeView()
    pushIdeStatus(status)
    return { ok: true, status }
  })

  /**
   * 渲染层上报工作台显示区。
   *
   * 数值要夹一下：渲染层是「我们自己的代码但仍是 Web 内容」，
   * 传一个 NaN 或 1e9 进来会让 setBounds 抛错或把窗口拖垮。
   * 夹取而不是拒绝 —— 尺寸算稍有偏差不该让面板黑屏。
   */
  ipcMain.handle('api:ideLayout', (_e, rect: IdeLayoutRect | null): { ok: boolean } => {
    applyIdeLayout(sanitizeRect(rect))
    return { ok: true }
  })

  /* ---- MCP（D6） ---- */

  /**
   * MCP 状态。tool-host 没起来时也要能答 —— 否则设置页会因为这个面板
   * 抛异常而整页报错，用户连「MCP 为什么不可用」都看不到。
   */
  ipcMain.handle('api:mcpStatus', async (): Promise<McpStatusSnapshot> => {
    if (!toolHost) {
      return {
        enabled: false,
        loading: false,
        configPath: null,
        configError: 'tool-host 未就绪，MCP 不可用',
        servers: []
      }
    }
    return await toolHost.request<McpStatusSnapshot>(TOOL_METHODS.mcpStatus, undefined, 10_000)
  })

  // 重读 mcp.json 并重连。可能很慢（要重新拉起所有 server 进程），超时给足
  ipcMain.handle('api:mcpReload', async (): Promise<McpStatusSnapshot> => {
    if (!toolHost) {
      return {
        enabled: false,
        loading: false,
        configPath: null,
        configError: 'tool-host 未就绪，MCP 不可用',
        servers: []
      }
    }
    const res = await toolHost.request<{
      configPath: string
      configError: string | null
      servers: McpStatusSnapshot['servers']
    }>(TOOL_METHODS.mcpReload, {}, 120_000)
    return {
      enabled: true,
      loading: false,
      configPath: res.configPath,
      configError: res.configError,
      servers: res.servers
    }
  })

  // 凭据保险箱可用性探针 —— 不可用时**必须 fail-stop，绝不回退明文文件存储**
  ipcMain.handle('api:vaultStatus', () => {
    const available = safeStorage.isEncryptionAvailable()
    return {
      available,
      backend: available ? safeStorage.getSelectedStorageBackend?.() ?? 'os-native' : null,
      // 明确对外语义：不可用即拒绝写入
      policy: available ? 'accept' : 'fail-stop'
    }
  })
}

/** 档位权重：数值越大越宽松，用于判定切换方向 */
function tierWeight(t: ApprovalTier): number {
  return t === 'strict' ? 0 : t === 'standard' ? 1 : 2
}

/* ------------------------------------------------------------------ *
 * 档位持久化（D9）
 * ------------------------------------------------------------------ */

function isApprovalTier(v: unknown): v is ApprovalTier {
  return v === 'strict' || v === 'standard' || v === 'autonomous'
}

/** 存住当前档位。失败只告警 —— 存不下不该让「切档位」这件事失败 */
async function persistTier(tier: ApprovalTier): Promise<void> {
  await agentHost
    ?.request(AGENT_METHODS.settingSet, { key: SETTING_KEYS.approvalTier, value: tier }, 5_000)
    .catch((e: Error) => console.warn(`[main] 档位落库失败（重启后会回到默认档）：${e.message}`))
}

/**
 * 启动时恢复档位。
 *
 * 三处必须一起改，少一处就会「界面显示严格档、实际按标准档裁决」：
 *   ① 本进程的 currentTier（bootInfo 与签票时的风险判断都读它）
 *   ② tool-host（真正的裁决方）
 *   ③ 渲染层（用户看到的那三个按钮）
 * 所以恢复完成后主动推一条 tierChanged，而不是等界面来问 —— 界面可能早就问过了。
 */
async function restoreTier(): Promise<void> {
  if (!agentHost) return
  try {
    const res = await agentHost.request<{ value: string | null }>(
      AGENT_METHODS.settingGet,
      { key: SETTING_KEYS.approvalTier },
      5_000
    )
    const saved = res.value
    if (!saved) return // 从没设过，用默认档

    if (!isApprovalTier(saved)) {
      console.warn(`[main] 忽略无法识别的档位设置「${saved}」，回落默认档 ${currentTier}`)
      return
    }
    if (!isTierAllowedOnPlatform(saved, process.platform)) {
      console.warn(`[main] 已保存的档位 ${saved} 在本平台不允许，回落默认档 ${currentTier}`)
      return
    }
    if (saved === currentTier) return

    const prev = currentTier
    currentTier = saved
    await toolHost?.request(TOOL_METHODS.setTier, { tier: saved }).catch((e: Error) => {
      console.warn(`[main] 恢复档位时推送 tool-host 失败：${e.message}`)
    })
    await agentHost.request(AGENT_METHODS.setTier, { tier: saved, prev }).catch(() => undefined)
    mainWindow?.webContents.send('evt:tier', saved)
    console.log(`[main] 档位已从库里恢复：${prev} → ${saved}`)
  } catch (e) {
    console.warn(`[main] 恢复档位失败，沿用默认档：${(e as Error).message}`)
  }
}

/**
 * tool-host 初始化握手：下发票据密钥与工作区根目录，并推送当前档位。
 *
 * 必须重试 —— 端口建立与子进程绑定 handler 之间存在时间差，
 * 一次性发送可能丢消息；丢弃初始化会让 tool-host 处于 fail-closed（拒绝一切执行），
 * 表现为「什么都跑不动」，比崩溃更难查。
 */
async function initToolHost(maxAttempts = 10, intervalMs = 300): Promise<boolean> {
  if (!toolHost) return false
  const payload: ToolInitParams = {
    ticketSecret: TICKET_SECRET,
    workspaceRoot: WORKSPACE_ROOT,
    dataDir: app.getPath('userData'),
    // MCP 配置放 userData 下，与事件库、凭据保险箱同处一地：
    // 既不在工作区里（不然会被 Agent 自己改掉），也不散落在系统各处
    mcpConfigPath: join(app.getPath('userData'), 'mcp.json')
  }
  for (let i = 1; i <= maxAttempts; i++) {
    try {
      await toolHost.request(TOOL_METHODS.init, payload)
      await toolHost.request(TOOL_METHODS.setTier, { tier: currentTier })
      console.log(`[init] tool-host 已就绪（workspaceRoot=${WORKSPACE_ROOT}，tier=${currentTier}）`)
      return true
    } catch {
      if (i === maxAttempts) {
        console.error('[init] tool-host 初始化失败：达到重试上限，执行通道将保持 fail-closed')
        return false
      }
      await new Promise((r) => setTimeout(r, intervalMs))
    }
  }
  return false
}

/**
 * 启动自检：验证 P-01 ↔ P-03 / P-04 的 MessagePort 链路真正打通。
 *
 * 判据不是「进程拉起来了」，而是「跨进程请求拿到了返回值」——
 * 子进程可能已 spawn 但端口未绑定，此时 request 会超时，必须如实失败。
 */
async function selfCheck(maxAttempts = 10, intervalMs = 400): Promise<void> {
  for (let i = 1; i <= maxAttempts; i++) {
    try {
      const [agent, tool] = await Promise.all([
        agentHost?.request<string>(AGENT_METHODS.ping),
        toolHost?.request<string>(TOOL_METHODS.ping)
      ])
      if (agent && tool) {
        console.log('[selfcheck] P-01 主进程 online')
        console.log(`[selfcheck] P-03 agent-host ${agent}`)
        console.log(`[selfcheck] P-04 tool-host ${tool}`)
        console.log('[selfcheck] 三进程链路就绪（MessagePort，未开本地端口）')
        return
      }
    } catch (err) {
      if (i === maxAttempts) {
        console.error(`[selfcheck] 链路未打通：${(err as Error).message}`)
        return
      }
    }
    await new Promise((r) => setTimeout(r, intervalMs))
  }
  console.error('[selfcheck] 链路未打通：达到重试上限')
}

/* ------------------------------------------------------------------ *
 * 生命周期
 * ------------------------------------------------------------------ */

/** 在渲染进程里跑一段 JS。探针专用，不要在业务路径上用 */
function jsOf(code: string): Promise<unknown> {
  if (!mainWindow) return Promise.reject(new Error('无窗口'))
  return mainWindow.webContents.executeJavaScript(code)
}

/**
 * 读事件流里「类型」那一列的文本。
 *
 * 为什么不用整页 innerText 做 substring：版面一变长就会假阴性 ——
 * 曾因为只读前 800 字符，事件流刚排到第 801 字符，断言就红了，而功能其实是对的。
 * 查 DOM 更准，也更能说明「屏幕上真渲染了这一行」。
 */
async function eventTypesOnScreen(): Promise<string[]> {
  const raw = (await jsOf(
    "JSON.stringify([...document.querySelectorAll('.stream-type')].map(e => e.textContent ?? ''))"
  )) as string
  return JSON.parse(raw) as string[]
}

/**
 * 冒烟自检（`electron . --smoke`）：等待链路与首屏数据都就绪后，读取界面实际渲染文本并退出。
 * 判据是「界面上真有这些字」，不是「页面加载成功」——白屏也能加载成功。
 */
async function smoke(): Promise<void> {
  await selfCheck()
  await verifyToolChain()

  // 首屏只截前 800 字符用于打印 —— 排版好看，但**不能用来做断言**（见 eventTypesOnScreen）
  const readText = (): Promise<string> =>
    mainWindow?.webContents.executeJavaScript(
      'document.body.innerText.replace(/\\n{2,}/g, "\\n").slice(0, 800)'
    ) ?? Promise.resolve('')

  // 首屏依赖异步 IPC（bootInfo / vaultStatus / ping），等它落定再取值
  let text = ''
  for (let i = 0; i < 20; i++) {
    text = await readText()
    if (text.includes('在线') || text.includes('离线')) break
    await new Promise((r) => setTimeout(r, 250))
  }

  console.log('---SMOKE-TEXT-BEGIN---')
  console.log(text || '(空)')
  console.log('---SMOKE-TEXT-END---')

  // 可选：走一次真实 run，验证「渲染层 → 主进程 → agent-host → 事件回流渲染层」闭环
  if (process.argv.includes('--smoke-run')) {
    const started = await mainWindow?.webContents.executeJavaScript(
      "window.openintj.startRun('给这个仓库补上缺失的单元测试并让 CI 通过').then(r => JSON.stringify(r))"
    )
    console.log(`[probe] run 已启动 → ${started}`)

    // 断言读的是事件流的类型列，不是整页文本 ——
    // 整页 innerText 做 substring 匹配会随版面变长而失效（曾经就是这么假阴性的）
    let types: string[] = []
    for (let i = 0; i < 60; i++) {
      types = await eventTypesOnScreen()
      if (types.includes('GoalContractDrafted')) break
      await new Promise((r) => setTimeout(r, 500))
    }
    console.log(
      types.includes('GoalContractDrafted')
        ? `[probe] 事件已回流渲染层 → 界面事件流共 ${types.length} 条，含 GoalContractDrafted`
        : `[probe] 失败：事件未回流渲染层 → 界面只有 ${types.length} 条：${types.join(',') || '(空)'}`
    )
  }

  // 可选：D4 审批闭环 —— 真的点卡片上的按钮
  if (process.argv.includes('--smoke-approval')) {
    await verifyApprovalLoop()
  }

  // 可选：真实模型 —— 起草目标契约并上屏
  if (process.argv.includes('--smoke-llm')) {
    await verifyLlm()
  }

  // 可选：Agent 主循环 —— 起草 → 确认 → 模型自己驱动多步执行
  if (process.argv.includes('--smoke-agent')) {
    await verifyAgentLoop()
  }

  // 可选：凭据中心 —— 存进保险箱、读回只见掩码、落盘是密文、清掉能回落
  if (process.argv.includes('--smoke-cred')) {
    await verifyCredentialCenter()
  }

  // 可选：脱敏管线 —— 密钥必遮、路径可控、且不许误伤 commit hash
  if (process.argv.includes('--smoke-redact')) {
    verifyRedaction()
  }

  // 可选：MCP —— 真拉起一个 server 子进程，走完连接/发现/分级/裁决/调用/审计
  if (process.argv.includes('--smoke-mcp')) {
    await verifyMcp()
  }

  // 可选：档位持久化 —— 切换即落库；「重启」后要三处一起恢复（主进程 / 裁决方 / 界面）
  if (process.argv.includes('--smoke-tier')) {
    await verifyTier()
  }

  // 可选：内嵌终端 —— 真拉起 PTY，验「终端不是绕过策略层的通道」
  if (process.argv.includes('--smoke-term')) {
    await verifyTerminal()
  }

  // 可选：会话面板 —— 验 IDE 右栏「对话 / 任务 / 变更 / 预览」四个分页签
  if (process.argv.includes('--smoke-session')) {
    await verifySessionPanel()
  }

  // 可选：内嵌 IDE —— 真拉起 VSCodium 服务端，验外部产品边界被收紧
  // 放在靠后：它会把界面切到编辑器页，且快照探针要在最后拍
  if (process.argv.includes('--smoke-ide')) {
    await verifyIde()
  }

  // 可选：截一张界面图存盘（用 Electron 自带 capturePage，不依赖系统截图工具）
  // 放在所有探针之后 —— 前置探针会把界面切到它要验的那一页，截图才拍得到
  if (process.argv.includes('--snapshot')) {
    await captureSnapshot()
  }

  app.exit(0)
}

/**
 * D4 审批闭环探针：读的是界面上真有卡片、按钮真能点、点了真能跑。
 *
 * 用 `dd if=/dev/zero of=/dev/null count=0` 作 L3 样本：命中 R4-02 判 L3，
 * 但执行起来是空操作 —— 验证闸门不该以「真的干了坏事」为代价。
 */
async function verifyApprovalLoop(): Promise<void> {
  if (!mainWindow) {
    console.error('[probe] 无窗口，跳过审批闭环探针')
    return
  }
  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)
  const readText = (): Promise<string> =>
    mainWindow!.webContents.executeJavaScript('document.body.innerText.replace(/\\n{2,}/g, "\\n")')

  const waitFor = async (needle: string, attempts = 25): Promise<boolean> => {
    for (let i = 0; i < attempts; i++) {
      if ((await readText()).includes(needle)) return true
      await new Promise((r) => setTimeout(r, 200))
    }
    return false
  }

  const L3 = 'dd if=/dev/zero of=/dev/null bs=1 count=0'

  // ① 起一个 run，但**不花模型 token** 起草契约（手写）
  const runId = (await js(
    "window.openintj.startRun('D4 审批闭环验证', { draft: false }).then(r => r.runId)"
  )) as string
  console.log(`[probe] ⑦ run 就绪（跳过模型起草）→ ${runId}`)

  // ② IC-07 硬闸门：契约未确认，工具调用必须被拒
  const blocked = (await js(
    `window.openintj.invokeTool({ runId: '${runId}', command: ${JSON.stringify(L3)} }).then(r => JSON.stringify(r))`
  )) as string
  console.log(
    blocked.includes('IC-07')
      ? `[probe] ⑧ 契约未确认即被拒 → ${blocked.slice(0, 100)}…`
      : `[probe] ⑧ 失败：闸门没拦住 → ${blocked.slice(0, 140)}`
  )

  // ③ 手写一份契约并确认
  const manualContract = {
    objective: '验证审批闭环（含闸门、放行、拒绝、批量）',
    successCriteria: ['契约未确认时工具调用被拒', '允许后命令真的执行', '拒绝后不产生进程'],
    nonGoals: ['不做真实危险操作'],
    assumptions: ['工作区根为当前工程目录'],
    risks: ['探针命令选错可能误判闸门'],
    needsClarification: []
  }
  const confirmed = (await js(
    `window.openintj.confirmContract({ runId: '${runId}', accepted: true, contract: ${JSON.stringify(
      manualContract
    )} }).then(r => JSON.stringify(r))`
  )) as string
  console.log(
    confirmed.includes('"ok":true')
      ? `[probe] ⑨ 契约已确认（手写）→ ${confirmed.slice(0, 80)}`
      : `[probe] ⑨ 失败：契约确认失败 → ${confirmed.slice(0, 140)}`
  )

  // ④ 发起 L3 调用（不 await，让它停在审批上）
  await js(
    `(window.__inv1 = window.openintj.invokeTool({ runId: '${runId}', command: ${JSON.stringify(
      L3
    )} }), 'started')`
  )
  const cardShown = await waitFor('待审批 1')
  console.log(cardShown ? '[probe] ⑩ 审批卡已上屏（含风险徽标与三档按钮）' : '[probe] ⑩ 失败：审批卡未上屏')

  // ⑤ 点「允许一次」→ 应真的执行
  const clicked = (await js(
    `(() => { const b = [...document.querySelectorAll('button')].find(x => x.textContent.trim() === '允许一次'); if (!b) return 'no-button'; b.click(); return 'clicked' })()`
  )) as string
  const approved = (await js('window.__inv1.then(r => JSON.stringify(r))')) as string
  console.log(
    clicked === 'clicked' && approved.includes('"ok":true')
      ? `[probe] ⑪ 允许后执行 → ${approved.slice(0, 110)}…`
      : `[probe] ⑪ 失败：click=${clicked} result=${approved.slice(0, 160)}`
  )

  // ⑥ 再发起一次，点「拒绝」→ 应不执行
  await js(
    `(window.__inv2 = window.openintj.invokeTool({ runId: '${runId}', command: ${JSON.stringify(
      L3
    )} }), 'started')`
  )
  await waitFor('待审批 1')
  const rejected = (await js(
    `(() => { const b = [...document.querySelectorAll('button')].find(x => x.textContent.trim() === '拒绝'); if (!b) return 'no-button'; b.click(); return 'clicked' })()`
  )) as string
  const denied = (await js('window.__inv2.then(r => JSON.stringify(r))')) as string
  console.log(
    rejected === 'clicked' && denied.includes('"ok":false')
      ? `[probe] ⑫ 拒绝后未执行 → ${denied.slice(0, 110)}…`
      : `[probe] ⑫ 失败：click=${rejected} result=${denied.slice(0, 160)}`
  )

  // ⑦ 并发 4 条 → 超过阈值应聚合为批量卡
  await js(
    `(window.__batch = Promise.all([1,2,3,4].map(() => window.openintj.invokeTool({ runId: '${runId}', command: ${JSON.stringify(
      L3
    )} }))), 'started')`
  )
  const batchShown = await waitFor('待审批 4')
  const hasBatchBtn = (await readText()).includes('批量处理')
  console.log(
    batchShown && hasBatchBtn
      ? '[probe] ⑬ 4 条并发 → 聚合为批量审批卡'
      : `[probe] ⑬ 失败：pending4=${batchShown} batchBtn=${hasBatchBtn}`
  )

  // 收尾：全部拒绝，别把卡片留在界面上
  await js(
    `(() => { const b = [...document.querySelectorAll('button')].find(x => x.textContent.trim() === '批量处理'); if (b) b.click(); return 'ok' })()`
  )
  await new Promise((r) => setTimeout(r, 300))
  await js(
    `(() => { const b = [...document.querySelectorAll('button')].find(x => x.textContent.includes('全部拒绝')); if (b) b.click(); return 'ok' })()`
  )
  await new Promise((r) => setTimeout(r, 500))
  const cleared = !(await readText()).includes('待审批')
  console.log(cleared ? '[probe] ⑭ 批量拒绝后队列清空' : '[probe] ⑭ 失败：队列未清空')
}

/**
 * 真实模型探针：走一遍「一句话目标 → 模型起草契约 → 界面上能看到」。
 *
 * 需要 OPENINTJ_LLM_* 环境变量；缺任何一个就明确报「未配置」，绝不假装成功。
 */
async function verifyLlm(): Promise<void> {
  if (!mainWindow) {
    console.error('[probe] 无窗口，跳过模型探针')
    return
  }
  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)

  const status = (await js('window.openintj.llmStatus().then(r => JSON.stringify(r))')) as string
  console.log(`[probe] ⑮ 模型接入 → ${status}`)
  if (status.includes('"configured":false')) {
    console.log('[probe] ⑮ 模型未配置，跳过契约生成（不伪造输出）')
    return
  }

  // ⑮-b 协议必须与声明一致。
  // 加第三协议时专门补的一条：协议名写错、或解析里漏了一个分支，旧代码会**静默**
  // 按 anthropic 跑 —— 探针照样全绿，但测的根本不是想测的那条通道。
  // 「绿得毫无意义」比红更难发现，所以这条断言存在的价值就是让它红。
  const declaredRaw = process.env['OPENINTJ_LLM_PROTOCOL']?.trim()
  const effective = (JSON.parse(status) as { protocol?: string | null }).protocol ?? null
  if (declaredRaw && effective !== parseProtocol(declaredRaw)) {
    console.log(`[probe] ⑮-b 失败：声明协议「${declaredRaw}」，实际生效「${effective}」`)
  } else {
    console.log(`[probe] ⑮-b 协议一致 → ${effective ?? '(环境变量未声明协议)'}`)
  }

  const started = (await js(
    "window.openintj.startRun('给这个仓库补上缺失的单元测试，并让 CI 通过').then(r => JSON.stringify(r))"
  )) as string
  const parsed = JSON.parse(started) as { runId: string; state: string; contract?: Record<string, unknown> }

  if (!parsed.contract) {
    console.log(`[probe] ⑯ 失败：契约未生成（state=${parsed.state}）→ ${started.slice(0, 300)}`)
    return
  }

  const c = parsed.contract as {
    objective?: string
    successCriteria?: string[]
    nonGoals?: string[]
    assumptions?: string[]
    risks?: string[]
    needsClarification?: string[]
  }
  console.log(`[probe] ⑯ 契约已由真实模型起草 → ${parsed.runId} / state=${parsed.state}`)
  console.log(`[probe]    目标：${c.objective ?? '(缺)'}`)
  console.log(`[probe]    成功判据 ${c.successCriteria?.length ?? 0} 条：${(c.successCriteria ?? [])[0] ?? '(空)'}`)
  console.log(
    `[probe]    不做 ${c.nonGoals?.length ?? 0} / 假设 ${c.assumptions?.length ?? 0} / 风险 ${
      c.risks?.length ?? 0
    } / 待澄清 ${c.needsClarification?.length ?? 0}`
  )

  // 界面必须真的把它渲染出来，而不是只在返回值里。
  // 查的是契约卡片的分栏 DOM，不是整页文本 —— 后者会随版面变长而假阴性
  const sections = (await jsOf(
    "JSON.stringify([...document.querySelectorAll('.contract-section-title')].map(e => e.textContent ?? ''))"
  )) as string
  const titles = JSON.parse(sections) as string[]
  const shown = titles.includes('成功判据') && titles.includes('待用户澄清')
  console.log(
    shown
      ? `[probe] ⑰ 契约卡片已上屏（${titles.length} 个分栏：${titles.join(' / ')}）`
      : `[probe] ⑰ 失败：契约卡片未上屏 → 分栏=${titles.join(',') || '(空)'}`
  )
}

/**
 * 截一张界面图。
 *
 * 用 Electron 自带的 `webContents.capturePage()`，不走系统截图工具 ——
 * 后者在受限环境里会被安全策略拦掉（`Add-Type` 一类），而这是进程内 API，无外部依赖。
 * 顺带说明：这比截屏更可靠的地方在于它截的是**渲染结果**，不受窗口是否被遮挡影响。
 */
/**
 * 截**整屏**（含原生 view）。
 *
 * `capturePage()` 只能拍到渲染层 —— 工作台是独立的 WebContentsView，
 * 是盖在网页上面的另一层，capturePage 的画面里根本没有它。
 * 而「右栏被工作台盖住」恰恰是两层之间的问题：DOM 查询全绿、
 * bounds 数字也对，但用户看到的是工作台铺满整行。
 * 这种事只能用 desktopCapturer 截屏亲眼看，数据骗不了人，但数据也不全。
 */
async function captureScreen(tag: string): Promise<void> {
  try {
    // 整屏截图拍的是「屏幕上最上面那层」—— 不把自己窗口顶到最前，
    // 拍到的就是用户正在看的别的应用，而不是我们的布局。
    mainWindow?.show()
    mainWindow?.moveTop()
    mainWindow?.focus()
    await new Promise((r) => setTimeout(r, 600))
    const sources = await desktopCapturer.getSources({
      types: ['window'],
      thumbnailSize: { width: 2560, height: 1440 },
      fetchWindowIcons: false
    })
    // 按标题找自己的窗口：截屏不需要抢前台（前台权在 Windows 上受限制），
    // 而窗口捕获走的是系统合成结果，原生 view 也在里面。
    const mine =
      sources.find((s) => (s.name ?? '').includes('openINTJ')) ??
      sources.find((s) => (s.name ?? '').includes('Agent Console')) ??
      sources[0]
    const shot = mine?.thumbnail
    if (!shot || shot.isEmpty()) {
      console.error('[probe] 整屏截图失败：无可用屏幕源')
      return
    }
    const dir = join(process.cwd(), '.snapshots')
    mkdirSync(dir, { recursive: true })
    const file = join(dir, `openintj-screen-${tag}-${Date.now()}.png`)
    writeFileSync(file, shot.toPNG())
    const size = shot.getSize()
    console.log(`[probe] 整屏截图已保存 → ${file}（${size.width}×${size.height}）`)
  } catch (e) {
    console.error(`[probe] 整屏截图失败：${(e as Error).message}`)
  }
}

async function captureSnapshot(tag = ''): Promise<void> {
  if (!mainWindow) {
    console.error('[probe] 无窗口，无法截图')
    return
  }
  // 等一帧，确保 React 把最新状态画上去了
  await new Promise((r) => setTimeout(r, 600))

  /**
   * 截图前先把叠加的工作台摘下来。
   *
   * 两个原因，第二个是踩出来的：
   *   1. capturePage() 只拍主 webContents，拍不到叠加的 WebContentsView；
   *   2. 更要命的是 —— 叠加着一个已加载的第三方工作台时，
   *      **capturePage() 会一直不返回**（实测 8 分钟仍在跑）。
   * 所以先摘、拍完挂回去。
   */
  const overlay =
    ideView && mainWindow.contentView.children.includes(ideView) ? ideView : null
  if (overlay) mainWindow.contentView.removeChildView(overlay)

  const image = await mainWindow.webContents.capturePage()
  const dir = join(process.cwd(), '.snapshots')
  mkdirSync(dir, { recursive: true })
  const file = join(dir, `openintj${tag ? `-${tag}` : ''}-${Date.now()}.png`)
  writeFileSync(file, image.toPNG())
  const size = image.getSize()
  console.log(`[probe] 界面截图已保存 → ${file}（${size.width}×${size.height}）`)

  if (overlay) mainWindow.contentView.addChildView(overlay)

  // 工作台是独立 WebContents，上面那张拍不到它 —— 单独拍一张作为证据
  if (ideView && (ideView.webContents.getURL() ?? '').startsWith('http://127.0.0.1:')) {
    const wb = await ideView.webContents.capturePage()
    const wbFile = join(dir, `openintj-workbench-${Date.now()}.png`)
    writeFileSync(wbFile, wb.toPNG())
    console.log(`[probe] 工作台截图已保存 → ${wbFile}`)
  }
}

/**
 * 脱敏管线探针（D8）。
 *
 * 两条最容易写错的地方，都单独验：
 *   ① `redaction: 'none'` 时密钥仍然必须被遮 —— 底线不能依赖调用方标对级别
 *   ② 不能误伤 —— 尤其 git commit hash（40 位 hex），遮掉它排障就废了
 */
/**
 * 内嵌终端探针（US-15）。
 *
 * 核心要证的不是「终端能跑命令」——那太容易了 —— 而是**终端不能绕过策略层**：
 *   ① 只读命令自动放行并拿到真实 PTY 输出；
 *   ② 严格档下同一条命令立刻要审批，**在批准之前命令没有执行**；
 *   ③ 拒绝后终端明示原因，且审计里留下 executed=false；
 *   ④ 票据绑定命令原文，换一条命令就失效；
 *   ⑤ resize / 中断 / 关闭都有真实返回值。
 *
 * 全程在**真实渲染进程里点真实按钮、读真实渲染出来的字**，不 mock 中间层。
 */
async function verifyTerminal(): Promise<void> {
  if (!mainWindow || !toolHost) {
    console.error('[probe] 无窗口或 tool-host 未就绪，跳过终端探针')
    return
  }
  await initToolHost()

  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)
  const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

  /** 轮询直到条件成立；返回最后一次取到的值，便于失败时打印现场 */
  const until = async <T>(fn: () => Promise<T>, ok: (v: T) => boolean, tries = 40): Promise<T> => {
    let last = await fn()
    for (let i = 0; i < tries && !ok(last); i++) {
      await sleep(250)
      last = await fn()
    }
    return last
  }

  const termText = async (): Promise<string> =>
    String(await js("document.querySelector('.term-out .xterm-rows')?.innerText ?? ''"))
  const statusText = async (): Promise<string> =>
    String(await js("document.querySelector('.term-status-text')?.textContent ?? ''"))

  /** 往输入行里写字。直接改 value 不行 —— React 受控组件只认原生 setter + input 事件 */
  const typeCommand = async (cmd: string): Promise<string> =>
    String(
      await js(`(() => {
        const el = document.querySelector('.term-input')
        if (!el) return 'no-input'
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set
        setter.call(el, ${JSON.stringify(cmd)})
        el.dispatchEvent(new Event('input', { bubbles: true }))
        return 'ok'
      })()`)
    )

  const clickExec = async (): Promise<string> =>
    String(
      await js(`(() => {
        const btns = [...document.querySelectorAll('.term-input-row .pill')]
        const b = btns[btns.length - 1]
        if (!b) return 'no-button'
        b.click()
        return 'ok'
      })()`)
    )

  const clickApproval = async (label: string): Promise<string> =>
    String(
      await js(`(() => {
        const b = [...document.querySelectorAll('.term-approval .pill')]
          .find(x => (x.textContent || '').includes(${JSON.stringify(label)}))
        if (!b) return 'no-button'
        b.click()
        return 'ok'
      })()`)
    )

  const submit = async (cmd: string): Promise<void> => {
    await typeCommand(cmd)
    await sleep(60)
    await clickExec()
  }

  const sessions = await toolHost.request<Array<{ sessionId: string }>>(TOOL_METHODS.terminalList, {}, 5_000)
  for (const s of sessions) {
    await toolHost.request(TOOL_METHODS.terminalClose, { sessionId: s.sessionId }, 5_000).catch(() => undefined)
  }

  const originalTier = currentTier

  try {
    // 55 打开终端页 → 真拉起 PTY 会话
    await js("[...document.querySelectorAll('.tab')].find(b => (b.textContent||'').includes('终端'))?.click() ?? 'ok'")
    const ready = await until(statusText, (t) => /powershell|bash|zsh|sh/i.test(t), 40)
    const list = await toolHost.request<Array<{ sessionId: string; shell: string; cols: number; rows: number }>>(
      TOOL_METHODS.terminalList,
      {},
      5_000
    )
    const sid = list[0]?.sessionId ?? null
    console.log(
      sid
        ? `[probe] 55 终端会话已建立 → ${sid} · ${ready} · ${list[0]?.cols}x${list[0]?.rows}`
        : `[probe] 55 失败：没有活跃会话（状态栏「${ready}」）`
    )
    if (!sid) return

    // 56 只读命令自动放行，输出是**真 PTY 输出**（有 ANSI，说明 TTY 语义在）
    await submit('node -v')
    const afterReadonly = await until(termText, (t) => /v\d+\.\d+\.\d+/.test(t), 40)
    console.log(
      /v\d+\.\d+\.\d+/.test(afterReadonly)
        ? `[probe] 56 只读命令自动放行 → 输出里出现 ${/v\d+\.\d+\.\d+/.exec(afterReadonly)?.[0]}`
        : `[probe] 56 失败：终端里没看到版本号 → ${JSON.stringify(afterReadonly.slice(-160))}`
    )

    // 57 严格档：同一条只读命令立刻要审批 —— 证明终端和 shell 调用走同一张真值表
    await js("window.openintj.setApprovalTier('strict').then(() => 'ok')")
    await sleep(500)
    const before = await termText()
    await submit('node -v')
    const approvalShown = await until(
      () => js("Boolean(document.querySelector('[data-testid=\"term-approval\"]'))"),
      (v) => v === true,
      20
    )
    const termApproval = approvalShown === true
    console.log(
      termApproval
        ? '[probe] 57 严格档下终端命令被拦在审批条前（终端不是旁路）'
        : '[probe] 57 失败：严格档下没有出现内联审批条'
    )

    // 58 审批等待期间命令**不能已经执行** —— 只比「有没有审批条」是不够的
    await sleep(400)
    const during = await termText()
    const grew = during.length - before.length
    // 允许 pty 自己回显提示符，但不允许出现新的版本号输出
    const leaked = during.slice(before.length).match(/v\d+\.\d+\.\d+/g)?.length ?? 0
    console.log(
      leaked === 0
        ? `[probe] 58 批准前未执行（新增输出 ${grew} 字符，无命令结果泄漏）`
        : `[probe] 58 失败：批准前终端里已经出现了执行结果（${leaked} 处）`
    )

    // 59 拒绝 → 面板明示原因（提示条，不写进 pty 区域），审计留下 executed=false
    await clickApproval('拒绝')
    const noticeText = async (): Promise<string> =>
      String(await js("document.querySelector('[data-testid=\"term-notice\"]')?.textContent ?? ''"))
    const afterDeny = await until(noticeText, (t) => /已拒绝/.test(t), 20)
    const auditRun = terminalRunId(sid)
    const rows = await toolHost.request<
      Array<{ command: string; executed: boolean; basis: string; riskClass: string }>
    >(TOOL_METHODS.auditRecent, { runId: auditRun, limit: 20 }, 5_000)
    // 注意：auditRecent 透出来的是布尔。写成 `executed === 0` 会永远匹配不到 ——
    // 断言写错的表现和功能坏掉一模一样，都是红。
    const deniedRow = rows.find((r) => r.command === 'node -v' && r.executed === false)
    console.log(
      /已拒绝/.test(afterDeny) && deniedRow
        ? `[probe] 59 拒绝后明示原因并留痕（executed=0 / ${deniedRow.riskClass} / ${deniedRow.basis}）`
        : `[probe] 59 失败：提示条「${afterDeny.slice(0, 60)}」；审计 ${JSON.stringify(rows.slice(0, 3))}`
    )

    // 60 允许一次 → 命令真的执行（同一会话里继续，`cd` 之类状态本该保留）
    await submit('node -v')
    await until(
      () => js("Boolean(document.querySelector('[data-testid=\"term-approval\"]'))"),
      (v) => v === true,
      20
    )
    await clickApproval('允许一次')
    const afterAllow = await until(termText, (t) => {
      const m = t.match(/v\d+\.\d+\.\d+/g)
      return (m?.length ?? 0) >= 2 // 56 已经有过一次，这里要再出现一次
    }, 40)
    const vCount = afterAllow.match(/v\d+\.\d+\.\d+/g)?.length ?? 0
    console.log(
      vCount >= 2
        ? `[probe] 60 允许后真的执行 → 终端里累计出现 ${vCount} 次版本号`
        : `[probe] 60 失败：允许后没有新的执行结果（出现 ${vCount} 次）`
    )

    // 趁终端里真有输出+真被允许执行过，留一张存证。
    // 不放在最末尾：62 的 resize 会让 PowerShell 重绘提示符、清掉上面的历史。
    if (process.argv.includes('--snapshot')) {
      await captureSnapshot('term')
    }

    // 61 票据绑定命令：拿一张写着 `node -v` 的票去提交 `whoami`，必须被拒
    const wrongCmdTicket = issueTicket(TICKET_SECRET, {
      runId: auditRun,
      command: 'node -v',
      riskClass: 'L0',
      scope: 'once'
    })
    const swapped = await toolHost.request<TerminalSubmitResult>(
      TOOL_METHODS.terminalInput,
      { sessionId: sid, command: 'whoami', approvalTicket: wrongCmdTicket },
      10_000
    )
    console.log(
      swapped.kind === 'denied'
        ? `[probe] 61 票据绑定命令生效（换命令即失效）→ ${swapped.reason?.slice(0, 48)}…`
        : `[probe] 61 失败：换命令后仍然返回 ${swapped.kind}`
    )

    // 62 resize 真的下发到 pty
    const rz = await toolHost.request<{ ok: boolean; cols: number; rows: number }>(
      TOOL_METHODS.terminalResize,
      { sessionId: sid, cols: 100, rows: 40 },
      10_000
    )
    console.log(
      rz.ok && rz.cols === 100 && rz.rows === 40
        ? `[probe] 62 resize 生效 → ${rz.cols}x${rz.rows}`
        : `[probe] 62 失败：${JSON.stringify(rz)}`
    )

    // 63 中断（不裁决，只减副作用）
    const int = await toolHost.request<{ ok: boolean }>(TOOL_METHODS.terminalInterrupt, { sessionId: sid }, 10_000)
    console.log(int.ok ? '[probe] 63 中断已下发（Ctrl+C 不经裁决）' : '[probe] 63 失败：中断未生效')

    // 64 关闭后不能再提交（会话不残留）
    await toolHost.request(TOOL_METHODS.terminalClose, { sessionId: sid }, 10_000)
    const afterClose = await toolHost.request<TerminalSubmitResult>(
      TOOL_METHODS.terminalInput,
      { sessionId: sid, command: 'node -v' },
      10_000
    )
    const remaining = await toolHost.request<unknown[]>(TOOL_METHODS.terminalList, {}, 5_000)
    console.log(
      afterClose.kind === 'no-session' && remaining.length === 0
        ? '[probe] 64 关闭后会话已回收（再提交返回 no-session，列表为空）'
        : `[probe] 64 失败：${afterClose.kind} / 剩余 ${remaining.length} 个会话`
    )
  } finally {
    // 档位还原：探针把用户设置改成严格档，不还原就是「自检把产品弄坏了」
    await toolHost.request(TOOL_METHODS.setTier, { tier: originalTier }).catch(() => undefined)
    currentTier = originalTier
    const left = await toolHost.request<Array<{ sessionId: string }>>(TOOL_METHODS.terminalList, {}, 5_000).catch(() => [])
    for (const s of left) {
      await toolHost.request(TOOL_METHODS.terminalClose, { sessionId: s.sessionId }, 5_000).catch(() => undefined)
    }
  }
}

function verifyRedaction(): void {
  const FAKE_KEY = 'sk-2EFKK7hyMf0bvgyynovoRigbT0xQ9alIzj3Xog5VlhJ4tEbM' // scan:ignore(脱敏探针合成 key，非凭据)

  // ㉜ 声明 none，密钥也必须被脱敏
  const r1 = applyRedaction(
    { command: `curl -H "Authorization: Bearer ${FAKE_KEY}" https://example.com` },
    'none'
  )
  const s1 = JSON.stringify(r1.value)
  const masked = !s1.includes(FAKE_KEY) && s1.includes('***')
  console.log(
    masked
      ? `[probe] ㉜ 声明 none 仍遮住密钥（级别自动升为 ${r1.level}）→ ${(r1.value as { command: string }).command.slice(0, 60)}…`
      : `[probe] ㉜ 失败：密钥未被脱敏 → ${s1}`
  )

  // ㉝ 私钥块、JWT、连接串密码、URL 凭据
  const r2 = applyRedaction(
    {
      pem: '-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----', // scan:ignore(私钥夹具，中间是字面量省略号，非密钥)
      jwt: 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U', // scan:ignore(JWT 夹具，故意不成形，非令牌)
      db: 'postgres://admin:s3cr3tpass@db.internal:5432/app',
      kv: 'API_KEY=abcdef1234567890'
    },
    'none'
  )
  const s2 = JSON.stringify(r2.value)
  const clean =
    !s2.includes('MIIEow') &&
    !s2.includes('dozjgNryP4J3jVmNHl0w5N') &&
    !s2.includes('s3cr3tpass') &&
    !s2.includes('abcdef1234567890')
  console.log(
    clean
      ? `[probe] ㉝ 私钥 / JWT / 连接串密码 / kv 形式凭据均已遮蔽（命中 ${r2.hits.map((h) => h.ruleId).join(',')}）`
      : `[probe] ㉝ 失败：仍有敏感内容残留 → ${s2}`
  )

  // ㉞ 不许误伤 commit hash —— 40 位 hex 是 git 的命脉，遮了就没法排障
  const COMMIT = 'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'
  const r3 = applyRedaction({ note: `fixed in ${COMMIT}`, sha256: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' }, 'content')
  const s3 = JSON.stringify(r3.value)
  console.log(
    s3.includes(COMMIT)
      ? '[probe] ㉞ commit hash 未被误伤（40 位 hex 原样保留）'
      : `[probe] ㉞ 失败：commit hash 被当成密钥遮掉了 → ${s3}`
  )

  // ㉟ 路径脱敏：家目录换 ~，但保留目录结构
  const home = homedir()
  const r4 = applyRedaction({ p: `${home}/projects/app/src/index.ts` }, 'path', home)
  const p4 = (r4.value as { p: string }).p
  console.log(
    p4.startsWith('~') && p4.includes('src/index.ts')
      ? `[probe] ㉟ 路径脱敏为 ${p4}（结构保留，排障仍可读）`
      : `[probe] ㉟ 失败：路径脱敏结果异常 → ${p4}`
  )

  // ㊱ 结构不许被破坏 —— 回放要靠它
  const r5 = applyRedaction({ a: [{ b: FAKE_KEY, n: 1 }], s: 'plain' }, 'none')
  const v5 = r5.value as { a: { b: string; n: number }[]; s: string }
  const shapeOk =
    Array.isArray(v5.a) && v5.a.length === 1 && v5.a[0]!.n === 1 && v5.s === 'plain' && !v5.a[0]!.b.includes(FAKE_KEY)
  console.log(
    shapeOk
      ? '[probe] ㊱ 嵌套结构与数字/普通文本保持原样，只有密钥值被替换'
      : `[probe] ㊱ 失败：payload 结构被改坏 → ${JSON.stringify(r5.value)}`
  )

  // ㊲ 终端控制序列清理（只在 content 档）
  const r6 = applyRedaction({ out: '\u001b[31mred\u001b[0m text' }, 'content')
  const out6 = (r6.value as { out: string }).out
  console.log(
    out6 === 'red text' ? '[probe] ㊲ ANSI 控制序列已清理' : `[probe] ㊲ 失败：残留控制字符 → ${JSON.stringify(out6)}`
  )
}

/**
 * 凭据中心探针（IC-06）。
 *
 * 最重要的一条断言是 ㉖：**落盘文件里搜不到明文密钥**。
 * 这条不能靠「代码里写了 safeStorage 所以应该加密了」来保证 ——
 * 必须把文件字节读出来，去里面找那把 key。找得到就是没加密。
 *
 * 测试会临时写一份凭据进保险箱，结束时**按原字节还原**，不留副作用。
 */
async function verifyCredentialCenter(): Promise<void> {
  if (!mainWindow) {
    console.error('[probe] 无窗口，跳过凭据中心探针')
    return
  }
  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)
  const dataDir = app.getPath('userData')
  const file = credentialFilePath(dataDir)

  // 不伪造凭据：没有真实 key 就跳过。拿假 key 存进去只能测「加密了没」，测不出「能不能用」
  const envKey = process.env['OPENINTJ_LLM_API_KEY']?.trim()
  const envUrl = process.env['OPENINTJ_LLM_BASE_URL']?.trim()
  const envModel = process.env['OPENINTJ_LLM_MODEL']?.trim()
  const envProtocol = process.env['OPENINTJ_LLM_PROTOCOL']?.toLowerCase()
  if (!envKey || !envUrl || !envModel) {
    console.log('[probe] ㉔ 环境里没有完整的 OPENINTJ_LLM_*，跳过凭据中心探针（不伪造凭据）')
    return
  }

  const original = existsSync(file) ? readFileSync(file) : null

  try {
    // ㉔ 保存。协议名原样透传 —— 由 shared 的 parseProtocol 统一判定，
    // 探针这里再写一份映射就等于给自己造了第二套真相。
    const input = {
      protocol: envProtocol ?? 'anthropic',
      baseUrl: envUrl,
      model: envModel,
      apiKey: envKey,
      disableThinking: true
    }
    const saved = JSON.parse(
      (await js(`window.openintj.credentialSave(${JSON.stringify(input)}).then(r => JSON.stringify(r))`)) as string
    ) as { ok: boolean; reason?: string; status?: CredentialStatus }
    console.log(
      saved.ok
        ? `[probe] ㉔ 已保存进保险箱 → ${saved.status?.maskedKey}（来源 ${saved.status?.source}）`
        : `[probe] ㉔ 失败：${saved.reason}`
    )
    if (!saved.ok) return

    // ㉕ 读回只有掩码 —— 明文绝不能被回传渲染层
    const shown = JSON.stringify(saved.status ?? {})
    const leaked = shown.includes(envKey)
    console.log(
      leaked
        ? '[probe] ㉕ 失败：返回给渲染层的数据里出现了明文密钥'
        : `[probe] ㉕ 回传只有掩码（${saved.status?.maskedKey}），明文未出主进程`
    )

    // ㉖ 落盘必须是密文：直接把文件字节读出来搜 key
    const raw = readFileSync(file)
    const isPlain = raw.includes(Buffer.from(envKey, 'utf8'))
    console.log(
      isPlain
        ? '[probe] ㉖ 失败：落盘文件里能直接搜到明文密钥'
        : `[probe] ㉖ 落盘为密文（${raw.length} 字节，未包含明文密钥）`
    )

    // ㉗ 生效来源切到 vault
    const status = JSON.parse(
      (await js('window.openintj.credentialStatus().then(r => JSON.stringify(r))')) as string
    ) as CredentialStatus
    console.log(
      status.source === 'vault'
        ? '[probe] ㉗ 生效来源已切到 vault（保险箱优先于环境变量）'
        : `[probe] ㉗ 失败：来源为 ${status.source}，应为 vault`
    )

    // ㉘ 用保险箱里的凭据真的调一次模型
    const test = JSON.parse(
      (await js('window.openintj.llmTest().then(r => JSON.stringify(r))')) as string
    ) as LlmTestResult
    console.log(
      test.ok
        ? `[probe] ㉘ 测试连接通过 → ${test.latencyMs}ms · ${test.model} · 回复「${test.sample}」`
        : `[probe] ㉘ 失败：${test.reason}`
    )

    // ㉙ 清除前先把界面切到设置页 —— 此时保险箱里有凭据、来源是 vault，
    // 这是「凭据中心真的能用」最直观的证据，截图要趁这一刻
    //
    // 按文字找标签页，不用 index：加一个「终端」标签就会让 `.tab[1]` 指错地方，
    // 而且是静默指错（断言在下一行才红，看起来像页面渲染坏了）。
    await js(
      "[...document.querySelectorAll('.tab')].find(b => (b.textContent||'').includes('设置'))?.click() ?? null"
    )
    await new Promise((r) => setTimeout(r, 700))
    if (process.argv.includes('--snapshot')) {
      await captureSnapshot()
    }

    // ㉚ 设置页确实渲染出来了（查真实 DOM，不靠整页文本）
    const fields = (await js(
      "JSON.stringify([...document.querySelectorAll('.field > label')].map(e => e.textContent ?? ''))"
    )) as string
    const labels = JSON.parse(fields) as string[]
    const pageOk = labels.includes('Base URL') && labels.includes('API Key')
    console.log(
      pageOk
        ? `[probe] ㉚ 设置与凭据中心已上屏（${labels.join(' / ')}）`
        : `[probe] ㉚ 失败：设置页字段缺失 → ${labels.join(',') || '(空)'}`
    )

    // ㉛ 清除后回落环境变量
    const cleared = JSON.parse(
      (await js('window.openintj.credentialClear().then(r => JSON.stringify(r))')) as string
    ) as { ok: boolean; status?: CredentialStatus }
    console.log(
      cleared.ok && cleared.status?.source === 'env'
        ? '[probe] ㉛ 清除后回落为 env 来源（环境变量仍然可用）'
        : `[probe] ㉛ 失败：清除后来源为 ${cleared.status?.source}`
    )
  } finally {
    // 还原到测试前的状态，不在用户机器上留下探针写进去的东西
    if (original) writeFileSync(file, original)
    else if (existsSync(file)) rmSync(file, { force: true })
    console.log('[probe]    已还原保险箱到测试前状态')
  }
}

/**
 * Agent 主循环探针：起草 → 确认 → 模型自己决定跑什么命令 → 走裁决链 → 收尾。
 *
 * 这是 D3 的端到端验收。三件事必须同时成立才算通过：
 *   ① 循环能靠模型自己推进（不是被脚本喂命令）
 *   ② 模型请求的每条命令都过了 tool-host 的分级（事件流里有 riskClass）
 *   ③ 结果卡片真的渲染出来，而不是只躺在返回值里
 *
 * 目标刻意选得极小（问个版本号），一次只读命令就能完成 —— 探针要证明的是链路通，
 * 不是让模型去改仓库，更不该在自检里烧掉几万 token。
 *
 * 审批用测试替身自动放行：D4 探针已经验过真人点击的真实路径，
 * 这里若还要等人，自检就永远跑不完。
 */
async function verifyAgentLoop(): Promise<void> {
  if (!mainWindow) {
    console.error('[probe] 无窗口，跳过循环探针')
    return
  }
  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)

  const status = (await js('window.openintj.llmStatus().then(r => JSON.stringify(r))')) as string
  if (status.includes('"configured":false')) {
    console.log('[probe] ⑱ 模型未配置，跳过循环（不伪造模型输出）')
    return
  }

  const startedRaw = (await js(
    "window.openintj.startRun('告诉我这台机器上 node 的版本号').then(r => JSON.stringify(r))"
  )) as string
  const started = JSON.parse(startedRaw) as { runId: string; state: string; contract?: unknown }
  if (!started.contract) {
    console.log(`[probe] ⑱ 失败：契约未生成（state=${started.state}）`)
    return
  }
  console.log(`[probe] ⑱ 契约已起草 → ${started.runId}`)

  const confRaw = (await js(
    `window.openintj.confirmContract({runId:${JSON.stringify(started.runId)},accepted:true}).then(r => JSON.stringify(r))`
  )) as string
  const conf = JSON.parse(confRaw) as { ok: boolean; state?: string; reason?: string }
  if (!conf.ok) {
    console.log(`[probe] ⑲ 失败：契约确认被拒 → ${conf.reason}`)
    return
  }
  console.log(`[probe] ⑲ 契约已确认 → state=${conf.state}，交给模型驱动`)

  // 自动应答任何审批（测试替身）—— 走的是与界面按钮完全相同的路径
  let autoApproved = 0
  const responder = setInterval(() => {
    for (const id of pendingApprovals.keys()) {
      autoApproved++
      void mainWindow?.webContents.executeJavaScript(
        `window.openintj.respondApproval(${JSON.stringify(id)}, true, false)`
      )
    }
  }, 300)

  // 关键：**不能 await** startAgentLoop —— executeJavaScript 会等 Promise 解析，
  // 而循环要跑好几步还可能等审批，直接 await 会把探针卡死在这里（D4 踩过的坑）
  await js(
    [
      'window.__loopRes = null;',
      `window.openintj.startAgentLoop(${JSON.stringify(started.runId)})`,
      ".then(r => { window.__loopRes = JSON.stringify(r) })",
      ".catch(e => { window.__loopRes = JSON.stringify({ outcome: 'failed', steps: 0, reason: e.message }) });",
      "'started'"
    ].join('')
  )

  let raw: unknown = null
  for (let i = 0; i < 240; i++) {
    raw = await js('window.__loopRes')
    if (raw) break
    await new Promise((r) => setTimeout(r, 1000))
  }
  clearInterval(responder)

  if (!raw) {
    console.log('[probe] ⑳ 失败：循环 4 分钟未收尾')
    return
  }

  const res = JSON.parse(raw as string) as {
    outcome: string
    steps: number
    reason?: string
    summary?: string
    satisfied?: unknown[]
    unmet?: unknown[]
    usage?: { inputTokens: number; outputTokens: number; cacheReadTokens: number }
  }

  console.log(
    `[probe] ⑳ 循环收尾 → outcome=${res.outcome} steps=${res.steps} ` +
      `输入 ${res.usage?.inputTokens ?? 0} + 缓存命中 ${res.usage?.cacheReadTokens ?? 0} / 输出 ${
        res.usage?.outputTokens ?? 0
      }` +
      (res.reason ? ` reason=${res.reason}` : '')
  )
  if (res.summary) console.log(`[probe]    小结：${res.summary.slice(0, 160)}`)
  if (res.satisfied?.length) console.log(`[probe]    已满足判据 ${res.satisfied.length} 条`)
  if (res.unmet?.length) console.log(`[probe]    未满足判据 ${res.unmet.length} 条`)
  if (autoApproved > 0) console.log(`[probe]    自动放行 ${autoApproved} 条审批（测试替身）`)

  // 事件流必须证明：模型真说过话、真请求过命令、命令真过了分级
  const types = await eventTypesOnScreen()
  const hasModelMsg = types.includes('ModelMessage')
  const hasToolInvoked = types.includes('ToolInvoked')
  // riskClass 在 payload 里，是 tool-host 事后分级的结果 —— 它出现就说明命令真的过了一遍裁决
  const hasRisk = Boolean(
    await jsOf("document.querySelector('.stream')?.innerText.includes('riskClass') ?? false")
  )
  console.log(
    hasModelMsg && hasToolInvoked && hasRisk
      ? `[probe] ㉑ 模型请求的命令确实过了裁决链（事件流 ${types.length} 条，含 ModelMessage + ToolInvoked + riskClass）`
      : `[probe] ㉑ 失败：事件流缺证据（ModelMessage=${hasModelMsg} ToolInvoked=${hasToolInvoked} riskClass=${hasRisk}）`
  )

  const cardShown = Boolean(await jsOf("Boolean(document.querySelector('.delivery'))"))
  console.log(cardShown ? '[probe] ㉒ 执行结果卡片已上屏' : '[probe] ㉒ 失败：执行结果卡片未上屏')

  // 循环收尾后不该还挂着审批卡 —— 那些卡片背后已经没有等待者，
  // 用户点下去只会得到「审批请求不存在或已超时」，比不显示更让人困惑。
  // 收卡是**异步**的：agent-host 发 approval.cancel → 主进程转发 → React 重渲染，
  // 所以这里必须等，不能查一次就下结论（查一次就是假阴性）
  let leftover = -1
  for (let i = 0; i < 25; i++) {
    leftover = (await jsOf("document.querySelectorAll('.approval').length")) as number
    if (leftover === 0) break
    await new Promise((r) => setTimeout(r, 150))
  }
  console.log(
    leftover === 0
      ? '[probe] ㉓ 收尾后无残留审批卡（僵尸卡片已收回）'
      : `[probe] ㉓ 失败：收尾后仍挂着 ${leftover} 条审批卡`
  )
}

/**
 * 工具链探针：验证「分级 → 裁决 → 票据 → 执行」整条链真的在位。
 *
 * 这些不是日志，每一条都是可失败的断言。重点验两条防线：
 *   ① 调用方无法「声明已批准」（无票据即拒）
 *   ② 调用方无法「伪造批准」（密钥只在 P-01 / P-04，agent-host 没有）
 */
async function verifyToolChain(): Promise<void> {
  if (!toolHost) {
    console.error('[probe] tool-host 未就绪，跳过工具链探针')
    return
  }
  const cwd = WORKSPACE_ROOT
  await initToolHost()

  type Probe = { ok: true; result: ToolExecResult } | { ok: false; err: string }
  const tryExec = async (command: string, ticket?: ApprovalTicket, runId = 'smoke'): Promise<Probe> => {
    try {
      const result = await toolHost!.request<ToolExecResult>(TOOL_METHODS.exec, {
        runId,
        agentId: 'main',
        command,
        cwd,
        approvalTicket: ticket
      })
      return { ok: true, result }
    } catch (e) {
      return { ok: false, err: (e as Error).message }
    }
  }

  // ① 分级：危险命令必须判 L3
  const dangerous = await toolHost
    .request<{ riskClass: string; matchedRuleId: string }>(TOOL_METHODS.classify, {
      command: 'rm -rf /tmp/openintj-probe',
      cwd
    })
    .catch((e: Error) => ({ riskClass: `err:${e.message}`, matchedRuleId: '-' }))
  console.log(`[probe] ① 分级 rm -rf → ${dangerous.riskClass}（${dangerous.matchedRuleId}）`)

  // ② 无票据执行 L3 → 必须拒绝，且不产生进程
  //    注意别用 echo 试：它在只读白名单（R1-05）里，本就该自动放行
  const noTicket = await tryExec('chmod 777 README.md')
  console.log(
    noTicket.ok
      ? '[probe] ② 失败：无票据的 L3 命令被执行了'
      : `[probe] ② 无票据被拒 → ${noTicket.err.slice(0, 70)}…`
  )

  // ③ 伪造票据（用错误密钥签）→ 必须拒绝，证明 agent-host 无法自行造票
  const forged = issueTicket('attacker-guessed-secret', {
    runId: 'smoke',
    command: 'chmod 777 README.md',
    riskClass: 'L3',
    scope: 'once'
  })
  const forgedTry = await tryExec('chmod 777 README.md', forged)
  console.log(
    forgedTry.ok
      ? '[probe] ③ 失败：伪造票据通过了校验'
      : `[probe] ③ 伪造票据被拒 → ${forgedTry.err.slice(0, 70)}…`
  )

  // ④ 票据绑定命令：签 A 用 B → 必须拒绝
  const boundToOther = issueTicket(TICKET_SECRET, {
    runId: 'smoke',
    command: 'cp README.md ../other-target.txt',
    riskClass: 'L2',
    scope: 'once'
  })
  const swapped = await tryExec('cp README.md ../openintj-probe-escape.txt', boundToOther)
  console.log(
    swapped.ok ? '[probe] ④ 失败：票据可跨命令复用' : `[probe] ④ 票据绑定命令生效 → ${swapped.err.slice(0, 70)}…`
  )

  // ⑤ 正例：L0 只读，标准档自动放行，不需要票据
  const readOnly = await tryExec('node -v')
  console.log(
    readOnly.ok
      ? `[probe] ⑤ 只读自动放行 → exit=${readOnly.result.exitCode} stdout=${readOnly.result.stdout.trim()}（${readOnly.result.risk.outcome}）`
      : `[probe] ⑤ 失败：只读命令被拦 → ${readOnly.err}`
  )

  // ⑥ 会话授权：首次凭 session 票据，之后同任务同级免票据
  const writeCmd = 'git init --quiet openintj-smoke-tmp'
  const sessionTicket = issueTicket(TICKET_SECRET, {
    runId: 'smoke-session',
    command: writeCmd,
    riskClass: 'L1',
    scope: 'session'
  })
  const first = await tryExec(writeCmd, sessionTicket, 'smoke-session')
  const second = await tryExec(writeCmd, undefined, 'smoke-session')
  console.log(
    first.ok && second.ok
      ? `[probe] ⑥ 会话授权 → 首次 ${first.result.risk.outcome}，复用免票据（basis=${second.result.risk.basis}）`
      : `[probe] ⑥ 失败：首次=${first.ok ? 'ok' : first.err}，复用=${second.ok ? 'ok' : second.err}`
  )
  try {
    rmSync(join(WORKSPACE_ROOT, 'openintj-smoke-tmp'), { recursive: true, force: true })
  } catch {
    /* 清理失败不影响结论 */
  }
}

/**
 * D6 MCP 探针。
 *
 * 不 mock、不打桩：真的拉起一个 stdio MCP server 子进程（scripts/fixtures 里那个），
 * 走完「读配置 → 连接 → 发现工具 → 分级 → 裁决 → 调用 → 审计入库」整条链。
 *
 * 关键断言是**票据也要绑定 MCP 调用**：MCP 没有命令行，如果票据不绑参数，
 * 一次批准就能被换成任意参数重放 —— 那等于给模型开了一个免审批通道。
 */
async function verifyMcp(): Promise<void> {
  if (!toolHost) {
    console.error('[probe] tool-host 未就绪，跳过 MCP 探针')
    return
  }
  await initToolHost()

  const dataDir = app.getPath('userData')
  const cfgPath = join(dataDir, 'mcp.json')
  const serverScript = join(dataDir, 'openintj-mcp-probe-server.mjs')
  const fixture = join(app.getAppPath(), 'scripts', 'fixtures', 'mcp-probe-server.mjs')

  if (!existsSync(fixture)) {
    console.error(`[probe] 找不到 MCP 探针 server 脚本：${fixture}，跳过`)
    return
  }

  // 用户可能已经有自己的 mcp.json —— 探针不许把人家的配置弄丢
  const backup: Buffer | null = existsSync(cfgPath) ? readFileSync(cfgPath) : null

  // ⓪ 初始化阶段就该把 mcp.json 路径下发到 tool-host。
  //    这一条常被忽略：如果 init 没传路径，后面 reload 也一样能跑，
  //    但真实用户重启应用后 MCP 会「凭空消失」—— 因为他们不会去点「重新加载」。
  const boot = await toolHost.request<McpStatusSnapshot>(TOOL_METHODS.mcpStatus, undefined, 10_000)
  console.log(
    boot.enabled && boot.configPath === cfgPath && boot.configError === null
      ? `[probe] ⓪ init 已下发 MCP 配置路径 → ${boot.configPath}（当前 ${boot.servers.length} 个 server）`
      : `[probe] ⓪ 失败：${JSON.stringify(boot)}`
  )

  writeFileSync(serverScript, readFileSync(fixture))

  const launch = {
    command: process.execPath,
    args: [serverScript],
    // 探针环境里只有 Electron 自带的 node，用这个变量让它以纯 Node 方式跑脚本
    env: { ELECTRON_RUN_AS_NODE: '1' },
    connectTimeoutMs: 20_000
  }
  writeFileSync(
    cfgPath,
    JSON.stringify(
      {
        mcpServers: {
          // 声明只读 → L0 自动放行
          probeRo: { ...launch, trust: 'readonly' },
          // 默认 confirm → L2 逐次审批
          probeConfirm: { ...launch, trust: 'confirm' }
        }
      },
      null,
      2
    )
  )

  type ReloadRes = { configPath: string; configError: string | null; servers: McpServerStatus[] }
  const tryCall = async (
    params: Record<string, unknown>
  ): Promise<{ ok: true; r: McpCallOutcome } | { ok: false; err: string }> => {
    try {
      const r = await toolHost!.request<McpCallOutcome>(TOOL_METHODS.mcpCall, {
        agentId: 'main',
        ...params
      })
      return { ok: true, r }
    } catch (e) {
      return { ok: false, err: (e as Error).message }
    }
  }

  try {
    // ① 连接：两个 server 都必须真的连上
    const reload = await toolHost.request<ReloadRes>(TOOL_METHODS.mcpReload, {}, 120_000)
    const connected = reload.servers.filter((s) => s.status === 'connected')
    console.log(
      connected.length === reload.servers.length && reload.servers.length === 2
        ? `[probe] ㊳ 两个 MCP server 已连接（工具数 ${reload.servers.map((s) => `${s.name}:${s.toolCount}`).join(', ')}）`
        : `[probe] ㊳ 失败：连接状态 ${JSON.stringify(reload.servers)}；configError=${reload.configError ?? '-'}`
    )

    // ② 发现：工具名必须带 mcp__ 前缀，且能从左侧无歧义拆回 server/tool
    const tools = await toolHost.request<McpToolInfo[]>(TOOL_METHODS.mcpList, undefined, 35_000)
    const echoRo = tools.find((t) => t.qualifiedName === 'mcp__probeRo__echo')
    const echoConfirm = tools.find((t) => t.qualifiedName === 'mcp__probeConfirm__echo')
    const parsed = echoRo ? parseMcpToolName(echoRo.qualifiedName) : null
    console.log(
      tools.length === 4 && echoRo && echoConfirm && parsed?.server === 'probeRo' && parsed.tool === 'echo'
        ? `[probe] ㊴ 工具发现 ${tools.length} 个，命名与解析往返一致（${echoRo.qualifiedName}）`
        : `[probe] ㊴ 失败：工具 ${tools.map((t) => t.qualifiedName).join(',') || '(空)'}`
    )

    // ③ 只读 server：无票据 → L0 自动放行，且真的拿到了 server 的返回
    const ro = await tryCall({
      runId: 'smoke-mcp-ro',
      server: 'probeRo',
      tool: 'echo',
      args: { text: 'hello' }
    })
    console.log(
      ro.ok && ro.r.ok && ro.r.text === 'pong:hello' && ro.r.risk.riskClass === 'L0' && ro.r.risk.outcome === 'AUTO_APPROVE'
        ? `[probe] ㊵ 只读自动放行 → "${ro.r.text}"（${ro.r.risk.riskClass}/${ro.r.risk.outcome}，${ro.r.durationMs}ms）`
        : `[probe] ㊵ 失败：${ro.ok ? JSON.stringify({ t: ro.r.text, risk: ro.r.risk }) : ro.err}`
    )

    // ④ confirm server：无票据 → 必须拒绝
    const noTicket = await tryCall({
      runId: 'smoke-mcp-deny',
      server: 'probeConfirm',
      tool: 'echo',
      args: { text: 'no-ticket' }
    })
    console.log(
      !noTicket.ok && noTicket.err.includes('拒绝执行')
        ? `[probe] ㊶ confirm 级无票据被拒 → ${noTicket.err.slice(0, 60)}…`
        : `[probe] ㊶ 失败：${noTicket.ok ? '被放行了' : noTicket.err}`
    )

    // ⑤ 被拒的调用也要入库，且风险等级按 MCP 的规则判（不是 shell 分类器的结果）
    //
    //    审计库跨次运行是累积的，所以每次用一个全新的 runId —— 否则第二次跑这个探针
    //    会因为「上一次的记录还在」而假阴性。（这类断言写过一次就得记住。）
    const auditRun = `smoke-mcp-audit-${Date.now()}`
    await toolHost
      .request(TOOL_METHODS.mcpDenial, {
        runId: auditRun,
        server: 'probeConfirm',
        tool: 'echo',
        args: { text: 'denied' },
        basis: 'user-deny'
      })
      .catch(() => undefined)
    const rows = await toolHost.request<
      { command: string; riskClass: string; matchedRuleId: string; executed: boolean; basis: string }[]
    >(TOOL_METHODS.auditRecent, { runId: auditRun, limit: 5 })
    const row = rows[0]
    console.log(
      rows.length === 1 && row && !row.executed && row.riskClass === 'L2' && row.matchedRuleId === 'R6-mcp-call'
        ? `[probe] ㊷ 被拒调用已入库 → executed=${row.executed} ${row.riskClass}/${row.matchedRuleId} basis=${row.basis}`
        : `[probe] ㊷ 失败：审计 ${JSON.stringify(rows)}`
    )

    // ⑥ 持合法票据 → 执行成功（证明票据链对 MCP 同样成立）
    const argsOk = { text: 'with-ticket' }
    const ticketOk = issueTicket(TICKET_SECRET, {
      runId: 'smoke-mcp-ok',
      command: mcpCommandString('probeConfirm', 'echo', argsOk),
      riskClass: 'L2',
      scope: 'once'
    })
    const passed = await tryCall({
      runId: 'smoke-mcp-ok',
      server: 'probeConfirm',
      tool: 'echo',
      args: argsOk,
      approvalTicket: ticketOk
    })
    console.log(
      passed.ok && passed.r.text === 'pong:with-ticket'
        ? `[probe] ㊸ 持票据执行成功 → "${passed.r.text}"（${passed.r.risk.outcome}/${passed.r.risk.basis}）`
        : `[probe] ㊸ 失败：${passed.ok ? passed.r.text : passed.err}`
    )

    // ⑦ 票据绑定参数：签的是 text=A，实际传 text=B → 必须拒绝
    const ticketA = issueTicket(TICKET_SECRET, {
      runId: 'smoke-mcp-swap',
      command: mcpCommandString('probeConfirm', 'echo', { text: 'A' }),
      riskClass: 'L2',
      scope: 'once'
    })
    const swapped = await tryCall({
      runId: 'smoke-mcp-swap',
      server: 'probeConfirm',
      tool: 'echo',
      args: { text: 'B' },
      approvalTicket: ticketA
    })
    console.log(
      !swapped.ok
        ? `[probe] ㊹ 票据绑定参数生效（换参数即失效）→ ${swapped.err.slice(0, 60)}…`
        : `[probe] ㊹ 失败：票据可跨参数复用`
    )

    // ⑧ server 返回 isError 时，要如实标记而不是当成成功
    const boom = await tryCall({
      runId: 'smoke-mcp-ro',
      server: 'probeRo',
      tool: 'boom',
      args: {}
    })
    console.log(
      boom.ok && boom.r.ok && boom.r.isError
        ? `[probe] ㊺ isError 如实透传 → "${boom.r.text}"`
        : `[probe] ㊺ 失败：${boom.ok ? JSON.stringify({ ok: boom.r.ok, isError: boom.r.isError }) : boom.err}`
    )

    // ⑨ 界面：设置页必须真的把这些 server 列出来（连上了但界面看不见，等于没连）
    if (mainWindow) {
      // 先切回控制台、再进设置页 —— 强制设置页重新挂载。
      // 原因：MCP 面板只在挂载时拉一次状态（其余靠用户点「刷新」）。
      // 若前一个探针（例如 --smoke-cred 的 ㉚）已经把它留在设置页，这里直接点「设置」
      // 是一次空操作（React 状态没变 → 不重渲染 → 面板还是 mcp.json 为空时的旧数据），
      // 断言就会误红。这是探针之间的状态耦合，不是产品缺陷 —— 真人操作时页面是重新进来的。
      await jsOf(
        "[...document.querySelectorAll('.tab')].find(b => (b.textContent||'').includes('控制台'))?.click() ?? 'ok'"
      )
      await new Promise((r) => setTimeout(r, 150))
      await jsOf(
        "[...document.querySelectorAll('.tab')].find(b => (b.textContent||'').includes('设置'))?.click() ?? 'ok'"
      )
      let rows = 0
      for (let i = 0; i < 16; i++) {
        const n = Number(await jsOf("document.querySelectorAll('.mcp-row').length"))
        rows = Number.isFinite(n) ? n : 0
        if (rows >= 2) break
        await new Promise((r) => setTimeout(r, 250))
      }
      const names = (await jsOf(
        "JSON.stringify([...document.querySelectorAll('.mcp-row .mcp-name')].map(e => e.textContent))"
      )) as string
      console.log(
        rows >= 2
          ? `[probe] ㊻ 设置页已列出 ${rows} 个 MCP server → ${names}`
          : `[probe] ㊻ 失败：设置页只列出 ${rows} 个（界面没跟上后端状态）`
      )
      // 趁面板上真有 server，留一张存证（只在显式要求截图时做）
      if (process.argv.includes('--snapshot')) {
        // 面板在页面底部，不滚进视野的话截图里只有半个卡片
        await jsOf("document.querySelector('[data-testid=\"mcp-card\"]')?.scrollIntoView({block:'center'}) ?? 'ok'")
        await captureSnapshot('mcp')
      }
      // 切回控制台，保持后续探针与截图的页面语义不变
      await jsOf(
        "[...document.querySelectorAll('.tab')].find(b => (b.textContent||'').includes('控制台'))?.click() ?? 'ok'"
      )
    }

    // ⑩ 模型侧：MCP 工具必须真的进了 tool spec（连上了但模型看不见，等于没连）
    const catalog = await agentHost?.request<{
      builtin: string[]
      mcp: { name: string; server: string; trust: string }[]
      total: number
    }>(AGENT_METHODS.toolCatalog, undefined, 40_000)
    console.log(
      catalog && catalog.mcp.length === 4 && catalog.mcp.some((t) => t.name === 'mcp__probeConfirm__echo')
        ? `[probe] ㊼ 工具目录 ${catalog.total} 项（内置 ${catalog.builtin.join('/')} + MCP ${catalog.mcp.length}）`
        : `[probe] ㊼ 失败：工具目录 ${JSON.stringify(catalog)}`
    )

    // ⑪ 配置写坏时必须报 configError，而不是静默当成「没配」
    writeFileSync(cfgPath, '{ 这不是 JSON }')
    const broken = await toolHost.request<ReloadRes>(TOOL_METHODS.mcpReload, {}, 60_000)
    console.log(
      broken.configError && broken.servers.length === 0
        ? `[probe] ㊽ 坏配置如实上报 → ${broken.configError}`
        : `[probe] ㊽ 失败：坏配置未上报（${JSON.stringify(broken)}）`
    )

    // ⑫ 命名规范：前缀识别与非法 server 名都要经得住
    console.log(
      isMcpToolName('mcp__a__b') && !isMcpToolName('run_command') && parseMcpToolName('mcp__') === null
        ? '[probe] ㊾ MCP 命名规范校验通过（前缀识别 / 空名拒绝）'
        : '[probe] ㊾ 失败：命名规范校验异常'
    )
  } finally {
    // 还原用户原本的 mcp.json（没有就删掉），并清理探针脚本
    try {
      if (backup) writeFileSync(cfgPath, backup)
      else rmSync(cfgPath, { force: true })
      rmSync(serverScript, { force: true })
    } catch {
      /* 清理失败不影响结论 */
    }
    // 让运行时状态与磁盘一致：重载一次，把探针进程收掉
    await toolHost.request(TOOL_METHODS.mcpReload, {}, 60_000).catch(() => undefined)
  }
}

/**
 * D9 探针：档位持久化。
 *
 * 「重启后还在」没法在同一个进程里真的重启来验，所以这里用一种更严的验法：
 * 把三处状态**全部**打回默认（主进程内存 / tool-host 裁决方 / 界面），再调一次恢复逻辑。
 *
 * 为什么必须验三处：只恢复一处的结果是「界面显示严格档，实际按标准档裁决」——
 * 那比不持久化更糟，因为用户会以为自己已经收紧了，然后把真正的风险放过去。
 */
async function verifyTier(): Promise<void> {
  if (!agentHost || !toolHost) {
    console.error('[probe] 子进程未就绪，跳过档位持久化探针')
    return
  }
  await initToolHost()

  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)
  const savedTier = async (): Promise<string | null> => {
    const r = await agentHost!.request<{ value: string | null }>(
      AGENT_METHODS.settingGet,
      { key: SETTING_KEYS.approvalTier },
      5_000
    )
    return r.value
  }
  const runtimeTier = async (): Promise<string> => {
    const p = await toolHost!.request<string>(TOOL_METHODS.ping)
    return /tier=([a-z]+)/.exec(p)?.[1] ?? '(未知)'
  }
  const activePill = async (): Promise<string> =>
    String(await js("document.querySelector('[data-tier].active')?.getAttribute('data-tier') ?? '(无)'"))

  const original = await savedTier()

  try {
    // 51 走界面路径切到严格档（和用户点按钮完全一样）
    await js("document.querySelector('[data-tier=\"strict\"]')?.click() ?? 'ok'")
    await new Promise((r) => setTimeout(r, 700))
    const dbAfter = await savedTier()
    const rtAfter = await runtimeTier()
    console.log(
      dbAfter === 'strict' && rtAfter === 'strict'
        ? `[probe] 51 切换即落库 → 库里 ${dbAfter}，裁决方 tier=${rtAfter}`
        : `[probe] 51 失败：库里 ${dbAfter}，裁决方 tier=${rtAfter}`
    )

    // 52 模拟重启：三处全部打回默认（重载渲染层是干净做法，比改 React state 可靠）
    currentTier = DEFAULT_TIER
    await toolHost.request(TOOL_METHODS.setTier, { tier: DEFAULT_TIER })
    const loaded = new Promise<void>((resolve) =>
      mainWindow!.webContents.once('did-finish-load', () => resolve())
    )
    mainWindow?.webContents.reload()
    await loaded
    await new Promise((r) => setTimeout(r, 900))
    const beforeRestore = await activePill()
    console.log(
      beforeRestore === 'standard' && (await runtimeTier()) === 'standard'
        ? `[probe] 52 「重启」后回到默认档（界面 ${beforeRestore}，裁决方 standard）`
        : `[probe] 52 失败：界面 ${beforeRestore}，裁决方 ${await runtimeTier()}`
    )

    // 53 恢复：三处必须一起变回严格档
    await restoreTier()
    await new Promise((r) => setTimeout(r, 500))
    const rtRestored = await runtimeTier()
    const pillRestored = await activePill()
    console.log(
      currentTier === 'strict' && rtRestored === 'strict' && pillRestored === 'strict'
        ? `[probe] 53 三处一起恢复 → 主进程 ${currentTier}，裁决方 ${rtRestored}，界面 ${pillRestored}`
        : `[probe] 53 失败：主进程 ${currentTier}，裁决方 ${rtRestored}，界面 ${pillRestored}`
    )

    // 54 脏数据容错：库里被塞了一个不存在的档位，必须回落默认而不是照着用
    await agentHost.request(AGENT_METHODS.settingSet, {
      key: SETTING_KEYS.approvalTier,
      value: 'god-mode'
    })
    currentTier = DEFAULT_TIER
    await toolHost.request(TOOL_METHODS.setTier, { tier: DEFAULT_TIER })
    await restoreTier()
    console.log(
      currentTier === DEFAULT_TIER
        ? `[probe] 54 非法档位被忽略 → 仍为 ${currentTier}（不做静默猜测）`
        : `[probe] 54 失败：非法值被采用了 → ${currentTier}`
    )
  } finally {
    // 复原：库写回原值（原本没有就写默认档，行为等价），运行时回默认并通知界面
    await agentHost
      .request(AGENT_METHODS.settingSet, {
        key: SETTING_KEYS.approvalTier,
        value: original ?? DEFAULT_TIER
      })
      .catch(() => undefined)
    currentTier = DEFAULT_TIER
    await toolHost.request(TOOL_METHODS.setTier, { tier: DEFAULT_TIER }).catch(() => undefined)
    mainWindow?.webContents.send('evt:tier', DEFAULT_TIER)
  }
}

/**
 * D5 内嵌 IDE 探针：验「外部产品边界真的被收紧了」，而不只是「页面打开了」。
 *
 * 断言清单
 *   65 编辑器标签页可用，面板挂载
 *   66 服务端就位（版本来自 reh-web 的 product.json）
 *   67 点启动 → ready，且启动耗时有值
 *   68 错令牌 → 403（令牌校验不是摆设）
 *   69 正确令牌 → 302（种 cookie 后 200 拿工作台）
 *   70 只绑 127.0.0.1 —— 非回环地址访问不到；且非默认 8000 端口
 *   71 工作台真的渲染出来了（独立 view 里能查到 .monaco-workbench）
 *   72 点停止 → stopped
 *   73 端口释放 —— 进程树被杀干净，不留常驻
 *   74 界面文案不含禁用商标（C2）
 */
/**
 * 可选：会话面板（IDE 右栏）—— 验「对话 / 任务 / 变更 / 预览」各司其职。
 *
 * 这组探针关心的不是「按钮能不能点」，而是三件容易做错的事：
 *   1. 右栏与工作台**并排**，不是盖在上面（三栏的本义是同时可见）
 *   2. 「预览」签只在真有预览目标时才出现 —— 常驻一个空白签，
 *      用户点进去会以为功能坏了，而不是「还没发生」
 *   3. 消息存在 agent-host：换会话再切回来还在。
 *      若消息只活在渲染层，刷新/切页就丢，而 run 还在跑 —— 那种
 *      「执行在继续、对话已清空」的错位比什么都难解释。
 */
async function verifySessionPanel(): Promise<void> {
  if (!mainWindow) {
    console.error('[probe] 无窗口，跳过会话面板探针')
    return
  }
  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)
  const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))
  const until = async <T>(
    fn: () => Promise<T>,
    ok: (v: T) => boolean,
    tries = 40,
    gap = 250
  ): Promise<T> => {
    let last = await fn()
    for (let i = 0; i < tries && !ok(last); i++) {
      await sleep(gap)
      last = await fn()
    }
    return last
  }
  /** 用原生 setter 写值再派发 input —— React 监听的是合成事件 */
  const setInput = async (selector: string, value: string): Promise<void> => {
    await js(`(() => {
      const el = document.querySelector(${JSON.stringify(selector)})
      if (!el) return
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
      setter.call(el, ${JSON.stringify(value)})
      el.dispatchEvent(new Event('input', { bubbles: true }))
      return
    })()`)
  }
  const clickTab = async (label: string): Promise<void> => {
    await js(`(() => {
      const b = [...document.querySelectorAll('[data-testid="ide-side-tabs"] .side-tab')]
        .find(x => (x.textContent || '').includes(${JSON.stringify(label)}))
      if (b) b.click()
      return
    })()`)
  }

  const tabLabels = async (): Promise<string[]> =>
    JSON.parse(
      String(
        await js(
          `JSON.stringify([...document.querySelectorAll('[data-testid="ide-side-tabs"] .side-tab')].map(b => (b.textContent || '').trim()))`
        )
      )
    )

  console.log('[probe] ── 会话面板（IDE 右栏）──')

  await clickIdeTab()
  await sleep(400)

  // 75 —— 右栏与工作台并排，且默认开好一个会话
  const labels = await until(async () => tabLabels(), (v) => v.length === 3)
  console.log(
    labels.length === 3 && !labels.some((t) => t.includes('预览'))
      ? `[probe] 75 右栏三签就位 → ${labels.join(' / ')}（无预览时不亮「预览」签）`
      : `[probe] 75 失败：功能签=${labels.join('/')}`
  )

  const sessionCount = Number(
    String(
      await js(
        `document.querySelectorAll('[data-testid="ide-session-tabs"] .session-tab').length`
      )
    )
  )
  console.log(
    sessionCount >= 2
      ? `[probe] 76 会话签就绪 → ${sessionCount - 1} 个会话 ＋ 新建入口`
      : `[probe] 76 失败：会话签节点数=${sessionCount}`
  )

  // 77 —— 发一句话：没有模型也会落库（起草会失败，但「用户说过什么」不能丢）
  const line = '看看这个仓库现在缺什么测试'
  await setInput('[data-testid="ide-chat-input"]', line)
  await js(`document.querySelector('[data-testid="ide-chat-send"]').click()`)
  const chatText = await until(
    async () => String(await js(`document.querySelector('[data-testid="ide-chat-list"]').innerText`)),
    (v) => v.includes(line)
  )
  console.log(
    chatText.includes(line)
      ? '[probe] 77 消息落库 → 发出的话已出现在对话列表'
      : `[probe] 77 失败：对话列表里没有这句话（${chatText.slice(0, 60)}）`
  )

  // 78 —— 外部地址必须被拒 —— 预览入口不能变成「打开任意网址」的口子
  await setInput('[data-testid="ide-preview-url"]', 'https://example.com/evil')
  await js(`document.querySelector('[data-testid="ide-preview-register"]')?.click()`)
  const external = await until(
    async () =>
      String(await js(`document.querySelector('[data-testid="ide-notice"]')?.innerText || ''`)),
    (v) => v.includes('回环')
  )
  const externalTabs = await tabLabels()
  console.log(
    external.includes('回环') && externalTabs.length === 3
      ? `[probe] 78 外部地址被拒 → ${external.trim().slice(0, 40)}（仍 ${externalTabs.length} 个签）`
      : `[probe] 78 失败：提示=${external.slice(0, 40)} 签数=${externalTabs.length}`
  )

  // 79 —— 登记一个回环地址后，「预览」签才亮出来
  await setInput('[data-testid="ide-preview-url"]', 'http://127.0.0.1:5173')
  await js(`document.querySelector('[data-testid="ide-preview-register"]')?.click()`)
  const after = await until(async () => tabLabels(), (v) => v.length === 4)
  console.log(
    after.includes('预览')
      ? `[probe] 79 预览签按需出现 → ${after.join(' / ')}`
      : `[probe] 79 失败：登记后签=${after.join('/')}`
  )

  // 80 —— 切到预览签，能看到登记进去的地址
  await clickTab('预览')
  const previewText = await until(
    async () => textOfSide(),
    (v) => v.includes('127.0.0.1:5173')
  )
  console.log(
    previewText.includes('127.0.0.1:5173')
      ? '[probe] 80 预览内容上屏 → 已列出 127.0.0.1:5173'
      : `[probe] 80 失败：预览页未见该地址（${previewText.slice(0, 50)}）`
  )

  // 81 —— 新建会话是空的；切回上一个，消息还在（证明存在 agent-host 而非渲染层）
  await js(`document.querySelector('[data-testid="ide-session-new"]').click()`)
  await sleep(400)
  const fresh = await textOfSide()
  await js(`document.querySelector('[data-testid="ide-session-tabs"] .session-tab').click()`)
  const back = await until(async () => textOfSide(), (v) => v.includes(line))
  console.log(
    fresh.includes('还没有对话') && back.includes(line)
      ? '[probe] 81 会话隔离与回溯 → 新会话为空，切回来消息仍在'
      : `[probe] 81 失败：新会话空=${fresh.includes('还没有对话')} 回溯=${back.includes(line)}`
  )

  // 82 —— 任务签 / 变更签：切过去之后各自渲染，且内容不同（不是一块空白）
  await clickTab('任务')
  await sleep(350)
  const taskPanel = await textOfSide()
  await clickTab('变更')
  await sleep(350)
  const changePanel = await textOfSide()
  console.log(
    taskPanel.length > 0 && changePanel.length > 0 && taskPanel !== changePanel
      ? '[probe] 82 任务签 / 变更签可切换 → 两页各自渲染，内容不同'
      : `[probe] 82 失败：任务=${taskPanel.slice(0, 20)} 变更=${changePanel.slice(0, 20)}`
  )
}

/** 切到编辑器页，并等面板挂载 */
async function clickIdeTab(): Promise<void> {
  if (!mainWindow) return
  await mainWindow.webContents.executeJavaScript(
    `(() => {
      const b = [...document.querySelectorAll('.tab')].find(x => (x.textContent || '').includes('编辑器'))
      if (b) b.click()
      return
    })()`
  )
}

/**
 * 取右栏的文本。
 *
 * 用 textContent 而不是 innerText：innerText 只返回**可见**文本，
 * 面板一旦被折叠/裁剪（或还没布局完）就返回空串，断言会假阴性。
 */
async function textOfSide(): Promise<string> {
  if (!mainWindow) return ''
  return String(
    await mainWindow.webContents.executeJavaScript(
      `document.querySelector('.ide-side')?.textContent || ''`
    )
  )
}

async function verifyIde(): Promise<void> {
  if (!mainWindow) {
    console.error('[probe] 无窗口，跳过 IDE 探针')
    return
  }
  const js = (code: string): Promise<unknown> => mainWindow!.webContents.executeJavaScript(code)
  const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))
  const until = async <T>(
    fn: () => Promise<T>,
    ok: (v: T) => boolean,
    tries = 40,
    gap = 250
  ): Promise<T> => {
    let last = await fn()
    for (let i = 0; i < tries && !ok(last); i++) {
      await sleep(gap)
      last = await fn()
    }
    return last
  }
  /** 在工作台那个独立 view 里执行 JS —— 与主界面是两个 WebContents，不能混用 */
  const viewJs = async (code: string): Promise<string> => {
    if (!ideView) return 'no-view'
    try {
      return String(await ideView.webContents.executeJavaScript(code))
    } catch (e) {
      return `err:${(e as Error).message.slice(0, 60)}`
    }
  }
  const tryFetch = async (url: string, timeoutMs = 1500): Promise<number> => {
    try {
      const r = await fetch(url, { redirect: 'manual', signal: AbortSignal.timeout(timeoutMs) })
      return r.status
    } catch {
      return -1 // 连不上
    }
  }

  // 65 —— 按文字找标签。位置选择器在会长大的 UI 里不能用（加一个标签就指错）
  //
  // 这里踩过一个坑：`?.click() ?? 'no-tab'` 里的 ?? **永远会命中** ——
  // click() 返回 undefined，于是无论点没点到都显示 'no-tab'。
  // 判定必须写成「先找、找不到早退、点完返回确定值」。
  const clicked = String(
    await js(`(() => {
      const b = [...document.querySelectorAll('.tab')].find(x => (x.textContent||'').includes('编辑器'))
      if (!b) return 'no-tab'
      b.click()
      return 'ok'
    })()`)
  )
  const panelShown = String(
    await until(
      async () => js("document.querySelector('[data-testid=\"ide-panel\"]') ? 'yes' : 'no'"),
      (v) => v === 'yes'
    )
  )
  console.log(
    clicked !== 'no-tab' && panelShown === 'yes'
      ? '[probe] 65 编辑器标签页可用 → 面板已挂载'
      : `[probe] 65 失败：点击=${clicked} 面板=${panelShown}`
  )

  const host = getIdeHost()

  // 66 —— 安装态。找不到就到此为止，后面的断言没有意义
  const s0 = host.status()
  console.log(
    s0.phase !== 'not-installed'
      ? `[probe] 66 服务端就位 → ${s0.version ?? 'unknown'}（phase=${s0.phase}）`
      : `[probe] 66 失败：未找到 VSCodium 服务端，先跑 npm run vscodium:fetch → ${s0.error ?? ''}`
  )
  if (s0.phase === 'not-installed') return

  // 67 —— 启动
  await js("document.querySelector('[data-testid=\"ide-start\"]')?.click() ?? null")
  const s1 = await until(
    async () => host.status(),
    (s) => s.phase === 'ready' || s.phase === 'failed',
    60,
    500
  )
  console.log(
    s1.phase === 'ready'
      ? `[probe] 67 IDE 已就绪 → ${s1.url} · 启动耗时 ${s1.elapsedMs}ms`
      : `[probe] 67 失败：phase=${s1.phase} ${s1.error ?? ''}`
  )
  if (s1.phase !== 'ready' || !s1.url || !s1.port) return

  // 68 —— 错令牌必须被拒。这是「边界收紧」里最该验的一条
  const wrongCode = await tryFetch(host.urlWithTokenFor('definitely-not-the-token')!, 3000)
  console.log(
    wrongCode === 403
      ? '[probe] 68 错令牌被拒 → HTTP 403（连接令牌真的在校验）'
      : `[probe] 68 失败：错令牌返回 ${wrongCode}（期望 403）`
  )

  // 69 —— 正确令牌：302 种 cookie，或已带 cookie 时直接 200
  const okCode = await tryFetch(host.urlWithToken()!, 3000)
  console.log(
    okCode === 302 || okCode === 200
      ? `[probe] 69 正确令牌放行 → HTTP ${okCode}`
      : `[probe] 69 失败：正确令牌返回 ${okCode}（期望 302/200）`
  )

  // 70 —— 只绑回环。真去连一次非回环地址：绑了 0.0.0.0 的话这里会通
  const u = new URL(s1.url)
  const lanIps: string[] = []
  for (const list of Object.values(networkInterfaces())) {
    for (const i of list ?? []) {
      // 只要非回环的 IPv4 —— 服务端若绑了 0.0.0.0，这些地址就能连上
      if (i && !i.internal && String(i.family).includes('4')) lanIps.push(i.address)
    }
  }
  let reachable = ''
  for (const ip of lanIps) {
    const code = await tryFetch(`http://${ip}:${s1.port}/`, 1200)
    if (code !== -1) {
      reachable = `${ip} → HTTP ${code}`
      break
    }
  }
  const notDefaultPort = s1.port !== 8000
  console.log(
    !reachable && notDefaultPort
      ? `[probe] 70 只绑回环 → ${lanIps.length} 个非回环地址均不可达；端口 ${s1.port}（非默认 8000）`
      : lanIps.length === 0
        ? `[probe] 70 只绑回环（本机无非回环地址，改用 host 判定）→ ${u.hostname}`
        : `[probe] 70 失败：${reachable || `端口落在默认值 ${s1.port}`}`
  )

  // 71 —— 工作台真的渲染出来了。只验 URL 不够：URL 对了但页面白屏也算过就太松了
  const viewUrl = ideView?.webContents.getURL() ?? ''
  const workbench = await until(
    () => viewJs("document.querySelector('.monaco-workbench') ? 'yes' : document.readyState"),
    (v) => v === 'yes',
    60,
    500
  )
  console.log(
    viewUrl.startsWith('http://127.0.0.1:') && workbench === 'yes'
      ? `[probe] 71 工作台已渲染 → ${viewUrl.slice(0, 40)}… · .monaco-workbench 就位`
      : `[probe] 71 失败：viewUrl=${viewUrl.slice(0, 60)} workbench=${workbench}`
  )

  // 72 —— 布局同步：view 的实际矩形必须贴着渲染层占位区，且不能盖住右栏
  //
  // 这条是三栏布局的命门：右栏（对话/任务/变更/预览）和 view 是**两层**东西 ——
  // DOM 在渲染层，工作台是盖在上面的原生 view。view 一旦比占位区宽，
  // 右栏不会报错、探针 75~82 也照样绿（它们查的是 DOM），
  // 但用户看到的是「工作台铺满整行、右栏不见了」。所以必须拿两边的实测数字对。
  const rects = JSON.parse(
    String(
      await js(`(() => {
        const pick = (el) => {
          if (!el) return null
          const r = el.getBoundingClientRect()
          return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), right: Math.round(r.right) }
        }
        return JSON.stringify({
          stage: pick(document.querySelector('[data-testid="ide-stage"]')),
          side: pick(document.querySelector('[data-testid="ide-side"]')),
          win: { w: window.innerWidth }
        })
      })()`)
    )
  )
  const viewBounds = ideView?.getBounds() ?? null
  const stage = rects?.stage
  const side = rects?.side
  const aligned =
    viewBounds && stage
      ? Math.abs(viewBounds.x - stage.x) <= 2 &&
        Math.abs(viewBounds.width - stage.w) <= 2 &&
        Boolean(side) &&
        side.x >= viewBounds.x + viewBounds.width
      : false
  console.log(
    aligned
      ? `[probe] 72 布局同步 → 占位区 x=${stage.x} w=${stage.w}，view x=${viewBounds!.x} w=${viewBounds!.width}，右栏从 ${side.x} 起（未被盖住）`
      : `[probe] 72 失败：占位区=${JSON.stringify(stage)} 右栏=${JSON.stringify(side)} view=${JSON.stringify(viewBounds)} 窗口宽=${rects?.win?.w}`
  )
  await captureScreen('ide')
  if (process.argv.includes('--snapshot')) {
    // 渲染层 + 工作台各拍一张：都不依赖窗口是否在前台，
    // 两张的宽度对上，两层的位置关系就清楚了
    await captureSnapshot('layout')
  }

  // 73 —— 窗口尺寸变化（最大化/还原）后 view 也要跟住。
  // 用户几乎必然先开工作台再最大化 —— 若 resize 通知丢了，
  // view 停在旧尺寸，比占位区宽出的那一条正好压在右栏上。
  const measure = async (): Promise<{ stage: { x: number; w: number } | null; side: { x: number } | null; win: number }> =>
    JSON.parse(
      String(
        await js(`(() => {
          const pick = (el) => {
            if (!el) return null
            const r = el.getBoundingClientRect()
            return { x: Math.round(r.x), w: Math.round(r.width) }
          }
          return JSON.stringify({
            stage: pick(document.querySelector('[data-testid="ide-stage"]')),
            side: pick(document.querySelector('[data-testid="ide-side"]')),
            win: window.innerWidth
          })
        })()`)
      )
    )

  const snap = async (label: string): Promise<void> => {
    const before = await measure()
    mainWindow?.maximize()
    await new Promise((r) => setTimeout(r, 900))
    const after = await measure()
    const vb = ideView?.getBounds() ?? null
    const ok =
      vb && after.stage
        ? Math.abs(vb.width - after.stage.w) <= 2 && (after.side?.x ?? 0) >= vb.x + vb.width
        : false
    console.log(
      ok
        ? `[probe] ${label} 最大化后仍同步 → 窗口 ${before.win}→${after.win}，view w=${vb!.width}，右栏从 ${after.side!.x} 起`
        : `[probe] ${label} 失败：窗口 ${before.win}→${after.win} view=${JSON.stringify(vb)} 占位区=${JSON.stringify(after.stage)} 右栏=${JSON.stringify(after.side)}`
    )
    mainWindow?.unmaximize()
    await new Promise((r) => setTimeout(r, 900))
    if (!ok) await captureScreen('ide-max')
  }

  await snap('72b')
  // 再来一次还原态的确认 —— unmaximize 之后也要回到贴合格局
  {
    const m = await measure()
    const vb = ideView?.getBounds() ?? null
    const ok =
      vb && m.stage
        ? Math.abs(vb.width - m.stage.w) <= 2 && (m.side?.x ?? 0) >= vb.x + vb.width
        : false
    console.log(
      ok
        ? `[probe] 72c 还原后仍同步 → view w=${vb!.width}，右栏从 ${m.side!.x} 起`
        : `[probe] 72c 失败：view=${JSON.stringify(vb)} 占位区=${JSON.stringify(m.stage)} 右栏=${JSON.stringify(m.side)}`
    )
  }

  // 74 —— 停止
  await js("document.querySelector('[data-testid=\"ide-stop\"]')?.click() ?? null")
  const s2 = await until(async () => host.status(), (s) => s.phase === 'stopped', 40, 250)
  console.log(
    s2.phase === 'stopped'
      ? '[probe] 73 已停止 → phase=stopped'
      : `[probe] 73 失败：phase=${s2.phase}`
  )

  // 73 —— 进程树真的没了。只杀父进程会留下占着端口的孤儿，这里正是抓那种情况
  const deadCode = await until(
    () => tryFetch(`http://127.0.0.1:${s1.port}/`, 1500),
    (v) => v === -1,
    40,
    250
  )
  console.log(
    deadCode === -1
      ? `[probe] 74 端口已释放 → :${s1.port} 不再应答（进程树杀干净了）`
      : `[probe] 74 失败：:${s1.port} 仍在应答（HTTP ${deadCode}），有孤儿进程`
  )

  // 74 —— C2 商标：我们渲染的每一个字都必须干净
  const bodyText = String(await js('document.body.innerText'))
  const hits = FORBIDDEN_TRADEMARKS.filter((t) => bodyText.includes(t))
  console.log(
    hits.length === 0
      ? '[probe] 83 界面文案无禁用商标 → 仅出现 VSCodium'
      : `[probe] 83 失败：界面出现 ${hits.join(' / ')}`
  )
}

app.whenReady().then(() => {
  registerIpc()
  startWorkers()
  createWindow()
  const boot = async (): Promise<void> => {
    await initToolHost()
    await selfCheck()
    // 恢复放在链路就绪之后：要读库、还要把档位推给 tool-host，
    // 那时 tool-host 若还没握手完成，推送会丢，用户就得到「界面严格、实际标准」
    await restoreTier()
  }

  if (process.argv.includes('--smoke')) {
    mainWindow?.webContents.once('did-finish-load', () => void smoke())
  } else {
    void boot()
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  agentHost?.stop()
  toolHost?.stop()
  // IDE 是第三方进程树，必须显式杀干净 —— 否则退出后仍有一个 93MB 的
  // node 常驻，还占着端口，下次启动会拿到新端口而界面还指着旧的
  ideHost?.dispose()
})

/** 审批请求类型收窄（供渲染层类型推导，不参与运行） */
export type { ApprovalRequest }
