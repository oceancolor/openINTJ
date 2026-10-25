/**
 * 内嵌 IDE 契约（F27 / T2 方案 B：内嵌 VSCodium + Open VSX）
 *
 * ── 先说清楚一个例外 ──────────────────────────────────────────────
 * 本项目有一条硬约束「进程间通信不开本地 HTTP 端口」，那条约束管的是
 * **我们自己的进程**（P-01 / P-03 / P-04 之间一律 MessagePort）。
 * VSCodium 不是我们的进程 —— 它是第三方产品，把工作台提供给外部的唯一方式
 * 就是 HTTP。所以这里不是「违反约束」，而是**引入了一个外部产品边界**。
 *
 * 边界必须被收紧，四条缺一不可：
 *   1. 只绑 127.0.0.1（不绑 0.0.0.0，网络上其它机器访问不到）
 *   2. 每次启动生成随机连接令牌，不复用、不落盘明文、不出主进程
 *   3. 应用退出时必须把它杀干净 —— 留下一个常驻 Node 进程是最难归因的资源泄漏
 *   4. 令牌不过渲染层：界面只拿得到 origin，拿不到令牌
 *
 * ── 为什么直启 server-main.js，而不用 `codium-tunnel serve-web` ────
 * 这是实测出来的，不是偏好：
 *
 * `serve-web` 的首次启动**会去联网下载 server 本体**（实测 94,038,270 字节，
 * 本机耗时 99 秒），下载期间每个 HTTP 请求都返回 202 + 一页
 * "The latest version of the VSCodium Server is downloading"。也就是说：
 *   - 首次打开是 100 秒白屏，而不是 3 秒；
 *   - 断网 / 内网环境直接不可用；
 *   - 版本号由远端决定，不可复现 —— 同一份代码两次跑出两个 server。
 *
 * 而下载下来的那份东西（VSCodium 的 `reh-web` 包）**本身就是一个可独立运行
 * 的 HTTP server**：`node.exe out/server-main.js`，参数与 serve-web 几乎一致
 * （实测支持 --host / --port 0 / --connection-token / --server-data-dir /
 *   --accept-server-license-terms / --telemetry-level）。
 *
 * 所以我们自己把 reh-web 包拉到 vendor/（scripts/vscodium/fetch.mjs），
 * 运行时直启它。代价是版本要自己管，收益是：秒起、离线可用、版本可复现。
 * 对一个「内嵌 IDE」来说，这个交换是划算的 —— 我们要的是确定的启动行为，
 * 不是自动更新。
 *
 * ── HTTP 握手（实测，别凭印象改）───────────────────────────────────
 *   无令牌             → 403 Forbidden（纯文本 "Forbidden."）
 *   错令牌             → 403 Forbidden
 *   正确 ?tkn=<token>  → 302 → Location: / 且下发 cookie vscode-tkn
 *   仅带 cookie 再取 / → 200 + workbench HTML
 *
 * 也就是说令牌校验是真的生效的，不是摆设。Electron 侧只要加载
 * `/?tkn=<token>`，让 302 把 cookie 种上即可，不必手工塞 cookie。
 */

/** IDE 的安装/运行状态 */
export type IdePhase =
  /** 没装（vendor 里没有，也没用环境变量指定） */
  | 'not-installed'
  /** 装了但没启动 */
  | 'idle'
  /** 正在拉起（等 stdout 出端口 / 等 HTTP 就绪） */
  | 'starting'
  /** 可用 */
  | 'ready'
  /** 启动失败或超时（BD-11：禁用面板 + 给重试入口） */
  | 'failed'
  /** 用户主动关闭 */
  | 'stopped'

export interface IdeStatus {
  phase: IdePhase
  /** 可执行文件位置（reh-web 包里的 node.exe）；未安装时为 null */
  binaryPath: string | null
  /** 从 reh-web 的 product.json 取到的版本串；取不到为 null */
  version: string | null
  /** 版本对应的 commit —— 定位「拉的是哪一份」全靠它 */
  commit: string | null
  /**
   * 工作台 **origin**（形如 `http://127.0.0.1:58101/`），**不含令牌**。
   *
   * 为什么不给完整 URL：令牌是秘密，它只该存在于主进程。
   * 渲染层拿 origin 够用了 —— 真正加载页面的是主进程里的 WebContentsView。
   */
  url: string | null
  /** 实际监听端口（仅 ready 时有值） */
  port: number | null
  /** 打开的工作区目录 */
  workspaceFolder: string | null
  /** 失败原因 —— 界面必须原样显示，不能只说「打不开」 */
  error: string | null
  /** 启动耗时（毫秒）；超时判定与排障都要看它 */
  elapsedMs: number | null
}

/** 启动参数 */
export interface IdeStartParams {
  /** 要打开的工作区目录；缺省用工作区根 */
  workspaceFolder?: string
}

export interface IdeStartResult {
  ok: boolean
  status: IdeStatus
}

export interface IdeStopResult {
  ok: boolean
  status: IdeStatus
}

/**
 * 工作台窗口区（DIP，相对窗口内容区左上角）。
 *
 * 由渲染层上报、主进程据此 setBounds。传 null 表示「当前不显示」
 * （切走了标签页）—— 此时必须把 view 移出，而不是叠一个透明层：
 * 叠层会让工作台继续在后台跑着接收键盘事件。
 */
export interface IdeLayoutRect {
  x: number
  y: number
  width: number
  height: number
}

/** 目标：把 IDE 拉到可运行状态的耗时上限（BD-11 定的是 10 秒） */
export const IDE_START_TIMEOUT_MS = 10_000

/** 就绪探测间隔 */
export const IDE_READY_POLL_MS = 250

/**
 * 面板里允许出现的品牌词只有 VSCodium。
 *
 * C2 商标约束：我们自己的文案里不得出现 "Visual Studio Code" 与 "VS Code"。
 * VSCodium 自己界面里写什么是它的事，但**我们的**字符串必须干净 ——
 * 探针会去我们的界面文本里搜这两个词（与终端的 AC-6 同源）。
 */
export const FORBIDDEN_TRADEMARKS = ['Visual Studio Code', 'VS Code'] as const

/** IDE 事件通道（主进程 → 渲染层） */
export const IDE_CHANNELS = {
  status: 'ide.status'
} as const

/** 空闲态：装了但没启动 */
export function idleStatus(binaryPath: string | null, version: string | null, commit: string | null): IdeStatus {
  return {
    phase: binaryPath ? 'idle' : 'not-installed',
    binaryPath,
    version,
    commit,
    url: null,
    port: null,
    workspaceFolder: null,
    error: null,
    elapsedMs: null
  }
}
