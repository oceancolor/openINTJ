/**
 * 内嵌 IDE 宿主（F27 / D5）
 *
 * 直接拉起 VSCodium 的 reh-web 包：
 *   node.exe out/server-main.js --host 127.0.0.1 --port 0 --connection-token <t> ...
 *
 * 为什么不是 `codium-tunnel serve-web`，见 ../shared/ide.ts 的头注释 ——
 * 那条路会在首次请求时联网下载 94MB，实测 99 秒。这里是秒起。
 *
 * 三条安全收紧（与契约一致，缺一不可）：
 *   1. --host 127.0.0.1 —— 不对外暴露
 *   2. 每次启动现生成随机令牌，只留在内存里
 *   3. 停止时用 taskkill /T /F 杀整棵树 —— 它会派生扩展宿主进程，
 *      只杀父进程会留下孤儿，而孤儿还占着端口
 */

import { spawn, type ChildProcess } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { app } from 'electron'
import {
  IDE_READY_POLL_MS,
  IDE_START_TIMEOUT_MS,
  idleStatus,
  type IdeStartParams,
  type IdeStatus
} from '../shared/ide'

/** 定位结果 */
interface ServerLayout {
  /** reh-web 包根目录 */
  root: string
  /** 包内自带的 node 运行时（93MB，不依赖系统 node 版本） */
  nodeExe: string
  /** server 入口 */
  main: string
}

/** 版本号来源：包里的 product.json */
interface ServerProduct {
  version?: string
  commit?: string
}

/**
 * 候选位置。
 *
 * 顺序有讲究：环境变量优先，因为它是「我明确要指定一份」的意图，
 * 应该能盖过打包时带的那一份（换版本调试时不该重新打包）。
 */
function layoutCandidates(): string[] {
  const out: string[] = []
  const env = process.env['OPENINTJ_VSCODIUM_SERVER']
  if (env) out.push(env)
  if (app.isPackaged) {
    // 打包态：vendor 随资源一起分发到 resources/
    out.push(join(process.resourcesPath, 'vscodium-server'))
  } else {
    // 开发态：__dirname = <root>/out/main
    out.push(join(__dirname, '../../vendor/vscodium-server'))
  }
  return out
}

/**
 * 包内自带的 node 运行时叫什么 —— 与 scripts/vscodium/fetch.mjs 的
 * nodeRelative() 必须一致，改一边忘另一边会「拉下来了却显示未安装」。
 */
function nodeName(): string {
  return process.platform === 'win32' ? 'node.exe' : 'node'
}

/** 找到一份可用的 reh-web 包；找不到返回 null（界面据此显示 not-installed） */
export function locateServer(): ServerLayout | null {
  for (const root of layoutCandidates()) {
    const nodeExe = join(root, nodeName())
    const main = join(root, 'out', 'server-main.js')
    if (existsSync(nodeExe) && existsSync(main)) return { root, nodeExe, main }
  }
  return null
}

/** 读版本号。读不出来不算致命 —— 界面显示 unknown 也比打不开 IDE 好 */
function readProduct(root: string): ServerProduct {
  try {
    return JSON.parse(readFileSync(join(root, 'product.json'), 'utf8')) as ServerProduct
  } catch {
    return {}
  }
}

/**
 * 杀掉整棵进程树。
 *
 * 为什么不能只 `proc.kill()`：server 会派生扩展宿主（Extension host agent），
 * 那些子进程不跟着父进程退出。留下来的后果不只是占内存 —— 它们还占着端口，
 * 下次启动会拿到另一个端口，而界面上旧的 WebContentsView 还指着死掉的那个。
 */
function killTree(proc: ChildProcess): void {
  const pid = proc.pid
  if (!pid) return
  try {
    if (process.platform === 'win32') {
      // /T 杀子树，/F 强制。缺 /T 就会留下孤儿
      spawn('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' })
    } else {
      // 负号 = 进程组；配合 detached: true 才会是组 leader
      process.kill(-pid, 'SIGTERM')
    }
  } catch {
    /* 已经退出了，忽略 */
  }
  try {
    proc.kill('SIGKILL')
  } catch {
    /* 同上 */
  }
}

/** 工作台地址。folder 走查询参数 —— server-main.js 没有 --default-folder 这个参数 */
function workbenchUrl(port: number, folder: string | null, token: string | null): string {
  const base = `http://127.0.0.1:${port}/`
  const qs: string[] = []
  if (folder) qs.push(`folder=${encodeURIComponent(folder)}`)
  if (token) qs.push(`tkn=${encodeURIComponent(token)}`)
  return qs.length ? `${base}?${qs.join('&')}` : base
}

export class VscodiumHost {
  private proc: ChildProcess | null = null
  private token: string | null = null
  private port: number | null = null
  private folder: string | null = null
  private phase: IdeStatus['phase'] = 'not-installed'
  private error: string | null = null
  private lastElapsedMs: number | null = null
  private buf = ''
  private starting: Promise<IdeStatus> | null = null

  constructor(private readonly onStatus?: (s: IdeStatus) => void) {}

  /** 当前状态快照 */
  status(): IdeStatus {
    const layout = locateServer()
    const product = layout ? readProduct(layout.root) : {}
    const version = product.version ?? null
    const commit = product.commit ?? null

    if (!layout) {
      return { ...idleStatus(null, null, null), phase: 'not-installed', error: this.error }
    }

    return {
      phase: this.phase === 'not-installed' ? 'idle' : this.phase,
      binaryPath: layout.nodeExe,
      version,
      commit,
      url: this.port ? `http://127.0.0.1:${this.port}/` : null,
      port: this.port,
      workspaceFolder: this.folder,
      error: this.error,
      elapsedMs: this.lastElapsedMs
    }
  }

  /**
   * 拉起 IDE。已经在跑就直接返回当前状态 —— 重复点「启动」不该起第二个进程。
   */
  async start(params: IdeStartParams = {}): Promise<IdeStatus> {
    if (this.starting) return await this.starting
    if (this.phase === 'ready') return this.status()

    this.starting = this.doStart(params).finally(() => {
      this.starting = null
    })
    return await this.starting
  }

  private async doStart(params: IdeStartParams): Promise<IdeStatus> {
    const t0 = Date.now()
    const layout = locateServer()
    if (!layout) {
      this.phase = 'not-installed'
      this.error =
        '未找到 VSCodium 服务端。先跑 `npm run vscodium:fetch` 拉取（约 94MB），' +
        '或用环境变量 OPENINTJ_VSCODIUM_SERVER 指向已有的 reh-web 目录。'
      return this.settle()
    }

    this.phase = 'starting'
    this.error = null
    this.emit()

    const token = randomBytes(24).toString('base64url')
    this.token = token
    this.folder = params.workspaceFolder ?? null

    // server 启动时会去扫这两个目录；不存在就刷一堆非致命报错，
    // 把真正的失败原因淹掉。先建好。
    const dataDir = join(app.getPath('userData'), 'ide', 'server')
    for (const sub of ['extensions', join('data', 'User')]) {
      mkdirSync(join(dataDir, sub), { recursive: true })
    }

    const proc = spawn(
      layout.nodeExe,
      [
        layout.main,
        '--host',
        '127.0.0.1',
        '--port',
        '0', // 交给我们随机，避免与本机已有服务撞端口
        '--connection-token',
        token,
        '--accept-server-license-terms',
        '--telemetry-level',
        'off',
        '--server-data-dir',
        dataDir
      ],
      {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
        // POSIX 下成组，stop() 才能整组杀掉
        detached: process.platform !== 'win32',
        env: { ...process.env }
      }
    )
    this.proc = proc
    this.buf = ''

    const collect = (d: Buffer): void => {
      this.buf += d.toString()
      // 只留最后 8KB：端口行早就出来了，留全量没有意义
      if (this.buf.length > 8192) this.buf = this.buf.slice(-8192)
    }
    proc.stdout?.on('data', collect)
    proc.stderr?.on('data', collect)

    proc.on('exit', () => {
      if (this.proc === proc) {
        this.proc = null
        this.port = null
        this.token = null
        // 正常停止时 phase 已被 stop() 置为 stopped，这里只处理「自己挂了」
        if (this.phase !== 'stopped') {
          this.phase = 'failed'
          this.error = this.error ?? 'IDE 进程意外退出'
          this.emit()
        }
      }
    })

    // 第一步：等它把端口报出来
    const port = await this.waitForPort(IDE_START_TIMEOUT_MS)
    if (!port) {
      this.phase = 'failed'
      this.error = `等待端口上报超时（${IDE_START_TIMEOUT_MS}ms）。服务端输出：${this.tailOfLog()}`
      killTree(proc)
      return this.settle(t0)
    }
    this.port = port

    // 第二步：等 HTTP 真的能答。端口 listening ≠ 工作台能开 ——
    // 中间还有一段扩展宿主初始化的时间，只用端口判定会让界面白屏
    const ready = await this.waitForHttp(IDE_START_TIMEOUT_MS - (Date.now() - t0))
    if (!ready.ok) {
      this.phase = 'failed'
      this.error = `工作台未就绪：${ready.reason}`
      killTree(proc)
      this.port = null
      return this.settle(t0)
    }

    this.phase = 'ready'
    this.error = null
    return this.settle(t0)
  }

  /** 从 stdout 里取端口。两种行都要认，不同版本写法不一样 */
  private async waitForPort(budgetMs: number): Promise<number | null> {
    const deadline = Date.now() + budgetMs
    const re = /Server bound to [\d.]+:(\d+)/ // 例：Server bound to 127.0.0.1:58101 (IPv4)
    const re2 = /Web UI available at https?:\/\/[^:]+:(\d+)/ // 例：Web UI available at http://localhost:58101
    while (Date.now() < deadline) {
      const m = re.exec(this.buf) ?? re2.exec(this.buf)
      if (m) return Number(m[1])
      if (this.proc?.exitCode !== null && this.proc?.exitCode !== undefined) return null
      await sleep(100)
    }
    return null
  }

  /**
   * 轮询工作台。
   *
   * 判定「就绪」的口径是**拿到 302**（正确令牌 → 302 种 cookie），
   * 而不是「连得上」。连得上但还没初始化完的时候它也会答，只是给的是半成品。
   */
  private async waitForHttp(budgetMs: number): Promise<{ ok: boolean; reason: string }> {
    const deadline = Date.now() + Math.max(1000, budgetMs)
    let last = '未发起请求'
    while (Date.now() < deadline) {
      try {
        const res = await fetch(workbenchUrl(this.port!, this.folder, this.token), {
          redirect: 'manual'
        })
        if (res.status === 302 || res.status === 200) return { ok: true, reason: '' }
        last = `HTTP ${res.status}`
      } catch (e) {
        last = (e as Error).message
      }
      await sleep(IDE_READY_POLL_MS)
    }
    return { ok: false, reason: `超时仍不可用于加载：${last}` }
  }

  /** 停止。杀不干净比不停止更糟 —— 所以 /T 是必需的 */
  async stop(): Promise<IdeStatus> {
    if (this.proc) {
      killTree(this.proc)
      this.proc = null
    }
    this.port = null
    this.token = null
    this.phase = 'stopped'
    this.error = null
    return this.settle()
  }

  /** 退出前必须调用：不做的话会留下一个常驻的 93MB node 进程 */
  dispose(): void {
    if (this.proc) {
      killTree(this.proc)
      this.proc = null
    }
  }

  /** 供主进程拼地址用（带令牌）。令牌绝不出主进程，所以没有对外暴露 token 的口子 */
  urlWithToken(): string | null {
    if (!this.port || !this.token) return null
    return workbenchUrl(this.port, this.folder, this.token)
  }

  /** 仅探针使用：验证令牌校验确实生效（拿错令牌应当被拒） */
  urlWithTokenFor(token: string): string | null {
    if (!this.port) return null
    return workbenchUrl(this.port, this.folder, token)
  }

  private settle(t0?: number): IdeStatus {
    if (t0 !== undefined) this.lastElapsedMs = Date.now() - t0
    const s = this.status()
    this.emit()
    return s
  }

  private emit(): void {
    this.onStatus?.(this.status())
  }

  /** 失败时给一段服务端输出，否则用户只看到「超时」三个字，无从下手 */
  private tailOfLog(): string {
    const t = this.buf.replace(/\s+/g, ' ').trim()
    if (!t) return '(无输出)'
    return t.slice(-300)
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}
