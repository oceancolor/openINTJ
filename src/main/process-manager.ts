/**
 * P-01 主进程 · 子进程管理器
 *
 * 职责：以 utilityProcess 承载 P-03 agent-host 与 P-04 tool-host，
 * 通过 MessagePort 双向通信（**不开本地 HTTP 端口**，见《系统设计》§3.1 与 C8）。
 *
 * 设计要点：
 *   - 每个子进程一条 MessageChannel，主进程持有远端 port
 *   - 请求/响应以 requestId 关联，超时即 reject（不无限等待）
 *   - 子进程崩溃由主进程监护并上报状态，不静默失败
 */

import { utilityProcess, MessageChannelMain, type MessagePortMain, type UtilityProcess } from 'electron'
import type { WireEnvelope } from '../shared/ipc'

type PendingResolver = {
  resolve: (v: unknown) => void
  reject: (e: Error) => void
  timer: NodeJS.Timeout
}

export interface WorkerHostOptions {
  /** 打包后是 out/ 下的 js 文件绝对路径 */
  modulePath: string
  serviceName: string
  /** 默认请求超时 */
  requestTimeoutMs?: number
  /** 追加到子进程环境变量（不透传主进程全部环境，按需显式给） */
  env?: Record<string, string>
}

export type WorkerStatus = 'starting' | 'online' | 'offline' | 'crashed'

/** 子进程反向发来的请求（P-03 要 P-04 做事时，经 P-01 转发） */
export type RequestHandler = (params: unknown) => Promise<unknown>

export class WorkerHost {
  private child: UtilityProcess | null = null
  private port: MessagePortMain | null = null
  private pending = new Map<string, PendingResolver>()
  private handlers = new Map<string, RequestHandler>()
  private seq = 0
  private eventHandlers = new Set<(channel: string, payload: unknown) => void>()
  private statusHandlers = new Set<(status: WorkerStatus, reason?: string) => void>()

  status: WorkerStatus = 'starting'

  constructor(private readonly opts: WorkerHostOptions) {}

  get serviceName(): string {
    return this.opts.serviceName
  }

  start(): void {
    const child = utilityProcess.fork(this.opts.modulePath, [], {
      serviceName: this.opts.serviceName,
      stdio: 'inherit',
      env: { ...process.env, ...(this.opts.env ?? {}) }
    })
    this.child = child

    const { port1, port2 } = new MessageChannelMain()

    // port1 交给子进程（随消息转移所有权）
    child.postMessage({ kind: 'port-init' }, [port1])

    this.port = port2
    port2.start()
    port2.on('message', (msg) => {
      this.handleMessage(msg.data as WireEnvelope)
    })

    child.on('spawn', () => {
      this.setStatus('online')
    })

    child.on('exit', (code) => {
      // 非零退出视为崩溃；零退出视为正常下线
      this.setStatus(code === 0 ? 'offline' : 'crashed', code === 0 ? undefined : `exit code ${code}`)
      this.rejectAll(new Error(`${this.opts.serviceName} 已退出（code=${code}）`))
      this.port = null
      this.child = null
    })
  }

  stop(): void {
    if (this.child) {
      this.child.kill()
    }
  }

  onEvent(handler: (channel: string, payload: unknown) => void): () => void {
    this.eventHandlers.add(handler)
    return () => this.eventHandlers.delete(handler)
  }

  onStatus(handler: (status: WorkerStatus, reason?: string) => void): () => void {
    this.statusHandlers.add(handler)
    return () => this.statusHandlers.delete(handler)
  }

  /**
   * 注册子进程可反向调用的方法。
   *
   * 用途：P-03 需要 P-04 执行命令，但**不开 P-03 ↔ P-04 直连通道** ——
   * 一切跨进程调用都经 P-01 路由，这样「谁调用了什么」在主进程有唯一入口。
   */
  handle(method: string, handler: RequestHandler): void {
    this.handlers.set(method, handler)
  }

  /** 发请求并等待响应；超时 reject，绝不无限等待 */
  request<T = unknown>(method: string, params?: unknown, timeoutMs?: number): Promise<T> {
    if (!this.port) {
      return Promise.reject(new Error(`${this.opts.serviceName} 尚未就绪`))
    }
    const requestId = `${this.opts.serviceName}-${++this.seq}`
    const waitMs = timeoutMs ?? this.opts.requestTimeoutMs ?? 30_000

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId)
        reject(new Error(`${this.opts.serviceName} 请求超时：${method}`))
      }, waitMs)

      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
        timer
      })

      const envelope: WireEnvelope = { kind: 'req', requestId, method, params }
      this.port!.postMessage(envelope)
    })
  }

  private handleMessage(env: WireEnvelope): void {
    if (env.kind === 'res') {
      const p = this.pending.get(env.requestId)
      if (!p) return
      clearTimeout(p.timer)
      this.pending.delete(env.requestId)
      if (env.ok) p.resolve(env.result)
      else p.reject(new Error(`${env.error.code}: ${env.error.message}`))
      return
    }
    if (env.kind === 'evt') {
      for (const h of this.eventHandlers) h(env.channel, env.payload)
      return
    }
    if (env.kind === 'req') {
      // 子进程反向请求：找不到处理器要明确回错，不能静默丢弃（否则子进程侧只能等到超时）
      const handler = this.handlers.get(env.method)
      if (!handler) {
        this.port?.postMessage({
          kind: 'res',
          requestId: env.requestId,
          ok: false,
          error: { code: 'NO_HANDLER', message: `主进程未注册方法：${env.method}` }
        } satisfies WireEnvelope)
        return
      }
      handler(env.params).then(
        (result) => {
          this.port?.postMessage({
            kind: 'res',
            requestId: env.requestId,
            ok: true,
            result
          } satisfies WireEnvelope)
        },
        (err: Error) => {
          this.port?.postMessage({
            kind: 'res',
            requestId: env.requestId,
            ok: false,
            error: { code: 'FORWARD_ERROR', message: err.message }
          } satisfies WireEnvelope)
        }
      )
    }
  }

  private setStatus(status: WorkerStatus, reason?: string): void {
    if (this.status === status) return
    this.status = status
    for (const h of this.statusHandlers) h(status, reason)
  }

  private rejectAll(err: Error): void {
    for (const [, p] of this.pending) {
      clearTimeout(p.timer)
      p.reject(err)
    }
    this.pending.clear()
  }
}
