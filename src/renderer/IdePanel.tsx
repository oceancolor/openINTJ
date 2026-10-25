/**
 * 内嵌 IDE 面板（F27 / D5）
 *
 * 关键形态：**这里只有一个占位区，真正的工作台是主进程盖在上面的一层
 * WebContentsView**。
 *
 * 为什么不让渲染层自己 iframe 一个工作台：
 *   1. CSP —— 第三方工作台要放松 CSP 才能跑，而 CSP 是**整个渲染进程**一份的。
 *      为一个面板把主界面的安全基线降下来，得不偿失。
 *   2. 隔离 —— 主进程给工作台单独一个 session 分区（persist:ide），
 *      它的 cookie / localStorage 与我们的界面天然隔开。
 *
 * 代价（写清楚，不是缺陷）：工作台区域对渲染层来说是「一块透明的洞」，
 * 我们无法在上面叠自己的 DOM，也无法截获它的键盘事件。所以面板状态条
 * 放在占位区**上方**，而不是盖在工作台上。
 *
 * 商标约束（C2）：我们自己的文案里只出现 VSCodium。
 * 工作台界面里写什么由它自己决定，但**我们渲染的每一个字**必须是干净的。
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { IdePhase, IdeStatus } from '../shared/ide'

interface Props {
  /** 页面是否可见；不可见时必须上报 null，否则工作台会盖住别的页面 */
  active: boolean
}

const PHASE_LABEL: Record<IdePhase, string> = {
  'not-installed': '未安装 VSCodium 服务端',
  idle: '已安装，未启动',
  starting: '正在启动 …',
  ready: '工作台已就绪',
  failed: '启动失败',
  stopped: '已停止'
}

const PHASE_DOT: Record<IdePhase, string> = {
  'not-installed': 'off',
  idle: 'idle',
  starting: 'idle',
  ready: 'on',
  failed: 'err',
  stopped: 'off'
}

export default function IdePanel({ active }: Props): React.ReactElement {
  const [status, setStatus] = useState<IdeStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  /** 占位区 —— 它的矩形就是工作台该出现的地方 */
  const stageRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    let alive = true
    void window.openintj.ideStatus().then((s) => {
      if (alive) setStatus(s)
    })
    const off = window.openintj.onIdeStatus(setStatus)
    return () => {
      alive = false
      off()
    }
  }, [])

  /**
   * 把占位区的矩形报给主进程。
   *
   * 三件事会让它变：切标签页、窗口缩放、侧栏折叠。所以既监听 ResizeObserver
   * 也监听 window resize —— 只挂 ResizeObserver 的话，窗口整体缩放但占位区
   * 尺寸恰好按同比例变化时，某些浏览器不会回调。
   */
  useEffect(() => {
    const el = stageRef.current
    if (!el) return

    const report = (): void => {
      const visible = active && status?.phase === 'ready'
      if (!visible) {
        void window.openintj.ideLayout(null)
        return
      }
      const r = el.getBoundingClientRect()
      void window.openintj.ideLayout({ x: r.left, y: r.top, width: r.width, height: r.height })
    }

    report()
    const ro = new ResizeObserver(report)
    ro.observe(el)
    window.addEventListener('resize', report)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', report)
      // 收摊时一定要收回去：不报 null 的话工作台会一直盖在别的页面上
      void window.openintj.ideLayout(null)
    }
  }, [active, status?.phase])

  const start = useCallback(async (): Promise<void> => {
    setBusy(true)
    setNotice(null)
    try {
      const res = await window.openintj.ideStart({})
      setStatus(res.status)
      if (!res.ok) setNotice(res.status.error ?? '启动失败，原因未知')
    } catch (e) {
      setNotice(`启动失败：${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }, [])

  const stop = useCallback(async (): Promise<void> => {
    setBusy(true)
    try {
      setStatus((await window.openintj.ideStop()).status)
      setNotice(null)
    } finally {
      setBusy(false)
    }
  }, [])

  const phase: IdePhase = status?.phase ?? 'idle'
  const canStart = !busy && (phase === 'idle' || phase === 'stopped' || phase === 'failed')
  const canStop = !busy && (phase === 'ready' || phase === 'starting')

  return (
    <div className="ide-wrap" data-testid="ide-panel">
      <div className="ide-status">
        <span className="mono ide-dot" data-state={PHASE_DOT[phase]} />
        <span className="mono ide-status-text">{PHASE_LABEL[phase]}</span>
        {status?.version && <span className="mono ide-version">VSCodium {status.version}</span>}
        {status?.elapsedMs != null && phase === 'ready' && (
          <span className="mono ide-version">启动耗时 {status.elapsedMs}ms</span>
        )}
        <span className="ide-status-actions">
          <button
            className="pill"
            onClick={() => void start()}
            disabled={!canStart}
            data-testid="ide-start"
          >
            {busy && phase === 'starting' ? '启动中 …' : '启动'}
          </button>
          <button className="pill" onClick={() => void stop()} disabled={!canStop} data-testid="ide-stop">
            停止
          </button>
        </span>
      </div>

      {notice && (
        <div className="ide-notice" data-testid="ide-notice">
          <span>{notice}</span>
        </div>
      )}

      {/* 工作台就盖在这块区域上。渲染层只负责留位置和报坐标。 */}
      <div className="ide-stage" ref={stageRef} data-testid="ide-stage">
        {phase !== 'ready' && (
          <div className="ide-placeholder">
            {phase === 'not-installed' ? (
              <>
                <p>
                  未找到 VSCodium 服务端。先跑 <code>npm run vscodium:fetch</code> 拉取（约 94MB），
                  或用环境变量 <code>OPENINTJ_VSCODIUM_SERVER</code> 指向已有的 reh-web 目录。
                </p>
                <p className="ide-placeholder-dim">
                  我们直启的是 VSCodium 的 reh-web 包，不走 <code>serve-web</code> ——
                  后者首次启动会联网下载服务端本体（实测 99 秒），断网环境直接不可用。
                </p>
              </>
            ) : phase === 'failed' ? (
              <p>{status?.error ?? '启动失败'}</p>
            ) : (
              <p>点「启动」打开内嵌工作台，工作区为当前项目根目录。</p>
            )}
          </div>
        )}
      </div>

      <div className="ide-hint">
        工作台跑在 <code>127.0.0.1</code> 的随机端口上，每次启动现生成随机连接令牌，令牌不出主进程；
        停止时会连同子进程一起杀干净。扩展市场走 Open VSX。
      </div>
    </div>
  )
}
