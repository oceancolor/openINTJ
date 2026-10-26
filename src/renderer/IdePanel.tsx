/**
 * 内嵌 IDE 面板（F27 / D5）
 *
 * 关键形态：**左侧是占位区，真正的工作台是主进程盖在上面的一层
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
 * ── 三栏分工 ────────────────────────────────────────────────────────
 *   左两栏：工作台自己负责（资源管理器 + 编辑器），我们不动它，只给它留位置
 *   右一栏：本面板新增 —— 对话 / 任务 / 变更 / 预览 四个分页签
 *
 * 右栏用分页签而不是「再开一栏」，是因为宽度有限：固定 380px 的栏位里
 * 并排放对话和预览，两个都看不清。分页签让它们**复用同一块面积**，
 * 代价是一次只能看一个 —— 而「边看代码边看对话」才是真实用法，
 * 预览只在需要时切过去看一眼。
 *
 * 「预览」签只在**真的有预览目标时**才出现。常驻一个空白签的结果是：
 * 用户点进去看到空面板，会以为功能坏了，而不是「还没发生」。
 *
 * 商标约束（C2）：我们自己的文案里只出现 VSCodium。
 * 工作台界面里写什么由它自己决定，但**我们渲染的每一个字**必须是干净的。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { IdePhase, IdeStatus } from '../shared/ide'
import type { ApprovalRequest } from '../shared/approval'
import { TIER_LABELS } from '../shared/approval'
import type { GoalContract } from '../shared/contract'
import type {
  AgentSession,
  ChatMessage,
  FileChange,
  PreviewTarget,
  TaskItem
} from '../shared/session'

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

const STATE_TONE: Record<string, string> = {
  CONTRACT_PENDING: 'pending',
  CONFIRMED: 'ok',
  RUNNING: 'ok',
  PAUSED: 'warn',
  INTERRUPTED: 'warn',
  CRITIC_REVIEW: 'ok',
  REPLANNING: 'warn',
  DELIVERED: 'done',
  FAILED: 'err'
}

const CHANGE_LABEL: Record<FileChange['kind'], string> = {
  added: '新增',
  modified: '修改',
  deleted: '删除'
}

export default function IdePanel({ active }: Props): React.ReactElement {
  const [status, setStatus] = useState<IdeStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  /** 占位区 —— 它的矩形就是工作台该出现的地方 */
  const stageRef = useRef<HTMLDivElement | null>(null)

  /* ---- 右栏：会话 ---- */

  const [sessions, setSessions] = useState<AgentSession[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [tasks, setTasks] = useState<TaskItem[]>([])
  const [changes, setChanges] = useState<FileChange[]>([])
  const [previews, setPreviews] = useState<PreviewTarget[]>([])
  const [tab, setTab] = useState<'chat' | 'tasks' | 'changes' | 'preview'>('chat')
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)
  const [running, setRunning] = useState(false)
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([])
  const [responding, setResponding] = useState<string[]>([])
  const [manualUrl, setManualUrl] = useState('')
  const [tier, setTier] = useState<string>('standard')

  /** 当前会话。注意别叫 active —— 那一个是「页面是否可见」的 prop */
  const currentSession = useMemo(
    () => sessions.find((s) => s.sessionId === activeId) ?? null,
    [sessions, activeId]
  )

  const refreshSessions = useCallback(async (): Promise<AgentSession[]> => {
    const list = await window.openintj.sessionList()
    setSessions(list)
    return list
  }, [])

  const refreshMessages = useCallback(async (sessionId: string): Promise<void> => {
    setMessages(await window.openintj.sessionMessages(sessionId))
  }, [])

  const refreshPanels = useCallback(async (sessionId: string): Promise<void> => {
    const [t, c, p] = await Promise.all([
      window.openintj.sessionTasks(sessionId),
      window.openintj.sessionChanges(sessionId),
      window.openintj.sessionPreview(sessionId)
    ])
    setTasks(t)
    setChanges(c)
    setPreviews(p)
  }, [])

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

  /**
   * 登记一个预览地址（只接受回环地址）。
   *
   * 手动入口的意义：dev server 的提示行不总出现在命令输出里
   * （后台起的服务、别人起的服务），这时候用户得能自己指一个地址。
   */
  const addPreview = useCallback(async (): Promise<void> => {
    if (!activeId || !manualUrl.trim()) return
    try {
      const res = await window.openintj.sessionAddPreview({ sessionId: activeId, url: manualUrl })
      if (!res.ok) setNotice(res.reason ?? '登记失败')
      else setManualUrl('')
    } catch (e) {
      setNotice(`登记失败：${(e as Error).message}`)
    }
    await refreshPanels(activeId)
  }, [activeId, manualUrl, refreshPanels])

  /** 用户签收一条文件变化 —— 只打标记，不回滚、不动文件 */
  const confirmChange = useCallback(
    async (changeId: string): Promise<void> => {
      await window.openintj.sessionConfirmChange(changeId)
      if (activeId) await refreshPanels(activeId)
    },
    [activeId, refreshPanels]
  )

  /** 首次挂载：没有会话就先开一个 —— 空白的右栏比没有右栏更让人不知所措 */
  useEffect(() => {
    void (async () => {
      const list = await refreshSessions()
      if (list.length === 0) {
        const created = await window.openintj.sessionCreate('会话 1')
      setSessions([created])
      setActiveId(created.sessionId)
      } else {
        setActiveId((prev) => prev ?? list[0]!.sessionId)
      }
    })()
  }, [refreshSessions])

  /** 切换会话：消息、任务、变更、预览都要跟着换，不能留在上一个会话的视图上 */
  useEffect(() => {
    if (!activeId) return
    void (async () => {
      await refreshMessages(activeId)
      await refreshPanels(activeId)
    })()
  }, [activeId, refreshMessages, refreshPanels])

  /**
   * 执行流有动静就刷新派生面板。
   *
   * 只认「本会话的 run」：别的会话（或控制台手动发起的 run）在跑时，
   * 右栏不该跟着跳 —— 那会让人以为自己的对话在执行别的任务。
   */
  useEffect(() => {
    const off = window.openintj.onEvent((e) => {
      if (!activeId) return
      if (e.runId !== activeId && e.runId !== (currentSession?.runId ?? '')) return
      void refreshPanels(activeId)
    })
    return off
  }, [activeId, currentSession?.runId, refreshPanels])

  /** 文件变化：只收本会话的 run 的 */
  useEffect(() => {
    const off = window.openintj.onFileChanged((c) => {
      if (!activeId) return
      if (c.runId !== (currentSession?.runId ?? '')) return
      void refreshPanels(activeId)
    })
    return off
  }, [activeId, currentSession?.runId, refreshPanels])

  /** 审批卡片：卡片是否消失由 ApprovalDecided 决定，不由「谁点的按钮」决定 */
  useEffect(() => {
    const offReq = window.openintj.onApprovalRequest((r) => {
      setApprovals((prev) => (prev.some((x) => x.requestId === r.requestId) ? prev : [...prev, r]))
    })
    const offCancel = window.openintj.onApprovalCancel((p) => {
      const gone = new Set(p.requestIds)
      setApprovals((prev) => prev.filter((x) => !gone.has(x.requestId)))
    })
    return () => {
      offReq()
      offCancel()
    }
  }, [])

  useEffect(() => {
    void window.openintj.getApprovalTier().then(setTier).catch(() => undefined)
  }, [])

  /** IDE 状态订阅（原有） */
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
   * 三件事会让它变：切标签页、窗口缩放、侧栏折叠（这里指右栏宽度变化）。
   * 所以既监听 ResizeObserver 也监听 window resize —— 只挂 ResizeObserver
   * 的话，窗口整体缩放但占位区尺寸恰好按同比例变化时，某些浏览器不会回调。
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

  const send = useCallback(async (): Promise<void> => {
    const text = draft.trim()
    if (!text || sending) return
    setSending(true)
    setNotice(null)
    try {
      const res = await window.openintj.sessionSend({ sessionId: activeId ?? undefined, text })
      setActiveId(res.session.sessionId)
      setSessions(await window.openintj.sessionList())
      setMessages(res.messages)
      await refreshPanels(res.session.sessionId)
      setDraft('')
    } catch (e) {
      setNotice(`发送失败：${(e as Error).message}`)
    } finally {
      setSending(false)
    }
  }, [draft, sending, activeId, refreshPanels])

  const confirmContract = useCallback(
    async (accepted: boolean): Promise<void> => {
      if (!activeId) return
      setBusy(true)
      try {
        const res = await window.openintj.sessionConfirmContract({
          sessionId: activeId,
          accepted
        })
        if (!res.ok) setNotice(res.reason ?? '契约确认失败')
        else {
          setSessions(await window.openintj.sessionList())
          await refreshMessages(activeId)
          await refreshPanels(activeId)
        }
      } catch (e) {
        setNotice(`契约确认失败：${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }, [activeId, refreshMessages, refreshPanels])

  const runLoop = useCallback(async (): Promise<void> => {
    if (!currentSession?.runId || running) return
    setRunning(true)
    setNotice(null)
    try {
      const res = await window.openintj.startAgentLoop(currentSession.runId)
      if (res.outcome !== 'delivered') {
        setNotice(`本轮未交付（${res.outcome}）：${res.reason ?? '无说明'}`)
      }
      await refreshMessages(activeId!)
      await refreshPanels(activeId!)
    } catch (e) {
      setNotice(`执行失败：${(e as Error).message}`)
    } finally {
      setRunning(false)
    }
  }, [currentSession?.runId, running, activeId, refreshMessages, refreshPanels])

  const newSession = useCallback(async (): Promise<void> => {
    const created = await window.openintj.sessionCreate(`会话 ${sessions.length + 1}`)
    setSessions(await window.openintj.sessionList())
    setActiveId(created.sessionId)
    setTab('chat')
  }, [sessions.length])

  const respond = useCallback(async (requestId: string, approved: boolean): Promise<void> => {
    setResponding((prev) => [...prev, requestId])
    try {
      await window.openintj.respondApproval(requestId, approved, false)
    } finally {
      setResponding((prev) => prev.filter((x) => x !== requestId))
    }
  }, [])

  /** 当前会话自己的审批卡 —— 别的会话的请求不该出现在这里 */
  const myApprovals = useMemo(
    () =>
      currentSession?.runId
        ? approvals.filter((a) => a.runId === currentSession.runId)
        : [],
    [approvals, currentSession?.runId]
  )

  const phase: IdePhase = status?.phase ?? 'idle'
  const canStart = !busy && (phase === 'idle' || phase === 'stopped' || phase === 'failed')
  const canStop = !busy && (phase === 'ready' || phase === 'starting')
  const hasPreview = previews.length > 0

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

      {/* 左两栏 = 工作台占位区；右一栏 = 对话 / 任务 / 变更 / 预览 */}
      <div className="ide-split">
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

        <aside className="ide-side" data-testid="ide-side">
          {/* 会话分页签：一段对话一个签，可新建 */}
          <div className="ide-side-sessions" data-testid="ide-session-tabs">
            {sessions.map((s, i) => (
              <button
                key={s.sessionId}
                className={'session-tab' + (s.sessionId === activeId ? ' active' : '')}
                onClick={() => setActiveId(s.sessionId)}
                title={s.title}
                data-session-id={s.sessionId}
              >
                {s.title || `会话 ${i + 1}`}
              </button>
            ))}
            <button className="session-tab new" onClick={() => void newSession()} data-testid="ide-session-new">
              ＋ 新会话
            </button>
          </div>

          {/* 功能分页签：复用右栏这一块面积 */}
          <div className="ide-side-tabs" data-testid="ide-side-tabs">
            <button className={'side-tab' + (tab === 'chat' ? ' active' : '')} onClick={() => setTab('chat')}>
              对话
            </button>
            <button className={'side-tab' + (tab === 'tasks' ? ' active' : '')} onClick={() => setTab('tasks')}>
              任务{tasks.length > 0 && <span className="tab-count">{tasks.length}</span>}
            </button>

            <button
              className={'side-tab' + (tab === 'changes' ? ' active' : '')}
              onClick={() => setTab('changes')}
            >
              变更{changes.length > 0 && <span className="tab-count">{changes.length}</span>}
            </button>
            {/* 没有预览目标就整个不出现 —— 常驻一个空白签会让人以为功能坏了 */}
            {hasPreview && (
              <button
                className={'side-tab' + (tab === 'preview' ? ' active' : '')}
                onClick={() => setTab('preview')}
              >
                预览
              </button>
            )}
          </div>

          <div className="ide-side-body" data-testid="ide-side-body">
            {tab === 'chat' && (
              <div className="chat">
                <div className="chat-list" data-testid="ide-chat-list">
                  {messages.length === 0 && (
                    <div className="empty">
                      还没有对话。在下面说一句话目标，模型会先把它起草成一份契约等你确认。
                    </div>
                  )}
                  {messages.map((m) => (
                    <ChatRow key={m.messageId} message={m} />
                  ))}

                  {/*
                    还在等待应答的审批 —— 单独列，不混进历史消息。
                    原因：审批是「正在发生」的事，而 messages 是「已经发生」的事。
                    混在一起的话，卡片会随着消息刷新被重排，用户正要点的按钮会跳走。
                  */}
                  {myApprovals.length > 0 && (
                    <div className="chat-pending" data-testid="ide-approvals">
                      {myApprovals.map((a) => (
                        <ApprovalCard
                          key={a.requestId}
                          request={a}
                          onRespond={respond}
                          busy={responding.includes(a.requestId)}
                          tier={tier}
                        />
                      ))}
                    </div>
                  )}
                </div>

                {/*
                  契约闸门的操作条：确认与否决都在这里。
                  确认**不会**自动开跑 —— 契约成立与「开始执行」是两件事，
                  合在一起的后果是用户点一下确认，命令就已经跑完了，那不叫确认，叫追认。
                */}
                {currentSession && (
                  <div className="chat-actions" data-testid="ide-chat-actions">
                    {currentSession.state === 'contract-pending' ? (
                      <>
                        <button
                          className="pill ghost"
                          disabled={busy || sending || running}
                          onClick={() => void confirmContract(false)}
                          data-testid="ide-contract-reject"
                        >
                          否决
                        </button>
                        <button
                          className="pill primary"
                          disabled={busy || sending || running}
                          onClick={() => void confirmContract(true)}
                          data-testid="ide-contract-accept"
                        >
                          确认契约
                        </button>
                      </>
                    ) : (
                      <button
                        className="pill primary"
                        disabled={!currentSession.runId || busy || sending || running}
                        onClick={() => void runLoop()}
                        data-testid="ide-run-loop"
                      >
                        {running ? '执行中…' : '让模型开始执行'}
                      </button>
                    )}
                    <span className="chat-state mono">
                      {currentSession.runId ? `run ${currentSession.runId.slice(-6)}` : '未起 run'}
                    </span>
                  </div>
                )}

                {/*
                  登记预览地址。
                  入口放在对话页而不是「预览」签里 —— 签只在有预览时才出现，
                  把入口放进签内等于永远点不到（鸡生蛋）。
                */}
                {currentSession && !hasPreview && (
                  <div className="chat-preview-add" data-testid="ide-preview-add">
                    <input
                      value={manualUrl}
                      onChange={(e) => setManualUrl(e.target.value)}
                      placeholder="登记预览地址，如 http://127.0.0.1:5173"
                      className="mono"
                      data-testid="ide-preview-url"
                    />
                    <button
                      className="pill ghost small"
                      disabled={!manualUrl.trim()}
                      onClick={() => void addPreview()}
                      data-testid="ide-preview-register"
                    >
                      登记
                    </button>
                  </div>
                )}

                <div className="chat-composer">
                  <input
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') void send()
                    }}
                    placeholder={
                      currentSession?.state === 'contract-pending'
                        ? '补充约束（会并入契约重新起草），或直接确认契约'
                        : '说一句话目标，例如：给这个仓库补上缺失的单元测试并让 CI 通过'
                    }
                    disabled={sending || running}
                    data-testid="ide-chat-input"
                  />
                  <button
                    className="pill primary"
                    onClick={() => void send()}
                    disabled={sending || running || !draft.trim()}
                    data-testid="ide-chat-send"
                  >
                    {sending ? '进行中…' : running ? '执行中…' : '发送'}
                  </button>
                </div>
              </div>
            )}

            {tab === 'tasks' && <TaskList tasks={tasks} />}
            {tab === 'changes' && <ChangeList changes={changes} onConfirm={confirmChange} />}
            {tab === 'preview' && hasPreview && (
              <PreviewList
                previews={previews}
                onOpen={(u) => void window.openintj.openExternal(u)}
              />
            )}
          </div>
        </aside>
      </div>

      <div className="ide-hint">
        工作台跑在 <code>127.0.0.1</code> 的随机端口上，每次启动现生成随机连接令牌，令牌不出主进程；
        停止时会连同子进程一起杀干净。扩展市场走 Open VSX。右栏的「变更」只在工作区内生效，上报的是文件路径与增删改状态，
        <strong>不含文件内容</strong> —— 内容要落盘或喂模型前必须先过脱敏。
      </div>
    </div>
  )
}

/** 会话里的一条消息 —— 结构化消息（契约 / 审批）要有可操作的卡片，不能只是一段文本 */
function ChatRow({ message }: { message: ChatMessage }): React.ReactElement {
  const time = new Date(message.at).toLocaleTimeString('zh-CN', { hour12: false })

  return (
    <div className="chat-msg" data-role={message.role} data-kind={message.kind}>
      <div className="chat-meta mono">
        {message.role === 'user' ? '你' : message.role === 'agent' ? 'Agent' : '系统'} · {time}
      </div>
      <div className="chat-bubble">{message.text}</div>

      {message.kind === 'contract-draft' && (
        <ContractCard message={message} />
      )}

      {message.kind === 'delivery' && (
        <DeliveryCard payload={message.payload} />
      )}
    </div>
  )
}

/** 契约草稿卡 —— IC-07 硬闸门的确认入口就在这里 */
function ContractCard({ message }: { message: ChatMessage }): React.ReactElement | null {
  const p = message.payload as { contract?: GoalContract | null; llm?: { configured: boolean } } | null
  const contract = p?.contract
  if (!contract) return null

  return (
    <div className="side-card contract-mini">
      <div className="contract-obj">{contract.objective}</div>
      <ContractLines title="成功判据" items={contract.successCriteria} tone="ok" />
      <ContractLines title="明确不做" items={contract.nonGoals} />
      <ContractLines title="依赖假设" items={contract.assumptions} />
      <ContractLines title="已知风险" items={contract.risks} tone="warn" />
      <ContractLines
        title="待用户澄清"
        items={contract.needsClarification}
        tone="danger"
        emptyText="无"
      />
    </div>
  )
}

function ContractLines({
  title,
  items,
  tone,
  emptyText = '未列出'
}: {
  title: string
  items: string[]
  tone?: 'ok' | 'warn' | 'danger'
  emptyText?: string
}): React.ReactElement {
  return (
    <div className="contract-lines">
      <div className={'contract-section-title' + (tone ? ' ' + tone : '')}>{title}</div>
      {items.length === 0 ? (
        <div className="contract-empty">{emptyText}</div>
      ) : (
        <ul>
          {items.map((it, i) => (
            <li key={i}>{it}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

/** 审批卡 —— 与控制台同一条应答链，拒绝 / 允许一次 / 本任务内允许 */
function ApprovalCard({
  request,
  onRespond,
  busy,
  tier
}: {
  request: ApprovalRequest
  onRespond: (requestId: string, approved: boolean) => Promise<void>
  busy: boolean
  tier: string
}): React.ReactElement {
  return (
    <div className="side-card approval-mini">
      <code className="approval-cmd">{request.command}</code>
      <div className="side-row-meta mono">
        风险 {request.riskClass} · cwd {request.cwd} · 档位 {TIER_LABELS[tier as keyof typeof TIER_LABELS] ?? tier}
      </div>
      <div className="approval-actions">
        <button className="pill ghost" disabled={busy} onClick={() => void onRespond(request.requestId, false)}>
          拒绝
        </button>
        <button className="pill primary" disabled={busy} onClick={() => void onRespond(request.requestId, true)}>
          允许一次
        </button>
      </div>
    </div>
  )
}

/** 执行结果 —— 逐条列出已满足 / 未满足，不粉饰 */
function DeliveryCard({ payload }: { payload: unknown }): React.ReactElement | null {
  const p = payload as
    | {
        summary?: string
        satisfied?: { criterion: string; evidence: string }[]
        unmet?: { criterion: string; reason: string }[]
        steps?: number
      }
    | null
  if (!p) return null

  return (
    <div className="side-card delivery-mini">
      {p.summary && <div className="contract-obj">{p.summary}</div>}
      {(p.satisfied?.length ?? 0) > 0 && (
        <div className="contract-lines">
          <div className="contract-section-title ok">已满足</div>
          <ul>
            {p.satisfied!.map((s, i) => (
              <li key={i}>
                {s.criterion}
                <span className="mono evidence">{s.evidence}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {(p.unmet?.length ?? 0) > 0 && (
        <div className="contract-lines">
          <div className="contract-section-title danger">未满足</div>
          <ul>
            {p.unmet!.map((u, i) => (
              <li key={i}>
                {u.criterion}
                <span className="mono evidence">{u.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function TaskList({ tasks }: { tasks: TaskItem[] }): React.ReactElement {
  if (tasks.length === 0) {
    return <div className="empty">还没有任务。在「对话」里说一句话目标，确认契约后即生成。</div>
  }
  return (
    <div className="side-list" data-testid="ide-task-list">
      {tasks.map((t) => (
        <div className="side-row" key={t.taskId}>
          <div className="side-row-title">{t.title}</div>
          <div className="side-row-meta mono">
            <span className={'state-chip ' + (STATE_TONE[t.state] ?? 'pending')}>{t.state}</span>
            <span>{t.steps} 步</span>
            <span>{new Date(t.at).toLocaleTimeString('zh-CN', { hour12: false })}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function ChangeList({
  changes,
  onConfirm
}: {
  changes: FileChange[]
  onConfirm: (changeId: string) => Promise<void>
}): React.ReactElement {
  if (changes.length === 0) {
    return (
      <div className="empty">
        还没有文件变化。监听器只盯工作区内的文件；<code>.git</code> / <code>node_modules</code> 之类目录不参与。
      </div>
    )
  }
  return (
    <div className="side-list" data-testid="ide-change-list">
    {changes.map((c) => (
      <div className="side-row" key={c.changeId}>
        <div className="side-row-meta mono">
          <span className={'change-kind ' + c.kind}>{CHANGE_LABEL[c.kind]}</span>
          <span>{new Date(c.at).toLocaleTimeString('zh-CN', { hour12: false })}</span>
        </div>
        <div className="side-row-title mono">{c.path}</div>
        <div className="side-row-actions">
          {c.confirmed ? (
            <span className="mono confirmed">已确认</span>
          ) : (
            <button className="pill ghost small" onClick={() => void onConfirm(c.changeId)}>
              确认
            </button>
          )}
        </div>
      </div>
    ))}
  </div>
  )
}

function PreviewList({
  previews,
  onOpen
}: {
  previews: PreviewTarget[]
  onOpen: (url: string) => void
}): React.ReactElement {
  return (
    <div className="side-list" data-testid="ide-preview-list">
      {previews.map((p) => (
        <div className="side-row" key={p.previewId}>
          <div className="side-row-title mono">{p.url}</div>
          <div className="side-row-meta mono">{p.title}</div>
          <div className="side-row-actions">
            <button className="pill ghost small" onClick={() => onOpen(p.url)}>
              在系统浏览器打开
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
