import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ApprovalRequest, ApprovalTier } from '../shared/approval'
import { TIER_LABELS } from '../shared/approval'
import type { EventEnvelope } from '../shared/events'
import type { GoalContract } from '../shared/contract'
import type { LlmStatus, TokenUsage } from '../shared/llm'
import type { BootInfo, CredentialStatus, RunLoopResult } from '../shared/ipc'
import SettingsPage from './SettingsPage'
import TerminalPanel from './TerminalPanel'
import IdePanel from './IdePanel'

const TIERS: ApprovalTier[] = ['strict', 'standard', 'autonomous']

const ZERO_USAGE: TokenUsage = {
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
  reasoningTokens: 0
}

/** 超过这个条数就聚合为批量审批卡（《系统设计》§3.2.M10） */
const BATCH_THRESHOLD = 3

const RISK_LABEL: Record<string, string> = {
  L0: '只读',
  L1: '工作区内写入',
  L2: '工作区外写入',
  L3: '危险命令',
  'R5-network': '出网访问',
  'R0-unknown': '无法归类'
}

const RISK_TONE: Record<string, string> = {
  L0: 'calm',
  L1: 'warn',
  L2: 'alert',
  L3: 'danger',
  'R5-network': 'warn',
  'R0-unknown': 'danger'
}

export default function App() {
  const [boot, setBoot] = useState<BootInfo | null>(null)
  const [ping, setPing] = useState<{ main: string; agentHost: string; toolHost: string } | null>(null)
  const [tier, setTier] = useState<ApprovalTier>('standard')
  const [tierMsg, setTierMsg] = useState<string | null>(null)
  const [events, setEvents] = useState<EventEnvelope[]>([])
  const [goal, setGoal] = useState('')
  const [command, setCommand] = useState('node -v')
  const [runId, setRunId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [vault, setVault] = useState<{ available: boolean; policy: string } | null>(null)
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([])
  const [responding, setResponding] = useState<string[]>([])
  const [invokeMsg, setInvokeMsg] = useState<{ tone: 'ok' | 'warn' | 'error'; text: string } | null>(null)
  const [batchOpen, setBatchOpen] = useState(false)
  const [llm, setLlm] = useState<LlmStatus | null>(null)
  const [contract, setContract] = useState<GoalContract | null>(null)
  const [contractMsg, setContractMsg] = useState<string | null>(null)
  const [runState, setRunState] = useState<string>('CONTRACT_PENDING')
  const [loopRunning, setLoopRunning] = useState(false)
  const [loopResult, setLoopResult] = useState<RunLoopResult | null>(null)
  const [page, setPage] = useState<'console' | 'settings' | 'terminal' | 'ide'>('console')
  const [credStatus, setCredStatus] = useState<CredentialStatus | null>(null)
  /** 终端面板首次进入才挂载（隐藏状态下 xterm 算不出正确尺寸） */
  const [terminalMounted, setTerminalMounted] = useState(false)
  /**
   * IDE 面板同样首次进入才挂载。
   *
   * 原因和终端不一样，值得写清楚：不挂载就不上报矩形，主进程也就不会把
   * WebContentsView 显示出来 —— 这正好实现「没进过这一页就不起进程」。
   */
  const [ideMounted, setIdeMounted] = useState(false)

  // 初次加载：运行时信息 + 三进程探针 + 模型接入状态
  useEffect(() => {
    void (async () => {
      const info = await window.openintj.getBootInfo()
      setBoot(info)
      setTier(info.approval.tier)
      const v = await window.openintj.vaultStatus()
      setVault({ available: v.available, policy: v.policy })
      // 子进程启动有延迟，重试几次
      for (let i = 0; i < 6; i++) {
        const p = await window.openintj.ping()
        setPing(p)
        if (p.agentHost !== 'offline' && p.toolHost !== 'offline') break
        await new Promise((r) => setTimeout(r, 400))
      }
      try {
        setLlm(await window.openintj.llmStatus())
        setCredStatus(await window.openintj.credentialStatus())
      } catch {
        /* 模型状态拿不到不影响使用，界面显示「未知」 */
      }
    })()
  }, [])

  /**
   * 档位被非界面操作改动（启动时从库里恢复）时，界面必须跟着变。
   *
   * 不订阅这条通道的话，恢复会比首屏慢半步：用户看到的是默认档，
   * 会以为上次设的严格档没存住。
   */
  useEffect(() => window.openintj.onTierChanged((t) => setTier(t)), [])

  /** 凭据变了之后重新拉一次全局模型状态 —— 顶栏那行显示的就是它，不能停在旧值 */
  const reloadModelStatus = useCallback(async () => {
    try {
      setCredStatus(await window.openintj.credentialStatus())
      setLlm(await window.openintj.llmStatus())
    } catch (e) {
      pushUiError(setEvents, `重新读取模型状态失败：${(e as Error).message}`)
    }
  }, [])

  // 订阅事件流；顺带从事件里捞契约与状态，避免只依赖返回值
  useEffect(() => {
    const off = window.openintj.onEvent((e) => {
      setEvents((prev) => [...prev.slice(-199), e])

      // runId 也从事件信封里兜底 —— 否则「从检查点续跑」「别处发起的 run」这类入口
      // 会让顶栏一直显示「当前 Run 未启动」，而右边事件数却在涨，自相矛盾
      if (e.runId && e.runId !== 'system' && e.runId !== '-') {
        setRunId((prev) => prev ?? e.runId)
      }
      if (e.type === 'GoalContractDrafted') {
        const payload = e.payload as { contract?: GoalContract }
        if (payload?.contract) setContract(payload.contract)
      }
      if (e.type === 'ApprovalDecided') {
        // 卡片是否消失，由「这条审批已经有结论」决定，而不是由「谁点的按钮」决定。
        // 只在 respond() 里乐观移除是不够的：审批可能被别处应答（另一个窗口、
        // 自动化测试替身、程序自己超时收卡），那些路径都不经过按钮的 onClick，
        // 卡片就会永远贴在屏幕上 —— 用户点它只会得到「请求不存在」。
        const p = e.payload as { requestId?: string }
        if (p.requestId) setApprovals((prev) => prev.filter((x) => x.requestId !== p.requestId))
      }

      if (e.type === 'GoalContractConfirmed') setRunState('CONFIRMED')
      if (e.type === 'TaskStarted') setRunState('RUNNING')
      if (e.type === 'RunResumed') setRunState('RUNNING')
      if (e.type === 'RunPaused') setRunState('PAUSED')

      // 执行结果从事件流里派生，而不是只靠 startAgentLoop 的返回值 ——
      // 否则界面重开 / 刷新一次，结果卡片就凭空消失了（事件才是事实来源，F5）
      if (e.type === 'RunDelivered') {
        const p = e.payload as {
          summary?: string
          satisfied?: { criterion: string; evidence: string }[]
          unmet?: { criterion: string; reason: string }[]
          steps?: number
          usage?: TokenUsage
        }
        setRunState('DELIVERED')
        setLoopResult({
          outcome: 'delivered',
          steps: p.steps ?? 0,
          summary: p.summary,
          satisfied: p.satisfied ?? [],
          unmet: p.unmet ?? [],
          usage: p.usage ?? ZERO_USAGE
        })
      }
      if (e.type === 'RunFailed') {
        const p = e.payload as { reason?: string; steps?: number; usage?: TokenUsage }
        setRunState('FAILED')
        setLoopResult({
          outcome: 'failed',
          steps: p.steps ?? 0,
          reason: p.reason,
          usage: p.usage ?? ZERO_USAGE
        })
      }
    })
    return off
  }, [])

  // 订阅审批请求 —— D4 的关键：有发起方，卡片才上得来
  useEffect(() => {
    const off = window.openintj.onApprovalRequest((r) => {
      setApprovals((prev) => (prev.some((x) => x.requestId === r.requestId) ? prev : [...prev, r]))
    })
    return off
  }, [])

  // 审批被撤回（run 已收尾）：卡片必须跟着消失，否则用户点它只会得到「请求不存在」
  useEffect(() => {
    const off = window.openintj.onApprovalCancel((p) => {
      const gone = new Set(p.requestIds)
      setApprovals((prev) => prev.filter((x) => !gone.has(x.requestId)))
    })
    return off
  }, [])

  const onStart = useCallback(async () => {
    if (!goal.trim() || busy) return
    setBusy(true)
    setContractMsg(null)
    try {
      const res = await window.openintj.startRun(goal.trim())
      setRunId(res.runId)
      setRunState(res.state)
      if (res.contract) setContract(res.contract)
      if (res.llm) setLlm(res.llm)
      if (!res.contract) {
        setContractMsg(
          res.llm && !res.llm.configured
            ? `模型未配置（缺 ${res.llm.missing.join(' / ')}），契约无法生成`
            : '契约未生成成功，详见事件流'
        )
      }
    } catch (e) {
      pushUiError(setEvents, (e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [goal, busy])

  const onConfirmContract = useCallback(
    async (accepted: boolean) => {
      if (!runId) return
      setContractMsg(null)
      try {
        const res = await window.openintj.confirmContract({ runId, accepted })
        if (res.ok) {
          setRunState(res.state ?? (accepted ? 'CONFIRMED' : 'CONTRACT_PENDING'))
        } else {
          setContractMsg(res.reason ?? '确认失败')
        }
      } catch (e) {
        setContractMsg((e as Error).message)
      }
    },
    [runId]
  )

  /**
   * 让模型开跑。
   *
   * 注意这不是「一键自动完成」：循环里每一次命令都还会弹审批卡片，
   * 用户随时可以拒绝或暂停 —— 自动化的是「想」，不是「批准」。
   */
  const onRunLoop = useCallback(async () => {
    if (!runId || loopRunning) return
    setLoopRunning(true)
    setLoopResult(null)
    setInvokeMsg(null)
    try {
      const res = await window.openintj.startAgentLoop(runId)
      setLoopResult(res)
      setRunState(
        res.outcome === 'delivered'
          ? 'DELIVERED'
          : res.outcome === 'paused'
            ? 'PAUSED'
            : res.outcome === 'max-steps'
              ? 'FAILED'
              : 'FAILED'
      )
    } catch (e) {
      pushUiError(setEvents, (e as Error).message)
      setLoopResult({ outcome: 'failed', steps: 0, reason: (e as Error).message, usage: ZERO_USAGE })
    } finally {
      setLoopRunning(false)
    }
  }, [runId, loopRunning])

  const onPause = useCallback(async () => {
    if (!runId) return
    // 立即置灰，别让用户以为没点上
    setRunState('PAUSED')
    const res = await window.openintj.pauseRun(runId)
    if (!res.ok) {
      setRunState('RUNNING')
      pushUiError(setEvents, res.reason ?? '暂停失败')
    }
  }, [runId])

  const onTier = useCallback(async (t: ApprovalTier) => {
    setTierMsg(null)
    const res = await window.openintj.setApprovalTier(t)
    if (res.ok) setTier(t)
    else setTierMsg(res.reason ?? '切换失败')
  }, [])

  const respond = useCallback(async (request: ApprovalRequest, approved: boolean, remember: boolean) => {
    setResponding((prev) => [...prev, request.requestId])
    // 乐观移除：用户点了就得立刻有反馈，不能等 IPC 往返
    setApprovals((prev) => prev.filter((x) => x.requestId !== request.requestId))
    try {
      await window.openintj.respondApproval(request.requestId, approved, remember)
    } catch (e) {
      // 应答失败要把卡片放回去，否则用户以为已经处理过了
      setApprovals((prev) => [...prev, request])
      pushUiError(setEvents, `审批应答失败：${(e as Error).message}`)
    } finally {
      setResponding((prev) => prev.filter((id) => id !== request.requestId))
    }
  }, [])

  const respondAll = useCallback(
    async (approved: boolean) => {
      const list = [...approvals]
      setBatchOpen(false)
      for (const r of list) await respond(r, approved, false)
    },
    [approvals, respond]
  )

  const onInvoke = useCallback(async () => {
    if (!runId || !command.trim()) return
    setBusy(true)
    setInvokeMsg(null)
    try {
      const out = await window.openintj.invokeTool({ runId, command: command.trim() })
      if (out.ok) {
        setInvokeMsg({
          tone: 'ok',
          text: `执行完成 · exit=${out.result?.exitCode ?? '—'} · ${out.result?.durationMs ?? 0}ms${
            out.result?.truncated ? ' · 输出已截断' : ''
          }`
        })
      } else {
        setInvokeMsg({ tone: 'warn', text: out.reason ?? '未执行' })
      }
    } catch (e) {
      setInvokeMsg({ tone: 'error', text: (e as Error).message })
    } finally {
      setBusy(false)
    }
  }, [runId, command])

  const sandbox = boot?.sandbox
  const processes = useMemo(
    () => [
      { name: 'P-01 主进程', status: ping?.main ?? 'unknown', detail: '窗口 / IPC 路由 / 保险箱' },
      {
        name: 'P-03 agent-host',
        status: ping?.agentHost === 'offline' ? 'offline' : ping ? 'online' : 'unknown',
        detail: ping?.agentHost ?? '探测中'
      },
      {
        name: 'P-04 tool-host',
        status: ping?.toolHost === 'offline' ? 'offline' : ping ? 'online' : 'unknown',
        detail: ping?.toolHost ?? '探测中'
      }
    ],
    [ping]
  )

  const batchMode = approvals.length > BATCH_THRESHOLD
  /** 契约确认后才允许发起工具调用（IC-07） */
  const canInvoke = Boolean(runId) && (runState === 'CONFIRMED' || runState === 'RUNNING')
  /** 循环可以在暂停后重入，但前提是契约已过闸门且模型可用 */
  const canLoop =
    Boolean(runId) &&
    ['CONFIRMED', 'RUNNING', 'PAUSED'].includes(runState) &&
    Boolean(llm?.configured)

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">openINTJ</span>
          <span className="brand-sub">Agent Console · v{boot?.version ?? '…'}</span>
        </div>
        <div className="topbar-right">
          <span className="mono" style={{ color: 'var(--text-faint)' }}>
            {boot?.platform ?? ''}
          </span>
          <span className="mono" style={{ color: llm?.configured ? 'var(--cyan)' : 'var(--text-faint)' }}>
            {llm ? (llm.configured ? `${llm.model} · ${llm.protocol}` : '模型未配置') : '模型 …'}
          </span>
          <span className="mono" style={{ color: vault?.available ? 'var(--mint)' : 'var(--danger)' }}>
            保险箱 {vault ? (vault.available ? '可用' : '不可用 · fail-stop') : '…'}
          </span>
          {approvals.length > 0 && <span className="mono badge-pending">待审批 {approvals.length}</span>}
        </div>
      </header>

      <nav className="tabs">
        {(
          [
            { id: 'console', label: '控制台' },
            { id: 'terminal', label: '终端' },
            { id: 'ide', label: '编辑器' },
            { id: 'settings', label: '设置与凭据' }
          ] as const
        ).map((t) => (
          <button
            key={t.id}
            className={'tab' + (page === t.id ? ' active' : '')}
            onClick={() => {
              setPage(t.id)
              if (t.id === 'terminal') setTerminalMounted(true)
              if (t.id === 'ide') setIdeMounted(true)
              // 切到设置页就重新拉一次 —— 凭据可能在别处被改过，
              // 显示陈旧状态会让用户对着一张过期的卡片做决定
              if (t.id === 'settings') void reloadModelStatus()
            }}
          >
            {t.label}
          </button>
        ))}
        <span className="tabs-note">
          {credStatus
            ? `凭据来源：${credStatus.source === 'vault' ? '保险箱' : credStatus.source === 'env' ? '环境变量' : '未配置'}`
            : ''}
        </span>
      </nav>

      {page === 'settings' && (
        <SettingsPage status={credStatus} onChanged={reloadModelStatus} />
      )}

      {/*
        终端首次进入才挂载，之后整块用 CSS 隐藏而不是卸载 —— 卸载会关掉 PTY 会话，
        用户切去控制台看事件、再切回终端，会话就没了（`cd` 白敲）。
        首次挂载必须发生在「可见」的时候：隐藏状态下 xterm 算不出尺寸，
        fit() 会给出错误的 cols/rows，pty 一开局就按错宽度折行。
      */}
      {terminalMounted && (
        <div className={'page' + (page === 'terminal' ? '' : ' hidden')}>
          <TerminalPanel active={page === 'terminal'} />
        </div>
      )}

      {/* 首次进入才挂载：不挂载就不上报矩形，主进程也就不会显示工作台 */}
      {ideMounted && (
        <div className={'page' + (page === 'ide' ? '' : ' hidden')}>
          <IdePanel active={page === 'ide'} />
        </div>
      )}

      {/* 控制台整块用 CSS 隐藏，只为避免大段 JSX 重排；状态仍然保留，切回来不会丢 */}
      <div className={'page' + (page === 'console' ? '' : ' hidden')}>
        {/* C1 必显：Windows 隔离强度差异，不提供关闭开关 */}
        {sandbox?.notice && (
          <div className="notice">
            <strong>隔离降级</strong>
            <span>{sandbox.notice}</span>
          </div>
        )}

      <div className="grid-3">
        {processes.map((p) => (
          <div className="card" key={p.name}>
            <div className="metric-label">{p.name}</div>
            <div className="metric-value">
              <span
                className={
                  'dot ' + (p.status === 'online' ? 'online' : p.status === 'offline' ? 'error' : 'offline')
                }
              />
              {p.status === 'online' ? '在线' : p.status === 'offline' ? '离线' : '探测中'}
            </div>
            <div className="hint mono" style={{ color: 'var(--text-faint)' }}>
              {p.detail}
            </div>
          </div>
        ))}
      </div>

      <div className="grid-3">
        <div className="card">
          <div className="metric-label">审批档位（IC-08 · 按风险分级）</div>
          <div className="tier-row">
            {TIERS.map((t) => {
              const blocked = t === 'autonomous' && boot?.approval.autonomousBlocked
              return (
                <button
                  key={t}
                  data-tier={t}
                  className={'pill' + (tier === t ? ' active' : '')}
                  disabled={blocked}
                  title={blocked ? 'Windows 侧「宽松」档默认禁用：无 OS 级隔离兜底' : undefined}
                  onClick={() => void onTier(t)}
                >
                  {TIER_LABELS[t]}
                </button>
              )
            })}
          </div>
          <div className={'hint' + (tierMsg ? ' warn' : '')}>
            {tierMsg ?? '只读与工作区内写入不阻塞并发；工作区外写入与危险命令逐次审批。'}
          </div>
        </div>

        <div className="card">
          <div className="metric-label">沙箱原语</div>
          <div className="metric-value mono">{sandbox?.primitive ?? '…'}</div>
          <div className="hint">
            {sandbox?.isolated ? '具备 OS 级隔离' : '无 OS 级隔离，降级为审批在环 + 白名单 + 根目录约束'}
          </div>
        </div>

        <div className="card">
          <div className="metric-label">当前 Run</div>
          <div className="metric-value mono">{runId ?? '未启动'}</div>
          <div className="hint">{events.length} 条事件</div>
        </div>
      </div>

      {/* 目标契约 —— IC-07 硬闸门：未确认不得执行任何工具 */}
      {(contract || contractMsg) && (
        <div className="card contract">
          <div className="card-title">
            目标契约 · 未确认前不得执行
            <span className={'state-chip ' + (runState === 'CONTRACT_PENDING' ? 'pending' : 'done')}>
              {runState}
            </span>
          </div>

          {contract && (
            <>
              <div className="contract-obj">{contract.objective}</div>
              <div className="contract-grid">
                <ContractSection title="成功判据" items={contract.successCriteria} tone="ok" />
                <ContractSection title="明确不做" items={contract.nonGoals} />
                <ContractSection title="依赖假设" items={contract.assumptions} />
                <ContractSection title="已知风险" items={contract.risks} tone="warn" />
                <ContractSection
                  title="待用户澄清"
                  items={contract.needsClarification}
                  tone="danger"
                  emptyText="无"
                />
              </div>
            </>
          )}

          {contractMsg && <div className="hint warn">{contractMsg}</div>}

          <div className="contract-actions">
            {runState === 'CONTRACT_PENDING' ? (
              <>
                <button className="pill ghost" onClick={() => void onConfirmContract(false)}>
                  否决
                </button>
                <button
                  className="pill primary"
                  disabled={!contract}
                  onClick={() => void onConfirmContract(true)}
                >
                  确认契约并开始
                </button>
              </>
            ) : (
              <>
                <span className="mono contract-done">契约已确认 · 状态 {runState}</span>
                {loopRunning ? (
                  <button className="pill danger" onClick={() => void onPause()}>
                    暂停
                  </button>
                ) : (
                  <button
                    className="pill primary"
                    disabled={!canLoop}
                    title={!llm?.configured ? '模型未配置，循环无法启动' : undefined}
                    onClick={() => void onRunLoop()}
                  >
                    {loopResult ? '再跑一轮' : '让模型开始执行'}
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {/* 执行结果 —— 循环的产出。没做完就如实说哪条没做到，不粉饰 */}
      {loopResult && (
        <div className={'card delivery ' + loopResult.outcome}>
          <div className="card-title">
            执行结果
            <span className={'state-chip ' + (loopResult.outcome === 'delivered' ? 'done' : 'pending')}>
              {loopResult.outcome} · {loopResult.steps} 步 · 输入 {loopResult.usage.inputTokens}
              {loopResult.usage.cacheReadTokens > 0 ? ` +缓存 ${loopResult.usage.cacheReadTokens}` : ''} / 输出{' '}
              {loopResult.usage.outputTokens} tok
            </span>
          </div>

          {loopResult.summary && <div className="contract-obj">{loopResult.summary}</div>}
          {loopResult.reason && <div className="hint warn">{loopResult.reason}</div>}

          {loopResult.satisfied && loopResult.satisfied.length > 0 && (
            <div className="delivery-list">
              <div className="contract-section-title ok">已满足的判据</div>
              {loopResult.satisfied.map((s, i) => (
                <div className="delivery-item" key={`s${i}`}>
                  <div className="delivery-criterion">{s.criterion}</div>
                  <div className="delivery-evidence mono">{s.evidence}</div>
                </div>
              ))}
            </div>
          )}

          {loopResult.unmet && loopResult.unmet.length > 0 && (
            <div className="delivery-list">
              <div className="contract-section-title danger">未满足的判据</div>
              {loopResult.unmet.map((u, i) => (
                <div className="delivery-item" key={`u${i}`}>
                  <div className="delivery-criterion">{u.criterion}</div>
                  <div className="delivery-evidence mono warn-text">{u.reason}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 审批区 —— 有请求才出现，不占常驻版面 */}
      {approvals.length > 0 && (
        <div className="card approvals">
          <div className="card-title">
            待审批 {approvals.length} 条
            {batchMode && (
              <button className="pill ghost small" onClick={() => setBatchOpen((v) => !v)}>
                {batchOpen ? '收起' : '批量处理'}
              </button>
            )}
          </div>

          {batchMode && batchOpen && (
            <div className="batch">
              <div className="hint">
                超过 {BATCH_THRESHOLD} 条，聚合为批量审批。逐条命令仍可单独裁决。
              </div>
              <div className="batch-actions">
                <button className="pill danger" onClick={() => void respondAll(false)}>
                  全部拒绝（{approvals.length}）
                </button>
                <button className="pill primary" onClick={() => void respondAll(true)}>
                  全部允许一次（{approvals.length}）
                </button>
              </div>
            </div>
          )}

          {(!batchMode || !batchOpen) &&
            approvals.map((a) => (
              <div className="approval" key={a.requestId}>
                <div className={'risk-badge ' + (RISK_TONE[a.riskClass] ?? 'warn')}>
                  {a.riskClass} · {RISK_LABEL[a.riskClass] ?? '未知'}
                </div>
                <div className="approval-body">
                  <code className="approval-cmd">{a.command}</code>
                  <div className="approval-meta mono">
                    cwd {a.cwd} · 泳道 {a.parallelLaneId.slice(-6)} · 档位 {TIER_LABELS[tier]}
                  </div>
                </div>
                <div className="approval-actions">
                  <button
                    className="pill ghost"
                    disabled={responding.includes(a.requestId)}
                    onClick={() => void respond(a, false, false)}
                  >
                    拒绝
                  </button>
                  <button
                    className="pill"
                    disabled={responding.includes(a.requestId)}
                    onClick={() => void respond(a, true, false)}
                  >
                    允许一次
                  </button>
                  <button
                    className="pill primary"
                    disabled={responding.includes(a.requestId)}
                    onClick={() => void respond(a, true, true)}
                  >
                    本任务内允许
                  </button>
                </div>
              </div>
            ))}
        </div>
      )}

      <div className="card stream">
        <div className="card-title">实时执行流 · EventStream</div>
        <div className="stream-body">
          {events.length === 0 ? (
            <div className="empty">
              {llm && !llm.configured
                ? '暂无事件。本机尚未配置模型凭据，目标契约无法起草 —— 本产品不接受伪造的模型输出，请先补齐 OPENINTJ_LLM_* 环境变量。'
                : '暂无事件。在下方输入目标并启动 —— 模型会先把目标起草成一份契约（成功判据 / 明确不做 / 依赖假设 / 已知风险 / 待澄清），你确认之后才会放行任何工具调用。'}
            </div>
          ) : (
            events.map((e) => (
              <div className="stream-row" key={e.eventId + e.seq}>
                <span className="stream-seq">{e.seq > 0 ? e.seq : '—'}</span>
                <span className={'stream-type ' + (e.kind === 'observation' ? 'obs' : '')}>{e.type}</span>
                <span className="stream-payload">{JSON.stringify(e.payload)}</span>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="composer">
        <input
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void onStart()
          }}
          placeholder="下达一句话目标，例如：给这个仓库补上缺失的单元测试并让 CI 通过"
          disabled={busy}
        />
        <button className="pill primary" onClick={() => void onStart()} disabled={busy || !goal.trim()}>
          {busy ? '启动中…' : '开始'}
        </button>
      </div>

      <div className="composer sub">
        <input
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void onInvoke()
          }}
          placeholder={
            canInvoke
              ? '手动发起一次工具调用（与模型驱动同一条裁决链）'
              : '先确认目标契约，才能发起工具调用'
          }
          disabled={busy || !canInvoke}
          className="mono"
        />
        <button className="pill" onClick={() => void onInvoke()} disabled={busy || !canInvoke || !command.trim()}>
          发起调用
        </button>
      </div>
      {invokeMsg && <div className={'result ' + invokeMsg.tone}>{invokeMsg.text}</div>}
      </div>
    </div>
  )
}

/** 契约里的一个分栏 */
function ContractSection({
  title,
  items,
  tone,
  emptyText = '未列出'
}: {
  title: string
  items: string[]
  tone?: 'ok' | 'warn' | 'danger'
  emptyText?: string
}) {
  return (
    <div className="contract-section">
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

/** 界面级异常也要进事件流，不能只在 console 里一声不响 */
function pushUiError(
  setEvents: React.Dispatch<React.SetStateAction<EventEnvelope[]>>,
  message: string
): void {
  setEvents((prev) => [
    ...prev,
    {
      eventId: `ui-${Date.now()}`,
      seq: -1,
      runId: '-',
      traceId: '-',
      tenantId: 'local',
      agentId: 'main',
      kind: 'observation',
      type: 'ErrorObserved',
      at: Date.now(),
      redaction: 'none',
      payload: { message },
      schemaVersion: 1
    } as EventEnvelope
  ])
}
