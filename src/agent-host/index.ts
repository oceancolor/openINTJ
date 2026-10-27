/**
 * P-03 Agent 主循环宿主（Utility Process）
 *
 * 为什么独立进程（《系统设计》§3.1.1.1）：
 *   ① 长跑不阻塞 UI（QS-06）
 *   ② 崩溃可隔离，并可从检查点续跑（QS-01）
 *   ③ 可用 Node 原生模块，事件落盘路径最短
 *   ④ 可单独设内存与 CPU 限额
 *
 * 本文件当前范围：MessagePort 链路、事件流、run 状态机、**工具调用与审批在环**。
 * LLM 真实调用尚未接入（不伪造模型输出，N2）：当前工具调用由控制台手动发起，
 * 与将来模型驱动走同一条裁决链 —— 手动入口不是旁路。
 */

import { join } from 'node:path'
import { openStore, type AppendInput, type EventStoreLike } from './event-store'
import * as sessions from './session-store'
import { generateImage, placeholderSvg } from './image-gen'
import { hintOf, structureInput } from './input-structure'
import type { InputStructureHint } from '../shared/contract'
import {
  ARTIFACT_DIR,
  SESSION_METHODS,
  type AgentSession,
  type FileChange,
  type PreviewTarget,
  type SessionSendResult,
  type SessionState,
  type TaskItem
} from '../shared/session'
import { ulid } from '../shared/ulid'
import { applyRedaction } from '../shared/redaction'
import { complete, textOf, thinkingOf } from './llm'
import { runAgentLoop, type McpInvokeOutcome } from './agent-loop'
import { AGENT_TOOLS } from './tool-specs'
import type {
  ImageConfigResult,
  ImageGenerateResult,
  LlmConfigResult,
  LlmStatus,
  ToolSpec,
  UnifiedMessage
} from '../shared/llm'
import { describeLlm } from '../shared/llm'
import { extractContract, type GoalContract } from '../shared/contract'
import { mcpCommandString, type McpToolInfo } from '../shared/mcp'
import {
  AGENT_METHODS,
  CREDENTIAL_METHODS,
  EVENT_CHANNELS,
  TOOL_METHODS,
  type FsWriteParams,
  type FsWriteResult,
  type LlmTestResult,
  type McpCallOutcome,
  type PrecheckResult,
  type RunLoopResult,
  type ToolExecResult,
  type ToolInvokeOutcome,
  type WireEnvelope
} from '../shared/ipc'
import {
  DEFAULT_TIER,
  type ApprovalRequest,
  type ApprovalTier,
  type DecisionBasis
} from '../shared/approval'
import type { ApprovalTicket } from '../shared/approval-token'
import type { Checkpoint, TaskRunState } from '../shared/events'

/* ------------------------------------------------------------------ *
 * Utility Process 的 parentPort（Electron 专有，非 worker_threads）
 * ------------------------------------------------------------------ */

interface ParentPortLike {
  on(event: 'message', cb: (e: { data: unknown; ports: MessagePortLike[] }) => void): void
  postMessage(msg: unknown): void
}
interface MessagePortLike {
  on(event: 'message', cb: (e: { data: unknown }) => void): void
  postMessage(msg: unknown): void
  start(): void
}

const parentPort = (process as unknown as { parentPort: ParentPortLike }).parentPort

let outPort: MessagePortLike | null = null

/* ------------------------------------------------------------------ *
 * 运行时状态
 * ------------------------------------------------------------------ */

const store: EventStoreLike = (() => {
  const dataDir = process.env['OPENINTJ_DATA_DIR'] ?? join(process.cwd(), '.openintj-data')
  try {
    return openStore(dataDir)
  } catch (err) {
    // 事件流是状态的单一事实来源，落不了盘就不该继续跑（不静默退化为内存）
    console.error(`[agent-host] 事件存储打开失败，fail-stop：${(err as Error).message}`)
    process.exit(1)
  }
})()
let currentTier: ApprovalTier = DEFAULT_TIER

interface Run {
  runId: string
  traceId: string
  state: TaskRunState
  goal: string
  createdAt: number
  /** 已生成或用户手写的目标契约；未确认前不得执行任何工具（IC-07 硬闸门） */
  contract?: GoalContract
  /** 循环是否在跑 —— 防止同一 run 被并发驱动两次（会把事件流的 seq 搅乱） */
  loopRunning?: boolean
  /** 最近一轮循环走了几步 —— 任务列表里显示用，没有跑过为 0 */
  lastSteps?: number
}
const runs = new Map<string, Run>()

/**
 * 待决审批。
 *
 * 注意 resolve 的形状：用户答「允许」时，随决定一起带回**主进程签发的票据**。
 * agent-host 拿不到密钥签不出票，只能原样转交给 tool-host。
 */
interface PendingApproval {
  request: ApprovalRequest
  resolve: (d: ApprovalDecision) => void
  timer: NodeJS.Timeout
}

export interface ApprovalDecision {
  approved: boolean
  remember: boolean
  ticket?: ApprovalTicket
  basis: DecisionBasis
}

const pendingApprovals = new Map<string, PendingApproval>()

/**
 * 审批等待上限。N-04b（模型调用超时与重试阈值）尚未拍板，
 * 这里取保守值并**超时即拒绝**（basis='timeout'）—— 绝不「超时视为同意」。
 */
const APPROVAL_TIMEOUT_MS = 120_000

/* ------------------------------------------------------------------ *
 * 反向调用（P-03 → P-01 → P-04）
 * ------------------------------------------------------------------ */

interface MainCall {
  resolve: (v: unknown) => void
  reject: (e: Error) => void
  timer: NodeJS.Timeout
}
const mainCalls = new Map<string, MainCall>()
let callSeq = 0

/**
 * 经主进程转发调用（主进程持有 P-04 端口，P-03 与 P-04 之间不开直连通道）。
 * 超时必须 reject —— 卡住的调用会让整个 run 静默挂起。
 */
function callMain<T>(method: string, params: unknown, timeoutMs = 30_000): Promise<T> {
  if (!outPort) return Promise.reject(new Error('主进程端口未就绪'))
  const requestId = `agent-req-${++callSeq}`
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      mainCalls.delete(requestId)
      reject(new Error(`主进程调用超时：${method}`))
    }, timeoutMs)
    mainCalls.set(requestId, { resolve: resolve as (v: unknown) => void, reject, timer })
    outPort!.postMessage({ kind: 'req', requestId, method, params } satisfies WireEnvelope)
  })
}

function settleMainCall(env: Extract<WireEnvelope, { kind: 'res' }>): void {
  const call = mainCalls.get(env.requestId)
  if (!call) return
  clearTimeout(call.timer)
  mainCalls.delete(env.requestId)
  if (env.ok) call.resolve(env.result)
  else call.reject(new Error(`${env.error.code}: ${env.error.message}`))
}

/* ------------------------------------------------------------------ *
 * 凭据（IC-06）
 * ------------------------------------------------------------------ */

/**
 * 取当前生效的模型凭据。
 *
 * 每次用之前都问一遍主进程：IPC 往返不到 1 毫秒，换来的是「设置改完立刻生效」，
 * 不需要任何缓存失效逻辑 —— 而缓存失效是这类代码最容易长 bug 的地方。
 *
 * 凭据来源（保险箱优先 / 环境变量兜底）由 P-01 一处判断，子进程不重复实现，
 * 否则两边的优先级迟早会不一致。
 */
async function currentLlm(): Promise<LlmConfigResult> {
  try {
    return await callMain<LlmConfigResult>(CREDENTIAL_METHODS.get, undefined, 5_000)
  } catch (err) {
    // 不静默回退到环境变量：保险箱里可能有一份配置，
    // 静默回退会让用户在界面上看到「未配置」，却不知道自己配过的东西去哪了
    return { configured: false, missing: [`凭据服务不可用：${(err as Error).message}`] }
  }
}

/**
 * 用当前生效凭据发一次最小请求。
 *
 * 花这 20 来个 token 是值得的：凭据填错的表现是「第一次跑任务时莫名其妙失败」，
 * 用户分不清是 key 错、地址错还是模型名错。一次廉价调用就能把「连得上」和「用不了」切开。
 */
async function testLlm(): Promise<LlmTestResult> {
  const llm = await currentLlm()
  if (!llm.configured) {
    return { ok: false, latencyMs: 0, reason: `未配置：缺 ${llm.missing.join(' / ')}` }
  }
  const started = Date.now()
  try {
    const res = await complete(
      llm.config,
      {
        maxTokens: 24,
        messages: [{ role: 'user', content: [{ type: 'text', text: '只回复两个字符：ok' }] }]
      },
      30_000
    )
    return {
      ok: true,
      latencyMs: Date.now() - started,
      model: res.model,
      protocol: llm.config.protocol,
      sample: textOf(res).slice(0, 60)
    }
  } catch (err) {
    return { ok: false, latencyMs: Date.now() - started, reason: (err as Error).message }
  }
}

/* ------------------------------------------------------------------ *
 * 事件发送
 * ------------------------------------------------------------------ */

function emit(channel: string, payload: unknown): void {
  const env: WireEnvelope = { kind: 'evt', channel, payload }
  outPort?.postMessage(env)
}

function emitEvent(input: AppendInput): void {
  // 落盘前脱敏。**这是唯一的口子** —— 放在这里，就不存在「某条调用路径忘了脱敏」。
  // 如果让每个 emitEvent 的调用点自己脱敏，漏掉的那一处不会报错，只会静静地漏。
  const outcome = applyRedaction(input.payload, input.redaction ?? 'none')
  if (outcome.hits.length > 0) {
    console.warn(
      `[agent-host] 事件 ${input.type} 已脱敏：${outcome.hits.map((h) => `${h.ruleId}×${h.count}`).join(', ')}`
    )
  }

  // redaction 字段写**实际**级别：请求 none 但命中密钥时会是 secret
  const { seq } = store.append({ ...input, payload: outcome.value, redaction: outcome.level })
  const events = store.list(input.runId, seq - 1)
  const ev = events[0]
  if (ev) emit(EVENT_CHANNELS.agentEvent, ev)
}

/* ------------------------------------------------------------------ *
 * 主循环（首版骨架）
 * ------------------------------------------------------------------ */

/**
 * 启动一次 run。
 *
 * 当前实现只推进到「契约草稿已生成」并停下 —— 因为生成契约需要一次真实模型调用，
 * 而模型凭据尚未配置。**这里绝不伪造模型输出**（N2：不接 Mock，US-01 AC-5 的反向断言）。
 * 凭据就位后在此处接入 LLM 双协议适配层。
 */
/** 契约起草的系统提示 —— 只出 JSON，且不许把「猜的」写成「已知的」 */
const CONTRACT_SYSTEM_PROMPT = `你是 openINTJ 的目标契约起草器。要把用户的一句话目标转成一份可验证、可执行的契约。

只输出一个 JSON 对象。不要解释，不要代码围栏，不要多余文字。

字段定义：
- objective: string，一句话目标
- successCriteria: string[]，每条都必须能被判定真假；禁止「更好/更快/更完善」这类无法判定的说法
- nonGoals: string[]，明确不做的事
- assumptions: string[]，契约成立所依赖的假设
- risks: string[]，已知风险
- needsClarification: string[]，必须问过用户才能继续的问题；没有就给空数组

规则：
1. 不得编造用户没提过的约束；
2. 拿不准的放进 needsClarification，不要塞进 assumptions 冒充已知；
3. 宁可少写，也不要写无法验证的判据。
4. 用户的话里如果出现「这个/那个/之前/尽快/适当」这类指代词，或没说清交付物形态，
   needsClarification 必须提出**最关键**的那几个问题（最多 3 个），不要自行补全。`

/**
 * 把结构化提出的澄清问题并进契约的 `needsClarification`。
 *
 * 上限 3 条是主干定的（超过 3 个问题没人会逐条答，只会全部忽略），
 * 去重按文本完全相同判 —— 语义相近但不相同的两条都留着，
 * 因为「合并掉一个真问题」比「多显示一条」贵得多。
 */
function mergeClarifications(existing: string[], questions: string[]): string[] {
  // 结构化提的问题排前面：它们是「不看就一定会做错」的那几条，
  // 起草顺手写的问题次之。上限取 3 与已有条数的较大者 ——
  // 契约里已经列了 4 条时不该因为合并反而被砍到 3 条。
  const out: string[] = []
  for (const q of [...questions, ...existing]) {
    const t = q.trim()
    if (!t || out.some((x) => x.trim() === t)) continue
    out.push(t)
  }
  return out.slice(0, Math.max(3, existing.length))
}

/**
 * 启动一次 run 并起草目标契约。
 *
 * 契约由**真实模型**生成；模型不可用/解析失败时如实标记，绝不产出看似合理的假契约（N2）。
 */
async function startRun(
  goal: string,
  opts?: { draft?: boolean; history?: UnifiedMessage[] }
): Promise<{
  runId: string
  state: TaskRunState
  contract?: GoalContract
  llm: LlmStatus
  /** 输入结构化的结论（歧义分 / 澄清问题 / 这一步做了什么）；未触发或未配置时为空 */
  structure?: InputStructureHint
}> {
  const now = Date.now()
  const runId = ulid(now)
  const traceId = ulid(now)

  const run: Run = { runId, traceId, state: 'CONTRACT_PENDING', goal, createdAt: now }
  runs.set(runId, run)

  // 任务建立即落一个里程碑检查点：恢复时至少有「从契约待确认处重来」的落点
  const cp = store.createCheckpoint(runId, 'milestone-boundary', 'run-created')
  emitEvent({
    runId,
    traceId,
    agentId: 'main',
    kind: 'action',
    type: 'CheckpointCreated',
    payload: { checkpointId: cp.checkpointId, atSeq: cp.atSeq, scope: cp.scope, label: cp.label }
  })

  // 用户可以选择不花模型 token 起草契约（例如他很清楚自己要什么）
  if (opts?.draft === false) {
    emitEvent({
      runId,
      traceId,
      agentId: 'main',
      kind: 'action',
      type: 'GoalContractDrafted',
      payload: { goal, status: 'skipped', reason: '按调用方要求跳过模型起草，等待手写契约' }
    })
    return { runId, state: run.state, llm: describeLlm(await currentLlm()) }
  }

  const llm = await currentLlm()
  const status = describeLlm(llm)

  if (!llm.configured) {
    emitEvent({
      runId,
      traceId,
      agentId: 'main',
      kind: 'action',
      type: 'GoalContractDrafted',
      payload: {
        goal,
        status: 'awaiting-model',
        reason: `LLM 未配置，契约无法生成（缺少 ${llm.missing.join(' / ')}）。可手写契约后确认。`
      }
    })
    return { runId, state: run.state, llm: status }
  }

  /**
   * 输入结构化 —— 起草之前先量一下这句话有多含糊。
   *
   * 它失败时**必须继续起草**（主干的 pass-through 语义），不能因此拒绝用户：
   * 为了消歧把一句话挡回去，是比不问更糟的体验。
   */
  let structure: InputStructureHint | undefined
  try {
    const s = await structureInput(llm.config, { text: goal, history: opts?.history ?? [] })
    structure = hintOf(s)
    emitEvent({
      runId,
      traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ModelUsage',
      payload: {
        stage: 'input-structuring',
        mode: s.mode,
        ambiguityScore: s.ambiguityScore,
        questions: s.questions,
        tokensSpent: s.tokensSpent,
        durationMs: s.durationMs
      }
    })
  } catch (err) {
    // 结构化失败不阻断：它只是让契约更准的手段，不是契约的前提
    emitEvent({
      runId,
      traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ErrorObserved',
      payload: { stage: 'input-structuring', message: (err as Error).message }
    })
  }

  try {
    // 用结构化后的执行输入去起草（没触发时它就是原话）
    const draftInput = structure?.executionInput?.trim() || goal
    const res = await complete(llm.config, {
      system: CONTRACT_SYSTEM_PROMPT,
      maxTokens: 4000,
      messages: [{ role: 'user', content: [{ type: 'text', text: draftInput }] }]
    })
    const draft = extractContract(textOf(res))

    if (draft.status === 'drafted') {
      // 结构化提出的澄清问题并进契约：契约是用户要逐条过目的东西，
      // 问题不进这里就等于没问 —— 模型会在 assumption 里悄悄替他决定
      const merged: GoalContract = structure?.questions.length
        ? {
            ...draft.contract,
            needsClarification: mergeClarifications(draft.contract.needsClarification, structure.questions)
          }
        : draft.contract
      run.contract = merged
      emitEvent({
        runId,
        traceId,
        agentId: 'main',
        kind: 'action',
        type: 'GoalContractDrafted',
        payload: {
          goal,
          status: 'drafted',
          contract: merged,
          // 结构化结论一并落事件：事后要能回答「这几个问题是谁提的」
          structure: structure ?? null,
          model: res.model,
          stopReason: res.stopReason,
          usage: res.usage,
          // 思考长度单列：它直接决定这次调用花了多少 token
          thinkingChars: thinkingOf(res).length
        }
      })
      // 契约生成是里程碑：用户确认前停在这里，恢复也从这里起
      const cp2 = store.createCheckpoint(runId, 'milestone-boundary', 'contract-drafted')
      emitEvent({
        runId,
        traceId,
        agentId: 'main',
        kind: 'action',
        type: 'CheckpointCreated',
        payload: { checkpointId: cp2.checkpointId, atSeq: cp2.atSeq, scope: cp2.scope, label: cp2.label }
      })
      return {
        runId,
        state: run.state,
        contract: merged,
        llm: status,
        ...(structure ? { structure } : {})
      }
    }

    emitEvent({
      runId,
      traceId,
      agentId: 'main',
      kind: 'action',
      type: 'GoalContractDrafted',
      payload: {
        goal,
        status: 'unparseable',
        reason: draft.reason,
        rawText: draft.rawText,
        structure: structure ?? null,
        model: res.model,
        usage: res.usage
      }
    })
    return { runId, state: run.state, llm: status, ...(structure ? { structure } : {}) }
  } catch (err) {
    emitEvent({
      runId,
      traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ErrorObserved',
      payload: { stage: 'contract-draft', message: (err as Error).message }
    })
    return { runId, state: run.state, llm: status }
  }
}

/**
 * 确认（或否决）目标契约 —— IC-07 硬闸门。
 *
 * 允许用户**手写契约**：LLM 不可用时，用户自己写一份并确认，同样成立 ——
 * 闸门的所有者是用户，不是模型。
 */
function confirmContract(input: {
  runId: string
  accepted: boolean
  contract?: GoalContract
}): { ok: boolean; state?: TaskRunState; reason?: string } {
  const run = runs.get(input.runId)
  if (!run) return { ok: false, reason: '任务不存在' }

  if (!input.accepted) {
    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'action',
      type: 'InterventionRecorded',
      payload: { kind: 'contract-rejected', note: '用户否决了契约草稿，任务停在待确认' }
    })
    return { ok: true, state: run.state }
  }

  // 优先用用户手写的（用户是权威），其次用模型起草的
  const contract = input.contract ?? run.contract
  if (!contract) {
    return { ok: false, reason: '没有可确认的契约：模型尚未起草，且未提供手写契约' }
  }
  run.contract = contract
  run.state = 'CONFIRMED'
  emitEvent({
    runId: run.runId,
    traceId: run.traceId,
    agentId: 'main',
    kind: 'action',
    type: 'GoalContractConfirmed',
    payload: { contract, source: input.contract ? 'user-authored' : 'model-drafted' }
  })
  return { ok: true, state: run.state }
}

function pauseRun(runId: string): { ok: boolean; reason?: string } {
  const run = runs.get(runId)
  if (!run) return { ok: false, reason: '任务不存在' }
  run.state = 'PAUSED'
  emitEvent({
    runId,
    traceId: run.traceId,
    agentId: 'main',
    kind: 'action',
    type: 'RunPaused',
    payload: { at: Date.now() }
  })
  return { ok: true }
}

/* ------------------------------------------------------------------ *
 * Agent 主循环（模型驱动）
 * ------------------------------------------------------------------ */

/**
 * 契约确认之后，把任务交给模型自己推进。
 *
 * 这条路径与「手动发起调用」共用同一条裁决链 —— 模型也只能**请求执行**，
 * 不能声明风险、不能自带批准。唯一的区别是：发起方从人换成了模型。
 * 这恰恰是更需要闸门的场景，所以这里没有为模型开任何捷径。
 */
async function runLoop(
  runId: string,
  followUp?: string,
  history?: UnifiedMessage[]
): Promise<RunLoopResult> {
  const emptyUsage = {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
    reasoningTokens: 0
  }

  const run = runs.get(runId)
  if (!run) return { outcome: 'failed', steps: 0, reason: '任务不存在', usage: emptyUsage }

  // IC-07：契约是这个循环的许可证，没有就不许开跑
  if (!run.contract) {
    return { outcome: 'failed', steps: 0, reason: '没有已确认的目标契约，不得启动循环（IC-07）', usage: emptyUsage }
  }
  if (run.state !== 'CONFIRMED' && run.state !== 'RUNNING' && run.state !== 'PAUSED') {
    return {
      outcome: 'failed',
      steps: 0,
      reason: `当前状态 ${run.state} 不允许启动循环（需先确认契约）`,
      usage: emptyUsage
    }
  }
  if (run.loopRunning) {
    return { outcome: 'failed', steps: 0, reason: '该任务的循环已在运行，不重复启动', usage: emptyUsage }
  }

  const llm = await currentLlm()
  if (!llm.configured) {
    return {
      outcome: 'failed',
      steps: 0,
      reason: `LLM 未配置，循环无法启动（缺 ${llm.missing.join(' / ')}）`,
      usage: emptyUsage
    }
  }

  // 从 CONFIRMED / PAUSED 进入 RUNNING —— 模型可能一步就 finish，状态不能停在「待确认」
  if (run.state !== 'RUNNING') {
    const resumed = run.state === 'PAUSED'
    run.state = 'RUNNING'
    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'action',
      type: resumed ? 'RunResumed' : 'TaskStarted',
      payload: { at: Date.now(), by: 'model-loop' }
    })
  }

  run.loopRunning = true

  try {
    // MCP 工具是动态的（取决于用户配了几个 server），所以工具清单每次开跑时现取。
    // 取不到只是少几个工具，不影响 shell 那条主路径。
    const { specs, mcp } = await resolveToolSpecs()
    if (mcp.length > 0) {
      emitEvent({
        runId: run.runId,
        traceId: run.traceId,
        agentId: 'main',
        kind: 'action',
        type: 'ToolInvoked',
        payload: {
          stage: 'mcp-tools-loaded',
          count: mcp.length,
          tools: mcp.map((t) => ({ name: t.qualifiedName, server: t.server, trust: t.trust }))
        }
      })
    }

    const result = await runAgentLoop({
      llm: llm.config,
      contract: run.contract,
      goal: run.goal,
      // 多轮对话的追加指令：只在有内容时带上，不塞空串污染提示词
      followUps: followUp ? [followUp] : undefined,
      history: history ?? [],
      tools: specs,
      requestCommand: async (command, purpose) => invokeTool({ runId, command, purpose }),
      requestMcpTool: async (server, tool, args, purpose) =>
        invokeMcpTool({ runId, server, tool, args, purpose }),
      // 生图：与命令执行共用同一道闸门 —— 写文件那一步照旧要过 tool-host 裁决
      requestImage: async (prompt, purpose) => generateImageTask({ runId, prompt, purpose }),
      emit: (kind, type, payload) => {
        emitEvent({ runId: run.runId, traceId: run.traceId, agentId: 'main', kind, type, payload })
      },
      // 暂停是软中断：每步边界检查一次，不在命令执行到一半时掐断
      shouldContinue: () => runs.get(runId)?.state === 'RUNNING'
    })

    if (result.outcome === 'delivered') {
      run.state = 'DELIVERED'
      run.lastSteps = result.steps
      const cp = store.createCheckpoint(runId, 'milestone-boundary', 'run-delivered')
      emitEvent({
        runId,
        traceId: run.traceId,
        agentId: 'main',
        kind: 'action',
        type: 'CheckpointCreated',
        payload: { checkpointId: cp.checkpointId, atSeq: cp.atSeq, scope: cp.scope, label: cp.label }
      })
      emitEvent({
        runId,
        traceId: run.traceId,
        agentId: 'main',
        kind: 'action',
        type: 'RunDelivered',
        payload: {
          summary: result.summary,
          satisfied: result.satisfied ?? [],
          unmet: result.unmet ?? [],
          steps: result.steps,
          usage: result.usage
        }
      })
    } else if (result.outcome === 'failed' || result.outcome === 'max-steps') {
      run.state = 'FAILED'
      emitEvent({
        runId,
        traceId: run.traceId,
        agentId: 'main',
        kind: 'action',
        type: 'RunFailed',
        payload: { reason: result.reason ?? result.outcome, steps: result.steps, usage: result.usage }
      })
    }
    // paused：pauseRun 已经落过 RunPaused，这里不重复

    return result
  } catch (err) {
    const message = (err as Error).message
    run.state = 'FAILED'
    emitEvent({
      runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'action',
      type: 'RunFailed',
      payload: { reason: message, steps: 0 }
    })
    return { outcome: 'failed', steps: 0, reason: message, usage: emptyUsage }
  } finally {
    run.loopRunning = false
    releasePendingApprovals(runId)
  }
}

/**
 * 收回该 run 上还挂着的审批请求。
 *
 * 循环都结束了，那些卡片背后已经没有等待者 —— 不收回就会留下一张
 * 「点了没反应」的僵尸卡片。用户点它只会得到「审批请求不存在或已超时」，
 * 那种体验比不显示还糟：它让人以为自己被无视了。
 */
function releasePendingApprovals(runId: string): void {
  const stale: string[] = []
  for (const [requestId, p] of pendingApprovals) {
    if (p.request.runId !== runId) continue
    clearTimeout(p.timer)
    pendingApprovals.delete(requestId)
    stale.push(requestId)
    // 让仍在等的那一侧立刻返回，而不是干等到超时
    p.resolve({ approved: false, remember: false, basis: 'timeout' })
  }
  if (stale.length > 0) {
    emit(EVENT_CHANNELS.approvalCancel, { runId, requestIds: stale, reason: 'run-finished' })
  }
}

/* ------------------------------------------------------------------ *
 * 工具调用（审批在环）
 * ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ *
 * MCP 工具（D6）
 * ------------------------------------------------------------------ */

/**
 * 取当前可用的 MCP 工具。
 *
 * 取不到就当没有：MCP 是增量能力，一个 server 起不来不该让整个任务跑不动。
 * 但失败要打日志 —— 用户看到「工具少了一个」时，我们得答得上来为什么。
 */
async function listMcpTools(): Promise<McpToolInfo[]> {
  try {
    return await callMain<McpToolInfo[]>(TOOL_METHODS.mcpList, undefined, 40_000)
  } catch (err) {
    console.warn(`[agent-host] 获取 MCP 工具清单失败：${(err as Error).message}`)
    return []
  }
}

/**
 * MCP 工具 → 模型看得到的 tool spec。
 *
 * description 里特意写上 trust 与「可能被拒」：模型知道这一步会弹窗之后，
 * 就不会把「被拒」当成系统故障反复重试 —— 那是它最常见的错误反应。
 */
function toMcpToolSpec(t: McpToolInfo): ToolSpec {
  return {
    name: t.qualifiedName,
    description: [
      t.description || `MCP 工具 ${t.name}`,
      '',
      `来源：MCP server「${t.server}」，该 server 声明的信任级别为 ${t.trust}。`,
      t.trust === 'readonly'
        ? '它被声明为只读，调用通常会自动放行。'
        : '每次调用都需要用户审批，且可能被拒绝。被拒不是错误，是用户的决定。'
    ].join('\n'),
    inputSchema: t.inputSchema
  }
}

/* ------------------------------------------------------------------ *
 * 工具目录
 * ------------------------------------------------------------------ */

/**
 * 组装本次要发给模型的工具清单。
 *
 * `runLoop` 与 `toolCatalog` 都调这里 —— 一个函数产出一份清单，
 * 就不存在「界面说有、模型看不到」这种只能靠猜的偏差。
 */
async function resolveToolSpecs(): Promise<{ specs: ToolSpec[]; mcp: McpToolInfo[] }> {
  const mcp = await listMcpTools()
  return { specs: [...AGENT_TOOLS, ...mcp.map(toMcpToolSpec)], mcp }
}

/* ------------------------------------------------------------------ *
 * 工具调用（审批在环）
 * ------------------------------------------------------------------ */

/**
 * IC-07 硬闸门 + 状态推进。
 *
 * 返回 null 表示可以执行；返回字符串表示拒绝理由。
 * shell 与 MCP 共用 —— 闸门是任务的属性，不是某个工具类型的属性。
 */
function ensureExecutable(run: Run): string | null {
  if (run.state !== 'CONFIRMED' && run.state !== 'RUNNING') {
    return `目标契约尚未确认（IC-07 硬闸门），当前状态 ${run.state}，不得执行工具`
  }
  if (run.state === 'CONFIRMED') {
    run.state = 'RUNNING'
    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'action',
      type: 'TaskStarted',
      payload: { contractConfirmedAt: Date.now() }
    })
  }
  return null
}

/** 工作目录取工作区根（与 P-01 下发给 P-04 的 workspaceRoot 同源） */
function workspaceDir(): string {
  return process.env['OPENINTJ_WORKSPACE'] ?? process.cwd()
}

interface ApprovalOutcome {
  requestId: string
  approved: boolean
  ticket?: ApprovalTicket
  basis: DecisionBasis
}

/**
 * 走一次审批：落事件 → 弹卡片 → 等应答。
 *
 * shell 与 MCP 共用这一段。两条路径的审批**语义必须完全一致**，
 * 否则用户学到的「点允许意味着什么」在两种工具上会不一样。
 *
 * `command` 是审批卡片上显示、并且被签进票据的那串文字：
 * shell 是命令行，MCP 是 `mcp:server:tool {参数}`。
 */
async function requestApproval(
  run: Run,
  pre: PrecheckResult,
  command: string,
  cwd: string
): Promise<ApprovalOutcome> {
  const request: ApprovalRequest = {
    requestId: ulid(),
    runId: run.runId,
    agentId: 'main',
    // 同一任务内串行：泳道内不并发，泳道间才并行（US-06 AC-6）
    parallelLaneId: run.runId,
    command,
    cwd,
    riskClass: pre.riskClass,
    // 需要人点头，不能异步吞掉
    allowAsyncPrompt: false,
    at: Date.now()
  }

  emitEvent({
    runId: run.runId,
    traceId: run.traceId,
    agentId: 'main',
    kind: 'action',
    type: 'ApprovalRequested',
    payload: { requestId: request.requestId, riskClass: pre.riskClass, rule: pre.matchedRuleId }
  })
  emit(EVENT_CHANNELS.approvalRequest, request)

  const decision = await waitForDecision(request)

  emitEvent({
    runId: run.runId,
    traceId: run.traceId,
    agentId: 'main',
    kind: 'action',
    type: 'ApprovalDecided',
    payload: {
      requestId: request.requestId,
      approved: decision.approved,
      remember: decision.remember,
      basis: decision.basis,
      // 票据只记存在与否，不把内容写进事件流
      ticketIssued: Boolean(decision.ticket)
    }
  })

  if (!decision.approved) {
    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ApprovalRejected',
      payload: { requestId: request.requestId, basis: decision.basis }
    })
  }

  return {
    requestId: request.requestId,
    approved: decision.approved,
    ticket: decision.ticket,
    basis: decision.basis
  }
}

function describeDenial(basis: DecisionBasis): string {
  return basis === 'timeout'
    ? `审批超时（${APPROVAL_TIMEOUT_MS}ms）未响应，按拒绝处理`
    : '用户拒绝执行'
}

/**
 * 发起一次工具调用（shell）。
 *
 * 顺序刻意如此：
 *   ① 先向 P-04 要预判（它会自己查真值表，不采信我们说的任何结论）
 *   ② 需要授权才弹窗；**先落 ToolInvoked 事件再弹窗**，事后可查「当时它想干什么」
 *   ③ 用户点了允许，主进程签票，票据原样转交 P-04；P-04 校验通过才 spawn
 *
 * 任一步失败都不产生进程，且失败原因写进事件流，不吞。
 */
async function invokeTool(input: {
  runId: string
  command: string
  /** 模型声明的目的（来自 tool_use 的 purpose 参数）；只记录，不参与裁决 */
  purpose?: string
}): Promise<ToolInvokeOutcome> {
  const run = runs.get(input.runId)
  if (!run) return { ok: false, reason: '任务不存在' }

  const gate = ensureExecutable(run)
  if (gate) return { ok: false, reason: gate }

  const cwd = workspaceDir()

  let pre: PrecheckResult
  try {
    pre = await callMain<PrecheckResult>(TOOL_METHODS.precheck, {
      runId: run.runId,
      command: input.command,
      cwd
    })
  } catch (err) {
    return { ok: false, reason: `预判失败：${(err as Error).message}` }
  }

  emitEvent({
    runId: run.runId,
    traceId: run.traceId,
    agentId: 'main',
    kind: 'action',
    type: 'ToolInvoked',
    payload: {
      command: input.command,
      // 模型自己说的目的 —— 用户看审批卡片时能直接判断「这一步有没有必要」
      purpose: input.purpose ?? null,
      cwd,
      riskClass: pre.riskClass,
      matchedRuleId: pre.matchedRuleId,
      outcome: pre.outcome,
      needsTicket: pre.needsTicket
    }
  })

  let ticket: ApprovalTicket | undefined

  if (pre.needsTicket) {
    const decision = await requestApproval(run, pre, input.command, cwd)
    if (!decision.approved) {
      // 被拒不执行，但**照样要落审计** —— 「用户拒绝过什么」是审计里最该留的一类
      await callMain(TOOL_METHODS.recordDenial, {
        runId: run.runId,
        command: input.command,
        cwd,
        basis: decision.basis
      }).catch((err: Error) => {
        console.warn(`[agent-host] 拒绝审计写入失败：${err.message}`)
      })

      return {
        ok: false,
        reason: describeDenial(decision.basis),
        risk: {
          command: input.command,
          riskClass: pre.riskClass,
          tier: currentTier,
          outcome: pre.outcome,
          basis: decision.basis,
          matchedRuleId: pre.matchedRuleId
        }
      }
    }
    ticket = decision.ticket
  }

  try {
    const result = await callMain<ToolExecResult>(
      TOOL_METHODS.exec,
      { runId: run.runId, agentId: 'main', command: input.command, cwd, approvalTicket: ticket },
      150_000
    )

    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'CommandOutput',
      payload: {
        command: input.command,
        exitCode: result.exitCode,
        stdout: result.stdout,
        stderr: result.stderr,
        durationMs: result.durationMs,
        truncated: result.truncated,
        riskClass: result.risk.riskClass,
        outcome: result.risk.outcome,
        basis: result.risk.basis
      },
      // 输出里可能带路径与内容，落盘前按内容级脱敏
      redaction: 'content'
    })

    // 命令输出里若出现回环地址（dev server 的提示），登记为预览目标 ——
    // 「预览」分页签只在真有预览时才亮出来，没有就一次都不出现。
    for (const url of sessions.detectLoopbackUrls(`${result.stdout}\n${result.stderr}`)) {
      const saved = sessions.notePreview({
        previewId: ulid(),
        runId: run.runId,
        kind: 'url',
        url,
        title: input.command
      })
      if (saved) {
        emitEvent({
          runId: run.runId,
          traceId: run.traceId,
          agentId: 'main',
          kind: 'observation',
          type: 'CommandOutput',
          payload: { stage: 'preview-detected', url, from: input.command }
        })
      }
    }

    const { risk, ...rest } = result
    return { ok: true, risk, result: rest }
  } catch (err) {
    const message = (err as Error).message
    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ErrorObserved',
      payload: { command: input.command, message }
    })
    return { ok: false, reason: message }
  }
}

/**
 * 发起一次 MCP 工具调用。
 *
 * 与 invokeTool **完全同构**（同一个闸门、同一次审批、同一条审计链），
 * 差别只有两处：预判走 mcpPrecheck（按 server 的 trust 分级）、
 * 执行的载荷是结构化参数而不是命令行。
 *
 * 这里不为 MCP 开任何捷径 —— 「工具是外部进程提供的」恰恰是需要更谨慎的理由。
 */
async function invokeMcpTool(input: {
  runId: string
  server: string
  tool: string
  args: unknown
  purpose?: string
}): Promise<McpInvokeOutcome> {
  const run = runs.get(input.runId)
  if (!run) return { ok: false, text: '', isError: true, reason: '任务不存在' }

  const gate = ensureExecutable(run)
  if (gate) return { ok: false, text: '', isError: true, reason: gate }

  const cwd = workspaceDir()
  // 票据绑定这一串 —— 必须与 tool-host 侧算出的逐字节一致
  const command = mcpCommandString(input.server, input.tool, input.args)

  let pre: PrecheckResult
  try {
    pre = await callMain<PrecheckResult>(TOOL_METHODS.mcpPrecheck, {
      runId: run.runId,
      server: input.server,
      tool: input.tool
    })
  } catch (err) {
    return { ok: false, text: '', isError: true, reason: `预判失败：${(err as Error).message}` }
  }

  emitEvent({
    runId: run.runId,
    traceId: run.traceId,
    agentId: 'main',
    kind: 'action',
    type: 'ToolInvoked',
    payload: {
      command,
      mcp: { server: input.server, tool: input.tool },
      args: input.args,
      purpose: input.purpose ?? null,
      cwd,
      riskClass: pre.riskClass,
      matchedRuleId: pre.matchedRuleId,
      outcome: pre.outcome,
      needsTicket: pre.needsTicket
    }
  })

  let ticket: ApprovalTicket | undefined

  if (pre.needsTicket) {
    const decision = await requestApproval(run, pre, command, cwd)
    if (!decision.approved) {
      // 被拒也要留痕；用 MCP 专用的拒绝入口，否则 shell 分类器会给这个字符串
      // 算出一个毫无意义的风险等级，写进审计反而误导
      await callMain(TOOL_METHODS.mcpDenial, {
        runId: run.runId,
        server: input.server,
        tool: input.tool,
        args: input.args,
        basis: decision.basis
      }).catch((err: Error) => {
        console.warn(`[agent-host] MCP 拒绝审计写入失败：${err.message}`)
      })

      return { ok: false, text: '', isError: true, reason: describeDenial(decision.basis) }
    }
    ticket = decision.ticket
  }

  try {
    const result = await callMain<McpCallOutcome>(
      TOOL_METHODS.mcpCall,
      {
        runId: run.runId,
        agentId: 'main',
        server: input.server,
        tool: input.tool,
        args: input.args,
        approvalTicket: ticket
      },
      110_000
    )

    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ToolResult',
      payload: {
        server: input.server,
        tool: input.tool,
        ok: result.ok,
        isError: result.isError,
        durationMs: result.durationMs,
        // 外部 server 的返回什么都可能有，按内容级脱敏
        text: result.text,
        riskClass: result.risk.riskClass,
        outcome: result.risk.outcome,
        basis: result.risk.basis
      },
      redaction: 'content'
    })

    return {
      ok: result.ok,
      text: result.text,
      isError: result.isError,
      reason: result.ok ? undefined : result.text
    }
  } catch (err) {
    const message = (err as Error).message
    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ErrorObserved',
      payload: { command, message }
    })
    return { ok: false, text: '', isError: true, reason: message }
  }
}

/** 等待用户应答；超时按拒绝处理，绝不「超时视为同意」 */
function waitForDecision(request: ApprovalRequest): Promise<ApprovalDecision> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      if (!pendingApprovals.has(request.requestId)) return
      pendingApprovals.delete(request.requestId)
      resolve({ approved: false, remember: false, basis: 'timeout' })
    }, APPROVAL_TIMEOUT_MS)

    pendingApprovals.set(request.requestId, { request, resolve, timer })
  })
}

/* ------------------------------------------------------------------ *
 * 会话（IDE 右栏：对话 / 任务 / 变更 / 预览）
 *
 * 会话是「多轮」的载体：消息存在本进程，界面切走了也不丢。
 * 任务列表与文件变化则是**派生**数据 —— 前者从 runs 读，后者由主进程
 * 把 tool-host 的监听结果转交过来登记。事件流才是事实来源，
 * 这里不复制第二份。
 * ------------------------------------------------------------------ */

/**
 * run 状态 → 会话状态。
 *
 * 不直接拿 TaskRunState 当会话状态用：run 关心「执行到哪一步」，
 * 会话关心「界面现在该显示什么」。强行共用的结果是界面要写好几个
 * `if` 去反推「现在到底该不该显示契约卡」，而那种推导迟早错一次。
 */
function sessionStateOf(run: Run | undefined): SessionState {
  if (!run) return 'empty'
  switch (run.state) {
    case 'CONTRACT_PENDING':
      return 'contract-pending'
    case 'CONFIRMED':
    case 'RUNNING':
    case 'CRITIC_REVIEW':
    case 'REPLANNING':
      return 'running'
    case 'PAUSED':
    case 'INTERRUPTED':
      return 'paused'
    case 'DELIVERED':
      return 'delivered'
    case 'FAILED':
      return 'failed'
    default:
      return 'failed'
  }
}

/** run 状态变了之后把会话状态同步过去 —— 界面的按钮可用性全靠这个值 */
function syncSession(sessionId: string): AgentSession | undefined {
  const s = sessions.getSession(sessionId)
  if (!s) return undefined
  sessions.setState(sessionId, sessionStateOf(s.runId ? runs.get(s.runId) : undefined))
  return sessions.getSession(sessionId)
}

/** 取（必要时新建）会话 —— 支持「发第一句话自动开一个会话」 */
function resolveSession(sessionId?: string, firstText?: string): AgentSession {
  const found = sessionId ? sessions.getSession(sessionId) : undefined
  if (found) return found
  return sessions.createSession(firstText)
}

/**
 * 发一句话。
 *
 * 分支不是按「第几轮」分，而是按**契约的许可状态**分 —— 因为真正决定
 * 这句话怎么处理的，是「现在许不许动手」：
 *   - 还没有 run（第一轮）      → 起草契约，停在硬闸门
 *   - run 还在 CONTRACT_PENDING → 契约没被确认，这句并入补充约束重新起草
 *   - 其余（已确认 / 暂停 / 收尾）→ 当作追加指令，再驱动一轮循环
 */
async function sendMessage(input: { sessionId?: string; text: string }): Promise<SessionSendResult> {
  const text = input.text.trim()
  if (!text) throw new Error('消息为空')

  const session = resolveSession(input.sessionId, text)
  const run = session.runId ? runs.get(session.runId) : undefined

  // 先落用户消息再动：跑崩了也留得下「用户当时说了什么」
  sessions.appendMessage({ sessionId: session.sessionId, role: 'user', kind: 'text', text })

  if (!run || run.state === 'CONTRACT_PENDING') {
    const goal = run ? `${run.goal}\n补充约束：${text}` : text
    // 之前轮次的对话也交给结构化器：一句「再快一点」只有接在上文之后才消得掉歧义
    const res = await startRun(goal, { history: historyOf(session) })
    sessions.bindRun(session.sessionId, res.runId)

    const drafted = Boolean(res.contract)
    const st = res.structure
    sessions.appendMessage({
      sessionId: session.sessionId,
      role: 'agent',
      kind: 'contract-draft',
      text: drafted
        ? `契约已起草：${res.contract?.objective ?? ''}（成功判据 ${res.contract?.successCriteria.length ?? 0} 条${
            res.contract?.needsClarification.length
              ? `，待澄清 ${res.contract.needsClarification.length} 项`
              : ''
          }）`
        : (res.llm.missing.length > 0
            ? `未能起草契约：模型未配置（缺 ${res.llm.missing.join(' / ')}）`
            : '未能起草契约：模型返回无法解析'),
      payload: { contract: res.contract ?? null, llm: res.llm, structure: st ?? null }
    })

    const updated = syncSession(session.sessionId) ?? session
    return {
      session: updated,
      messages: sessions.messagesOf(session.sessionId),
      phase: drafted ? 'contract-pending' : 'awaiting-model'
    }
  }

  const result = await runLoop(run.runId, text, historyOf(session))
  const ok = result.outcome === 'delivered'

  sessions.appendMessage({
    sessionId: session.sessionId,
    role: 'agent',
    kind: ok ? 'delivery' : 'error',
    text: ok
      ? `完成（${result.steps} 步）${result.summary ? `：${result.summary}` : ''}`
      : (result.reason ?? result.outcome),
    payload: result
  })

  const updated = syncSession(session.sessionId) ?? session
  return {
    session: updated,
    messages: sessions.messagesOf(session.sessionId),
    phase: ok ? 'finished' : 'error'
  }
}

/**
 * 确认 / 否决当前会话的契约 —— IC-07 硬闸门的开关在用户手上。
 *
 * 确认这里**不自动开跑**：契约成立与「开始执行」是两件事，
 * 合在一起的后果是用户点一下确认，命令就已经跑完了 —— 那不叫确认，叫追认。
 * 所以这里只推进状态，跑循环是界面上「开始执行」的另一个动作。
 */
function confirmSessionContract(input: {
  sessionId: string
  accepted: boolean
  contract?: GoalContract
}): { ok: boolean; session?: AgentSession; reason?: string } {
  const session = sessions.getSession(input.sessionId)
  if (!session) return { ok: false, reason: '会话不存在' }
  if (!session.runId) return { ok: false, reason: '该会话还没有待确认的契约' }

  const res = confirmContract({
    runId: session.runId,
    accepted: input.accepted,
    contract: input.contract
  })
  if (!res.ok) return { ok: false, reason: res.reason }

  sessions.appendMessage({
    sessionId: session.sessionId,
    role: 'system',
    kind: 'contract-result',
    text: input.accepted ? '契约已确认，可以开始执行' : '契约已被否决，任务停在待确认',
    payload: { accepted: input.accepted, state: res.state }
  })

  return { ok: true, session: syncSession(session.sessionId) }
}

/**
 * 把会话里之前轮次的消息拼成模型看得懂的历史。
 *
 * 只带文本类消息：契约草稿、审批请求、执行结果这些**结构化**的东西不进历史，
 * 它们各自在界面上有卡片，塞进上下文只会占地方、还会让模型以为自己说过那些 JSON。
 *
 * 最后一条（当前这句）由 `followUps` 单独呈现，这里要去掉，否则会重复一遍。
 */
function historyOf(session: AgentSession): UnifiedMessage[] {
  const all = sessions.messagesOf(session.sessionId)
  const prior = all.slice(0, -1)
  const out: UnifiedMessage[] = []
  for (const m of prior) {
    if (m.role === 'user' && m.kind === 'text') {
      out.push({ role: 'user', content: [{ type: 'text', text: m.text }] })
    } else if (m.role === 'agent' && (m.kind === 'text' || m.kind === 'progress')) {
      out.push({ role: 'assistant', content: [{ type: 'text', text: m.text }] })
    }
  }
  return out
}

/** 会话的任务列表 —— 从 runs 直接派生，不另存一份 */
function tasksOf(sessionId: string): TaskItem[] {
  const s = sessions.getSession(sessionId)
  if (!s?.runId) return []
  const run = runs.get(s.runId)
  if (!run) return []

  return [
    {
      taskId: run.runId,
      runId: run.runId,
      title: run.goal,
      state: run.state,
      steps: run.lastSteps ?? 0,
      at: run.createdAt
    }
  ]
}

function changesOf(sessionId: string): FileChange[] {
  const s = sessions.getSession(sessionId)
  if (!s?.runId) return []
  return sessions.changesOf(s.runId)
}

function previewsOf(sessionId: string): PreviewTarget[] {
  const s = sessions.getSession(sessionId)
  if (!s?.runId) return []
  return sessions.previewsOf(s.runId)
}

/* ------------------------------------------------------------------ *
 * 工件落盘与生图
 * ------------------------------------------------------------------ */

/** 生图凭据 —— 与对话模型分开一条，见 shared/llm.ts 里 ImageConfig 的说明 */
async function currentImage(): Promise<ImageConfigResult> {
  try {
    return await callMain<ImageConfigResult>(CREDENTIAL_METHODS.getImage, undefined, 5_000)
  } catch (err) {
    // 拉不到就当没配：生图是可选能力，不能因为它把整个循环拖死
    console.warn(`[agent-host] 生图凭据获取失败：${(err as Error).message}`)
    return { configured: false, missing: ['credential.image.get 调用失败'] }
  }
}

export interface ArtifactWriteOutcome {
  ok: boolean
  /** 相对工作区根的路径 */
  path?: string
  bytes?: number
  reason?: string
}

/**
 * 把一个工件写进工作区 —— **必须过 tool-host 的裁决链**。
 *
 * 这里是最容易被「图省事」绕过去的地方：agent-host 手里已经有字节了，
 * 直接 `fs.writeFile` 三行就能落盘。但那样写进去的东西
 *   —— 没有风险分级、没有审计、没有「用户拒绝过什么」的记录，
 * 与这条产品最核心的承诺（动手必经裁决）直接冲突。
 * 所以哪怕只是落一张图，也老老实实走 precheck → 审批 → 写入 → 审计。
 */
async function writeArtifact(input: {
  runId: string
  path: string
  contentBase64: string
  mimeType: string
  purpose: string
}): Promise<ArtifactWriteOutcome> {
  const run = runs.get(input.runId)
  if (!run) return { ok: false, reason: '任务不存在' }

  const gate = ensureExecutable(run)
  if (gate) return { ok: false, reason: gate }

  let pre: PrecheckResult
  try {
    pre = await callMain<PrecheckResult>(TOOL_METHODS.precheck, {
      runId: run.runId,
      command: `writeFile ${input.path}`,
      cwd: workspaceDir()
    })
  } catch (err) {
    return { ok: false, reason: `预判失败：${(err as Error).message}` }
  }

  emitEvent({
    runId: run.runId,
    traceId: run.traceId,
    agentId: 'main',
    kind: 'action',
    type: 'ToolInvoked',
    payload: {
      command: `writeFile ${input.path}`,
      purpose: input.purpose,
      cwd: workspaceDir(),
      riskClass: pre.riskClass,
      matchedRuleId: pre.matchedRuleId,
      outcome: pre.outcome,
      needsTicket: pre.needsTicket
    }
  })

  let ticket: ApprovalTicket | undefined
  if (pre.needsTicket) {
    const decision = await requestApproval(run, pre, `writeFile ${input.path}`, workspaceDir())
    if (!decision.approved) {
      await callMain(TOOL_METHODS.recordDenial, {
        runId: run.runId,
        command: `writeFile ${input.path}`,
        cwd: workspaceDir(),
        basis: decision.basis
      }).catch((err: Error) => {
        console.warn(`[agent-host] 拒绝审计写入失败：${err.message}`)
      })
      return { ok: false, reason: describeDenial(decision.basis) }
    }
    ticket = decision.ticket
  }

  try {
    const res = await callMain<FsWriteResult>(
      TOOL_METHODS.fsWrite,
      {
        runId: run.runId,
        agentId: 'main',
        path: input.path,
        contentBase64: input.contentBase64,
        mimeType: input.mimeType,
        overwrite: true,
        approvalTicket: ticket
      } satisfies FsWriteParams,
      60_000
    )
    if (!res.ok) return { ok: false, reason: res.reason ?? '写入未通过' }

    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'CommandOutput',
      payload: {
        stage: 'artifact-written',
        path: res.path ?? input.path,
        bytes: res.bytes ?? 0,
        mimeType: input.mimeType
      }
    })
    return { ok: true, path: res.path ?? input.path, bytes: res.bytes ?? 0 }
  } catch (err) {
    const message = (err as Error).message
    emitEvent({
      runId: run.runId,
      traceId: run.traceId,
      agentId: 'main',
      kind: 'observation',
      type: 'ErrorObserved',
      payload: { stage: 'artifact-write', path: input.path, message }
    })
    return { ok: false, reason: message }
  }
}

/** 工件文件名：时间戳 + 来自 prompt 的短 slug，避免同一次任务里互相覆盖 */
function artifactName(prompt: string, ext: string): string {
  const slug =
    prompt
      .toLowerCase()
      .replace(/[^\p{Script=Han}\w]+/gu, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 24) || 'image'
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  return `${ARTIFACT_DIR}/${stamp}-${slug}.${ext}`
}

/**
 * 生图任务。
 *
 * 不论成功还是失败，产物都要落盘：
 *   成功 → 真正的图，登记为预览
 *   失败 → 一张写着失败原因的占位 SVG（**同样走审批**）
 *
 * 为什么失败也要落盘：让它「静默失败、什么都不留」的后果是用户以为
 * 自己在等一张图，而实际上早已放弃。留一张会说话的占位图，
 * 至少让「这一步到底发生了什么」是可查的。
 */
async function generateImageTask(input: {
  runId: string
  prompt: string
  purpose: string
}): Promise<{ ok: boolean; text: string; path?: string }> {
  const image = await currentImage()
  const name = artifactName(input.prompt, 'png')

  if (!image.configured) {
    const reason = `生图未配置（缺 ${image.missing.join(' / ')}）`
    const svg = placeholderSvg(input.prompt, reason)
    const written = await writeArtifact({
      runId: input.runId,
      path: artifactName(input.prompt, 'svg'),
      contentBase64: Buffer.from(svg, 'utf8').toString('base64'),
      mimeType: 'image/svg+xml',
      purpose: `${input.purpose}（生图不可用，落占位图）`
    })
    if (!written.ok) {
      return { ok: false, text: `${reason}；占位图也未能落盘：${written.reason}` }
    }
    sessions.noteArtifact({
      runId: input.runId,
      path: written.path!,
      title: `占位图：${input.prompt.slice(0, 40)}`,
      mimeType: 'image/svg+xml',
      size: svg.length
    })
    return {
      ok: false,
      text: `${reason}。已在工作区落下占位图 ${written.path}（说明失败原因），配置好生图凭据后可重试。`,
      path: written.path
    }
  }

  let result: ImageGenerateResult
  try {
    result = await generateImage(image.config, { prompt: input.prompt })
  } catch (err) {
    const reason = (err as Error).message
    const svg = placeholderSvg(input.prompt, reason)
    const written = await writeArtifact({
      runId: input.runId,
      path: artifactName(input.prompt, 'svg'),
      contentBase64: Buffer.from(svg, 'utf8').toString('base64'),
      mimeType: 'image/svg+xml',
      purpose: `${input.purpose}（生图失败，落占位图）`
    })
    if (!written.ok) {
      return { ok: false, text: `${reason}；占位图也未能落盘：${written.reason}` }
    }
    sessions.noteArtifact({
      runId: input.runId,
      path: written.path!,
      title: `占位图：${input.prompt.slice(0, 40)}`,
      mimeType: 'image/svg+xml',
      size: svg.length
    })
    return { ok: false, text: `${reason}。已落占位图 ${written.path}。`, path: written.path }
  }

  const written = await writeArtifact({
    runId: input.runId,
    path: name,
    contentBase64: result.bytesBase64,
    mimeType: result.mimeType,
    purpose: input.purpose
  })
  if (!written.ok) {
    return { ok: false, text: `图已生成但未能落盘：${written.reason}` }
  }

  sessions.noteArtifact({
    runId: input.runId,
    path: written.path!,
    title: input.prompt.slice(0, 60),
    mimeType: result.mimeType,
    size: result.byteLength
  })

  const revised = result.revisedPrompt ? `；网关改写后的 prompt：${result.revisedPrompt}` : ''
  return {
    ok: true,
    text: `已生成 ${written.path}（${result.byteLength} 字节，${result.size}，耗时 ${result.durationMs}ms）${revised}`,
    path: written.path
  }
}

/* ------------------------------------------------------------------ *
 * 请求处理
 * ------------------------------------------------------------------ */

async function handleRequest(method: string, params: unknown): Promise<unknown> {
  switch (method) {
    case AGENT_METHODS.ping:
      return `agent-host online (pid=${process.pid}, runs=${runs.size})`

    case AGENT_METHODS.getTier:
      return currentTier

    case AGENT_METHODS.setTier: {
      const { tier, prev } = params as { tier: ApprovalTier; prev: ApprovalTier }
      currentTier = tier
      // 档位切换必须落审计事件（PolicyTierSwitched）
      emit(EVENT_CHANNELS.agentEvent, {
        eventId: ulid(),
        seq: 0,
        runId: 'system',
        traceId: ulid(),
        tenantId: 'local',
        agentId: 'main',
        kind: 'action',
        type: 'PolicyTierSwitched',
        at: Date.now(),
        redaction: 'none',
        payload: { from: prev, to: tier },
        schemaVersion: 1
      })
      return { ok: true }
    }

    case AGENT_METHODS.startRun: {
      const { goal, draft } = params as { goal: string; draft?: boolean }
      return await startRun(goal, { draft })
    }

    case AGENT_METHODS.confirmContract: {
      return confirmContract(params as { runId: string; accepted: boolean; contract?: GoalContract })
    }

    case AGENT_METHODS.llmStatus: {
      return describeLlm(await currentLlm())
    }

    case AGENT_METHODS.settingGet: {
      const { key } = params as { key: string }
      return { key, value: store.getSetting(key) }
    }

    case AGENT_METHODS.settingSet: {
      const { key, value } = params as { key: string; value: string }
      store.setSetting(key, value)
      return { ok: true }
    }

    case AGENT_METHODS.llmTest: {
      return await testLlm()
    }

    case AGENT_METHODS.pauseRun: {
      const { runId } = params as { runId: string }
      return pauseRun(runId)
    }

    case AGENT_METHODS.listCheckpoints: {
      const { runId } = params as { runId: string }
      return store.listCheckpoints(runId) as Checkpoint[]
    }

    case AGENT_METHODS.approvalDecision: {
      const { requestId, approved, remember, ticket } = params as {
        requestId: string
        approved: boolean
        remember?: boolean
        ticket?: ApprovalTicket
      }
      const pending = pendingApprovals.get(requestId)
      if (!pending) return { ok: false, reason: '审批请求不存在或已超时' }
      clearTimeout(pending.timer)
      pendingApprovals.delete(requestId)
      pending.resolve({
        approved,
        remember: Boolean(remember),
        ticket,
        basis: approved ? 'user-grant' : 'user-deny'
      })
      return { ok: true }
    }

    case AGENT_METHODS.invokeTool: {
      const input = params as { runId: string; command: string; purpose?: string }
      return invokeTool(input)
    }

    case AGENT_METHODS.runLoop: {
      const { runId } = params as { runId: string }
      return await runLoop(runId)
    }

    case AGENT_METHODS.toolCatalog: {
      const { specs, mcp } = await resolveToolSpecs()
      return {
        builtin: specs.filter((s) => !s.name.startsWith('mcp__')).map((s) => s.name),
        mcp: mcp.map((t) => ({ name: t.qualifiedName, server: t.server, trust: t.trust })),
        total: specs.length
      }
    }

    /* ---------------- 会话（IDE 右栏） ---------------- */

    case SESSION_METHODS.list: {
      return sessions.listSessions()
    }

    case SESSION_METHODS.create: {
      const { title } = (params ?? {}) as { title?: string }
      return sessions.createSession(title)
    }

    case SESSION_METHODS.messages: {
      const { sessionId } = params as { sessionId: string }
      return sessions.messagesOf(sessionId)
    }

    case SESSION_METHODS.confirmContract: {
      const input = params as { sessionId: string; accepted: boolean; contract?: GoalContract }
      return confirmSessionContract(input)
    }

    case SESSION_METHODS.send: {
      const input = params as { sessionId?: string; text: string }
      return await sendMessage(input)
    }

    case SESSION_METHODS.tasks: {
      const { sessionId } = params as { sessionId: string }
      return tasksOf(sessionId)
    }

    case SESSION_METHODS.changes: {
      const { sessionId } = params as { sessionId: string }
      return changesOf(sessionId)
    }

    case SESSION_METHODS.noteChange: {
      const p = params as { runId: string; path: string; kind: FileChange['kind'] }
      return sessions.noteChange(p)
    }

    case SESSION_METHODS.confirmChange: {
      const { changeId } = params as { changeId: string }
      return { ok: sessions.confirmChange(changeId) }
    }

    case SESSION_METHODS.preview: {
      const { sessionId } = params as { sessionId: string }
      return previewsOf(sessionId)
    }

    /**
     * 手动登记预览地址。
     *
     * 地址是用户自己填的，仍然只接受回环地址 ——
     * 不是不信任用户，而是「预览」这个入口一旦能指向外部，
     * 就再没有一个地方能说清「应用里显示的到底是谁的内容」。
     */
    case SESSION_METHODS.previewAdd: {
      const input = params as { sessionId: string; url: string; title?: string }
      const session = sessions.getSession(input.sessionId)
      if (!session?.runId) return { ok: false, reason: '该会话还没有执行中的 run' }
      const saved = sessions.notePreview({
        previewId: ulid(),
        runId: session.runId,
        kind: 'url',
        url: input.url,
        title: input.title ?? input.url
      })
      if (!saved) return { ok: false, reason: '只接受回环地址（127.0.0.1 / localhost / ::1）' }
      return { ok: true, preview: saved }
    }

    default:
      throw new Error(`未知方法：${method}`)
  }
}

/* ------------------------------------------------------------------ *
 * 端口接入
 * ------------------------------------------------------------------ */

function bindPort(port: MessagePortLike): void {
  outPort = port
  port.on('message', (e) => {
    const env = e.data as WireEnvelope

    // 主进程回来的响应（我们反向请求的）
    if (env.kind === 'res') {
      settleMainCall(env)
      return
    }
    if (env.kind !== 'req') return

    handleRequest(env.method, env.params).then(
      (result) => {
        port.postMessage({ kind: 'res', requestId: env.requestId, ok: true, result } satisfies WireEnvelope)
      },
      (err: Error) => {
        port.postMessage({
          kind: 'res',
          requestId: env.requestId,
          ok: false,
          error: { code: 'AGENT_HOST_ERROR', message: err.message }
        } satisfies WireEnvelope)
      }
    )
  })
  port.start()

  // 上线即告知主进程
  port.postMessage({
    kind: 'evt',
    channel: EVENT_CHANNELS.processStatus,
    payload: { service: 'agentHost', status: 'online' }
  } satisfies WireEnvelope)
}

if (!parentPort) {
  console.error('[agent-host] 必须以 Electron Utility Process 方式启动')
  process.exit(1)
}

parentPort.on('message', (e) => {
  const data = e.data as { kind?: string }
  if (data?.kind === 'port-init' && e.ports.length > 0) {
    bindPort(e.ports[0]!)
  }
})
