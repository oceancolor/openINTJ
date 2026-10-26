/**
 * Agent 主循环 —— 模型驱动的「想一步、做一步、看结果、再想」
 *
 * 这条链路的核心不是「让模型调工具」，而是**让模型的每一次动手都经过裁决**：
 *   模型输出 tool_use → 本循环调 invokeTool → tool-host 自行分级 →（可能需要审批）→ 执行
 *   执行结果作为 tool_result 回灌模型 → 下一步。
 *
 * 关键点：本循环**没有**任何直接执行能力。它只会「请求执行」，也从不知道票据长什么样 ——
 * 票据由主进程签发、直达 tool-host，agent-host 只是搬运工。
 *
 * 另外两个刻意的选择：
 *   ① 步数上限 MAX_STEPS —— 模型会绕圈，绕圈烧的是真钱，必须有硬上限。
 *   ② `finish` 是唯一的正常出口 —— 模型「话说完但没调 finish」不算交付，判为 failed。
 *      否则「跑了几步没报错」会被误当成成功，这正是 Agent 产品最常见的不诚实。
 */

import type { ContentPart, TokenUsage, UnifiedMessage, LlmConfig, ToolSpec } from '../shared/llm'
import type { GoalContract } from '../shared/contract'
import type { ToolInvokeOutcome } from '../shared/ipc'
import type { ObservationType, ActionType } from '../shared/events'
import { isMcpToolName, parseMcpToolName } from '../shared/mcp'
import { complete, textOf } from './llm'
import { AGENT_SYSTEM_PROMPT, MAX_COMMANDS_PER_TURN } from './tool-specs'

/** 步数上限。12 步足够一次「看清现状 → 改 → 验」的常规任务；再多基本是在绕圈 */
export const MAX_STEPS = 12

/** 单步 model 调用的超时。N-04b 未拍板，取比契约起草更宽的值（带工具的多步推理更慢） */
export const STEP_TIMEOUT_MS = 180_000

/** MCP 调用结果 —— 与 shell 的形状刻意不同：MCP 没有 exitCode，只有文本与 isError */
export interface McpInvokeOutcome {
  ok: boolean
  text: string
  isError: boolean
  /** ok=false 时的原因（未执行 / 被拒 / 链路错误） */
  reason?: string
}

export interface AgentLoopDeps {
  llm: LlmConfig
  contract: GoalContract
  goal: string
  /**
   * 多轮对话里用户确认契约之后追加的指令。
   *
   * 契约是本轮的授权边界，追加指令只在契约范围内生效 ——
   * 它不能扩大授权，只能改变在授权范围内怎么做。
   * 界面上要照实说明这一点，否则用户会以为「补一句话」就能让它去做契约外的事。
   */
  followUps?: string[]
  /**
   * 之前轮次的对话（用户 → 模型），按时间顺序。
   *
   * 为什么需要：没有它的话每一轮都是一次全新的对话，模型不知道自己上一步
   * 改了什么、为什么这么改，于是会重复劳动 —— 表现为「它又做了一遍」。
   */
  history?: UnifiedMessage[]
  /**
   * 本次要发给模型的工具清单。由编排层组装 —— 内置工具 + 当前连上的 MCP 工具。
   * 循环自己不认识 MCP server，它只按名字前缀决定往哪个回调上分发。
   */
  tools: ToolSpec[]
  /**
   * 请求执行一条命令。由编排层注入 —— 循环本身不认识 tool-host。
   * 返回 ok=false 表示被拒或出错，原因必须如实回灌给模型。
   */
  requestCommand: (command: string, purpose: string) => Promise<ToolInvokeOutcome>
  /** 请求调用一个 MCP 工具。与 requestCommand 走同一条裁决链 */
  requestMcpTool: (
    server: string,
    tool: string,
    args: unknown,
    purpose: string
  ) => Promise<McpInvokeOutcome>
  /** 落事件。循环不直接碰 store，避免与编排层的 seq 逻辑打架 */
  emit: (kind: 'action' | 'observation', type: ActionType | ObservationType, payload: unknown) => void
  /** 每步开始前问一次「还该继续吗」（用于暂停 / 中断） */
  shouldContinue: () => boolean
}

export interface DeliveredItem {
  criterion: string
  evidence: string
}

export interface UnmetItem {
  criterion: string
  reason: string
}

export interface AgentLoopResult {
  outcome: 'delivered' | 'failed' | 'paused' | 'max-steps'
  steps: number
  /** 模型在 finish 里给的总结；非 delivered 时为空 */
  summary?: string
  satisfied?: DeliveredItem[]
  unmet?: UnmetItem[]
  /** 非 delivered 的原因（给界面看的一句话） */
  reason?: string
  /** 整个循环累计的 token 用量 —— 成本核算靠它 */
  usage: TokenUsage
}

const ZERO_USAGE: TokenUsage = {
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
  reasoningTokens: 0
}

function addUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
    cacheCreationTokens: a.cacheCreationTokens + b.cacheCreationTokens,
    reasoningTokens: a.reasoningTokens + b.reasoningTokens
  }
}

/** 开场白：把契约原样交给模型，并把「只有这些是你被授权做的事」讲清楚 */
function kickoffText(goal: string, contract: GoalContract, followUps?: string[]): string {
  const lines = [
    '用户的一句话目标：',
    goal,
    '',
    '用户已确认的目标契约（这是你的授权边界）：',
    JSON.stringify(contract, null, 2)
  ]

  if (followUps && followUps.length > 0) {
    lines.push(
      '',
      '用户在此之后追加的要求（按时间顺序，后说的优先；不得超出上面的契约范围）：',
      followUps.map((f, i) => `${i + 1}. ${f}`).join('\n')
    )
  }

  lines.push('', '现在开始。第一步建议先只读地看清现状，再决定要不要动手。')
  return lines.join('\n')
}

/** 工具返回结果 → 回灌给模型的文本。太长的输出要截断，否则上下文会被一条命令吃掉 */
function resultText(out: ToolInvokeOutcome): string {
  if (!out.ok) {
    return `【未执行】${out.reason ?? '未知原因'}`
  }
  const r = out.result
  if (!r) return '【已执行】没有输出'
  const parts = [`【已执行】exitCode=${r.exitCode} 耗时=${r.durationMs}ms`]
  if (r.stdout.trim()) parts.push(`--- stdout ---\n${clip(r.stdout)}`)
  if (r.stderr.trim()) parts.push(`--- stderr ---\n${clip(r.stderr)}`)
  if (r.truncated) parts.push('（输出超过 512KB 已截断）')
  if (!r.stdout.trim() && !r.stderr.trim()) parts.push('（无输出）')
  return parts.join('\n')
}

/** 单条工具输出上限 —— 8000 字符约 2~3k token，够看清错误信息，又不至于撑爆上下文 */
const OUTPUT_CLIP = 8000
function clip(s: string): string {
  if (s.length <= OUTPUT_CLIP) return s
  return `${s.slice(0, OUTPUT_CLIP)}\n…（已截断，原始长度 ${s.length} 字符）`
}

/**
 * 从工具入参里取 purpose。
 *
 * MCP 工具的 schema 由外部 server 定义，不保证有 purpose 字段；
 * 有就记录，没有就是空 —— 不能为了记一个 purpose 去改别人的 schema。
 */
function purposeOf(input: unknown): string {
  if (typeof input !== 'object' || input === null) return ''
  const p = (input as Record<string, unknown>)['purpose']
  return typeof p === 'string' ? p.trim() : ''
}

interface FinishInput {
  summary: string
  satisfied: DeliveredItem[]
  unmet: UnmetItem[]
}

/** 解析 finish 的入参。模型可能给错形状 —— 给错就如实说，不猜 */
function parseFinish(input: unknown): FinishInput | null {
  if (typeof input !== 'object' || input === null) return null
  const o = input as Record<string, unknown>
  if (typeof o['summary'] !== 'string') return null

  const toPairs = <K1 extends string, K2 extends string>(
    v: unknown,
    k1: K1,
    k2: K2
  ): Record<K1 | K2, string>[] => {
    if (!Array.isArray(v)) return []
    return v
      .filter((i): i is Record<string, unknown> => typeof i === 'object' && i !== null)
      .filter((i) => typeof i[k1] === 'string' && typeof i[k2] === 'string')
      .map((i) => ({ [k1]: i[k1] as string, [k2]: i[k2] as string }) as Record<K1 | K2, string>)
  }

  return {
    summary: o['summary'],
    satisfied: toPairs(o['satisfied'], 'criterion', 'evidence') as unknown as DeliveredItem[],
    unmet: toPairs(o['unmet'], 'criterion', 'reason') as unknown as UnmetItem[]
  }
}

/**
 * 跑完整循环。
 *
 * 返回的 outcome 一定是这四种之一，不存在「不知道算不算成功」的第五种。
 */
export async function runAgentLoop(deps: AgentLoopDeps): Promise<AgentLoopResult> {
  // 历史在前、开场白在后：模型先看到「之前说过什么」，再看到「现在要做什么」。
  // 顺序反过来的话，多轮上下文会被当成背景噪声，行为退化成单轮。
  const messages: UnifiedMessage[] = [
    ...(deps.history ?? []),
    { role: 'user', content: [{ type: 'text', text: kickoffText(deps.goal, deps.contract, deps.followUps) }] }
  ]

  let usage = ZERO_USAGE
  let steps = 0

  for (let step = 1; step <= MAX_STEPS; step++) {
    if (!deps.shouldContinue()) {
      return {
        outcome: 'paused',
        steps,
        reason: '任务已被暂停，循环在步边界停下',
        usage
      }
    }

    steps = step

    let res
    try {
      res = await complete(
        deps.llm,
        {
          system: AGENT_SYSTEM_PROMPT,
          messages,
          tools: deps.tools,
          maxTokens: 4000
        },
        STEP_TIMEOUT_MS
      )
    } catch (err) {
      const message = (err as Error).message
      deps.emit('observation', 'ErrorObserved', { stage: 'agent-step', step, message })
      return { outcome: 'failed', steps, reason: `第 ${step} 步模型调用失败：${message}`, usage }
    }

    usage = addUsage(usage, res.usage)
    messages.push({ role: 'assistant', content: res.content })

    const toolUses = res.content.filter(
      (p): p is Extract<ContentPart, { type: 'tool_use' }> => p.type === 'tool_use'
    )
    const stepText = textOf(res)

    // 把这一步的「模型说了什么」原样记下来 —— 用户要能看到它在想什么，而不是只看到命令
    deps.emit('observation', 'ModelMessage', {
      step,
      text: stepText,
      stopReason: res.stopReason,
      toolCalls: toolUses.map((t) => ({ id: t.id, name: t.name, input: t.input }))
    })
    deps.emit('observation', 'ModelUsage', { step, usage: res.usage, cumulative: usage })

    if (toolUses.length === 0) {
      // 模型既没调工具也没 finish，只说了一段话 —— 这不是交付
      return {
        outcome: 'failed',
        steps,
        reason: '模型未调用 finish 就停止输出，任务状态未知（不视为完成）',
        usage
      }
    }

    // 一次返回多个工具调用时按顺序串行执行：同一 run 内不并发（US-06 AC-6）
    const results: ContentPart[] = []
    let finish: FinishInput | null = null
    // shell 命令与 MCP 调用共用一个配额：两者都是「动手」，
    // 分开计数只会让人用 MCP 绕过节奏限制
    let callCount = 0

    for (const tu of toolUses) {
      if (finish) {
        // 已经宣布结束了，后面多出来的调用一律不执行，但协议上必须回应
        results.push({
          type: 'tool_result',
          toolUseId: tu.id,
          content: '【已忽略】本轮已调用 finish，该调用未执行',
          isError: true
        })
        continue
      }

      if (tu.name === 'finish') {
        const parsed = parseFinish(tu.input)
        if (parsed) {
          finish = parsed
          results.push({ type: 'tool_result', toolUseId: tu.id, content: '【已记录】交付说明已收到' })
        } else {
          results.push({
            type: 'tool_result',
            toolUseId: tu.id,
            content: 'finish 的入参不符合要求：需要 summary（字符串）、satisfied（数组）、unmet（数组）',
            isError: true
          })
        }
        continue
      }

      if (++callCount > MAX_COMMANDS_PER_TURN) {
        results.push({
          type: 'tool_result',
          toolUseId: tu.id,
          content: `单轮最多声明 ${MAX_COMMANDS_PER_TURN} 次工具调用，本条及之后未执行。请分批。`,
          isError: true
        })
        continue
      }

      /* ---- MCP 工具：按名字前缀分派，走的是与 shell 同一条链路 ---- */
      if (isMcpToolName(tu.name)) {
        const parsed = parseMcpToolName(tu.name)
        if (!parsed) {
          results.push({
            type: 'tool_result',
            toolUseId: tu.id,
            content: `工具名格式不合法：${tu.name}（应为 mcp__<server>__<tool>）`,
            isError: true
          })
          continue
        }
        const out = await deps.requestMcpTool(
          parsed.server,
          parsed.tool,
          tu.input ?? {},
          purposeOf(tu.input)
        )
        results.push({
          type: 'tool_result',
          toolUseId: tu.id,
          content: out.ok ? clip(out.text || '（无返回内容）') : `【未执行】${out.reason ?? '未知原因'}`,
          isError: !out.ok || out.isError
        })
        continue
      }

      if (tu.name !== 'run_command') {
        results.push({
          type: 'tool_result',
          toolUseId: tu.id,
          content: `未知工具：${tu.name}。本次可用工具为：${deps.tools.map((t) => t.name).join('、')}`,
          isError: true
        })
        continue
      }

      const args = (tu.input ?? {}) as Record<string, unknown>
      const command = typeof args['command'] === 'string' ? args['command'].trim() : ''
      const purpose = typeof args['purpose'] === 'string' ? args['purpose'].trim() : ''

      if (!command) {
        results.push({
          type: 'tool_result',
          toolUseId: tu.id,
          content: 'run_command 缺少 command 参数',
          isError: true
        })
        continue
      }

      // purpose 缺失不阻断执行，但要记下来 —— 它能反映模型是否真的在按契约做事
      if (!purpose) {
        deps.emit('observation', 'ErrorObserved', {
          step,
          message: '模型未给出 purpose，命令仍照常走裁决链',
          command
        })
      }

      const out = await deps.requestCommand(command, purpose)
      results.push({
        type: 'tool_result',
        toolUseId: tu.id,
        content: resultText(out),
        isError: !out.ok || (out.result?.exitCode ?? 0) !== 0
      })
    }

    messages.push({ role: 'user', content: results })

    if (finish) {
      return {
        outcome: 'delivered',
        steps,
        summary: finish.summary,
        satisfied: finish.satisfied,
        unmet: finish.unmet,
        usage
      }
    }
  }

  return {
    outcome: 'max-steps',
    steps: MAX_STEPS,
    reason: `已达步数上限 ${MAX_STEPS}，任务未收尾`,
    usage
  }
}
