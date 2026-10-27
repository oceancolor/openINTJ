/**
 * 输入结构化 —— 契约起草的前置一步
 *
 * ── 它在整条链上的位置 ──────────────────────────────────────────
 *   用户一句话 →【输入结构化】→ 结构化后的执行输入 + 歧义分 + ≤3 个澄清问题
 *              → 契起草（含澄清问题进 needsClarification）→ IC-07 硬闸门
 *
 * ── 为什么要这一步 ─────────────────────────────────────────────
 * 「帮我把这个优化一下」这种话，直接拿去起草契约的结果只有两种：
 * 要么模型自己脑补约束写进 assumptions（用户看不见，等于替他做了决定），
 * 要么起草出一份空泛到无法判定的契约（判据全是「性能有所提升」）。
 * 结构化做的是第三件事：**先量一下这句话有多含糊，含糊就问**。
 *
 * ── 为什么不自己实现 ───────────────────────────────────────────
 * 主干 `core/input-structuring.ts` 里那套（歧义线索词、policy=adaptive、
 * 阈值 0.62、最多 3 问、超时/解析失败一律回退 pass-through）是实测调过的。
 * 尤其「失败必须回退成 pass-through 而不是拒绝执行」这一条 ——
 * 为了消歧把用户的话挡回去，是比不问更糟的体验。照搬，不改。
 */

import {
  resolveInputStructuringConfig,
  structureUserInput,
  type InputStructureResult,
  type LlmClient
} from '@openintj/core'
import { complete, textOf } from './llm'
import type { LlmConfig, UnifiedMessage } from '../shared/llm'
import type { InputStructureHint } from '../shared/contract'

/**
 * 结构化器只会发纯文本，但主干的 `ChatMessageContent` 允许图片片段 ——
 * 真带图片进来时**如实丢弃并标注**，而不是当它不存在：
 * 「这段里有你没处理的内容」必须看得见，否则排查时会以为是模型答错了。
 */
const plainOf = (content: { type: string; text?: string }[]): string =>
  content.map((c) => (c.type === 'text' ? (c.text ?? '') : '[非文本内容，已忽略]')).join('\n')

/**
 * 把我们的 `complete()` 包成主干要的 `LlmClient`。
 *
 * 主干只用到 `chat`，另外两个方法（`visionChat` / `getStatus`）是它面向
 * 完整 ReAct 循环定义的，输入结构化这条路上不会调用 —— 但不能不给，
 * 给了就必须给真话：视觉能力**当前没有**，声明支持等于埋一个将来必炸的雷。
 */
function toLlmClient(cfg: LlmConfig): LlmClient {
  return {
    chat: async (messages, opts): Promise<string> => {
      const system = messages
        .filter((m) => m.role === 'system')
        .map((m) => (typeof m.content === 'string' ? m.content : plainOf(m.content)))
        .join('\n')

      const rest: UnifiedMessage[] = []
      for (const m of messages) {
        if (m.role === 'system') continue
        const text = typeof m.content === 'string' ? m.content : plainOf(m.content)
        const role = m.role === 'assistant' ? 'assistant' : 'user'
        const last = rest[rest.length - 1]
        if (last && last.role === role) last.content.push({ type: 'text', text })
        else rest.push({ role, content: [{ type: 'text', text }] })
      }
      if (rest.length === 0) throw new Error('结构化请求里没有非 system 消息')

      const res = await complete(
        cfg,
        {
          ...(system ? { system } : {}),
          messages: rest,
          maxTokens: opts?.maxTokens ?? 512,
          temperature: opts?.temperature ?? 0
        },
        20_000
      )
      return textOf(res)
    },
    visionChat: async () => {
      throw new Error('输入结构化不需要视觉能力，且当前未接入视觉模型')
    },
    getStatus: () => ({
      provider: cfg.protocol,
      model: cfg.model,
      baseUrl: cfg.baseUrl,
      available: true,
      mode: 'live',
      status: 'connected',
      visionSupported: false
    })
  }
}

export async function structureInput(
  cfg: LlmConfig,
  input: {
    text: string
    /** 之前轮次的对话，用于消歧（主干只取最近 6 条） */
    history?: UnifiedMessage[]
  }
): Promise<InputStructureResult> {
  const conf = resolveInputStructuringConfig()

  const history = (input.history ?? []).flatMap((m) => {
    const text = m.content
      .filter((c): c is { type: 'text'; text: string } => c.type === 'text')
      .map((c) => c.text)
      .join('\n')
    if (!text) return []
    return [{ role: m.role as 'user' | 'assistant', content: text }]
  })

  return await structureUserInput({
    input: input.text,
    llm: toLlmClient(cfg),
    ...(history.length > 0 ? { history } : {}),
    policy: conf.policy,
    ambiguityThreshold: conf.ambiguityThreshold,
    maxTokens: conf.maxTokens,
    timeoutMs: conf.timeoutMs
  })
}

/**
 * 结构化结果 → 契约里要展示的那几样东西。
 *
 * 只带这几样：歧义分、澄清问题、以及「这一步到底做了什么」（mode）。
 * 结构化的其余字段（context / relations / deliverables …）不进契约 ——
 * 契约是给用户**逐条否掉**的，字段越多越没人看，也就越没人否。
 *
 * 形状定义在 shared/contract.ts：渲染层也要认得它，不能只存在于 agent-host 里。
 */
export function hintOf(r: InputStructureResult): InputStructureHint {
  return {
    ambiguityScore: r.ambiguityScore,
    questions: r.questions,
    mode: r.mode,
    executionInput: r.executionInput,
    tokensSpent: r.tokensSpent,
    durationMs: r.durationMs
  }
}
