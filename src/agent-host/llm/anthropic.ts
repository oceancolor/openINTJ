/**
 * Anthropic Messages 协议客户端（原生实现）
 *
 * 端点：POST {baseUrl}/messages
 * 头：x-api-key + anthropic-version: 2023-06-01
 *
 * 实测要点（tokenhub 网关，hy4-preview）：
 *   - 回放的 thinking 块**没有 signature**，服务端也不校验；但带上会让前缀缓存失效，
 *     所以 toAnthropicContent 主动丢弃 thinking。
 *   - tools 生效，stop_reason 为 'tool_use'，tool_use.id 形如 call_xxx。
 */

import type { ContentPart, LlmConfig, UnifiedRequest, UnifiedResponse, StopReason } from '../../shared/llm'
import type { TokenUsage } from '../../shared/llm'

interface AnthropicBlock {
  type: string
  text?: string
  thinking?: string
  id?: string
  name?: string
  input?: unknown
}

interface AnthropicResponse {
  id: string
  model: string
  stop_reason: string | null
  content: AnthropicBlock[]
  usage?: {
    input_tokens?: number
    output_tokens?: number
    cache_read_input_tokens?: number
    cache_creation_input_tokens?: number
  }
}

/** 统一片段 → Anthropic content 块。thinking 故意不回放（见文件头说明） */
function toAnthropicContent(parts: ContentPart[]): unknown[] {
  const out: unknown[] = []
  for (const p of parts) {
    switch (p.type) {
      case 'text':
        if (p.text.trim()) out.push({ type: 'text', text: p.text })
        break
      case 'tool_use':
        out.push({ type: 'tool_use', id: p.id, name: p.name, input: p.input ?? {} })
        break
      case 'tool_result':
        out.push({
          type: 'tool_result',
          tool_use_id: p.toolUseId,
          content: p.content,
          ...(p.isError ? { is_error: true } : {})
        })
        break
      case 'thinking':
        // 丢弃：无 signature 可回传，且会破坏前缀缓存
        break
    }
  }
  // Anthropic 不接受空的 content 数组
  return out.length > 0 ? out : [{ type: 'text', text: '(空)' }]
}

function mapStopReason(r: string | null): StopReason {
  switch (r) {
    case 'end_turn':
    case 'tool_use':
    case 'max_tokens':
    case 'stop_sequence':
      return r
    default:
      return 'unknown'
  }
}

function normalizeBlocks(blocks: AnthropicBlock[]): ContentPart[] {
  const parts: ContentPart[] = []
  for (const b of blocks) {
    if (b.type === 'text' && typeof b.text === 'string') {
      parts.push({ type: 'text', text: b.text })
    } else if (b.type === 'thinking' && typeof b.thinking === 'string') {
      parts.push({ type: 'thinking', text: b.thinking })
    } else if (b.type === 'tool_use') {
      parts.push({ type: 'tool_use', id: String(b.id ?? ''), name: String(b.name ?? ''), input: b.input })
    }
  }
  return parts
}

export async function callAnthropic(
  cfg: LlmConfig,
  req: UnifiedRequest,
  timeoutMs: number
): Promise<UnifiedResponse> {
  const buildBody = (withThinkingOff: boolean): Record<string, unknown> => {
    const body: Record<string, unknown> = {
      model: cfg.model,
      max_tokens: req.maxTokens,
      messages: req.messages.map((m) => ({ role: m.role, content: toAnthropicContent(m.content) }))
    }
    if (req.system) body['system'] = req.system
    if (typeof req.temperature === 'number') body['temperature'] = req.temperature
    if (withThinkingOff) body['thinking'] = { type: 'disabled' }
    if (req.tools?.length) {
      body['tools'] = req.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.inputSchema
      }))
    }
    return body
  }

  const post = async (body: Record<string, unknown>): Promise<{ status: number; text: string }> => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const res = await fetch(`${cfg.baseUrl}/messages`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': cfg.apiKey,
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify(body),
        signal: controller.signal
      })
      return { status: res.status, text: await res.text() }
    } finally {
      clearTimeout(timer)
    }
  }

  let out = await post(buildBody(cfg.disableThinking))

  // 有的模型不认 thinking 参数（实测 glm-5.3-flash 直接 400）——
  // 不要因此判整个请求失败，去掉该参数重试一次，并在日志里说清楚为什么
  if (out.status === 400 && cfg.disableThinking && /thinking/i.test(out.text)) {
    console.warn('[llm] 该模型不接受 thinking 参数，去掉后重试一次')
    out = await post(buildBody(false))
  }

  if (out.status < 200 || out.status >= 300) {
    throw new Error(`Anthropic 调用失败 HTTP ${out.status}：${out.text.slice(0, 400)}`)
  }

  const text = out.text
  let data: AnthropicResponse
  try {
    data = JSON.parse(text) as AnthropicResponse
  } catch {
    throw new Error(`Anthropic 响应不是合法 JSON：${text.slice(0, 200)}`)
  }

  const usage: TokenUsage = {
    inputTokens: data.usage?.input_tokens ?? 0,
    outputTokens: data.usage?.output_tokens ?? 0,
    cacheReadTokens: data.usage?.cache_read_input_tokens ?? 0,
    cacheCreationTokens: data.usage?.cache_creation_input_tokens ?? 0,
    reasoningTokens: 0
  }

  return {
    id: data.id,
    model: data.model,
    stopReason: mapStopReason(data.stop_reason),
    content: normalizeBlocks(data.content ?? []),
    usage,
    raw: data
  }
}
