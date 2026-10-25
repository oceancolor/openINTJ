/**
 * OpenAI Chat Completions 协议客户端（原生实现）
 *
 * 端点：POST {baseUrl}/chat/completions
 * 头：Authorization: Bearer <key>
 *
 * 实测要点（tokenhub 网关，hy4-preview）：
 *   - 思考内容在 `message.reasoning_content`，**不是** content 的一部分；
 *   - 工具调用在 `message.tool_calls[]`，参数是 **JSON 字符串**，需要自己 parse；
 *   - usage 里 `prompt_tokens_details.cached_tokens` 是缓存命中量，
 *     `completion_tokens_details.reasoning_tokens` 是思考 token；
 *   - **关思考用 `thinking:{type:'disabled'}`**，这个通道认它：
 *     同一句「你好」，带思考 8.8s / 231 tok（其中 221 是思考），关掉后 1.24s / 7 tok。
 *     注意这与 Anthropic 通道（/v1/messages）上的结论不同 —— 那边 hy4-preview 关不掉。
 *     同一个参数在两套协议上的行为可能不一样，所以关思考必须两套各自实现、各自实测。
 */

import type {
  ContentPart,
  LlmConfig,
  StopReason,
  TokenUsage,
  UnifiedRequest,
  UnifiedResponse
} from '../../shared/llm'

interface OpenAiToolCall {
  id: string
  type?: string
  function?: { name?: string; arguments?: string }
}

interface OpenAiMessage {
  role: string
  content?: string | null
  reasoning_content?: string | null
  tool_calls?: OpenAiToolCall[]
}

interface OpenAiResponse {
  id: string
  model: string
  choices?: Array<{ message?: OpenAiMessage; finish_reason?: string | null }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    prompt_tokens_details?: { cached_tokens?: number }
    completion_tokens_details?: { reasoning_tokens?: number }
  }
}

/** 统一片段 → OpenAI messages（工具结果要拆成 role:'tool' 的独立消息） */
function toOpenAiMessages(
  system: string | undefined,
  messages: UnifiedRequest['messages']
): OpenAiMessage[] {
  const out: OpenAiMessage[] = []
  if (system) out.push({ role: 'system', content: system })

  for (const m of messages) {
    if (m.role === 'user') {
      // 用户消息里可能混有 tool_result，OpenAI 要求它们单独成 role:'tool'
      const texts: string[] = []
      for (const p of m.content) {
        if (p.type === 'text') texts.push(p.text)
        else if (p.type === 'tool_result') {
          out.push({
            role: 'tool',
            content: p.content,
            // OpenAI 规范里 tool 消息用 tool_call_id 关联
            ...({ tool_call_id: p.toolUseId } as Record<string, unknown>)
          })
        }
        // thinking 不回放（与 Anthropic 侧同理：省 token，且不是对话内容）
      }
      if (texts.length > 0) out.push({ role: 'user', content: texts.join('\n') })
      continue
    }

    // assistant
    const texts: string[] = []
    const calls: OpenAiToolCall[] = []
    for (const p of m.content) {
      if (p.type === 'text') texts.push(p.text)
      else if (p.type === 'tool_use') {
        calls.push({
          id: p.id,
          type: 'function',
          function: { name: p.name, arguments: JSON.stringify(p.input ?? {}) }
        })
      }
    }
    const msg: OpenAiMessage = { role: 'assistant', content: texts.join('') || null }
    if (calls.length > 0) msg.tool_calls = calls
    out.push(msg)
  }
  return out
}

function mapFinishReason(r: string | null | undefined): StopReason {
  switch (r) {
    case 'stop':
      return 'end_turn'
    case 'tool_calls':
      return 'tool_use'
    case 'length':
      return 'max_tokens'
    default:
      return 'unknown'
  }
}

/** 参数是 JSON 字符串，解析失败不能让整轮崩 —— 保留原文交给上层判断 */
function parseArgs(raw: string | undefined): unknown {
  if (!raw) return {}
  try {
    return JSON.parse(raw)
  } catch {
    return { __unparsable_arguments: raw }
  }
}

export async function callOpenAi(
  cfg: LlmConfig,
  req: UnifiedRequest,
  timeoutMs: number
): Promise<UnifiedResponse> {
  const buildBody = (withThinkingOff: boolean): Record<string, unknown> => {
    const body: Record<string, unknown> = {
      model: cfg.model,
      max_tokens: req.maxTokens,
      messages: toOpenAiMessages(req.system, req.messages)
    }
    if (typeof req.temperature === 'number') body['temperature'] = req.temperature
    // 关思考。这个参数不是 OpenAI 官方规范，但在兼容网关上有效（见文件头实测）。
    // 网关不认时会被 400 挡下，下面去掉重试一次。
    if (withThinkingOff) body['thinking'] = { type: 'disabled' }
    if (req.tools?.length) {
      body['tools'] = req.tools.map((t) => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: t.inputSchema }
      }))
    }
    return body
  }

  const post = async (body: Record<string, unknown>): Promise<{ status: number; text: string }> => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const res = await fetch(`${cfg.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${cfg.apiKey}`
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

  // 有的模型 / 网关不认 thinking 参数（社区里 OpenAI 官方端点就会拒）——
  // 别因此判整个请求失败，去掉该参数重试一次，并说清楚为什么
  if (out.status === 400 && cfg.disableThinking && /thinking/i.test(out.text)) {
    console.warn('[llm] 该模型/网关不接受 thinking 参数，去掉后重试一次')
    out = await post(buildBody(false))
  }

  if (out.status < 200 || out.status >= 300) {
    throw new Error(`OpenAI 调用失败 HTTP ${out.status}：${out.text.slice(0, 400)}`)
  }

  const text = out.text
  let data: OpenAiResponse
  try {
    data = JSON.parse(text) as OpenAiResponse
  } catch {
    throw new Error(`OpenAI 响应不是合法 JSON：${text.slice(0, 200)}`)
  }

  const choice = data.choices?.[0]
  const msg = choice?.message ?? { role: 'assistant' }

  const parts: ContentPart[] = []
  // reasoning 放到最前，和 Anthropic 的块顺序保持一致
  if (typeof msg.reasoning_content === 'string' && msg.reasoning_content.trim()) {
    parts.push({ type: 'thinking', text: msg.reasoning_content })
  }
  if (typeof msg.content === 'string' && msg.content.trim()) {
    parts.push({ type: 'text', text: msg.content })
  }
  for (const call of msg.tool_calls ?? []) {
    parts.push({
      type: 'tool_use',
      id: call.id,
      name: call.function?.name ?? '',
      input: parseArgs(call.function?.arguments)
    })
  }

  // 口径归一 —— 这里有个很容易算错成本的地方：
  //   OpenAI 的 prompt_tokens 是**总输入**，cached_tokens 是其中命中缓存的部分（子集）；
  //   Anthropic 的 input_tokens 是**未命中缓存**的部分，cache_read 另算（并列，不是子集）。
  // 若两边都直接塞进 inputTokens，OpenAI 侧会把缓存命中重复计一次全额费用。
  // 统一到 Anthropic 口径：inputTokens = 未命中缓存的输入，cacheReadTokens = 命中部分。
  const cachedTokens = data.usage?.prompt_tokens_details?.cached_tokens ?? 0
  const totalPrompt = data.usage?.prompt_tokens ?? 0
  const usage: TokenUsage = {
    inputTokens: Math.max(0, totalPrompt - cachedTokens),
    outputTokens: data.usage?.completion_tokens ?? 0,
    cacheReadTokens: cachedTokens,
    cacheCreationTokens: 0,
    reasoningTokens: data.usage?.completion_tokens_details?.reasoning_tokens ?? 0
  }

  return {
    id: data.id,
    model: data.model,
    stopReason: mapFinishReason(choice?.finish_reason),
    content: parts,
    usage,
    raw: data
  }
}
