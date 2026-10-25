/**
 * OpenAI Responses API 客户端（原生实现）
 *
 * 端点：POST {baseUrl}/responses
 * 头：Authorization: Bearer <key>
 *
 * 为什么单开一个文件而不是复用 openai.ts —— 线格式差得比名字差得远：
 *
 *   Chat Completions                    Responses
 *   ------------------------------      ------------------------------------------
 *   messages[]（system 是一条消息）      instructions（顶层字段）+ input
 *   choices[0].message.content          output[]（多个 item，按顺序）
 *   message.tool_calls[]                output[] 里 type:'function_call' 的 item
 *   tool 角色消息带 tool_call_id        {type:'function_call_output', call_id, output}
 *   tools[].function{name,desc,params}  tools[] 是**扁平**的 {type,name,description,parameters}
 *   finish_reason                       **没有这个字段**（要自己从 output 推导）
 *   thinking:{type:'disabled'} 关思考   reasoning:{effort:'none'}（这才是官方规范）
 *
 * 实测（tokenhub 网关，hy4-preview）：
 *   - `reasoning:{effort:'none'}` 有效：同一句「你好」1.30s / 7 token，思考被真正关掉；
 *   - 思考内容被降级成 `output[].reasoning.summary[].text` —— **只有摘要，不是全文**，
 *     所以这里归一到 thinking 片段时语义上是有损的（够用，但不能当完整思维链）；
 *   - usage 的 `input_tokens` **含**缓存命中（与 Chat Completions 同口径，不是 Anthropic 口径）。
 */

import type {
  ContentPart,
  LlmConfig,
  StopReason,
  TokenUsage,
  UnifiedRequest,
  UnifiedResponse
} from '../../shared/llm'

/** Responses 的请求 item —— 与 Chat Completions 的 message 不是一个东西 */
type InputItem =
  | { role: 'user' | 'assistant'; content: string }
  | { type: 'function_call'; call_id: string; name: string; arguments: string }
  | { type: 'function_call_output'; call_id: string; output: string }

interface OutputItem {
  type?: string
  /** message item */
  role?: string
  content?: Array<{ type?: string; text?: string }>
  /** reasoning item */
  summary?: Array<{ type?: string; text?: string }>
  /** function_call item */
  call_id?: string
  name?: string
  arguments?: string
}

interface ResponsesPayload {
  id?: string
  model?: string
  status?: string
  output?: OutputItem[]
  usage?: {
    input_tokens?: number
    output_tokens?: number
    input_tokens_details?: { cached_tokens?: number }
    output_tokens_details?: { reasoning_tokens?: number }
  }
  /** 出错时网关可能给这个 */
  error?: { message?: string; type?: string }
}

/**
 * 统一片段 → Responses 的 input 序列。
 *
 * 与 Chat Completions 最大的不同：assistant 的工具调用**不是** message 的一部分，
 * 而是一个独立的 `function_call` item（和 assistant 的文本消息并列、同层级）。
 * 工具结果也不是 `role:'tool'`，而是 `function_call_output`。
 */
function toInputItems(
  system: string | undefined,
  messages: UnifiedRequest['messages']
): { instructions?: string; input: InputItem[] } {
  const input: InputItem[] = []

  for (const m of messages) {
    if (m.role === 'user') {
      const texts: string[] = []
      for (const p of m.content) {
        if (p.type === 'text') texts.push(p.text)
        else if (p.type === 'tool_result') {
          // 工具结果必须单独成 item，且 call_id 要和当初的 function_call 对上
          input.push({ type: 'function_call_output', call_id: p.toolUseId, output: p.content })
        }
        // thinking 不回放：省 token，且它本来就不是对话内容
      }
      if (texts.length > 0) input.push({ role: 'user', content: texts.join('\n') })
      continue
    }

    // assistant：文本与工具调用拆成两个层级的 item
    const texts: string[] = []
    for (const p of m.content) {
      if (p.type === 'text') texts.push(p.text)
    }
    if (texts.length > 0) input.push({ role: 'assistant', content: texts.join('') })
    for (const p of m.content) {
      if (p.type === 'tool_use') {
        input.push({
          type: 'function_call',
          call_id: p.id,
          name: p.name,
          arguments: JSON.stringify(p.input ?? {})
        })
      }
    }
  }

  // system 在 Responses 里是顶层 instructions，不是 input 里的一条消息
  return { instructions: system, input }
}

/**
 * 推 StopReason。
 *
 * Responses **没有 finish_reason**，所以只能从 output 的内容反推：
 *   有 function_call      → tool_use（上层据此继续跑工具）
 *   status === 'incomplete' → max_tokens / 内容被截断
 *   否则                  → end_turn
 * 这个推导是这套协议最容易写错的地方：漏了 function_call 判断，
 * 模型一调工具就被当成「说完了」，循环直接退出。
 */
function deriveStopReason(data: ResponsesPayload, hasFunctionCall: boolean): StopReason {
  if (hasFunctionCall) return 'tool_use'
  if (data.status === 'incomplete') {
    const reason = (data as { incomplete_details?: { reason?: string } }).incomplete_details?.reason
    if (reason === 'max_output_tokens') return 'max_tokens'
    return 'unknown'
  }
  return 'end_turn'
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

export async function callResponses(
  cfg: LlmConfig,
  req: UnifiedRequest,
  timeoutMs: number
): Promise<UnifiedResponse> {
  const { instructions, input } = toInputItems(req.system, req.messages)

  const buildBody = (withThinkingOff: boolean): Record<string, unknown> => {
    const body: Record<string, unknown> = {
      model: cfg.model,
      // 注意这里是 max_output_tokens，不是 max_tokens
      max_output_tokens: req.maxTokens,
      input
    }
    if (instructions) body['instructions'] = instructions
    if (typeof req.temperature === 'number') body['temperature'] = req.temperature
    // 关思考用官方规范。与 Chat Completions 的 thinking:{type:'disabled'} 完全是两套写法 ——
    // 这也是「不能复用 openai.ts」的具体原因之一。
    if (withThinkingOff) body['reasoning'] = { effort: 'none' }
    if (req.tools?.length) {
      // 扁平结构：没有 function:{} 包裹
      body['tools'] = req.tools.map((t) => ({
        type: 'function',
        name: t.name,
        description: t.description,
        parameters: t.inputSchema
      }))
    }
    return body
  }

  const post = async (body: Record<string, unknown>): Promise<{ status: number; text: string }> => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      const res = await fetch(`${cfg.baseUrl}/responses`, {
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

  // 有的模型 / 网关不认 reasoning 参数 —— 去掉重试一次，而不是判整轮失败。
  // （与 openai.ts 同样的软开关策略，只是触发词不同）
  if (out.status === 400 && cfg.disableThinking && /reason/i.test(out.text)) {
    console.warn('[llm] 该模型/网关不接受 reasoning.effort 参数，去掉后重试一次')
    out = await post(buildBody(false))
  }

  if (out.status < 200 || out.status >= 300) {
    throw new Error(`Responses 调用失败 HTTP ${out.status}：${out.text.slice(0, 400)}`)
  }

  const text = out.text
  let data: ResponsesPayload
  try {
    data = JSON.parse(text) as ResponsesPayload
  } catch {
    throw new Error(`Responses 响应不是合法 JSON：${text.slice(0, 200)}`)
  }

  if (data.error?.message) {
    throw new Error(`Responses 返回错误：${data.error.message}`)
  }

  const parts: ContentPart[] = []
  let hasFunctionCall = false

  for (const item of data.output ?? []) {
    if (item.type === 'reasoning') {
      // 只有 summary，没有完整思维链 —— 归一成 thinking 片段但知道它是有损的
      const summary = (item.summary ?? [])
        .map((s) => s.text ?? '')
        .filter((s) => s.trim())
        .join('\n')
      if (summary) parts.push({ type: 'thinking', text: summary })
      continue
    }

    if (item.type === 'message') {
      // content[] 里是 output_text 类型的片段
      const txt = (item.content ?? [])
        .filter((c) => c.type === 'output_text' || typeof c.text === 'string')
        .map((c) => c.text ?? '')
        .join('')
      if (txt.trim()) parts.push({ type: 'text', text: txt })
      continue
    }

    if (item.type === 'function_call') {
      hasFunctionCall = true
      parts.push({
        type: 'tool_use',
        // call_id 就是后续 function_call_output 要回填的那个 id
        id: item.call_id ?? '',
        name: item.name ?? '',
        input: parseArgs(item.arguments)
      })
      continue
    }

    // 其他 item 类型（web_search_call 等）暂不处理，但不能让它们把整轮搞崩
  }

  // usage 口径归一：input_tokens **含**缓存命中，必须减掉才是「未命中输入」，
  // 否则缓存命中的部分会按全额单价再计一次费（与 openai.ts 同一处坑）。
  const cachedTokens = data.usage?.input_tokens_details?.cached_tokens ?? 0
  const totalInput = data.usage?.input_tokens ?? 0
  const usage: TokenUsage = {
    inputTokens: Math.max(0, totalInput - cachedTokens),
    outputTokens: data.usage?.output_tokens ?? 0,
    cacheReadTokens: cachedTokens,
    cacheCreationTokens: 0,
    reasoningTokens: data.usage?.output_tokens_details?.reasoning_tokens ?? 0
  }

  return {
    id: data.id ?? '',
    model: data.model ?? cfg.model,
    stopReason: deriveStopReason(data, hasFunctionCall),
    content: parts,
    usage,
    raw: data
  }
}
