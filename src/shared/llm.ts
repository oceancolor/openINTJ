/**
 * LLM 接入契约 —— 多协议归一化
 *
 * 依据《系统设计》§3.2.M11 与用户裁决（Anthropic + OpenAI 双协议，后追加 Responses）：
 *   各协议**各自原生实现**，再归一到本文件的统一结构。
 *   不做「用一个协议去模拟另一个」—— 模拟会丢掉 prompt caching、thinking block、
 *   tool_result 的结构语义，这些恰恰是成本与正确性的关键。
 *
 * 目前三套：
 *   anthropic         /v1/messages          （thinking block / tool_use / cache_read 原生）
 *   openai            /v1/chat/completions  （reasoning_content / tool_calls / choices[].finish_reason）
 *   openai-responses  /v1/responses         （output[] 扁平 item / function_call / **无 finish_reason**）
 *
 * 归一化要点（实测腾讯 tokenhub 网关得出，见 delivery/Phase7-首步LLM接入实测.md）：
 *   1. Anthropic 的 thinking 块只有 `{type,thinking}`，**没有 signature**；
 *   2. OpenAI Chat 侧把思考放在 `message.reasoning_content`，Responses 侧放在
 *      `output[].reasoning.summary[].text`（且只有摘要，不是全文），Anthropic 侧放在 content 数组里；
 *   3. 两边 usage 字段名不同，但都能给出 cached tokens（缓存命中量）；
 *   4. 历史里塞 thinking 会让前缀缓存失效（cache_read 归零）—— 所以**回放时丢弃 thinking**。
 *   5. 同一个「关思考」语义，三套通道的写法与效果都不同（见 LlmConfig.disableThinking 注释）。
 */

/**
 * 协议类型
 *
 * anthropic         → POST {baseUrl}/messages
 * openai            → POST {baseUrl}/chat/completions
 * openai-responses  → POST {baseUrl}/responses
 *
 * 后两者同属 OpenAI 家族但不是同一套线格式：Chat Completions 是 `choices[].message`，
 * Responses 是 `output[]`（且**没有 finish_reason**）。所以必须各自实现，不能共用。
 */
export type LlmProtocol = 'anthropic' | 'openai' | 'openai-responses'

/** 允许从环境变量写的协议名（含几个常见别名，避免手滑写错时被静默降级） */
const PROTOCOL_ALIASES: Record<string, LlmProtocol> = {
  anthropic: 'anthropic',
  claude: 'anthropic',
  messages: 'anthropic',
  openai: 'openai',
  'chat-completions': 'openai',
  chat: 'openai',
  'openai-responses': 'openai-responses',
  responses: 'openai-responses'
}

/**
 * 解析协议名 —— 全工程**唯一**的判定点。
 *
 * 环境变量、保险箱凭据、设置页三条路径都必须走这里。
 * 之前 env 和 vault 各写了一份 `raw === 'openai' ? 'openai' : 'anthropic'` 的三元，
 * 加第三协议时就会漏掉其中一处，症状是「界面上选了 Responses，实际打 /v1/messages」。
 * 认不出的返回 null（**不猜**），由调用方决定是报「缺少配置」还是「字段不合法」。
 */
export function parseProtocol(raw: string | undefined): LlmProtocol | null {
  const key = (raw ?? '').trim().toLowerCase()
  return PROTOCOL_ALIASES[key] ?? null
}

/** 供界面提示用：可选协议名清单（不含别名） */
export function protocolOptions(): LlmProtocol[] {
  return ['anthropic', 'openai', 'openai-responses']
}

export interface LlmConfig {
  protocol: LlmProtocol
  /** 例如 https://tokenhub.tencentmaas.com/v1（不带结尾斜杠） */
  baseUrl: string
  apiKey: string
  model: string
  /**
   * 是否请求关闭思考。
   *
   * 实测（tokenhub 网关）：同一个「关思考」语义，三套通道的实现方式各不相同，
   * 而且**行为也不同** —— 这正是必须三套各自实现、各自实测的原因：
   *   /v1/messages        hy4-preview **关不掉**（传 thinking.disabled 无效果）
   *   /v1/chat/completions 用 `thinking:{type:'disabled'}`，8.81s/231tok → 1.24s/7tok（有效）
   *   /v1/responses        用官方规范 `reasoning:{effort:'none'}`，1.30s/7tok（有效）
   * deepseek-flash 关掉后 2.8s/313token、不关 5.2s/726token；glm-5.3-flash **400 拒绝**该参数。
   * 所以这是个「能关就关、关不掉就重试一次」的软开关。
   */
  disableThinking: boolean
}

/** 内容片段 —— 两套协议归一后的最小公共语义 */
export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: 'tool_result'; toolUseId: string; content: string; isError?: boolean }

export interface UnifiedMessage {
  role: 'user' | 'assistant'
  content: ContentPart[]
}

export interface ToolSpec {
  name: string
  description: string
  /** JSON Schema */
  inputSchema: Record<string, unknown>
}

export interface UnifiedRequest {
  system?: string
  messages: UnifiedMessage[]
  tools?: ToolSpec[]
  maxTokens: number
  temperature?: number
}

export type StopReason = 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence' | 'unknown'

/**
 * Token 用量。
 *
 * **口径约定**：`inputTokens` 一律是「**未命中缓存的输入**」，`cacheReadTokens` 是
 * 「命中缓存的那部分」，两者**并列相加**才等于总输入。
 *
 * 为什么必须定这个约定：两套协议原生字段的语义是反的 ——
 *   Anthropic `input_tokens` 不含缓存命中，`cache_read_input_tokens` 另算；
 *   OpenAI `prompt_tokens` 含缓存命中，`cached_tokens` 是它的子集。
 * 若不做减法直接映射，OpenAI 侧会把缓存命中按全额单价重复计一次费。
 *
 * 成本 = inputTokens×P + cacheReadTokens×0.1P（+ cacheCreationTokens×1.25P，仅 Anthropic）
 */
export interface TokenUsage {
  /** 未命中缓存的输入 token */
  inputTokens: number
  outputTokens: number
  /** 命中前缀缓存的输入 token 数 —— 成本核算的关键指标 */
  cacheReadTokens: number
  /** 缓存写入（Anthropic 专有；OpenAI 侧恒为 0） */
  cacheCreationTokens: number
  /** 推理/思考 token（OpenAI 侧单列，Anthropic 侧已含在 output 里） */
  reasoningTokens: number
}

export interface UnifiedResponse {
  id: string
  model: string
  stopReason: StopReason
  content: ContentPart[]
  usage: TokenUsage
  /** 原始响应，排障用；不进事件流 */
  raw: unknown
}

/* ------------------------------------------------------------------ *
 * 配置读取
 * ------------------------------------------------------------------ */

export const LLM_ENV = {
  protocol: 'OPENINTJ_LLM_PROTOCOL',
  baseUrl: 'OPENINTJ_LLM_BASE_URL',
  apiKey: 'OPENINTJ_LLM_API_KEY',
  model: 'OPENINTJ_LLM_MODEL',
  /** 1/true 表示请求关闭思考；默认关闭思考（成本与延迟都差一个数量级） */
  disableThinking: 'OPENINTJ_LLM_DISABLE_THINKING'
} as const

export type LlmConfigResult =
  | { configured: true; config: LlmConfig }
  | { configured: false; missing: string[] }

/**
 * 从环境变量读配置。
 *
 * 为什么先走环境变量而不是写进配置文件：凭据只应存在 OS 保险箱（safeStorage），
 * 「设置与凭据中心」（IC-06）尚未实现前，环境变量是唯一不会落明文盘的路径。
 */
export function readLlmConfig(env: NodeJS.ProcessEnv): LlmConfigResult {
  const baseUrl = env[LLM_ENV.baseUrl]?.trim()
  const apiKey = env[LLM_ENV.apiKey]?.trim()
  const model = env[LLM_ENV.model]?.trim()

  const missing: string[] = []
  if (!baseUrl) missing.push(LLM_ENV.baseUrl)
  if (!apiKey) missing.push(LLM_ENV.apiKey)
  if (!model) missing.push(LLM_ENV.model)
  if (missing.length > 0) return { configured: false, missing }

  const protocolRaw = env[LLM_ENV.protocol] ?? 'anthropic'
  // 认不出的协议名**不能静默降级成 anthropic** —— 那会让人以为在测 Responses，
  // 实际打的是 /v1/messages，得出的结论全是错的（这类错最难查：不报错，只是"不对"）。
  // 也不能抛异常把主进程启动带崩。所以：不给猜测，按「配置不完整」处理，
  // 并把真实原因写进 missing，让设置页原样显示。
  const protocol = parseProtocol(protocolRaw)
  if (!protocol) {
    return {
      configured: false,
      missing: [
        `${LLM_ENV.protocol}=${protocolRaw.trim()}（未知协议，可选 ${protocolOptions().join(' / ')}）`
      ]
    }
  }

  const disableRaw = (env[LLM_ENV.disableThinking] ?? '1').trim().toLowerCase()
  const disableThinking = disableRaw !== '0' && disableRaw !== 'false' && disableRaw !== 'off'

  return {
    configured: true,
    config: {
      protocol,
      baseUrl: baseUrl!.replace(/\/+$/, ''),
      apiKey: apiKey!,
      model: model!,
      disableThinking
    }
  }
}

/** 对外只暴露「配好了没、用哪个模型」，绝不吐密钥 */
export interface LlmStatus {
  configured: boolean
  protocol: LlmProtocol | null
  model: string | null
  baseUrlHost: string | null
  missing: string[]
}

export function describeLlm(result: LlmConfigResult): LlmStatus {
  if (!result.configured) {
    return { configured: false, protocol: null, model: null, baseUrlHost: null, missing: result.missing }
  }
  let host = result.config.baseUrl
  try {
    host = new URL(result.config.baseUrl).host
  } catch {
    /* 配置写错时原样显示，便于排错 */
  }
  return {
    configured: true,
    protocol: result.config.protocol,
    model: result.config.model,
    baseUrlHost: host,
    missing: []
  }
}
