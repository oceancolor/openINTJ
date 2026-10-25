/**
 * LLM 调用入口 —— 按协议分派
 *
 * 分工：
 *   anthropic.ts / openai.ts / responses.ts 各自说自己协议的话（原生实现）
 *   本文件只负责分派、超时兜底、错误归一
 *
 * 三套协议**没有**公共超集可复用：system 是消息还是顶层字段、工具调用在 message 里
 * 还是顶层 item、有没有 finish_reason、关思考怎么写，全都不一样。硬凑一个
 * 「通用 OpenAI 兼容层」只会让其中一套变成二等公民。
 */

import type { LlmConfig, UnifiedRequest, UnifiedResponse } from '../../shared/llm'
import { callAnthropic } from './anthropic'
import { callOpenAi } from './openai'
import { callResponses } from './responses'

/**
 * 单次调用超时。
 *
 * 实测：hy4-preview 生成 448 个输出 token（其中约 400 是思考）耗时 16 秒。
 * N-04b（超时与重试阈值）尚未拍板，这里先给足 120 秒 ——
 * 卡住的是「等模型」而不是「卡死」，超时后必须如实报错，不能重试到静默。
 */
export const DEFAULT_LLM_TIMEOUT_MS = 120_000

export async function complete(
  cfg: LlmConfig,
  req: UnifiedRequest,
  timeoutMs = DEFAULT_LLM_TIMEOUT_MS
): Promise<UnifiedResponse> {
  if (!req.messages.length) throw new Error('请求没有任何消息')
  switch (cfg.protocol) {
    case 'anthropic':
      return await callAnthropic(cfg, req, timeoutMs)
    case 'openai':
      return await callOpenAi(cfg, req, timeoutMs)
    case 'openai-responses':
      return await callResponses(cfg, req, timeoutMs)
    default:
      throw new Error(`不支持的协议：${String(cfg.protocol)}`)
  }
}

/** 从响应里取纯文本（跳过 thinking），供契约解析等场景用 */
export function textOf(res: UnifiedResponse): string {
  return res.content
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map((p) => p.text)
    .join('\n')
    .trim()
}

export function thinkingOf(res: UnifiedResponse): string {
  return res.content
    .filter((p): p is { type: 'thinking'; text: string } => p.type === 'thinking')
    .map((p) => p.text)
    .join('\n')
    .trim()
}
