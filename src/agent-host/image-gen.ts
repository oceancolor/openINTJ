/**
 * 生图（P-03 侧）
 *
 * ── 为什么走搬过来的 @openintj/llm-hunyuan 而不是自己再写一份 ──────
 * 生图这条链路真正难的不是发一个 POST，而是**错误分类**：
 *   402 = 未开通后付费且无免费额度（要给用户开通链接，不是让他重试）
 *   429 / 5xx = 可重试；401 / 403 = 凭据问题，重试也不会好
 *   响应里既有 `data[0].b64_json` 也有 `data[0].url` 两种形态，少认一种就会「返回了但没图」
 * 这些是主干上踩过的坑，重写一遍的收益是零，风险是漏掉其中一条。
 * 所以这里只做两件事：把主干的实现接上，把它的错误翻译成我们自己的说法。
 *
 * ── 产物不在这里落盘 ──────────────────────────────────────────────
 * 本文件只负责「拿到字节」。落盘一律交回 index.ts 走 tool-host 的裁决写入 ——
 * 生图产物也是**写工作区**，没有理由绕过那道闸门。
 */

import { generateHunyuanImage } from '@openintj/llm-hunyuan'
import type { ImageConfig, ImageGenerateResult } from '../shared/llm'

/** 单次生图最久等多久 —— 比主干默认再宽一点，网关排队时 120s 会显得很紧张 */
export const IMAGE_TIMEOUT_MS = 150_000

export async function generateImage(
  cfg: ImageConfig,
  input: { prompt: string; size?: string; signal?: AbortSignal }
): Promise<ImageGenerateResult> {
  const started = Date.now()
  const size = input.size?.trim() || cfg.size

  if (cfg.provider !== 'hunyuan') {
    throw new Error(`未支持的生图服务：${cfg.provider}（目前仅 hunyuan）`)
  }

  try {
    const res = await generateHunyuanImage({
      apiKey: cfg.apiKey,
      baseUrl: cfg.baseUrl,
      prompt: input.prompt,
      size,
      timeoutMs: IMAGE_TIMEOUT_MS,
      signal: input.signal
    })

    return {
      mimeType: res.mimeType,
      bytesBase64: Buffer.from(res.bytes).toString('base64'),
      byteLength: res.bytes.byteLength,
      ...(res.revisedPrompt ? { revisedPrompt: res.revisedPrompt } : {}),
      ...(res.url ? { sourceUrl: res.url } : {}),
      durationMs: Date.now() - started,
      model: cfg.model,
      size
    }
  } catch (err) {
    // 主干抛的是它自己的 AgentError，消息里已经带了可行动的提示（如 402 的开通链接）。
    // 这里原样透传，不另写一套措辞 —— 改写一次，那句「去 TokenHub 开通」就丢了。
    throw new Error(`生图失败：${(err as Error).message}`)
  }
}

/**
 * 占位图 —— 生图不可用时的兜底产物。
 *
 * 用 SVG 而不是「假装生成成功」：它是文本、可直接读、可预览，
 * 并且上面写着**为什么这里是一张占位图**。用户看到的是一张会说话的图，
 * 而不是一张与请求毫无关系的随机图 —— 后者才是真正的欺骗。
 *
 * 它同样要过审批才准落盘（见 index.ts 的 writeArtifact），
 * 兜底不是绕过闸门的理由。
 */
export function placeholderSvg(prompt: string, reason: string): string {
  const esc = (s: string): string =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

  const lines = wrap(esc(prompt.slice(0, 120)), 28).slice(0, 4)
  const reasonLines = wrap(esc(reason.slice(0, 160)), 40).slice(0, 4)

  return [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">',
    '<rect width="1280" height="720" fill="#151821"/>',
    '<rect x="24" y="24" width="1232" height="672" fill="none" stroke="#3a4356" stroke-dasharray="12 10"/>',
    '<text x="64" y="120" fill="#e6e9f0" font-family="sans-serif" font-size="34">占位图（未真正生成）</text>',
    ...lines.map(
      (l, i) =>
        `<text x="64" y="${190 + i * 42}" fill="#a8b0c0" font-family="sans-serif" font-size="26">${l}</text>`
    ),
    ...reasonLines.map(
      (l, i) =>
        `<text x="64" y="${430 + i * 32}" fill="#d98a5f" font-family="sans-serif" font-size="22">${l}</text>`
    ),
    '</svg>'
  ].join('\n')
}

function wrap(text: string, perLine: number): string[] {
  const out: string[] = []
  for (let i = 0; i < text.length; i += perLine) out.push(text.slice(i, i + perLine))
  return out.length > 0 ? out : ['']
}
