/** Hy4 / T1 把思考写在 reasoning_content；max_tokens 过小会把最终 content 挤空。 */
export const HUNYUAN_REASONING_MAX_TOKEN_FLOOR = 8192;
export const HUNYUAN_REASONING_MAX_TOKEN_RETRY = 16_384;
/** Hy4 思考可能超过 60s；TokenHub 文档建议给思考模型更长等待。 */
export const HUNYUAN_REASONING_TIMEOUT_MS_FLOOR = 180_000;

export const isHunyuanReasoningModel = (model: string): boolean =>
  /hy4|hunyuan-t1|(?:^|[^a-z0-9])t1-/i.test(model);

export const resolveHunyuanMaxTokens = (opts: { model: string; requested: number }): number =>
  isHunyuanReasoningModel(opts.model)
    ? Math.max(opts.requested, HUNYUAN_REASONING_MAX_TOKEN_FLOOR)
    : opts.requested;

export const resolveHunyuanTimeoutMs = (opts: { model: string; requested: number }): number =>
  isHunyuanReasoningModel(opts.model)
    ? Math.max(opts.requested, HUNYUAN_REASONING_TIMEOUT_MS_FLOOR)
    : opts.requested;

export const extractAssistantText = (message: unknown): string => {
  if (!message || typeof message !== "object") return "";
  return normalizeContent((message as { content?: unknown }).content);
};

export const extractReasoningText = (message: unknown): string => {
  if (!message || typeof message !== "object") return "";
  const raw = (message as { reasoning_content?: unknown }).reasoning_content;
  return typeof raw === "string" ? raw : "";
};

export const isTruncatedReasoning = (opts: {
  content: string;
  finishReason: string | undefined;
  reasoningContent: string;
}): boolean =>
  !opts.content.trim() &&
  Boolean(opts.reasoningContent.trim()) &&
  (opts.finishReason === "length" || opts.finishReason === "max_tokens");

const normalizeContent = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object" && "text" in part) {
        const text = (part as { text?: unknown }).text;
        return typeof text === "string" ? text : "";
      }
      return "";
    })
    .join("");
};
