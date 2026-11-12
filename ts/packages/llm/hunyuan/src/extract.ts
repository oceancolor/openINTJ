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

/**
 * 原生 tool_calls 转 ReAct 文本协议：模型走原生函数调用返回
 * message.tool_calls 时，content 往往为空；转成 Action/Action-Input
 * 让上层文本 ReAct 循环能真正执行，而不是判「无最终 content」死掉。
 */
export const extractToolCallsText = (message: unknown): string => {
  if (!message || typeof message !== "object") return "";
  const calls = (message as { tool_calls?: unknown }).tool_calls;
  if (!Array.isArray(calls)) return "";
  const lines: string[] = [];
  for (const call of calls) {
    if (!call || typeof call !== "object") continue;
    const fn = (call as { function?: { name?: unknown; arguments?: unknown }; name?: unknown; arguments?: unknown });
    const name =
      typeof fn.function?.name === "string"
        ? fn.function.name
        : typeof fn.name === "string"
          ? fn.name
          : undefined;
    if (!name) continue;
    const rawArgs = fn.function?.arguments ?? fn.arguments;
    const args =
      typeof rawArgs === "string" ? rawArgs : rawArgs ? JSON.stringify(rawArgs) : "{}";
    lines.push(`Action: ${name}`, `Action-Input: ${args}`);
  }
  return lines.join("\n");
};

export const isTruncatedReasoning = (opts: {
  content: string;
  finishReason: string | undefined;
  reasoningContent: string;
}): boolean =>
  !opts.content.trim() &&
  Boolean(opts.reasoningContent.trim()) &&
  (opts.finishReason === "length" ||
    opts.finishReason === "max_tokens" ||
    // 只回了 reasoning + tool_calls 意图却没给 content/tool_calls 明细时，
    // 加大额度重试一次通常能拿到最终 content。
    opts.finishReason === "tool_calls");

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
