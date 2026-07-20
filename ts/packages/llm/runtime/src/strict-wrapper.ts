import {
  AgentError,
  type ChatMessage,
  type ChatOptions,
  ErrorCode,
  type LlmClient,
  type LlmStatus,
} from "@openintj/core";

/**
 * 包装真实 LLM adapter：禁止内部静默 mock。
 * 网络/HTTP 失败直接抛错（显式选择 ollama/hunyuan 时使用）。
 */
export class StrictLlmWrapper implements LlmClient {
  constructor(
    private readonly inner: LlmClient,
    private readonly label: string,
  ) {}

  async chat(messages: ChatMessage[], opts?: ChatOptions): Promise<string> {
    try {
      const out = await this.inner.chat(messages, opts);
      const st = this.inner.getStatus();
      if (st.mode === "mock" || st.status === "degraded") {
        throw new AgentError({
          code: ErrorCode.INTERNAL_ERROR,
          message: `${this.label} 不可用（status=${st.status}, mode=${st.mode}）${st.lastError ? `: ${st.lastError}` : ""}`,
          retriable: true,
        });
      }
      return out;
    } catch (e) {
      if (opts?.signal?.aborted) {
        const reason = opts.signal.reason;
        if (reason instanceof Error) throw reason;
      }
      if (e instanceof AgentError) throw e;
      throw new AgentError({
        code: ErrorCode.INTERNAL_ERROR,
        message: `${this.label} 调用失败: ${e instanceof Error ? e.message : String(e)}`,
        retriable: true,
        cause: e instanceof Error ? e : undefined,
      });
    }
  }

  async visionChat(
    messages: ChatMessage[],
    image: { base64: string; mimeType: string },
    opts?: ChatOptions,
  ): Promise<string> {
    try {
      const out = await this.inner.visionChat(messages, image, opts);
      const st = this.inner.getStatus();
      if (st.mode === "mock" || st.status === "degraded") {
        throw new AgentError({
          code: ErrorCode.INTERNAL_ERROR,
          message: `${this.label} 不可用（status=${st.status}, mode=${st.mode}）${st.lastError ? `: ${st.lastError}` : ""}`,
          retriable: true,
        });
      }
      return out;
    } catch (e) {
      if (opts?.signal?.aborted) {
        const reason = opts.signal.reason;
        if (reason instanceof Error) throw reason;
      }
      if (e instanceof AgentError) throw e;
      throw new AgentError({
        code: ErrorCode.INTERNAL_ERROR,
        message: `${this.label} 视觉调用失败: ${e instanceof Error ? e.message : String(e)}`,
        retriable: true,
        cause: e instanceof Error ? e : undefined,
      });
    }
  }

  getStatus(): LlmStatus {
    return this.inner.getStatus();
  }
}
