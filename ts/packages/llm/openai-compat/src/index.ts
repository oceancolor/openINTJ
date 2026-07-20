import {
  AgentError,
  type ChatMessage,
  type ChatOptions,
  ErrorCode,
  type LlmClient,
  type LlmStatus,
} from "@openintj/core";

export interface OpenAICompatibleConfig {
  provider: string;
  apiKey: string;
  baseUrl: string;
  model: string;
  visionModel?: string;
  timeoutMs?: number;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  fetch?: typeof globalThis.fetch;
}

interface ErrorEnvelope {
  error?: { message?: unknown; code?: unknown; type?: unknown };
  message?: unknown;
}

interface CompletionEnvelope {
  choices?: Array<{ message?: { content?: unknown } }>;
}

const text = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

const safeBodyMessage = (body: unknown, fallback: string): string => {
  const envelope = body as ErrorEnvelope | undefined;
  return (
    text(envelope?.error?.message) ??
    text(envelope?.message) ??
    fallback.replace(/\s+/g, " ").slice(0, 200)
  );
};

const isAuthFailure = (status: number, body: unknown): boolean => {
  if (status === 401 || status === 403) return true;
  const envelope = body as ErrorEnvelope | undefined;
  const marker = `${String(envelope?.error?.code ?? "")} ${String(envelope?.error?.type ?? "")}`;
  return /unauthor|invalid[_-]?api[_-]?key|forbidden/i.test(marker);
};

const toMessages = (messages: ChatMessage[]): Array<Record<string, unknown>> =>
  messages.map((message) => ({
    role: message.role,
    content: message.content,
    ...(message.toolCallId ? { tool_call_id: message.toolCallId } : {}),
    ...(message.name ? { name: message.name } : {}),
  }));

/** Strict, non-streaming client for OpenAI-compatible Chat Completions APIs. */
export class OpenAICompatibleClient implements LlmClient {
  readonly config: Readonly<
    Required<Omit<OpenAICompatibleConfig, "visionModel" | "fetch">> & {
      visionModel?: string;
    }
  >;
  private readonly fetchImpl: typeof globalThis.fetch;
  private authFailed = false;
  private connected = true;
  private lastError = "";
  private lastErrorCode = "";
  private lastErrorType = "";

  constructor(config: OpenAICompatibleConfig) {
    this.config = {
      provider: config.provider.trim(),
      apiKey: config.apiKey.trim(),
      baseUrl: config.baseUrl.replace(/\/+$/, ""),
      model: config.model.trim(),
      ...(config.visionModel?.trim() ? { visionModel: config.visionModel.trim() } : {}),
      timeoutMs: config.timeoutMs ?? 60_000,
      maxTokens: config.maxTokens ?? 2048,
      temperature: config.temperature ?? 0.7,
      topP: config.topP ?? 0.9,
    };
    this.fetchImpl = config.fetch ?? globalThis.fetch;
  }

  get isAvailable(): boolean {
    return Boolean(this.config.apiKey) && !this.authFailed && this.connected;
  }

  async chat(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    return await this.request(messages, opts, opts.model ?? this.config.model);
  }

  async visionChat(
    messages: ChatMessage[],
    image: { base64: string; mimeType: string },
    opts: ChatOptions = {},
  ): Promise<string> {
    if (!this.config.visionModel) {
      throw new AgentError({
        code: ErrorCode.CONFIG_MISSING,
        message: `${this.config.provider} 未配置视觉模型`,
        retriable: false,
        details: { provider: this.config.provider, reason: "vision_model_missing" },
      });
    }
    const augmented = [...messages];
    const reverseIndex = [...augmented].reverse().findIndex((message) => message.role === "user");
    const index = reverseIndex >= 0 ? augmented.length - reverseIndex - 1 : augmented.length - 1;
    const target = augmented[index];
    if (target) {
      const parts =
        typeof target.content === "string"
          ? [{ type: "text" as const, text: target.content }]
          : target.content;
      augmented[index] = {
        ...target,
        content: [
          ...parts,
          {
            type: "image_url",
            image_url: { url: `data:${image.mimeType};base64,${image.base64}` },
          },
        ],
      };
    }
    return await this.request(augmented, opts, opts.model ?? this.config.visionModel);
  }

  getStatus(): LlmStatus {
    return {
      provider: this.config.provider,
      model: this.config.model,
      ...(this.config.visionModel ? { visionModel: this.config.visionModel } : {}),
      baseUrl: this.config.baseUrl,
      available: this.isAvailable,
      mode: !this.config.apiKey || this.authFailed ? "unauthorized" : "live",
      status: !this.config.apiKey
        ? "missing_api_key"
        : this.authFailed
          ? "unauthorized"
          : this.connected
            ? "connected"
            : "degraded",
      ...(this.lastError ? { lastError: this.lastError } : {}),
      ...(this.lastErrorCode ? { lastErrorCode: this.lastErrorCode } : {}),
      ...(this.lastErrorType ? { lastErrorType: this.lastErrorType } : {}),
      visionSupported: Boolean(this.config.visionModel),
    };
  }

  private async request(
    messages: ChatMessage[],
    opts: ChatOptions,
    model: string,
  ): Promise<string> {
    this.assertAvailable();
    const controller = new AbortController();
    let timedOut = false;
    const abortFromCaller = (): void => controller.abort(opts.signal?.reason);
    if (opts.signal?.aborted) abortFromCaller();
    else opts.signal?.addEventListener("abort", abortFromCaller, { once: true });
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.config.timeoutMs);

    try {
      const response = await this.fetchImpl(`${this.config.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages: toMessages(messages),
          temperature: opts.temperature ?? this.config.temperature,
          top_p: opts.topP ?? this.config.topP,
          max_tokens: opts.maxTokens ?? this.config.maxTokens,
          stream: false,
        }),
        signal: controller.signal,
      });
      const raw = await response.text();
      let body: unknown;
      try {
        body = JSON.parse(raw);
      } catch {
        body = undefined;
      }
      if (!response.ok) {
        this.connected = false;
        const envelope = body as ErrorEnvelope | undefined;
        this.lastError = safeBodyMessage(body, raw || response.statusText);
        this.lastErrorCode = text(envelope?.error?.code) ?? `HTTP_${response.status}`;
        this.lastErrorType = text(envelope?.error?.type) ?? "http_error";
        const authFailure = isAuthFailure(response.status, body);
        if (authFailure) this.authFailed = true;
        throw new AgentError({
          code: authFailure ? ErrorCode.CONFIG_MISSING : ErrorCode.INTERNAL_ERROR,
          message: `${this.config.provider} 调用失败 (${response.status}): ${this.lastError}`,
          retriable: !authFailure && (response.status === 429 || response.status >= 500),
          details: {
            provider: this.config.provider,
            status: response.status,
            errorCode: this.lastErrorCode,
            errorType: this.lastErrorType,
          },
        });
      }
      const content = (body as CompletionEnvelope | undefined)?.choices?.[0]?.message?.content;
      if (typeof content !== "string") {
        this.connected = false;
        this.lastError = "response is missing choices[0].message.content";
        this.lastErrorCode = "INVALID_RESPONSE";
        this.lastErrorType = "invalid_response";
        throw new AgentError({
          code: ErrorCode.INTERNAL_ERROR,
          message: `${this.config.provider} ${this.lastError}`,
          retriable: true,
          details: { provider: this.config.provider },
        });
      }
      this.connected = true;
      this.lastError = "";
      this.lastErrorCode = "";
      this.lastErrorType = "";
      return content;
    } catch (error) {
      if (error instanceof AgentError) throw error;
      if (opts.signal?.aborted) {
        this.connected = false;
        if (opts.signal.reason instanceof Error) throw opts.signal.reason;
        throw new AgentError({
          code: ErrorCode.EXECUTION_FAILED,
          message: `${this.config.provider} 调用已取消`,
          retriable: false,
          details: { provider: this.config.provider },
        });
      }
      const message = error instanceof Error ? error.message : String(error);
      this.connected = false;
      this.lastError = message;
      this.lastErrorCode = timedOut ? "TIMEOUT" : "NETWORK_ERROR";
      this.lastErrorType = timedOut ? "timeout" : "network_error";
      throw new AgentError({
        code: timedOut ? ErrorCode.TIMEOUT : ErrorCode.INTERNAL_ERROR,
        message: timedOut
          ? `${this.config.provider} 调用超时: ${message}`
          : `${this.config.provider} 网络错误: ${message}`,
        retriable: true,
        details: { provider: this.config.provider },
        cause: error,
      });
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", abortFromCaller);
    }
  }

  private assertAvailable(): void {
    if (this.config.apiKey && !this.authFailed) return;
    const reason = this.authFailed ? "configured credentials were rejected" : "missing API key";
    throw new AgentError({
      code: ErrorCode.CONFIG_MISSING,
      message: `${this.config.provider} 不可用: ${reason}`,
      retriable: false,
      details: { provider: this.config.provider, reason },
    });
  }
}
