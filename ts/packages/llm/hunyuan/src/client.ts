import {
  AgentError,
  type ChatMessage,
  type ChatOptions,
  ErrorCode,
  type LlmClient,
  type LlmStatus,
} from "@openintj/core";
import { generateMockResponse } from "./mock-responses.js";
import { type HunyuanConfig, HunyuanConfigSchema, loadHunyuanConfigFromEnv } from "./types.js";

interface OpenAIChatRequestMessage {
  role: ChatMessage["role"];
  content: unknown;
  tool_call_id?: string;
  name?: string;
}

interface OpenAIChatResponse {
  choices: Array<{
    message: { role: string; content: string };
    finish_reason?: string;
  }>;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

interface OpenAIErrorEnvelope {
  error?: { message?: string; code?: string; type?: string };
}

const isAuthError = (status: number, body: unknown): boolean => {
  if (status === 401 || status === 403) return true;
  const err = (body as OpenAIErrorEnvelope | undefined)?.error;
  const code = err?.code ?? "";
  return /unauthor|invalid_api_key|forbidden/i.test(code);
};

export class HunyuanClient implements LlmClient {
  readonly config: HunyuanConfig;
  private authFailed: boolean;
  private lastError = "";
  private lastErrorCode = "";
  private lastErrorType = "";

  constructor(cfg: Partial<HunyuanConfig> | undefined = undefined) {
    this.config = cfg !== undefined ? HunyuanConfigSchema.parse(cfg) : loadHunyuanConfigFromEnv();
    this.authFailed = false;
  }

  static fromEnv(env?: NodeJS.ProcessEnv): HunyuanClient {
    return new HunyuanClient(loadHunyuanConfigFromEnv(env));
  }

  get isMockMode(): boolean {
    return !this.config.apiKey || this.authFailed;
  }

  async chat(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    if (this.isMockMode) return generateMockResponse(messages);
    return await this.request(messages, opts, this.config.model);
  }

  async visionChat(
    messages: ChatMessage[],
    image: { base64: string; mimeType: string },
    opts: ChatOptions = {},
  ): Promise<string> {
    if (this.isMockMode) return generateMockResponse(messages);
    const augmented = [...messages];
    const lastUserIdx = [...augmented].reverse().findIndex((m) => m.role === "user");
    const idx = lastUserIdx >= 0 ? augmented.length - 1 - lastUserIdx : augmented.length - 1;
    const target = augmented[idx];
    if (target) {
      const dataUrl = `data:${image.mimeType};base64,${image.base64}`;
      const textPart =
        typeof target.content === "string"
          ? [{ type: "text" as const, text: target.content }]
          : target.content;
      augmented[idx] = {
        ...target,
        content: [...textPart, { type: "image_url", image_url: { url: dataUrl } }],
      };
    }
    return await this.request(augmented, opts, this.config.visionModel);
  }

  getStatus(): LlmStatus {
    const status: LlmStatus = {
      provider: "hunyuan",
      model: this.config.model,
      visionModel: this.config.visionModel,
      baseUrl: this.config.baseUrl,
      available: !this.isMockMode,
      mode: this.isMockMode ? (this.authFailed ? "unauthorized" : "mock") : "live",
      status: !this.config.apiKey
        ? "missing_api_key"
        : this.authFailed
          ? "unauthorized"
          : "connected",
      visionSupported: true,
    };
    if (this.lastError) status.lastError = this.lastError;
    if (this.lastErrorCode) status.lastErrorCode = this.lastErrorCode;
    if (this.lastErrorType) status.lastErrorType = this.lastErrorType;
    return status;
  }

  private async request(
    messages: ChatMessage[],
    opts: ChatOptions,
    model: string,
  ): Promise<string> {
    const url = `${this.config.baseUrl.replace(/\/$/, "")}/chat/completions`;
    const body = {
      model: opts.model ?? model,
      messages: this.toOpenAIMessages(messages),
      temperature: opts.temperature ?? this.config.temperature,
      top_p: opts.topP ?? this.config.topP,
      max_tokens: opts.maxTokens ?? this.config.maxTokens,
      stream: false,
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await res.text();
      let parsed: OpenAIChatResponse | OpenAIErrorEnvelope | null = null;
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = null;
      }
      if (!res.ok) {
        const err = (parsed as OpenAIErrorEnvelope | undefined)?.error;
        this.lastError = err?.message ?? text.slice(0, 200);
        this.lastErrorCode = err?.code ?? `HTTP_${res.status}`;
        this.lastErrorType = err?.type ?? "http_error";
        if (isAuthError(res.status, parsed)) {
          this.authFailed = true;
          // 鉴权失败：直接降级 mock，不抛
          return generateMockResponse(messages);
        }
        throw new AgentError({
          code: ErrorCode.INTERNAL_ERROR,
          message: `Hunyuan 调用失败 (${res.status}): ${this.lastError}`,
          retriable: res.status >= 500,
        });
      }
      const data = parsed as OpenAIChatResponse;
      return data.choices?.[0]?.message?.content ?? "";
    } catch (err) {
      if (err instanceof AgentError) throw err;
      const message = err instanceof Error ? err.message : String(err);
      this.lastError = message;
      this.lastErrorType = "network_error";
      throw new AgentError({
        code: ErrorCode.INTERNAL_ERROR,
        message: `Hunyuan 网络/超时: ${message}`,
        retriable: true,
        cause: err,
      });
    } finally {
      clearTimeout(timer);
    }
  }

  private toOpenAIMessages(messages: ChatMessage[]): OpenAIChatRequestMessage[] {
    return messages.map((m) => {
      const r: OpenAIChatRequestMessage = {
        role: m.role,
        content: m.content,
      };
      if (m.toolCallId) r.tool_call_id = m.toolCallId;
      if (m.name) r.name = m.name;
      return r;
    });
  }
}
