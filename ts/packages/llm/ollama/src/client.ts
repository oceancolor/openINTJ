import {
  AgentError,
  type ChatMessage,
  type ChatMessageContent,
  type ChatOptions,
  ErrorCode,
  type LlmClient,
  type LlmStatus,
} from "@openintj/core";
import { generateMockResponse } from "./mock-responses.js";
import { type OllamaConfig, OllamaConfigSchema, loadOllamaConfigFromEnv } from "./types.js";

const stringifyContent = (c: ChatMessageContent): string =>
  typeof c === "string" ? c : c.map((p) => (p.type === "text" ? p.text : "")).join(" ");

interface OllamaChatResponse {
  message?: { role: string; content: string };
  done: boolean;
  prompt_eval_count?: number;
  eval_count?: number;
}

export class OllamaClient implements LlmClient {
  readonly config: OllamaConfig;
  private connected = true;
  private lastError = "";
  private lastErrorType = "";

  constructor(cfg: Partial<OllamaConfig> | undefined = undefined) {
    this.config = cfg !== undefined ? OllamaConfigSchema.parse(cfg) : loadOllamaConfigFromEnv();
  }

  static fromEnv(env?: NodeJS.ProcessEnv): OllamaClient {
    return new OllamaClient(loadOllamaConfigFromEnv(env));
  }

  get isMockMode(): boolean {
    return !this.connected;
  }

  async chat(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    return await this.request(messages, opts, this.config.model, undefined);
  }

  async visionChat(
    messages: ChatMessage[],
    image: { base64: string; mimeType: string },
    opts: ChatOptions = {},
  ): Promise<string> {
    return await this.request(messages, opts, this.config.visionModel, image.base64);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const url = `${this.config.baseUrl.replace(/\/$/, "")}/api/tags`;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 3000);
      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timer);
      this.connected = res.ok;
      return res.ok;
    } catch (err) {
      this.connected = false;
      this.lastError = err instanceof Error ? err.message : String(err);
      this.lastErrorType = "network_error";
      return false;
    }
  }

  getStatus(): LlmStatus {
    const status: LlmStatus = {
      provider: "ollama",
      model: this.config.model,
      visionModel: this.config.visionModel,
      baseUrl: this.config.baseUrl,
      available: this.connected,
      mode: this.connected ? "live" : "mock",
      status: this.connected ? "connected" : "degraded",
      visionSupported: true,
    };
    if (this.lastError) status.lastError = this.lastError;
    if (this.lastErrorType) status.lastErrorType = this.lastErrorType;
    return status;
  }

  private async request(
    messages: ChatMessage[],
    opts: ChatOptions,
    model: string,
    imageBase64: string | undefined,
  ): Promise<string> {
    const url = `${this.config.baseUrl.replace(/\/$/, "")}/api/chat`;
    // Ollama 自有协议：messages 用 string content + images 数组放外层
    const formatted = messages.map((m, i) => {
      const base: { role: string; content: string; images?: string[] } = {
        role: m.role,
        content: stringifyContent(m.content),
      };
      if (imageBase64 !== undefined && i === messages.length - 1 && m.role === "user") {
        base.images = [imageBase64];
      }
      return base;
    });
    const body = {
      model: opts.model ?? model,
      messages: formatted,
      stream: false,
      options: {
        temperature: opts.temperature ?? this.config.temperature,
        top_p: opts.topP ?? this.config.topP,
        num_ctx: this.config.numCtx,
        ...(opts.maxTokens ? { num_predict: opts.maxTokens } : {}),
      },
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.config.timeoutMs);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await res.text();
      if (!res.ok) {
        this.lastError = text.slice(0, 200);
        this.lastErrorType = "http_error";
        // 服务不可达时降级 mock，避免 CLI 卡死
        this.connected = false;
        return generateMockResponse(messages);
      }
      const data: OllamaChatResponse = JSON.parse(text);
      this.connected = true;
      return data.message?.content ?? "";
    } catch (err) {
      this.connected = false;
      const message = err instanceof Error ? err.message : String(err);
      this.lastError = message;
      this.lastErrorType = "network_error";
      // 网络错误降级 mock（除非显式希望抛错）
      if (
        err instanceof DOMException &&
        (err.name === "AbortError" || err.name === "TimeoutError")
      ) {
        throw new AgentError({
          code: ErrorCode.TIMEOUT,
          message: `Ollama 调用超时: ${message}`,
          retriable: true,
          cause: err,
        });
      }
      return generateMockResponse(messages);
    } finally {
      clearTimeout(timer);
    }
  }
}
