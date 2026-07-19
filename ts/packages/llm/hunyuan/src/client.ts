import {
  AgentError,
  type ChatMessage,
  type ChatOptions,
  ErrorCode,
  type LlmClient,
  type LlmStatus,
  type ToolHandler,
} from "@openintj/core";
import { type HunyuanConfig, HunyuanConfigSchema, loadHunyuanConfigFromEnv } from "./types.js";

interface OpenAIChatRequestMessage {
  role: ChatMessage["role"];
  content: unknown;
  tool_call_id?: string;
  name?: string;
}

export interface HunyuanSearchSource {
  index?: number;
  title?: string;
  url?: string;
  text?: string;
}

export interface HunyuanSearchResult {
  query: string;
  /** 模型基于联网搜索结果生成的回答（命中搜索时含实时信息）。 */
  answer: string;
  /** 命中搜索时的来源列表（需服务端开启 search_info）。 */
  sources: HunyuanSearchSource[];
  mode: "live";
}

interface OpenAIChatResponse {
  choices: Array<{
    message: { role: string; content: string };
    finish_reason?: string;
  }>;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  /** 混元扩展：开启 search_info 且命中搜索时返回的来源列表。 */
  search_info?: { search_results?: HunyuanSearchSource[] };
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
  private connected = true;
  private lastError = "";
  private lastErrorCode = "";
  private lastErrorType = "";
  /** 最近一次命中联网搜索时返回的来源列表（需 enableEnhancement + searchInfo）。 */
  lastSearchSources: HunyuanSearchSource[] = [];

  constructor(cfg: Partial<HunyuanConfig> | undefined = undefined) {
    this.config = cfg !== undefined ? HunyuanConfigSchema.parse(cfg) : loadHunyuanConfigFromEnv();
    this.authFailed = false;
  }

  static fromEnv(env?: NodeJS.ProcessEnv): HunyuanClient {
    return new HunyuanClient(loadHunyuanConfigFromEnv(env));
  }

  get isAvailable(): boolean {
    return Boolean(this.config.apiKey) && !this.authFailed && this.connected;
  }

  async chat(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    this.assertAvailable();
    return await this.request(messages, opts, this.config.model);
  }

  /**
   * 联网搜索：对单次调用强制开启功能增强（enable_enhancement + force_search + search_info），
   * 不依赖全局 env 开关，返回模型回答 + 命中来源。用于把"真实 search 工具"接进 Agent。
   */
  async webSearch(query: string, opts: ChatOptions = {}): Promise<HunyuanSearchResult> {
    const messages: ChatMessage[] = [{ role: "user", content: query }];
    this.assertAvailable();
    this.lastSearchSources = [];
    const answer = await this.request(messages, opts, this.config.model, "force");
    return { query, answer, sources: [...this.lastSearchSources], mode: "live" };
  }

  async visionChat(
    messages: ChatMessage[],
    image: { base64: string; mimeType: string },
    opts: ChatOptions = {},
  ): Promise<string> {
    this.assertAvailable();
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
      available: this.isAvailable,
      mode: !this.config.apiKey || this.authFailed ? "unauthorized" : "live",
      status: !this.config.apiKey
        ? "missing_api_key"
        : this.authFailed
          ? "unauthorized"
          : this.connected
            ? "connected"
            : "degraded",
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
    searchMode: "config" | "force" = "config",
  ): Promise<string> {
    const url = `${this.config.baseUrl.replace(/\/$/, "")}/chat/completions`;
    // force 模式（webSearch 走这里）无视全局 env，强制联网；config 模式按配置。
    const forced = searchMode === "force";
    // forceSearch 隐含开启联网搜索；citation 依赖 enableEnhancement + searchInfo。
    const searchEnabled = forced || this.config.enableEnhancement || this.config.forceSearch;
    const forceSearch = forced || this.config.forceSearch;
    const wantSearchInfo = forced || (searchEnabled && this.config.searchInfo);
    const body = {
      model: opts.model ?? model,
      messages: this.toOpenAIMessages(messages),
      temperature: opts.temperature ?? this.config.temperature,
      top_p: opts.topP ?? this.config.topP,
      max_tokens: opts.maxTokens ?? this.config.maxTokens,
      stream: false,
      ...(searchEnabled ? { enable_enhancement: true } : {}),
      ...(forceSearch ? { force_search_enhancement: true } : {}),
      ...(wantSearchInfo ? { search_info: true } : {}),
      ...(wantSearchInfo && this.config.citation ? { citation: true } : {}),
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
        this.connected = false;
        const err = (parsed as OpenAIErrorEnvelope | undefined)?.error;
        this.lastError = err?.message ?? text.slice(0, 200);
        this.lastErrorCode = err?.code ?? `HTTP_${res.status}`;
        this.lastErrorType = err?.type ?? "http_error";
        if (isAuthError(res.status, parsed)) {
          this.authFailed = true;
          throw new AgentError({
            code: ErrorCode.CONFIG_MISSING,
            message: `Hunyuan 鉴权失败: ${this.lastError}`,
            retriable: false,
            details: {
              provider: "hunyuan",
              status: res.status,
              errorCode: this.lastErrorCode,
            },
          });
        }
        throw new AgentError({
          code: ErrorCode.INTERNAL_ERROR,
          message: `Hunyuan 调用失败 (${res.status}): ${this.lastError}`,
          retriable: res.status >= 500,
        });
      }
      const data = parsed as OpenAIChatResponse;
      const content = data?.choices?.[0]?.message?.content;
      if (typeof content !== "string") {
        this.connected = false;
        this.lastError = "Hunyuan response is missing choices[0].message.content";
        this.lastErrorCode = "INVALID_RESPONSE";
        this.lastErrorType = "invalid_response";
        throw new AgentError({
          code: ErrorCode.INTERNAL_ERROR,
          message: this.lastError,
          retriable: true,
          details: { provider: "hunyuan" },
        });
      }
      this.lastSearchSources = data.search_info?.search_results ?? [];
      this.connected = true;
      this.lastError = "";
      this.lastErrorCode = "";
      this.lastErrorType = "";
      return content;
    } catch (err) {
      if (err instanceof AgentError) throw err;
      const message = err instanceof Error ? err.message : String(err);
      this.connected = false;
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

  private assertAvailable(): void {
    if (this.config.apiKey && !this.authFailed) return;
    const reason = this.authFailed ? "configured credentials were rejected" : "missing API key";
    throw new AgentError({
      code: ErrorCode.CONFIG_MISSING,
      message: `Hunyuan 不可用: ${reason}`,
      retriable: false,
      details: { provider: "hunyuan", reason },
    });
  }
}

/**
 * 用混元联网搜索构造一个真实的 `search` 工具 handler，可注册到 ToolHub 的内置 search 槽。
 * - 调用时强制开启联网搜索（不依赖 HUNYUAN_ENABLE_SEARCH env）。
 * - 无 key / 鉴权失败时抛出结构化 AgentError，由 ToolHub 记录为显式工具失败。
 */
export const createHunyuanSearchTool =
  (client: HunyuanClient): ToolHandler =>
  async (params: Record<string, unknown>) => {
    const raw = params["query"];
    const query = (typeof raw === "string" ? raw : String(raw ?? "")).trim();
    if (!query) {
      return { ok: false, error: "search: 缺少 query 参数" };
    }
    const r = await client.webSearch(query);
    return {
      ok: true,
      mode: r.mode,
      query: r.query,
      answer: r.answer,
      sources: r.sources,
    };
  };
