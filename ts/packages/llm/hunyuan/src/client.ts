import {
  AgentError,
  type ChatMessage,
  type ChatOptions,
  ErrorCode,
  type LlmClient,
  type LlmStatus,
  type ToolHandler,
} from "@openintj/core";
import { generateMockResponse } from "./mock-responses.js";
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
  mode: "live" | "mock" | "unauthorized";
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

  get isMockMode(): boolean {
    return !this.config.apiKey || this.authFailed;
  }

  async chat(messages: ChatMessage[], opts: ChatOptions = {}): Promise<string> {
    if (this.isMockMode) return generateMockResponse(messages);
    return await this.request(messages, opts, this.config.model);
  }

  /**
   * 联网搜索：对单次调用强制开启功能增强（enable_enhancement + force_search + search_info），
   * 不依赖全局 env 开关，返回模型回答 + 命中来源。用于把"真实 search 工具"接进 Agent。
   */
  async webSearch(query: string, opts: ChatOptions = {}): Promise<HunyuanSearchResult> {
    const messages: ChatMessage[] = [{ role: "user", content: query }];
    if (this.isMockMode) {
      return {
        query,
        answer: generateMockResponse(messages),
        sources: [],
        mode: this.authFailed ? "unauthorized" : "mock",
      };
    }
    const answer = await this.request(messages, opts, this.config.model, "force");
    if (this.authFailed) {
      return { query, answer, sources: [], mode: "unauthorized" };
    }
    return { query, answer, sources: [...this.lastSearchSources], mode: "live" };
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
      this.lastSearchSources = data.search_info?.search_results ?? [];
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

/**
 * 用混元联网搜索构造一个真实的 `search` 工具 handler，可注册到 ToolHub 的内置 search 槽。
 * - 调用时强制开启联网搜索（不依赖 HUNYUAN_ENABLE_SEARCH env）。
 * - 无 key / 鉴权失败时返回 `mode: "mock"|"unauthorized"`，不抛错（与工具语义一致）。
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
      ok: r.mode === "live",
      mode: r.mode,
      query: r.query,
      answer: r.answer,
      sources: r.sources,
    };
  };
