import type { ToolHandler } from "@openintj/core";

/**
 * 外部联网搜索工具（provider-neutral）。
 *
 * 背景：腾讯混元旧平台内建联网搜索（`enable_enhancement` 等）随平台 2026-06-22 下线，
 * TokenHub 把搜索改成独立的 Responses API 产品。为保持 provider 中立、不绑定某家模型，
 * 这里走「Function Calling + 外部搜索后端」路线：模型决定调用 `search` 工具时，由本 handler
 * 请求一个真正的 Web Search API（Tavily / Brave）拿结果，再以观察值回填给模型。
 *
 * - 默认零开销：未配置 key 时 `resolveWebSearchConfig` 返回 undefined，装配端不挂本工具。
 * - 工具语义：失败不抛错，返回 `{ ok:false, error }`，交给 ReAct 循环决定下一步。
 */
export type WebSearchProvider = "tavily" | "brave";

export interface WebSearchResultItem {
  title: string;
  url: string;
  snippet: string;
}

export interface WebSearchOutput {
  ok: boolean;
  provider: WebSearchProvider;
  query: string;
  /** 部分 provider（Tavily）会直接给出综合答案；Brave 不给则为空。 */
  answer?: string;
  results: WebSearchResultItem[];
  error?: string;
}

export interface WebSearchToolOpts {
  provider: WebSearchProvider;
  apiKey: string;
  /** 返回结果条数上限（默认 5）。 */
  maxResults?: number;
  /** 单次请求超时（毫秒，默认 10s）。 */
  timeoutMs?: number;
  /** 可注入的 fetch（测试用；缺省走全局 fetch）。 */
  fetchImpl?: typeof fetch;
}

const DEFAULT_MAX_RESULTS = 5;
const DEFAULT_TIMEOUT_MS = 10_000;

/**
 * 从 env 解析搜索 provider + key。优先级：
 *  1. `OPENINTJ_SEARCH_PROVIDER` 显式指定，配合 `OPENINTJ_SEARCH_API_KEY` 或对应的 `*_API_KEY`
 *  2. 未显式指定时，按存在的 provider-specific key 推断（Tavily 优先于 Brave）
 * 都没有 → undefined（不挂工具）。
 */
export const resolveWebSearchConfig = (
  env: NodeJS.ProcessEnv = process.env,
): { provider: WebSearchProvider; apiKey: string } | undefined => {
  const explicit = env["OPENINTJ_SEARCH_PROVIDER"]?.trim().toLowerCase();
  const generic = env["OPENINTJ_SEARCH_API_KEY"]?.trim();
  const tavilyKey = env["TAVILY_API_KEY"]?.trim();
  const braveKey = env["BRAVE_API_KEY"]?.trim();

  if (explicit === "tavily") {
    const key = generic || tavilyKey;
    return key ? { provider: "tavily", apiKey: key } : undefined;
  }
  if (explicit === "brave") {
    const key = generic || braveKey;
    return key ? { provider: "brave", apiKey: key } : undefined;
  }
  // 未显式指定：按已配置的 key 推断。
  if (tavilyKey) return { provider: "tavily", apiKey: tavilyKey };
  if (braveKey) return { provider: "brave", apiKey: braveKey };
  return undefined;
};

interface TavilyResponse {
  answer?: string;
  results?: Array<{ title?: string; url?: string; content?: string }>;
}

interface BraveResponse {
  web?: { results?: Array<{ title?: string; url?: string; description?: string }> };
}

const searchTavily = async (
  query: string,
  opts: Required<Pick<WebSearchToolOpts, "apiKey" | "maxResults" | "timeoutMs">>,
  fetchImpl: typeof fetch,
): Promise<WebSearchOutput> => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs);
  try {
    const res = await fetchImpl("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${opts.apiKey}`,
      },
      body: JSON.stringify({
        query,
        max_results: opts.maxResults,
        include_answer: true,
        search_depth: "basic",
      }),
      signal: controller.signal,
    });
    const text = await res.text();
    if (!res.ok) {
      return {
        ok: false,
        provider: "tavily",
        query,
        results: [],
        error: `tavily HTTP ${res.status}: ${text.slice(0, 200)}`,
      };
    }
    const data = JSON.parse(text) as TavilyResponse;
    const results: WebSearchResultItem[] = (data.results ?? []).map((r) => ({
      title: r.title ?? "",
      url: r.url ?? "",
      snippet: r.content ?? "",
    }));
    return {
      ok: true,
      provider: "tavily",
      query,
      ...(data.answer ? { answer: data.answer } : {}),
      results,
    };
  } catch (e) {
    return {
      ok: false,
      provider: "tavily",
      query,
      results: [],
      error: e instanceof Error ? e.message : String(e),
    };
  } finally {
    clearTimeout(timer);
  }
};

const searchBrave = async (
  query: string,
  opts: Required<Pick<WebSearchToolOpts, "apiKey" | "maxResults" | "timeoutMs">>,
  fetchImpl: typeof fetch,
): Promise<WebSearchOutput> => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs);
  try {
    const url = new URL("https://api.search.brave.com/res/v1/web/search");
    url.searchParams.set("q", query);
    url.searchParams.set("count", String(opts.maxResults));
    const res = await fetchImpl(url.toString(), {
      method: "GET",
      headers: {
        Accept: "application/json",
        "X-Subscription-Token": opts.apiKey,
      },
      signal: controller.signal,
    });
    const text = await res.text();
    if (!res.ok) {
      return {
        ok: false,
        provider: "brave",
        query,
        results: [],
        error: `brave HTTP ${res.status}: ${text.slice(0, 200)}`,
      };
    }
    const data = JSON.parse(text) as BraveResponse;
    const results: WebSearchResultItem[] = (data.web?.results ?? [])
      .slice(0, opts.maxResults)
      .map((r) => ({
        title: r.title ?? "",
        url: r.url ?? "",
        snippet: r.description ?? "",
      }));
    return { ok: true, provider: "brave", query, results };
  } catch (e) {
    return {
      ok: false,
      provider: "brave",
      query,
      results: [],
      error: e instanceof Error ? e.message : String(e),
    };
  } finally {
    clearTimeout(timer);
  }
};

/**
 * 构造一个真实的 `search` 工具 handler，可注册到 ToolHub 的内置 search 槽。
 * 入参 `{ query: string }`；返回 `WebSearchOutput`（失败不抛错）。
 */
export const createWebSearchTool = (opts: WebSearchToolOpts): ToolHandler => {
  const resolved = {
    apiKey: opts.apiKey,
    maxResults: opts.maxResults ?? DEFAULT_MAX_RESULTS,
    timeoutMs: opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  };
  const fetchImpl = opts.fetchImpl ?? fetch;
  return async (params: Record<string, unknown>) => {
    const raw = params["query"];
    const query = (typeof raw === "string" ? raw : String(raw ?? "")).trim();
    if (!query) {
      return {
        ok: false,
        provider: opts.provider,
        query: "",
        results: [],
        error: "search: 缺少 query 参数",
      };
    }
    return opts.provider === "tavily"
      ? searchTavily(query, resolved, fetchImpl)
      : searchBrave(query, resolved, fetchImpl);
  };
};
