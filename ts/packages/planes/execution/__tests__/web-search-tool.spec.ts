import { describe, expect, it, vi } from "vitest";
import {
  type WebSearchOutput,
  createWebSearchTool,
  resolveWebSearchConfig,
} from "../src/web-search-tool.js";

const jsonResponse = (body: unknown, init: { ok?: boolean; status?: number } = {}): Response =>
  ({
    ok: init.ok ?? true,
    status: init.status ?? 200,
    text: async () => JSON.stringify(body),
  }) as unknown as Response;

describe("resolveWebSearchConfig", () => {
  it("无 key 时返回 undefined", () => {
    expect(resolveWebSearchConfig({})).toBeUndefined();
  });

  it("显式 provider=tavily + OPENINTJ_SEARCH_API_KEY", () => {
    const cfg = resolveWebSearchConfig({
      OPENINTJ_SEARCH_PROVIDER: "tavily",
      OPENINTJ_SEARCH_API_KEY: "tvly-x",
    });
    expect(cfg).toEqual({ provider: "tavily", apiKey: "tvly-x" });
  });

  it("显式 provider=brave 用 BRAVE_API_KEY", () => {
    const cfg = resolveWebSearchConfig({
      OPENINTJ_SEARCH_PROVIDER: "brave",
      BRAVE_API_KEY: "bsk-x",
    });
    expect(cfg).toEqual({ provider: "brave", apiKey: "bsk-x" });
  });

  it("未显式指定时按已配置 key 推断（tavily 优先）", () => {
    const cfg = resolveWebSearchConfig({ TAVILY_API_KEY: "tvly-a", BRAVE_API_KEY: "bsk-b" });
    expect(cfg).toEqual({ provider: "tavily", apiKey: "tvly-a" });
  });

  it("显式 provider 但缺对应 key → undefined", () => {
    expect(resolveWebSearchConfig({ OPENINTJ_SEARCH_PROVIDER: "tavily" })).toBeUndefined();
  });
});

describe("createWebSearchTool (tavily)", () => {
  it("解析 answer + results 并带正确请求", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({
        answer: "TS 是 JS 的超集",
        results: [
          { title: "TypeScript", url: "https://ts.dev", content: "类型化的 JS" },
          { title: "TS Handbook", url: "https://ts.dev/handbook", content: "文档" },
        ],
      }),
    );
    const tool = createWebSearchTool({
      provider: "tavily",
      apiKey: "tvly-x",
      maxResults: 5,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    const out = (await tool({ query: "什么是 TypeScript" })) as WebSearchOutput;
    expect(out.ok).toBe(true);
    expect(out.provider).toBe("tavily");
    expect(out.answer).toBe("TS 是 JS 的超集");
    expect(out.results).toHaveLength(2);
    expect(out.results[0]).toEqual({
      title: "TypeScript",
      url: "https://ts.dev",
      snippet: "类型化的 JS",
    });
    const [url, init] = fetchImpl.mock.calls[0]!;
    expect(url).toBe("https://api.tavily.com/search");
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).headers).toMatchObject({ Authorization: "Bearer tvly-x" });
  });

  it("HTTP 非 2xx → ok:false 且不抛错", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({ error: "bad key" }, { ok: false, status: 401 }),
    );
    const tool = createWebSearchTool({
      provider: "tavily",
      apiKey: "tvly-x",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    const out = (await tool({ query: "x" })) as WebSearchOutput;
    expect(out.ok).toBe(false);
    expect(out.error).toContain("401");
  });

  it("fetch 抛错（如超时）→ ok:false 捕获", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("aborted");
    });
    const tool = createWebSearchTool({
      provider: "tavily",
      apiKey: "tvly-x",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    const out = (await tool({ query: "x" })) as WebSearchOutput;
    expect(out.ok).toBe(false);
    expect(out.error).toBe("aborted");
  });
});

describe("createWebSearchTool (brave)", () => {
  it("映射 web.results.description → snippet 并限 maxResults", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({
        web: {
          results: [
            { title: "A", url: "https://a", description: "da" },
            { title: "B", url: "https://b", description: "db" },
            { title: "C", url: "https://c", description: "dc" },
          ],
        },
      }),
    );
    const tool = createWebSearchTool({
      provider: "brave",
      apiKey: "bsk-x",
      maxResults: 2,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    const out = (await tool({ query: "abc" })) as WebSearchOutput;
    expect(out.ok).toBe(true);
    expect(out.provider).toBe("brave");
    expect(out.results).toHaveLength(2);
    expect(out.results[1]).toEqual({ title: "B", url: "https://b", snippet: "db" });
    const [url, init] = fetchImpl.mock.calls[0]!;
    expect(String(url)).toContain("https://api.search.brave.com/res/v1/web/search");
    expect(String(url)).toContain("q=abc");
    expect((init as RequestInit).headers).toMatchObject({ "X-Subscription-Token": "bsk-x" });
  });
});

describe("createWebSearchTool (通用)", () => {
  it("缺 query → ok:false，不发请求", async () => {
    const fetchImpl = vi.fn();
    const tool = createWebSearchTool({
      provider: "tavily",
      apiKey: "tvly-x",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    const out = (await tool({})) as WebSearchOutput;
    expect(out.ok).toBe(false);
    expect(out.error).toContain("query");
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
