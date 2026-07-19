import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HunyuanClient, createHunyuanSearchTool } from "../src/index.js";

describe("HunyuanClient (strict configuration)", () => {
  it("fails closed when no api key is configured", async () => {
    const c = new HunyuanClient({ apiKey: "" });
    expect(c.isAvailable).toBe(false);
    await expect(c.chat([{ role: "user", content: "你好" }])).rejects.toMatchObject({
      code: "CONFIG_MISSING",
      retriable: false,
    });
  });

  it("getStatus reports missing_api_key", () => {
    const c = new HunyuanClient({ apiKey: "" });
    const s = c.getStatus();
    expect(s.status).toBe("missing_api_key");
    expect(s.mode).toBe("unauthorized");
    expect(s.available).toBe(false);
    expect(s.provider).toBe("hunyuan");
  });
});

describe("HunyuanClient (live mode with fetch mock)", () => {
  const originalFetch = globalThis.fetch;
  beforeEach(() => {
    // reset
    globalThis.fetch = originalFetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("calls openai-compat endpoint with bearer auth and returns content", async () => {
    const fetchMock = vi.fn(async (url: unknown, init: unknown) => {
      const reqInit = init as { headers: Record<string, string>; body: string };
      expect(String(url)).toMatch(/chat\/completions$/);
      expect(reqInit.headers["Authorization"]).toBe("Bearer test-key");
      const body = JSON.parse(reqInit.body);
      expect(body.model).toBe("hunyuan-turbos-latest");
      return new Response(
        JSON.stringify({
          choices: [{ message: { role: "assistant", content: "hi from llm" } }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "test-key" });
    const r = await c.chat([{ role: "user", content: "ping" }]);
    expect(r).toBe("hi from llm");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("omits search params by default", async () => {
    let body: Record<string, unknown> = {};
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      body = JSON.parse((init as { body: string }).body);
      return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), {
        status: 200,
      });
    }) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    await c.chat([{ role: "user", content: "x" }]);
    expect(body["enable_enhancement"]).toBeUndefined();
    expect(body["search_info"]).toBeUndefined();
  });

  it("sends enable_enhancement + search_info when web search is configured", async () => {
    let body: Record<string, unknown> = {};
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      body = JSON.parse((init as { body: string }).body);
      return new Response(
        JSON.stringify({
          choices: [{ message: { content: "今天天气晴" } }],
          search_info: {
            search_results: [{ index: 1, title: "天气", url: "https://example.com" }],
          },
        }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;
    const c = new HunyuanClient({
      apiKey: "k",
      enableEnhancement: true,
      searchInfo: true,
      citation: true,
    });
    const r = await c.chat([{ role: "user", content: "今天天气" }]);
    expect(r).toBe("今天天气晴");
    expect(body["enable_enhancement"]).toBe(true);
    expect(body["search_info"]).toBe(true);
    expect(body["citation"]).toBe(true);
    expect(c.lastSearchSources).toHaveLength(1);
    expect(c.lastSearchSources[0]?.url).toBe("https://example.com");
  });

  it("forceSearch implies enable_enhancement", async () => {
    let body: Record<string, unknown> = {};
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      body = JSON.parse((init as { body: string }).body);
      return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), {
        status: 200,
      });
    }) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k", forceSearch: true });
    await c.chat([{ role: "user", content: "x" }]);
    expect(body["enable_enhancement"]).toBe(true);
    expect(body["force_search_enhancement"]).toBe(true);
  });

  it("fails closed on 401 and updates status", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ error: { message: "bad key", code: "invalid_api_key" } }), {
        status: 401,
      })) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "bad" });
    await expect(c.chat([{ role: "user", content: "你好" }])).rejects.toMatchObject({
      code: "CONFIG_MISSING",
      retriable: false,
    });
    const s = c.getStatus();
    expect(s.mode).toBe("unauthorized");
    expect(s.status).toBe("unauthorized");
  });

  it("webSearch forces enhancement + search_info even when config search is off", async () => {
    let body: Record<string, unknown> = {};
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      body = JSON.parse((init as { body: string }).body);
      return new Response(
        JSON.stringify({
          choices: [{ message: { content: "实时答案" } }],
          search_info: { search_results: [{ index: 1, title: "源", url: "https://e.com" }] },
        }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" }); // enableEnhancement 默认 false
    const r = await c.webSearch("今天新闻");
    expect(body["enable_enhancement"]).toBe(true);
    expect(body["force_search_enhancement"]).toBe(true);
    expect(body["search_info"]).toBe(true);
    expect(r.mode).toBe("live");
    expect(r.answer).toBe("实时答案");
    expect(r.sources[0]?.url).toBe("https://e.com");
  });

  it("createHunyuanSearchTool returns live result with sources", async () => {
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          choices: [{ message: { content: "A" } }],
          search_info: { search_results: [{ title: "t", url: "https://u" }] },
        }),
        { status: 200 },
      )) as unknown as typeof fetch;
    const tool = createHunyuanSearchTool(new HunyuanClient({ apiKey: "k" }));
    const out = (await tool({ query: "x" })) as {
      ok: boolean;
      mode: string;
      sources: unknown[];
    };
    expect(out.ok).toBe(true);
    expect(out.mode).toBe("live");
    expect(out.sources).toHaveLength(1);
  });

  it("createHunyuanSearchTool rejects empty query without calling fetch", async () => {
    const fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const tool = createHunyuanSearchTool(new HunyuanClient({ apiKey: "k" }));
    const out = (await tool({ query: "   " })) as { ok: boolean; error?: string };
    expect(out.ok).toBe(false);
    expect(out.error).toContain("query");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("createHunyuanSearchTool fails visibly without a key", async () => {
    const tool = createHunyuanSearchTool(new HunyuanClient({ apiKey: "" }));
    await expect(tool({ query: "hello" })).rejects.toMatchObject({
      code: "CONFIG_MISSING",
    });
  });

  it("throws on 500 retriable", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ error: { message: "server error" } }), {
        status: 500,
      })) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    await expect(c.chat([{ role: "user", content: "x" }])).rejects.toMatchObject({
      retriable: true,
    });
    expect(c.getStatus()).toMatchObject({
      available: false,
      mode: "live",
      status: "degraded",
    });
  });

  it("rejects malformed success responses", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ choices: [] }), { status: 200 })) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    await expect(c.chat([{ role: "user", content: "x" }])).rejects.toMatchObject({
      code: "INTERNAL_ERROR",
      retriable: true,
    });
    expect(c.getStatus().lastErrorType).toBe("invalid_response");
  });

  it("vision chat injects image into last user message", async () => {
    let captured: { messages: Array<{ role: string; content: unknown }> } = {
      messages: [],
    };
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      captured = JSON.parse((init as { body: string }).body);
      return new Response(
        JSON.stringify({ choices: [{ message: { content: "I see an image" } }] }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    const r = await c.visionChat([{ role: "user", content: "describe" }], {
      base64: "AAA",
      mimeType: "image/png",
    });
    expect(r).toBe("I see an image");
    const last = captured.messages.at(-1);
    expect(Array.isArray(last?.content)).toBe(true);
    const parts = last?.content as Array<{ type: string; image_url?: { url: string } }>;
    const img = parts.find((p) => p.type === "image_url");
    expect(img?.image_url?.url).toMatch(/^data:image\/png;base64,/);
  });
});
