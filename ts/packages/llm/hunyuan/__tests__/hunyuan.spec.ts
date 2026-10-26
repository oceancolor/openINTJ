import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  HUNYUAN_DEFAULT_BASE_URL,
  HUNYUAN_DEFAULT_MODEL,
  HUNYUAN_REASONING_MAX_TOKEN_FLOOR,
  HUNYUAN_REASONING_MAX_TOKEN_RETRY,
  HUNYUAN_REASONING_TIMEOUT_MS_FLOOR,
  HunyuanClient,
  createHunyuanSearchTool,
  extractAssistantText,
  isHunyuanReasoningModel,
  loadHunyuanConfigFromEnv,
  resolveHunyuanMaxTokens,
  resolveHunyuanTimeoutMs,
} from "../src/index.js";

describe("Hy4 reasoning token helpers", () => {
  it("treats hy4-preview as a reasoning model and floors small budgets", () => {
    expect(isHunyuanReasoningModel("hy4-preview")).toBe(true);
    expect(isHunyuanReasoningModel("hy3")).toBe(false);
    expect(resolveHunyuanMaxTokens({ model: "hy4-preview", requested: 1024 })).toBe(
      HUNYUAN_REASONING_MAX_TOKEN_FLOOR,
    );
    expect(resolveHunyuanMaxTokens({ model: "hy3", requested: 1024 })).toBe(1024);
    expect(resolveHunyuanTimeoutMs({ model: "hy4-preview", requested: 60_000 })).toBe(
      HUNYUAN_REASONING_TIMEOUT_MS_FLOOR,
    );
    expect(resolveHunyuanTimeoutMs({ model: "hy3", requested: 60_000 })).toBe(60_000);
  });

  it("joins content parts and ignores null content", () => {
    expect(extractAssistantText({ content: null, reasoning_content: "think" })).toBe("");
    expect(
      extractAssistantText({ content: [{ type: "text", text: "A" }, { type: "text", text: "B" }] }),
    ).toBe("AB");
  });
});

describe("HunyuanClient (strict configuration)", () => {
  it("uses the official Hy4 Preview TokenHub defaults", () => {
    const c = new HunyuanClient({ apiKey: "key" });
    expect(c.config.baseUrl).toBe(HUNYUAN_DEFAULT_BASE_URL);
    expect(c.config.model).toBe(HUNYUAN_DEFAULT_MODEL);
    expect(c.config.model).toBe("hy4-preview");
    expect(c.config.maxTokens).toBe(8192);
  });

  it("maps known retired defaults with warnings but preserves custom values", () => {
    const warnings: string[] = [];
    const migrated = loadHunyuanConfigFromEnv(
      {
        HUNYUAN_BASE_URL: "https://api.hunyuan.cloud.tencent.com/v1",
        HUNYUAN_MODEL: "hy3-preview",
      },
      (message) => warnings.push(message),
    );
    expect(migrated).toMatchObject({
      baseUrl: HUNYUAN_DEFAULT_BASE_URL,
      model: HUNYUAN_DEFAULT_MODEL,
    });
    expect(warnings).toHaveLength(2);

    const custom = loadHunyuanConfigFromEnv(
      {
        HUNYUAN_BASE_URL: "https://gateway.example.test/v1",
        HUNYUAN_MODEL: "private-hy3-deployment",
      },
      (message) => warnings.push(message),
    );
    expect(custom).toMatchObject({
      baseUrl: "https://gateway.example.test/v1",
      model: "private-hy3-deployment",
    });
    expect(warnings).toHaveLength(2);
  });

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
      expect(body.model).toBe("hy4-preview");
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

  it("aborts an in-flight HTTP request when the caller cancels", async () => {
    let fetchSignal: AbortSignal | undefined;
    globalThis.fetch = ((_url: unknown, init?: RequestInit) => {
      const signal = init?.signal as AbortSignal;
      fetchSignal = signal;
      return new Promise<Response>((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
    }) as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    const controller = new AbortController();
    const pending = c.chat([{ role: "user", content: "wait" }], {
      signal: controller.signal,
    });
    const reason = new Error("caller cancelled Hunyuan");

    controller.abort(reason);

    await expect(pending).rejects.toBe(reason);
    expect(fetchSignal?.aborted).toBe(true);
  });

  it("keeps Hy4 waiting past the 60s config timeout", async () => {
    vi.useFakeTimers();
    try {
      let aborted = false;
      globalThis.fetch = ((_url: unknown, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener(
            "abort",
            () => {
              aborted = true;
              reject(init.signal?.reason ?? new Error("aborted"));
            },
            { once: true },
          );
        })) as typeof fetch;
      const pending = expect(
        new HunyuanClient({ apiKey: "k", timeoutMs: 60_000 }).chat([{ role: "user", content: "x" }]),
      ).rejects.toMatchObject({ code: "TIMEOUT" });
      await vi.advanceTimersByTimeAsync(60_000);
      expect(aborted).toBe(false);
      await vi.advanceTimersByTimeAsync(120_000);
      expect(aborted).toBe(true);
      await pending;
    } finally {
      vi.useRealTimers();
    }
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

  it("floors max_tokens for Hy4 even when the caller asks for 1024", async () => {
    let body: Record<string, unknown> = {};
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      body = JSON.parse((init as { body: string }).body);
      return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), {
        status: 200,
      });
    }) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    await c.chat([{ role: "user", content: "x" }], { maxTokens: 1024 });
    expect(body["model"]).toBe("hy4-preview");
    expect(body["max_tokens"]).toBe(HUNYUAN_REASONING_MAX_TOKEN_FLOOR);
  });

  it("does not floor max_tokens for Hy3", async () => {
    let body: Record<string, unknown> = {};
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      body = JSON.parse((init as { body: string }).body);
      return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), {
        status: 200,
      });
    }) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k", model: "hy3" });
    await c.chat([{ role: "user", content: "x" }], { maxTokens: 1024 });
    expect(body["max_tokens"]).toBe(1024);
  });

  it("reads assistant text from content parts", async () => {
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          choices: [{ message: { content: [{ type: "text", text: "分段答案" }] } }],
        }),
        { status: 200 },
      )) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    await expect(c.chat([{ role: "user", content: "x" }])).resolves.toBe("分段答案");
  });

  it("retries once when Hy4 spends the budget on reasoning_content", async () => {
    const fetchMock = vi.fn(async (_url: unknown, init: unknown) => {
      const body = JSON.parse((init as { body: string }).body) as { max_tokens: number };
      if (body.max_tokens < HUNYUAN_REASONING_MAX_TOKEN_RETRY) {
        return new Response(
          JSON.stringify({
            choices: [
              {
                finish_reason: "length",
                message: { role: "assistant", content: null, reasoning_content: "思考中……" },
              },
            ],
          }),
          { status: 200 },
        );
      }
      return new Response(
        JSON.stringify({
          choices: [{ finish_reason: "stop", message: { role: "assistant", content: "FINAL: 好" } }],
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    await expect(c.chat([{ role: "user", content: "你好" }], { maxTokens: 1024 })).resolves.toBe(
      "FINAL: 好",
    );
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("fails closed on empty assistant content", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content: "" } }] }), {
        status: 200,
      })) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "k" });
    await expect(c.chat([{ role: "user", content: "你好" }])).rejects.toMatchObject({
      code: "INTERNAL_ERROR",
      retriable: true,
    });
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

  it("generateImage calls TokenHub hy-image-v3 and downloads the PNG", async () => {
    const fetchMock = vi.fn(async (url: unknown, init?: unknown) => {
      const href = String(url);
      if (href.includes("wand/hunyuan-image")) {
        expect(href).toMatch(/\/wand\/hunyuan-image\/v3-generation$/);
        const body = JSON.parse((init as { body: string }).body) as { model: string; prompt: string };
        expect(body.model).toBe("hy-image-v3");
        expect(body.prompt).toContain("大厂病封面");
        return new Response(JSON.stringify({ data: [{ url: "https://img.example/cover.png" }] }), {
          status: 200,
        });
      }
      return new Response(Buffer.from("PNGDATA"), {
        status: 200,
        headers: { "Content-Type": "image/png" },
      });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const r = await new HunyuanClient({ apiKey: "k" }).generateImage("大厂病封面");
    expect(r.mimeType).toBe("image/png");
    expect(Buffer.from(r.bytes).toString()).toBe("PNGDATA");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("maps TokenHub 402 to a billing configuration error", async () => {
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          error: {
            message:
              "No free trial quota is available for the service and postpaid billing is not enabled",
          },
        }),
        { status: 402 },
      )) as unknown as typeof fetch;
    await expect(new HunyuanClient({ apiKey: "k" }).generateImage("封面")).rejects.toMatchObject({
      code: "CONFIG_MISSING",
      retriable: false,
      message: expect.stringContaining("后付费"),
    });
  });
});
