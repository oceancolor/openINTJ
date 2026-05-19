import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HunyuanClient, generateMockResponse } from "../src/index.js";

describe("generateMockResponse", () => {
  it("returns greet for hello", () => {
    const r = generateMockResponse([{ role: "user", content: "hello there" }]);
    expect(r).toContain("OpenINTJ");
  });

  it("returns help for 介绍", () => {
    const r = generateMockResponse([{ role: "user", content: "介绍一下你的功能" }]);
    expect(r).toContain("框架");
  });

  it("default for arbitrary input echoes truncated text", () => {
    const r = generateMockResponse([{ role: "user", content: "x".repeat(120) }]);
    expect(r).toContain("...");
  });
});

describe("HunyuanClient (mock mode)", () => {
  it("uses mock when no api key", async () => {
    const c = new HunyuanClient({ apiKey: "" });
    expect(c.isMockMode).toBe(true);
    const r = await c.chat([{ role: "user", content: "你好" }]);
    expect(r).toContain("OpenINTJ");
  });

  it("getStatus reports missing_api_key", () => {
    const c = new HunyuanClient({ apiKey: "" });
    const s = c.getStatus();
    expect(s.status).toBe("missing_api_key");
    expect(s.mode).toBe("mock");
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

  it("degrades to mock on 401 and updates status", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ error: { message: "bad key", code: "invalid_api_key" } }), {
        status: 401,
      })) as unknown as typeof fetch;
    const c = new HunyuanClient({ apiKey: "bad" });
    const r = await c.chat([{ role: "user", content: "你好" }]);
    expect(r).toContain("OpenINTJ");
    const s = c.getStatus();
    expect(s.mode).toBe("unauthorized");
    expect(s.status).toBe("unauthorized");
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
