import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OllamaClient } from "../src/index.js";

const originalFetch = globalThis.fetch;

describe("OllamaClient", () => {
  beforeEach(() => {
    globalThis.fetch = originalFetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("calls /api/chat with model and messages", async () => {
    const fetchMock = vi.fn(async (url: unknown, init: unknown) => {
      expect(String(url)).toMatch(/\/api\/chat$/);
      const body = JSON.parse((init as { body: string }).body);
      expect(body.model).toBe("qwen2.5:7b");
      expect(body.stream).toBe(false);
      return new Response(
        JSON.stringify({
          message: { role: "assistant", content: "ollama says hi" },
          done: true,
        }),
        { status: 200 },
      );
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const c = new OllamaClient({ baseUrl: "http://localhost:11434" });
    const r = await c.chat([{ role: "user", content: "hi" }]);
    expect(r).toBe("ollama says hi");
  });

  it("falls back to mock when network errors", async () => {
    globalThis.fetch = (async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    const c = new OllamaClient({ baseUrl: "http://localhost:11434" });
    const r = await c.chat([{ role: "user", content: "你好" }]);
    expect(r).toContain("OpenINTJ");
    expect(c.getStatus().mode).toBe("mock");
  });

  it("vision chat puts image in images array, not in content", async () => {
    let captured: { messages: Array<{ role: string; images?: string[] }> } = {
      messages: [],
    };
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      captured = JSON.parse((init as { body: string }).body);
      return new Response(JSON.stringify({ message: { content: "I see picture" }, done: true }), {
        status: 200,
      });
    }) as unknown as typeof fetch;
    const c = new OllamaClient();
    const r = await c.visionChat([{ role: "user", content: "describe" }], {
      base64: "ZZZ",
      mimeType: "image/png",
    });
    expect(r).toBe("I see picture");
    expect(captured.messages.at(-1)?.images).toEqual(["ZZZ"]);
  });

  it("getStatus returns connected after success and degraded on failure", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ message: { content: "ok" }, done: true }), {
        status: 200,
      })) as unknown as typeof fetch;
    const c = new OllamaClient();
    await c.chat([{ role: "user", content: "x" }]);
    expect(c.getStatus().status).toBe("connected");

    globalThis.fetch = (async () => {
      throw new Error("fail");
    }) as unknown as typeof fetch;
    await c.chat([{ role: "user", content: "y" }]);
    expect(c.getStatus().status).toBe("degraded");
  });

  it("healthCheck returns true on /api/tags 200", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ models: [] }), {
        status: 200,
      })) as unknown as typeof fetch;
    const c = new OllamaClient();
    expect(await c.healthCheck()).toBe(true);
  });
});
