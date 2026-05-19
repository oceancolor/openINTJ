import { afterEach, describe, expect, it, vi } from "vitest";
import { OllamaEmbedder, loadOllamaEmbedderConfigFromEnv } from "../src/index.js";

const mockFetch = (resp: Partial<Response>): Response => resp as unknown as Response;

describe("OllamaEmbedder", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("calls /api/embeddings and returns embedding", async () => {
    const e = new OllamaEmbedder({ model: "nomic-embed-text" });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockFetch({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({
          embedding: new Array(768).fill(0).map((_, i) => i / 768),
        }),
      }),
    );
    const v = await e.embed("hello");
    expect(v).toHaveLength(768);
    expect(e.dimension).toBe(768);
  });

  it("infers dimension from first call", async () => {
    const e = new OllamaEmbedder();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockFetch({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({ embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }),
      }),
    );
    expect(e.dimension).toBe(0);
    await e.embed("x");
    expect(e.dimension).toBe(5);
  });

  it("throws on dimension mismatch in subsequent calls", async () => {
    const e = new OllamaEmbedder({ dimension: 3 });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockFetch({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({ embedding: [1, 2, 3, 4] }),
      }),
    );
    await expect(e.embed("x")).rejects.toThrow(/dimension mismatch/);
  });

  it("throws on non-200 response", async () => {
    const e = new OllamaEmbedder();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockFetch({
        ok: false,
        status: 500,
        statusText: "Server Error",
        text: async () => "boom",
      }),
    );
    await expect(e.embed("x")).rejects.toThrow(/500/);
  });

  it("embedBatch processes serially", async () => {
    const e = new OllamaEmbedder({ dimension: 3 });
    let n = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      n++;
      return mockFetch({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({ embedding: [n, n, n] }),
      });
    });
    const out = await e.embedBatch(["a", "b", "c"]);
    expect(out).toEqual([
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3],
    ]);
  });

  it("healthCheck returns boolean", async () => {
    const e = new OllamaEmbedder();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(mockFetch({ ok: true } as Response));
    expect(await e.healthCheck()).toBe(true);
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("network"));
    expect(await e.healthCheck()).toBe(false);
  });

  it("loadFromEnv parses env vars", () => {
    const cfg = loadOllamaEmbedderConfigFromEnv({
      OLLAMA_EMBED_ENDPOINT: "http://1.2.3.4:99",
      OLLAMA_EMBED_MODEL: "nomic-embed-text",
      OLLAMA_EMBED_DIMENSION: "768",
    });
    expect(cfg.endpoint).toBe("http://1.2.3.4:99");
    expect(cfg.model).toBe("nomic-embed-text");
    expect(cfg.dimension).toBe(768);
  });
});
