import { describe, expect, it, vi } from "vitest";

// 我们 mock @xenova/transformers 因为完整模型下载在 CI 不可行
vi.mock("@xenova/transformers", () => {
  const fakePipe = (
    text: string | string[],
    _opts?: unknown,
  ): Promise<{ data: number[]; dims: number[] }> => {
    const t = Array.isArray(text) ? text.join(" ") : text;
    // fake 384 维输出，基于 char codes 简单生成
    const data = new Array(384).fill(0).map((_, i) => Math.sin(i * 0.1 + t.charCodeAt(0)) * 0.1);
    return Promise.resolve({ data, dims: [1, 384] });
  };
  return {
    env: { cacheDir: undefined },
    pipeline: vi.fn(async () => fakePipe),
  };
});

import { XenovaEmbedder, loadXenovaEmbedderConfigFromEnv } from "../src/index.js";

describe("XenovaEmbedder", () => {
  it("loads pipeline and returns 384-dim embedding", async () => {
    const e = new XenovaEmbedder();
    const v = await e.embed("hello world");
    expect(v).toHaveLength(384);
    expect(e.dimension).toBe(384);
    expect(e.name).toBe("xenova:Xenova/all-MiniLM-L6-v2");
  });

  it("embedBatch returns matrix of shape [N, 384]", async () => {
    const e = new XenovaEmbedder();
    const out = await e.embedBatch(["a", "b", "c"]);
    expect(out).toHaveLength(3);
    for (const row of out) expect(row).toHaveLength(384);
  });

  it("warmup pre-loads the pipeline", async () => {
    const e = new XenovaEmbedder();
    await expect(e.warmup()).resolves.toBeUndefined();
  });

  it("loadFromEnv parses config", () => {
    const cfg = loadXenovaEmbedderConfigFromEnv({
      XENOVA_MODEL: "Xenova/bge-small-en-v1.5",
      XENOVA_POOLING: "cls",
      XENOVA_NORMALIZE: "false",
      XENOVA_CACHE_DIR: "C:/tmp/cache",
    });
    expect(cfg.model).toBe("Xenova/bge-small-en-v1.5");
    expect(cfg.pooling).toBe("cls");
    expect(cfg.normalize).toBe(false);
    expect(cfg.cacheDir).toBe("C:/tmp/cache");
  });
});
