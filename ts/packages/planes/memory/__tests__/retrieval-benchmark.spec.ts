/**
 * 检索召回质量基准：
 *  1. simple@dim64 始终跑，做回归守护（"别比现在更差"）。
 *  2. 三方对比（simple vs xenova vs ollama）：默认只跑 simple；设 RUN_EMBED_COMPARE=1 时
 *     额外加载 xenova / ollama（需各自 peer dep / 本地服务），打印同一套 cases 的评分表对比。
 *
 * 复用 `benchmarkRetrieval`（可插拔 embedder + 维度自动探测 + 全异步）。
 */
import { SimpleEmbedder } from "@openintj/core";
import { describe, expect, it } from "vitest";
import {
  type BenchmarkResult,
  benchmarkEmbedderCosine,
  benchmarkRetrieval,
  formatBenchmarkRow,
} from "../src/eval/retrieval-benchmark.js";

describe("retrieval benchmark (pluggable embedder)", () => {
  it("simple@dim64 达到召回质量基线（MemoryRetriever 路径）", async () => {
    const r = await benchmarkRetrieval(new SimpleEmbedder(64));
    console.log(formatBenchmarkRow(r));
    expect(r.dimension).toBe(64);
    expect(r.summary.ndcg).toBeGreaterThanOrEqual(0.55);
    expect(r.summary.recall).toBeGreaterThanOrEqual(0.55);
    expect(r.summary.mrr).toBeGreaterThanOrEqual(0.6);
  });

  it("simple 纯 cosine 基准：维度不改变质量（词袋哈希无真语义）", async () => {
    const d32 = await benchmarkEmbedderCosine(new SimpleEmbedder(32));
    const d256 = await benchmarkEmbedderCosine(new SimpleEmbedder(256));
    console.log(`[cosine] ${formatBenchmarkRow(d32)}`);
    console.log(`[cosine] ${formatBenchmarkRow(d256)}`);
    // 哈希词袋：维度只影响碰撞概率，不引入语义 → nDCG 基本持平。
    expect(Math.abs(d32.summary.ndcg - d256.summary.ndcg)).toBeLessThan(0.15);
  });

  // 三方对比：默认 skip（避免 CI 下载模型 / 连本地服务）。
  const runCompare = process.env["RUN_EMBED_COMPARE"] === "1";
  it.runIf(runCompare)(
    "三方对比 simple vs xenova vs ollama（RUN_EMBED_COMPARE=1）",
    async () => {
      type Ctor = new () => import("@openintj/core").EmbeddingProvider;
      const embedders: Array<{
        label: string;
        make: () => import("@openintj/core").EmbeddingProvider;
      }> = [{ label: "simple", make: () => new SimpleEmbedder(64) }];

      try {
        const { XenovaEmbedder } = (await import("@openintj/embed-xenova")) as {
          XenovaEmbedder: Ctor;
        };
        embedders.push({ label: "xenova", make: () => new XenovaEmbedder() });
      } catch (e) {
        console.warn(`[retrieval-benchmark] skip xenova: ${(e as Error).message}`);
      }

      try {
        const { OllamaEmbedder } = (await import("@openintj/embed-ollama")) as {
          OllamaEmbedder: Ctor;
        };
        embedders.push({ label: "ollama", make: () => new OllamaEmbedder() });
      } catch (e) {
        console.warn(`[retrieval-benchmark] skip ollama: ${(e as Error).message}`);
      }

      // 两套基准并列：MemoryRetriever 路径（产品实际路径）+ 纯 cosine（隔离 embedder 语义）。
      const memResults: BenchmarkResult[] = [];
      const cosResults: BenchmarkResult[] = [];
      for (const e of embedders) {
        try {
          memResults.push(await benchmarkRetrieval(e.make()));
          cosResults.push(await benchmarkEmbedderCosine(e.make()));
        } catch (err) {
          console.warn(`[retrieval-benchmark] ${e.label} failed: ${(err as Error).message}`);
        }
      }
      console.log("--- MemoryRetriever 路径 ---");
      for (const r of memResults) console.log(formatBenchmarkRow(r));
      console.log("--- 纯 cosine（隔离 embedder 语义）---");
      for (const r of cosResults) console.log(`[cosine] ${formatBenchmarkRow(r)}`);
      // 至少 simple 跑出来了；对比数字进 CI 日志供人工判读。
      expect(memResults.length).toBeGreaterThanOrEqual(1);
    },
    120_000,
  );
});
