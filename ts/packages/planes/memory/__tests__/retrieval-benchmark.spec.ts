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
  benchmarkRetrieval,
  formatBenchmarkRow,
} from "../src/eval/retrieval-benchmark.js";

describe("retrieval benchmark (pluggable embedder)", () => {
  it("simple@dim64 达到召回质量基线", async () => {
    const r = await benchmarkRetrieval(new SimpleEmbedder(64));
    console.log(formatBenchmarkRow(r));
    expect(r.dimension).toBe(64);
    expect(r.summary.ndcg).toBeGreaterThanOrEqual(0.55);
    expect(r.summary.recall).toBeGreaterThanOrEqual(0.55);
    expect(r.summary.mrr).toBeGreaterThanOrEqual(0.6);
  });

  // 三方对比：默认 skip（避免 CI 下载模型 / 连本地服务）。
  const runCompare = process.env["RUN_EMBED_COMPARE"] === "1";
  it.runIf(runCompare)(
    "三方对比 simple vs xenova vs ollama（RUN_EMBED_COMPARE=1）",
    async () => {
      const results: BenchmarkResult[] = [];
      results.push(await benchmarkRetrieval(new SimpleEmbedder(64)));

      try {
        const { XenovaEmbedder } = (await import("@openintj/embed-xenova")) as {
          XenovaEmbedder: new () => import("@openintj/core").EmbeddingProvider;
        };
        results.push(await benchmarkRetrieval(new XenovaEmbedder()));
      } catch (e) {
        console.warn(`[retrieval-benchmark] skip xenova: ${(e as Error).message}`);
      }

      try {
        const { OllamaEmbedder } = (await import("@openintj/embed-ollama")) as {
          OllamaEmbedder: new () => import("@openintj/core").EmbeddingProvider;
        };
        results.push(await benchmarkRetrieval(new OllamaEmbedder()));
      } catch (e) {
        console.warn(`[retrieval-benchmark] skip ollama: ${(e as Error).message}`);
      }

      for (const r of results) console.log(formatBenchmarkRow(r));
      // 至少 simple 跑出来了；对比数字进 CI 日志供人工判读。
      expect(results.length).toBeGreaterThanOrEqual(1);
    },
    120_000,
  );
});
