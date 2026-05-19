import type { EmbeddingProvider } from "@openintj/core";
import { z } from "zod";

export const XenovaEmbedderConfigSchema = z.object({
  /**
   * 模型 ID（HuggingFace 格式）。默认 sentence-transformers/all-MiniLM-L6-v2 (384 维)。
   * 可选：Xenova/bge-small-en-v1.5 (384), Xenova/multilingual-e5-small (384)
   */
  model: z.string().default("Xenova/all-MiniLM-L6-v2"),
  /** 平均池化（推荐 mean）或取 CLS。 */
  pooling: z.enum(["mean", "cls"]).default("mean"),
  /** 是否对输出做 L2 归一化（推荐 true，方便 cosineSimilarity）。 */
  normalize: z.boolean().default(true),
  /** 模型缓存目录（默认 transformers 自定义路径）。 */
  cacheDir: z.string().optional(),
});
export type XenovaEmbedderConfig = z.infer<typeof XenovaEmbedderConfigSchema>;

type FeatureExtractionPipeline = (
  text: string | string[],
  opts?: { pooling?: "mean" | "cls"; normalize?: boolean },
) => Promise<{ data: Float32Array | number[]; dims: number[] }>;

interface XenovaModule {
  pipeline: (task: string, model: string) => Promise<FeatureExtractionPipeline>;
  env?: { cacheDir?: string };
}

/**
 * XenovaEmbedder —— 使用 @xenova/transformers 在本地（CPU/WebGPU）做嵌入。
 *
 * 注意：
 * - 首次使用会下载模型权重（~80MB-300MB），缓存在 ~/.cache/huggingface 或 cacheDir。
 * - @xenova/transformers 是 peerDependency，需要用户显式安装：
 *   `pnpm add @xenova/transformers`
 *
 * 使用：
 * ```ts
 * const embedder = new XenovaEmbedder();
 * await embedder.warmup();      // 可选预热
 * const v = await embedder.embed("hello");
 * ```
 */
export class XenovaEmbedder implements EmbeddingProvider {
  readonly name: string;
  readonly config: XenovaEmbedderConfig;
  private _dimension = 0;
  private pipelinePromise?: Promise<FeatureExtractionPipeline>;

  constructor(config: Partial<XenovaEmbedderConfig> = {}) {
    this.config = XenovaEmbedderConfigSchema.parse(config);
    this.name = `xenova:${this.config.model}`;
  }

  get dimension(): number {
    return this._dimension;
  }

  private async getPipeline(): Promise<FeatureExtractionPipeline> {
    if (!this.pipelinePromise) {
      this.pipelinePromise = (async () => {
        const mod = (await import("@xenova/transformers").catch((e) => {
          throw new Error(
            `XenovaEmbedder: failed to load @xenova/transformers (install it as peer dep). Cause: ${
              (e as Error).message
            }`,
          );
        })) as XenovaModule;
        if (this.config.cacheDir && mod.env) {
          mod.env.cacheDir = this.config.cacheDir;
        }
        return mod.pipeline("feature-extraction", this.config.model);
      })();
    }
    return this.pipelinePromise;
  }

  async warmup(): Promise<void> {
    await this.getPipeline();
  }

  async embed(text: string): Promise<number[]> {
    const pipe = await this.getPipeline();
    const result = await pipe(text, {
      pooling: this.config.pooling,
      normalize: this.config.normalize,
    });
    const arr = Array.from(result.data);
    if (this._dimension === 0) this._dimension = arr.length;
    return arr;
  }

  async embedBatch(texts: readonly string[]): Promise<number[][]> {
    const pipe = await this.getPipeline();
    const out: number[][] = [];
    for (const t of texts) {
      const r = await pipe(t, {
        pooling: this.config.pooling,
        normalize: this.config.normalize,
      });
      const arr = Array.from(r.data);
      if (this._dimension === 0) this._dimension = arr.length;
      out.push(arr);
    }
    return out;
  }
}

export const loadXenovaEmbedderConfigFromEnv = (
  env: Record<string, string | undefined> = process.env,
): XenovaEmbedderConfig => {
  const cfg: Partial<XenovaEmbedderConfig> = {};
  if (env.XENOVA_MODEL) cfg.model = env.XENOVA_MODEL;
  if (env.XENOVA_POOLING === "mean" || env.XENOVA_POOLING === "cls") {
    cfg.pooling = env.XENOVA_POOLING;
  }
  if (env.XENOVA_NORMALIZE !== undefined) {
    cfg.normalize = env.XENOVA_NORMALIZE !== "false";
  }
  if (env.XENOVA_CACHE_DIR) cfg.cacheDir = env.XENOVA_CACHE_DIR;
  return XenovaEmbedderConfigSchema.parse(cfg);
};
