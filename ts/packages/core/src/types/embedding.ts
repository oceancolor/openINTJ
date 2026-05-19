import { createHash } from "node:crypto";

/**
 * 嵌入提供方接口。
 * - sync 实现（如 simple/hash）可用 number[] 同步返回
 * - async 实现（如 transformers/ollama）返回 Promise<number[]>
 *
 * 调用方一律按 await 处理（Promise.resolve 包装同步结果）。
 */
export interface EmbeddingProvider {
  readonly name: string;
  readonly dimension: number;
  embed(text: string): number[] | Promise<number[]>;
  embedBatch(texts: readonly string[]): number[][] | Promise<number[][]>;
}

const computeSimpleEmbedding = (text: string, dim: number): number[] => {
  const h = createHash("sha256").update(text, "utf8").digest("hex");
  const values: number[] = new Array(dim);
  for (let i = 0; i < dim; i++) {
    const start = (i * 2) % h.length;
    const slice = h.slice(start, start + 2);
    const byteVal = Number.parseInt(slice, 16);
    values[i] = (byteVal / 255) * 2 - 1;
  }
  let norm = 0;
  for (const v of values) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < dim; i++) {
      values[i] = (values[i] ?? 0) / norm;
    }
  }
  return values;
};

/**
 * 默认 simple embedder（SHA-256 → 64 维 L2 归一化）。
 * 用于演示和 CI；不用于生产。
 */
export class SimpleEmbedder implements EmbeddingProvider {
  readonly name = "simple-sha256";
  readonly dimension: number;

  constructor(dim = 64) {
    this.dimension = dim;
  }

  embed(text: string): number[] {
    return computeSimpleEmbedding(text, this.dimension);
  }

  embedBatch(texts: readonly string[]): number[][] {
    return texts.map((t) => computeSimpleEmbedding(t, this.dimension));
  }
}

/** 共享余弦相似度（与 simpleEmbedding 兼容；任何 EmbeddingProvider 都可用）。 */
export const cosineSimilarity = (a: readonly number[], b: readonly number[]): number => {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0;
    const bi = b[i] ?? 0;
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

/** 旧 API 兼容（@deprecated 用 SimpleEmbedder）。 */
export const simpleEmbedding = (text: string, dim = 64): number[] =>
  computeSimpleEmbedding(text, dim);
