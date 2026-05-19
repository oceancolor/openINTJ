/**
 * Re-export embedding primitives from @openintj/core.
 *
 * 历史版本将 simpleEmbedding/cosineSimilarity 定义在此文件，现迁移至 core。
 * 保留 re-export 以维持向后兼容（外部代码 import {...} from "@openintj/plane-memory"）。
 */
export {
  cosineSimilarity,
  simpleEmbedding,
  SimpleEmbedder,
} from "@openintj/core";
export type { EmbeddingProvider } from "@openintj/core";
