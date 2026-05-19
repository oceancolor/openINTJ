import { z } from "zod";

/**
 * 向量存储行（持久化单元）。
 * 与 MemoryFragment 一一对应（fragmentId 作为 PK）。
 */
export const VectorRowSchema = z.object({
  fragmentId: z.string(),
  content: z.string(),
  embedding: z.array(z.number()),
  memoryType: z.enum(["short_term", "working", "long_term"]),
  importance: z.number().min(0).max(1),
  taskTags: z.array(z.string()).default([]),
  contentHash: z.string(),
  timestamp: z.number(),
  accessCount: z.number().int().nonnegative().default(0),
  lastAccessed: z.number().nonnegative().default(0),
  metadataJson: z.string().default("{}"),
  summariesJson: z.string().default("{}"),
});
export type VectorRow = z.infer<typeof VectorRowSchema>;

export interface VectorSearchResult {
  row: VectorRow;
  /** 相似度分数；具体含义由实现决定（cosine/L2）。值域可能不同。 */
  score: number;
  /** distance（如果实现返回的是距离），否则等于 1-score。 */
  distance: number;
}

export interface VectorSearchOpts {
  topK: number;
  /** 仅返回某些 memoryType 的结果。 */
  memoryTypes?: ReadonlyArray<"short_term" | "working" | "long_term">;
  /** 仅返回包含某些 taskTags 之一的结果。 */
  taskTags?: readonly string[];
  /** 最小 importance 过滤。 */
  minImportance?: number;
}

export interface VectorStore {
  readonly name: string;
  /** 存储维度。在第一次 upsert 后冻结。 */
  readonly dimension: number;
  /** 准备阶段：建表 / 加载已有数据。 */
  init(): Promise<void>;
  upsert(rows: readonly VectorRow[]): Promise<void>;
  delete(fragmentIds: readonly string[]): Promise<number>;
  search(queryEmbedding: readonly number[], opts: VectorSearchOpts): Promise<VectorSearchResult[]>;
  /** 全表扫描（重启时 hydrate 用）。 */
  scanAll(): Promise<VectorRow[]>;
  count(): Promise<number>;
  close(): Promise<void>;
}
