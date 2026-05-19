import { createHash, randomUUID } from "node:crypto";
import { z } from "zod";
import { LODLevel, type LODLevelType } from "./shader.js";

export const MemoryTypeSchema = z.enum(["short_term", "working", "long_term"]);
export type MemoryType = z.infer<typeof MemoryTypeSchema>;

export const MemoryFragmentSchema = z.object({
  fragmentId: z.string().default(() => randomUUID()),
  content: z.string().default(""),
  summaries: z.record(z.coerce.number(), z.string()).default({}),
  embedding: z.array(z.number()).default([]),
  importance: z.number().min(0).max(1).default(0.5),
  timestamp: z.number().default(() => Date.now() / 1000),
  taskTags: z.array(z.string()).default([]),
  accessCount: z.number().int().nonnegative().default(0),
  lastAccessed: z.number().default(() => Date.now() / 1000),
  metadata: z.record(z.string(), z.unknown()).default({}),
  memoryType: MemoryTypeSchema.default("short_term"),
});

export type MemoryFragment = z.infer<typeof MemoryFragmentSchema>;

export const contentHash = (fragment: Pick<MemoryFragment, "content">): string =>
  createHash("md5").update(fragment.content, "utf8").digest("hex");

export const getContentAtLod = (fragment: MemoryFragment, lod: LODLevelType): string => {
  if (lod === LODLevel.LOD_0) return fragment.content;
  return fragment.summaries[lod] ?? fragment.content;
};

export const estimateTokens = (text: string): number => Math.max(1, Math.floor(text.length / 4));

export const estimateFragmentTokens = (fragment: MemoryFragment, lod: LODLevelType): number =>
  estimateTokens(getContentAtLod(fragment, lod));

/**
 * 时间衰减：half_life_hours 是独立配置（修复 Python v2 把 max_summary_length/10
 * 误用作半衰期的 bug）。返回 importance × exp(-ln2 × age / halfLife)。
 */
export const decayImportance = (
  fragment: Pick<MemoryFragment, "importance" | "timestamp">,
  halfLifeHours = 24,
  nowSeconds = Date.now() / 1000,
): number => {
  const ageHours = Math.max(0, (nowSeconds - fragment.timestamp) / 3600);
  const decay = Math.exp((-Math.LN2 * ageHours) / Math.max(0.0001, halfLifeHours));
  return fragment.importance * decay;
};
