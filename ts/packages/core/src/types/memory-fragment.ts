import { createHash, randomUUID } from "node:crypto";
import { z } from "zod";
import { LODLevel, type LODLevelType } from "./shader.js";

export const MemoryTypeSchema = z.enum(["short_term", "working", "long_term"]);
export type MemoryType = z.infer<typeof MemoryTypeSchema>;

export const MemoryKindSchema = z.enum([
  "fact",
  "preference",
  "intention",
  "episode",
  "state_reference",
  "procedure",
  "note",
]);
export type MemoryKind = z.infer<typeof MemoryKindSchema>;

/** 断言来源：用户显式断言优先于推断；被引用陈述不等于本人断言（F08）。 */
export const MemoryAssertionSchema = z.enum(["user", "assistant", "inferred", "quoted", "tool"]);
export type MemoryAssertion = z.infer<typeof MemoryAssertionSchema>;

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
  // ---- F08 生命周期（默认值保证旧数据可解析） ----
  kind: MemoryKindSchema.default("note"),
  assertion: MemoryAssertionSchema.default("assistant"),
  /** 有效期（秒 epoch）；到期后在所有召回入口不可见。fact/preference 默认无 TTL。 */
  validUntil: z.number().optional(),
  /** 同一主张的规范键；纠正时按 (scope, canonicalKey) supersede 旧值。 */
  canonicalKey: z.string().optional(),
  /** 被哪条 fragment 取代；非空即已失效，不可召回。 */
  supersededBy: z.string().optional(),
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
