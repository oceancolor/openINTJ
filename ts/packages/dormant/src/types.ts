import { z } from "zod";

/** 蛰伏记忆的"被动层"事件：用户消息、agent 决策、tool 调用结果。 */
export const PassiveEventSchema = z.object({
  eventId: z.string(),
  ts: z.number(),
  source: z.enum(["user", "agent", "tool", "policy"]),
  text: z.string(),
  metadata: z.record(z.string(), z.unknown()).default({}),
});
export type PassiveEvent = z.infer<typeof PassiveEventSchema>;

/** 抽取出的 pattern（候选用户偏好/口头禅/工作习惯）。 */
export const DormantPatternSchema = z.object({
  patternId: z.string(),
  /** 描述（人类可读）。 */
  description: z.string(),
  /** 由哪些事件聚合而来。 */
  evidenceIds: z.array(z.string()),
  /** 出现次数。 */
  frequency: z.number().int().nonnegative(),
  /** 置信度 0-1。 */
  confidence: z.number().min(0).max(1),
  /** 蕴含的语义类别（hint）。 */
  category: z.enum(["preference", "phrase", "habit", "context", "other"]).default("other"),
  /** 提取时间。 */
  ts: z.number(),
});
export type DormantPattern = z.infer<typeof DormantPatternSchema>;

/** 内化为 PersonaConfig 时的待审批条目。 */
export const InternalizationProposalSchema = z.object({
  proposalId: z.string(),
  pattern: DormantPatternSchema,
  /** 拟写入 PersonaConfig 的字段路径（如 "preferences.tea"）。 */
  targetField: z.string(),
  /** 拟写入的值。 */
  value: z.unknown(),
  /**
   * 状态。
   * - pending：待审批
   * - applied：已批准并写入 PersonaConfig
   * - rejected：已拒绝（从未写入）
   * - revoked：曾 applied，后被用户撤销（已从 PersonaConfig 删除）
   * - approved：保留位（历史兼容）
   */
  status: z.enum(["pending", "approved", "rejected", "applied", "revoked"]).default("pending"),
  /** 创建时间。 */
  ts: z.number(),
  /** 用户决策时间。 */
  decidedAt: z.number().optional(),
});
export type InternalizationProposal = z.infer<typeof InternalizationProposalSchema>;

/** 用户人格配置 —— 内化的最终目的地。 */
export const PersonaConfigSchema = z.object({
  preferences: z.record(z.string(), z.unknown()).default({}),
  phrases: z.record(z.string(), z.string()).default({}),
  habits: z.record(z.string(), z.unknown()).default({}),
  context: z.record(z.string(), z.unknown()).default({}),
  /** 元信息（修改时间等）。 */
  meta: z
    .object({
      lastUpdated: z.number().default(() => Date.now()),
      version: z.number().int().nonnegative().default(0),
    })
    .default({}),
});
export type PersonaConfig = z.infer<typeof PersonaConfigSchema>;
