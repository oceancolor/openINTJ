import { z } from "zod";

export const ShaderMode = {
  HIGH_FIDELITY: "high_fidelity",
  LOW_FIDELITY: "low_fidelity",
  HYBRID: "hybrid",
  ADAPTIVE: "adaptive",
} as const;

export type ShaderModeType = (typeof ShaderMode)[keyof typeof ShaderMode];

export const ShaderModeSchema = z.enum([
  ShaderMode.HIGH_FIDELITY,
  ShaderMode.LOW_FIDELITY,
  ShaderMode.HYBRID,
  ShaderMode.ADAPTIVE,
]);

export const LODLevel = {
  LOD_0: 0,
  LOD_1: 1,
  LOD_2: 2,
  LOD_3: 3,
  LOD_4: 4,
} as const;

export type LODLevelType = (typeof LODLevel)[keyof typeof LODLevel];
export const LODLevelSchema = z.union([
  z.literal(0),
  z.literal(1),
  z.literal(2),
  z.literal(3),
  z.literal(4),
]);

export const TaskType = {
  CODE_GENERATION: "code_generation",
  TECHNICAL_WRITING: "technical_writing",
  GENERAL_CHAT: "general_chat",
  QUICK_RESPONSE: "quick_response",
  ANALYSIS: "analysis",
  PLANNING: "planning",
} as const;

export type TaskTypeType = (typeof TaskType)[keyof typeof TaskType];

export const TaskTypeSchema = z.enum([
  TaskType.CODE_GENERATION,
  TaskType.TECHNICAL_WRITING,
  TaskType.GENERAL_CHAT,
  TaskType.QUICK_RESPONSE,
  TaskType.ANALYSIS,
  TaskType.PLANNING,
]);

export const ShaderConfigSchema = z.object({
  mode: ShaderModeSchema.default(ShaderMode.ADAPTIVE),
  targetLod: LODLevelSchema.default(LODLevel.LOD_1),
  maxSummaryLength: z.number().int().positive().default(200),
  importanceThreshold: z.number().min(0).max(1).default(0.3),
  recencyWeight: z.number().min(0).max(1).default(0.4),
  relevanceWeight: z.number().min(0).max(1).default(0.4),
  importanceWeight: z.number().min(0).max(1).default(0.2),
  compactionThreshold: z.number().min(0).max(1).default(0.8),
  maxFragmentsPerQuery: z.number().int().positive().default(10),
  /**
   * 关键修复：v2 Python 把 max_summary_length / 10 误用作半衰期，这里独立成参数。
   * 默认 24h 与 Python framework_core.py:319 的 memory_half_life_hours 对齐。
   */
  recencyHalfLifeHours: z.number().positive().default(24),
});

export type ShaderConfig = z.infer<typeof ShaderConfigSchema>;

export const TASK_SHADER_MAP: Readonly<Record<TaskTypeType, ShaderModeType>> = Object.freeze({
  [TaskType.CODE_GENERATION]: ShaderMode.HIGH_FIDELITY,
  [TaskType.TECHNICAL_WRITING]: ShaderMode.HIGH_FIDELITY,
  [TaskType.GENERAL_CHAT]: ShaderMode.LOW_FIDELITY,
  [TaskType.QUICK_RESPONSE]: ShaderMode.LOW_FIDELITY,
  [TaskType.ANALYSIS]: ShaderMode.HYBRID,
  [TaskType.PLANNING]: ShaderMode.HYBRID,
});

export const getShaderForTask = (task: TaskTypeType): ShaderModeType => TASK_SHADER_MAP[task];

export const getLodForMode = (mode: ShaderModeType, budgetRatio: number): LODLevelType => {
  const ratio = Math.max(0, Math.min(1, budgetRatio));
  switch (mode) {
    case ShaderMode.HIGH_FIDELITY:
      return ratio < 0.6 ? LODLevel.LOD_0 : LODLevel.LOD_1;
    case ShaderMode.LOW_FIDELITY:
      return ratio < 0.9 ? LODLevel.LOD_3 : LODLevel.LOD_4;
    case ShaderMode.HYBRID:
      if (ratio < 0.5) return LODLevel.LOD_1;
      if (ratio < 0.8) return LODLevel.LOD_2;
      return LODLevel.LOD_3;
    case ShaderMode.ADAPTIVE:
    default:
      if (ratio < 0.3) return LODLevel.LOD_0;
      if (ratio < 0.5) return LODLevel.LOD_1;
      if (ratio < 0.7) return LODLevel.LOD_2;
      if (ratio < 0.9) return LODLevel.LOD_3;
      return LODLevel.LOD_4;
  }
};
