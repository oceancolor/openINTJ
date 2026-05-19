import { randomUUID } from "node:crypto";
import { TaskType, TaskTypeSchema } from "@openintj/core";
import { z } from "zod";

export const IntentSchema = z.enum([
  "create",
  "modify",
  "delete",
  "query",
  "execute",
  "plan",
  "general",
]);
export type Intent = z.infer<typeof IntentSchema>;

export const ParsedGoalSchema = z.object({
  goalId: z.string().default(() => randomUUID()),
  rawInput: z.string().default(""),
  taskType: TaskTypeSchema.default(TaskType.GENERAL_CHAT),
  intent: IntentSchema.default("general"),
  entities: z.array(z.string()).default([]),
  constraints: z.record(z.string(), z.unknown()).default({}),
  priority: z.number().int().min(1).max(10).default(5),
  createdAt: z.number().default(() => Date.now() / 1000),
});
export type ParsedGoal = z.infer<typeof ParsedGoalSchema>;

export const PlanStepStatusSchema = z.enum([
  "pending",
  "running",
  "completed",
  "failed",
  "skipped",
]);
export type PlanStepStatus = z.infer<typeof PlanStepStatusSchema>;

export const PlanStepSchema = z.object({
  stepId: z.string().default(() => randomUUID().slice(0, 8)),
  action: z.string().default(""),
  description: z.string().default(""),
  params: z.record(z.string(), z.unknown()).default({}),
  dependencies: z.array(z.string()).default([]),
  estimatedTokens: z.number().int().nonnegative().default(0),
  status: PlanStepStatusSchema.default("pending"),
});
export type PlanStep = z.infer<typeof PlanStepSchema>;
