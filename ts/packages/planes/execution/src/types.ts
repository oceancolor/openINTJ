import { randomUUID } from "node:crypto";
import { z } from "zod";

export const StepStateSchema = z.enum([
  "pending",
  "ready",
  "running",
  "completed",
  "failed",
  "skipped",
  "waiting_approval",
]);
export type StepState = z.infer<typeof StepStateSchema>;

export const ExecutionModeSchema = z.enum([
  "sequential",
  "parallel",
  "conditional",
  "human_approval",
]);
export type ExecutionMode = z.infer<typeof ExecutionModeSchema>;

export const StepSchema = z.object({
  stepId: z.string().default(() => randomUUID()),
  action: z.string().min(1),
  params: z.record(z.string(), z.unknown()).default({}),
  state: StepStateSchema.default("pending"),
  result: z.unknown().optional(),
  error: z.string().optional(),
  startedAt: z.number().default(0),
  finishedAt: z.number().default(0),
  retryCount: z.number().int().nonnegative().default(0),
  maxRetries: z.number().int().nonnegative().default(3),
  /** 步骤超时（毫秒）。 */
  timeoutMs: z.number().int().positive().default(30_000),
});

export type Step = z.infer<typeof StepSchema>;

export const stepDurationMs = (step: Step): number =>
  step.startedAt > 0 && step.finishedAt > 0
    ? Math.max(0, (step.finishedAt - step.startedAt) * 1000)
    : 0;

export interface ExecutionError {
  stepId: string;
  error: string;
  retryCount: number;
  errorCode?: string;
}

export interface ExecutionResult {
  success: boolean;
  mode: ExecutionMode;
  finishedSteps: string[];
  failedSteps: string[];
  skippedSteps: string[];
  errors: ExecutionError[];
  totalDurationMs: number;
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  recoveryTimeoutMs: number;
}

export const DEFAULT_BREAKER: CircuitBreakerConfig = {
  failureThreshold: 3,
  recoveryTimeoutMs: 60_000,
};
