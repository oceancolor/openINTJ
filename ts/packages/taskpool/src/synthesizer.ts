import type { TaoResult } from "@openintj/core";
import { TaskType, type TaskTypeType } from "@openintj/core";
import type { TaskPoolRecoverySummary, TaskRunResult } from "./task-pool.js";

/** 把 TaskPool 各步 TaoResult 合成单次 run 输出（MVD reducer）。 */
export const synthesizeTaskPoolAnswer = (
  poolResult: TaskRunResult<TaoResult>,
  originalQuery: string,
): TaoResult => {
  const ordered = [...poolResult.results.values()];
  const last = ordered[ordered.length - 1];
  const combined = ordered.map((r, i) => `### 步骤 ${i + 1}\n${r.finalAnswer}`).join("\n\n");
  const trajectory = ordered.flatMap((r) => r.trajectory);
  const status =
    poolResult.status === "completed"
      ? ("completed" as const)
      : poolResult.status === "cancelled"
        ? ("failed" as const)
        : ("failed" as const);

  return {
    traceId: last?.traceId ?? poolResult.runId,
    status,
    finalAnswer: combined || last?.finalAnswer || `[taskpool] 未能完成：${originalQuery}`,
    iterations: ordered.reduce((n, r) => n + r.iterations, 0),
    reactTotalSteps: ordered.reduce((n, r) => n + r.reactTotalSteps, 0),
    totalTokensSpent: ordered.reduce((n, r) => n + r.totalTokensSpent, 0),
    durationMs: ordered.reduce((n, r) => n + r.durationMs, 0),
    trajectory,
    taskType: last?.taskType ?? TaskType.PLANNING,
    shaderMode: last?.shaderMode ?? "adaptive",
    metrics: { taskpoolSteps: ordered.length, ...last?.metrics },
    ...(poolResult.status !== "completed"
      ? { failureReason: `taskpool ${poolResult.status}` }
      : {}),
  };
};

export const resolveTaskPoolEnabled = (
  explicit?: boolean,
  env: NodeJS.ProcessEnv = process.env,
): boolean => {
  if (explicit !== undefined) return explicit;
  return env["OPENINTJ_TASK_POOL"] === "1";
};

export interface TaskPoolActivationStatus {
  requested: boolean;
  active: boolean;
  classifierRequired: true;
  classifierEnabled: boolean;
  reason: "disabled" | "classifier_required" | "ready";
  eligibleTaskTypes: readonly ["planning", "analysis"];
  precedence: "taskpool-before-self-consistency";
  persistence: "none" | "sqlite";
  recovery: "unsupported" | TaskPoolRecoveryPolicy;
  recoverySummary?: TaskPoolRecoverySummary;
}

/** Public activation contract: routing requires the classifier even when the pool is constructed. */
export const resolveTaskPoolActivation = (
  requested: boolean,
  classifierEnabled: boolean,
  capabilities: {
    persistence?: TaskPoolActivationStatus["persistence"];
    recovery?: TaskPoolActivationStatus["recovery"];
    recoverySummary?: TaskPoolRecoverySummary;
  } = {},
): TaskPoolActivationStatus => ({
  requested,
  active: requested && classifierEnabled,
  classifierRequired: true,
  classifierEnabled,
  reason: !requested ? "disabled" : classifierEnabled ? "ready" : "classifier_required",
  eligibleTaskTypes: ["planning", "analysis"],
  precedence: "taskpool-before-self-consistency",
  persistence: capabilities.persistence ?? "none",
  recovery: capabilities.recovery ?? "unsupported",
  ...(capabilities.recoverySummary ? { recoverySummary: capabilities.recoverySummary } : {}),
});

export type TaskPoolRecoveryPolicy = "cancel" | "resume";

/**
 * Restart recovery is fail-closed by default. Re-running a task whose external
 * side effect completed just before a crash is not exactly-once safe, so
 * automatic resume requires an explicit opt-in.
 */
export const resolveTaskPoolRecoveryPolicy = (
  explicit?: TaskPoolRecoveryPolicy,
  env: NodeJS.ProcessEnv = process.env,
): TaskPoolRecoveryPolicy => {
  if (explicit !== undefined) return explicit;
  return env["OPENINTJ_TASK_POOL_RECOVERY"] === "resume" ? "resume" : "cancel";
};

/** planning/analysis 等复杂类走 TaskPool（opt-in 时）。 */
export const shouldUseTaskPool = (enabled: boolean, taskType?: TaskTypeType): boolean => {
  if (!enabled || !taskType) return false;
  return taskType === TaskType.PLANNING || taskType === TaskType.ANALYSIS;
};

export type OrchestrationMode = "taskpool" | "self-consistency" | "simple";

/** TaskPool is the explicit complex-task opt-in; self-consistency is fallback. */
export const resolveOrchestrationMode = (
  taskPoolEligible: boolean,
  selfConsistencyEnabled: boolean,
): OrchestrationMode =>
  taskPoolEligible ? "taskpool" : selfConsistencyEnabled ? "self-consistency" : "simple";
