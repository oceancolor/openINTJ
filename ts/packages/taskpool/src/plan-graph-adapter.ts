import type { PlanGraph } from "@openintj/plane-control";

/** TaskGraph 节点（MVD：来自 ControlPlane 模板 PlanGraph，非 LLM 动态拆图）。 */
export interface TaskGraphNode {
  id: string;
  deps: readonly string[];
  action: string;
  description: string;
  /** P1-4：透传 PlanStep.params，供 ToolHub.call(action, params)。 */
  params?: Record<string, unknown>;
  /** Overrides the pool watchdog for this node. */
  timeoutMs?: number;
}

export interface TaskGraph {
  planId: string;
  goalIntent: string;
  /**
   * Original user input required to reconstruct worker prompts after restart.
   * Optional only for snapshots created before restart recovery was wired.
   */
  goalInput?: string;
  nodes: readonly TaskGraphNode[];
}

/** 确定性 PlanGraph → TaskGraph 适配器（RFC-007）。 */
export const planGraphToTaskGraph = (plan: PlanGraph): TaskGraph => ({
  planId: plan.planId,
  goalIntent: plan.goal.intent,
  goalInput: plan.goal.rawInput,
  nodes: plan.steps.map((s) => ({
    id: s.stepId,
    deps: [...s.dependencies],
    action: s.action,
    description: s.description,
    ...(s.params && Object.keys(s.params).length > 0 ? { params: { ...s.params } } : {}),
  })),
});
