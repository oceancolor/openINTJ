import type { PlanGraph } from "@openintj/plane-control";

/** TaskGraph 节点（MVD：来自 ControlPlane 模板 PlanGraph，非 LLM 动态拆图）。 */
export interface TaskGraphNode {
  id: string;
  deps: readonly string[];
  action: string;
  description: string;
  /** Overrides the pool watchdog for this node. */
  timeoutMs?: number;
}

export interface TaskGraph {
  planId: string;
  goalIntent: string;
  nodes: readonly TaskGraphNode[];
}

/** 确定性 PlanGraph → TaskGraph 适配器（RFC-007）。 */
export const planGraphToTaskGraph = (plan: PlanGraph): TaskGraph => ({
  planId: plan.planId,
  goalIntent: plan.goal.intent,
  nodes: plan.steps.map((s) => ({
    id: s.stepId,
    deps: [...s.dependencies],
    action: s.action,
    description: s.description,
  })),
});
