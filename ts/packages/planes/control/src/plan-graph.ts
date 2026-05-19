import { randomUUID } from "node:crypto";
import { AgentError, ErrorCode } from "@openintj/core";
import { type ParsedGoal, type PlanStep, PlanStepSchema, type PlanStepStatus } from "./types.js";

export class PlanGraph {
  readonly planId: string;
  readonly goal: ParsedGoal;
  readonly createdAt: number;
  readonly steps: PlanStep[];

  constructor(opts: {
    planId?: string;
    goal: ParsedGoal;
    steps: PlanStep[];
    createdAt?: number;
  }) {
    this.planId = opts.planId ?? randomUUID();
    this.goal = opts.goal;
    this.steps = opts.steps;
    this.createdAt = opts.createdAt ?? Date.now() / 1000;
    this.assertNoCycle();
  }

  get totalSteps(): number {
    return this.steps.length;
  }

  get completedSteps(): number {
    return this.steps.filter((s) => s.status === "completed").length;
  }

  get progress(): number {
    if (this.steps.length === 0) return 0;
    return this.completedSteps / this.totalSteps;
  }

  /** 返回所有依赖已 completed 的 pending 步骤。 */
  getReadySteps(): PlanStep[] {
    const completedIds = new Set(
      this.steps.filter((s) => s.status === "completed").map((s) => s.stepId),
    );
    return this.steps.filter(
      (s) => s.status === "pending" && s.dependencies.every((d) => completedIds.has(d)),
    );
  }

  markStep(stepId: string, status: PlanStepStatus): void {
    const step = this.steps.find((s) => s.stepId === stepId);
    if (step) step.status = status;
  }

  /** 拓扑排序（用于顺序执行 fallback）。 */
  topoOrder(): PlanStep[] {
    const result: PlanStep[] = [];
    const visited = new Set<string>();
    const visiting = new Set<string>();
    const byId = new Map(this.steps.map((s) => [s.stepId, s]));

    const visit = (s: PlanStep): void => {
      if (visited.has(s.stepId)) return;
      if (visiting.has(s.stepId)) {
        throw new AgentError({
          code: ErrorCode.VALIDATION_ERROR,
          message: `依赖图存在循环: ${s.stepId}`,
        });
      }
      visiting.add(s.stepId);
      for (const dep of s.dependencies) {
        const d = byId.get(dep);
        if (d) visit(d);
      }
      visiting.delete(s.stepId);
      visited.add(s.stepId);
      result.push(s);
    };

    for (const s of this.steps) visit(s);
    return result;
  }

  private assertNoCycle(): void {
    this.topoOrder();
  }
}

export const buildPlanStep = (input: Partial<PlanStep>): PlanStep => PlanStepSchema.parse(input);
