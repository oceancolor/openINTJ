import { PlanGraph, buildPlanStep } from "./plan-graph.js";
import type { Intent, ParsedGoal, PlanStep } from "./types.js";

type IntentTemplate = (goal: ParsedGoal) => PlanStep[];

const DEFAULT_TEMPLATES: Record<Intent, IntentTemplate> = {
  create: () => [
    buildPlanStep({ stepId: "s1", action: "analyze", description: "分析需求" }),
    buildPlanStep({
      stepId: "s2",
      action: "design",
      description: "设计方案",
      dependencies: ["s1"],
    }),
    buildPlanStep({
      stepId: "s3",
      action: "implement",
      description: "实现功能",
      dependencies: ["s2"],
    }),
    buildPlanStep({
      stepId: "s4",
      action: "verify",
      description: "验证结果",
      dependencies: ["s3"],
    }),
  ],
  modify: () => [
    buildPlanStep({ stepId: "s1", action: "read", description: "读取现有内容" }),
    buildPlanStep({
      stepId: "s2",
      action: "analyze",
      description: "分析修改点",
      dependencies: ["s1"],
    }),
    buildPlanStep({
      stepId: "s3",
      action: "modify",
      description: "执行修改",
      dependencies: ["s2"],
    }),
    buildPlanStep({
      stepId: "s4",
      action: "verify",
      description: "验证修改",
      dependencies: ["s3"],
    }),
  ],
  delete: () => [
    buildPlanStep({
      stepId: "s1",
      action: "verify_existence",
      description: "确认目标存在",
    }),
    buildPlanStep({
      stepId: "s2",
      action: "request_approval",
      description: "请求审批",
      dependencies: ["s1"],
    }),
    buildPlanStep({
      stepId: "s3",
      action: "delete",
      description: "执行删除",
      dependencies: ["s2"],
    }),
  ],
  query: () => [
    buildPlanStep({
      stepId: "s1",
      action: "retrieve",
      description: "检索信息",
    }),
    buildPlanStep({
      stepId: "s2",
      action: "analyze",
      description: "分析结果",
      dependencies: ["s1"],
    }),
    buildPlanStep({
      stepId: "s3",
      action: "respond",
      description: "生成响应",
      dependencies: ["s2"],
    }),
  ],
  execute: () => [
    buildPlanStep({
      stepId: "s1",
      action: "validate_params",
      description: "验证参数",
    }),
    buildPlanStep({
      stepId: "s2",
      action: "execute",
      description: "执行操作",
      dependencies: ["s1"],
    }),
    buildPlanStep({
      stepId: "s3",
      action: "report",
      description: "报告结果",
      dependencies: ["s2"],
    }),
  ],
  plan: () => [
    buildPlanStep({
      stepId: "s1",
      action: "decompose",
      description: "目标分解",
    }),
    buildPlanStep({
      stepId: "s2",
      action: "evaluate",
      description: "方案评估",
      dependencies: ["s1"],
    }),
    buildPlanStep({
      stepId: "s3",
      action: "synthesize",
      description: "综合输出",
      dependencies: ["s2"],
    }),
  ],
  general: () => [
    buildPlanStep({ stepId: "s1", action: "think", description: "思考分析" }),
    buildPlanStep({
      stepId: "s2",
      action: "act",
      description: "执行操作",
      dependencies: ["s1"],
    }),
    buildPlanStep({
      stepId: "s3",
      action: "respond",
      description: "生成响应",
      dependencies: ["s2"],
    }),
  ],
};

export interface PlannerConfig {
  templates?: Partial<Record<Intent, IntentTemplate>>;
}

export class Planner {
  readonly name = "planner";
  private readonly templates: Record<Intent, IntentTemplate>;

  constructor(cfg: PlannerConfig = {}) {
    this.templates = { ...DEFAULT_TEMPLATES, ...cfg.templates };
  }

  createPlan(goal: ParsedGoal): PlanGraph {
    const template = this.templates[goal.intent] ?? this.templates.general;
    const steps = template(goal);
    return new PlanGraph({ goal, steps });
  }
}
