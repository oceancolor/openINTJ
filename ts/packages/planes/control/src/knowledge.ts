import type { PlanStep } from "./types.js";

export type KnowledgeSlot = "goal" | "target" | "environment" | "deliverable";

export interface PlanKnowledgeContext {
  query: string;
  entities?: readonly string[];
  structure?: {
    goal?: string;
    constraints?: readonly string[];
    deliverables?: readonly string[];
    dependencies?: readonly string[];
    assumptions?: readonly string[];
  };
}

export interface KnowledgeFact {
  slot: KnowledgeSlot;
  known: boolean;
  evidence: string;
  question: string;
}

export interface AnnotatedPlanStep {
  step: PlanStep;
  known: boolean;
  missing: KnowledgeSlot[];
  cost: number;
}

const SLOT_QUESTION: Record<KnowledgeSlot, string> = {
  goal: "请用一句话明确任务目标和完成标准。",
  target: "请给出具体目标（文件、路径、服务名或交付物名称）。",
  environment: "请确认目标环境、集群或主机、域名以及发布窗口。",
  deliverable: "请列出本阶段可验收的交付物。",
};

const ACTION_COST: Record<string, number> = {
  think: 2,
  analyze: 2,
  retrieve: 2,
  decompose: 2,
  design: 3,
  evaluate: 3,
  validate_params: 3,
  request_approval: 3,
  implement: 8,
  execute: 8,
  modify: 7,
  delete: 7,
  verify: 4,
  report: 4,
  respond: 3,
  synthesize: 4,
  act: 5,
};

const NEEDS_TARGET = new Set(["implement", "execute", "modify", "delete", "act"]);
const NEEDS_ENV = new Set(["execute", "delete"]);

const joined = (parts: readonly (string | undefined)[]): string =>
  parts.filter((p): p is string => Boolean(p?.trim())).join("\n");

const hasMatch = (text: string, pattern: RegExp): boolean => pattern.test(text);

export const collectKnowledgeFacts = (ctx: PlanKnowledgeContext): KnowledgeFact[] => {
  const blob = joined([
    ctx.query,
    ctx.structure?.goal,
    ...(ctx.structure?.constraints ?? []),
    ...(ctx.structure?.deliverables ?? []),
    ...(ctx.structure?.dependencies ?? []),
    ...(ctx.structure?.assumptions ?? []),
    ...(ctx.entities ?? []),
  ]);
  const goalKnown = blob.trim().length >= 8;
  const targetKnown =
    (ctx.entities?.length ?? 0) > 0 ||
    (ctx.structure?.deliverables?.length ?? 0) > 0 ||
    hasMatch(blob, /(?:文件|路径|目录|\.ts|\.json|\.md|服务|模块|仓库|CLI|API|path|file)/i);
  const environmentKnown = hasMatch(
    blob,
    /(?:环境|集群|域名|主机|staging|prod(?:uction)?|测试环境|发布窗口)/i,
  );
  const deliverableKnown =
    (ctx.structure?.deliverables?.length ?? 0) > 0 || hasMatch(blob, /交付物|验收|产出|milestone/i);
  const evidence = (ok: boolean, yes: string, no: string): string => (ok ? yes : no);
  return [
    {
      slot: "goal",
      known: goalKnown,
      evidence: evidence(goalKnown, "已给出任务描述", "缺少明确目标"),
      question: SLOT_QUESTION.goal,
    },
    {
      slot: "target",
      known: targetKnown,
      evidence: evidence(targetKnown, "已定位操作对象", "缺少具体目标对象"),
      question: SLOT_QUESTION.target,
    },
    {
      slot: "environment",
      known: environmentKnown,
      evidence: evidence(environmentKnown, "已给出环境约束", "缺少环境/集群/域名"),
      question: SLOT_QUESTION.environment,
    },
    {
      slot: "deliverable",
      known: deliverableKnown,
      evidence: evidence(deliverableKnown, "已提到交付物或验收", "未说明可验收产物"),
      question: SLOT_QUESTION.deliverable,
    },
  ];
};

export const stepCost = (action: string): number => ACTION_COST[action] ?? 5;

const slotsForAction = (action: string, query: string): KnowledgeSlot[] => {
  const slots: KnowledgeSlot[] = [];
  if (NEEDS_TARGET.has(action)) slots.push("target");
  if (NEEDS_ENV.has(action) && hasMatch(query, /部署|生产|发布|上线|delete|rm\s+-rf/i)) {
    slots.push("environment");
  }
  if (action === "synthesize" || action === "evaluate") slots.push("deliverable");
  return slots;
};

export const annotatePlanSteps = (
  steps: readonly PlanStep[],
  ctx: PlanKnowledgeContext,
): AnnotatedPlanStep[] => {
  const facts = collectKnowledgeFacts(ctx);
  const knownSlots = new Set(facts.filter((f) => f.known).map((f) => f.slot));
  return steps.map((step) => {
    const missing = slotsForAction(step.action, ctx.query).filter((slot) => !knownSlots.has(slot));
    return {
      step,
      known: missing.length === 0,
      missing,
      cost: stepCost(step.action),
    };
  });
};

export const questionsForMissing = (
  missing: readonly KnowledgeSlot[],
  facts: readonly KnowledgeFact[],
  limit = 3,
): string[] => {
  const wanted = new Set(missing);
  const out: string[] = [];
  for (const fact of facts) {
    if (!wanted.has(fact.slot) || fact.known) continue;
    if (!out.includes(fact.question)) out.push(fact.question);
    if (out.length >= limit) break;
  }
  return out;
};
