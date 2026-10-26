import { TaskType, type TaskTypeType } from "@openintj/core";
import type { ControlPlane } from "./control-plane.js";
import {
  type AnnotatedPlanStep,
  type KnowledgeFact,
  type PlanKnowledgeContext,
  annotatePlanSteps,
  collectKnowledgeFacts,
  questionsForMissing,
} from "./knowledge.js";
import { type PathSearchResult, evaluatePlanPath } from "./path-search.js";
import type { PlanGraph } from "./plan-graph.js";

export const PLAN_ADVICE_TASK_TYPES: readonly TaskTypeType[] = [
  TaskType.PLANNING,
  TaskType.ANALYSIS,
];

export const shouldAdvisePlan = (taskType: TaskTypeType): boolean =>
  PLAN_ADVICE_TASK_TYPES.includes(taskType);

export interface PlanAdvice {
  action: "clarify" | "proceed";
  known: KnowledgeFact[];
  unknown: KnowledgeFact[];
  annotated: AnnotatedPlanStep[];
  path: PathSearchResult;
  nextGoal: string;
  questions: string[];
}

export const advisePlan = (plan: PlanGraph, ctx: PlanKnowledgeContext): PlanAdvice => {
  const facts = collectKnowledgeFacts({
    ...ctx,
    entities: ctx.entities ?? plan.goal.entities,
  });
  const annotated = annotatePlanSteps(plan.steps, {
    ...ctx,
    entities: ctx.entities ?? plan.goal.entities,
  });
  const path = evaluatePlanPath(annotated);
  const next = annotated.find((item) => item.step.stepId === path.nextStepId) ?? annotated[0];
  const known = facts.filter((f) => f.known);
  const unknown = facts.filter((f) => !f.known);
  if (!next) {
    return {
      action: "proceed",
      known,
      unknown,
      annotated,
      path,
      nextGoal: plan.goal.rawInput,
      questions: [],
    };
  }
  const blocked = !next.known;
  const questions = blocked ? questionsForMissing(next.missing, facts) : [];
  return {
    action: blocked && questions.length > 0 ? "clarify" : "proceed",
    known,
    unknown,
    annotated,
    path,
    nextGoal: next.step.description || next.step.action,
    questions,
  };
};

export const formatNextGoalQuery = (executionQuery: string, advice: PlanAdvice): string => {
  const knownLine =
    advice.known.length > 0
      ? advice.known.map((f) => `${f.slot}:${f.evidence}`).join("；")
      : "（无）";
  const unknownLine =
    advice.unknown.length > 0
      ? advice.unknown.map((f) => `${f.slot}:${f.evidence}`).join("；")
      : "（无）";
  return [
    `【规划】下一轮目标：${advice.nextGoal}`,
    `【已知】${knownLine}`,
    `【未知】${unknownLine}`,
    "---",
    executionQuery,
  ].join("\n");
};

export type PlanAdviceRunDecision =
  | { kind: "skip" }
  | { kind: "clarify"; questions: string[]; advice: PlanAdvice }
  | { kind: "proceed"; executionQuery: string; advice: PlanAdvice };

export const resolvePlanAdviceForRun = (
  control: ControlPlane,
  input: {
    query: string;
    executionQuery: string;
    taskType: TaskTypeType;
    structure?: PlanKnowledgeContext["structure"];
  },
): PlanAdviceRunDecision => {
  if (!shouldAdvisePlan(input.taskType)) return { kind: "skip" };
  const { plan } = control.processInput(input.executionQuery, input.taskType);
  const advice = advisePlan(plan, {
    query: input.query,
    entities: plan.goal.entities,
    ...(input.structure ? { structure: input.structure } : {}),
  });
  if (advice.action === "clarify") {
    return { kind: "clarify", questions: advice.questions, advice };
  }
  return {
    kind: "proceed",
    executionQuery: formatNextGoalQuery(input.executionQuery, advice),
    advice,
  };
};
