import {
  type ClassifyResult,
  type ReinforcingClassifier,
  type RouteDecision,
  decideRoute,
} from "@openintj/classifier";
import {
  type HookBus,
  type InputStructureResult,
  type InputStructuringPolicy,
  type LlmClient,
  TaskType,
  type TaskTypeType,
  detectTaskType,
  inputClarificationFromPreflight,
  inputClarificationFromQuestions,
  resolveInputStructuringConfig,
  structureUserInput,
} from "@openintj/core";
import {
  type ControlPlane,
  formatPlanAdviceBlock,
  resolvePlanAdviceForRun,
} from "@openintj/plane-control";
import {
  type ProductBehaviorMemoryCandidate,
  resolveDeterministicProductBehaviorAnswer,
} from "@openintj/shared";

export interface RunPreludeClassifier {
  classify(query: string): Promise<ClassifyResult>;
}

export interface PrepareRunPreludeDeps {
  hooks: HookBus;
  llm: LlmClient;
  control: ControlPlane;
  productBehaviorEnabled: boolean;
  inputStructuring?: InputStructuringPolicy;
  /** Pre-resolved config wins over `inputStructuring` (hosts that already resolved env). */
  inputStructuringConfig?: ReturnType<typeof resolveInputStructuringConfig>;
  classifier?: RunPreludeClassifier;
  /** Desktop artifact requests skip classifier + plan advice and force a task type. */
  skipClassifierAndPlan?: boolean;
  forcedTaskType?: TaskTypeType;
}

export interface PrepareRunPreludeInput {
  query: string;
  history?: readonly import("@openintj/core").ChatMessage[];
  signal?: AbortSignal;
  llm?: LlmClient;
  memories?: readonly ProductBehaviorMemoryCandidate[];
}

export interface RunPrelude {
  preflight?: { answer: string; guard: string };
  inputStructure?: InputStructureResult;
  /** Possibly rewritten to a clarify structure by plan-unknowns. */
  structured?: InputStructureResult;
  executionQuery: string;
  /** Alias kept for hosts: equals executionQuery (plan advice no longer splices into it). */
  workQuery: string;
  planAdviceBlock?: string;
  usedPlanAdvice: boolean;
  cls?: ClassifyResult;
  route?: RouteDecision;
  routedTaskType: TaskTypeType;
}

/**
 * RFC-011 §5/§6: the single run() prelude shared by CLI/server/desktop —
 * preflight → RFC-008 structure → classifier route → plan advice.
 * Slash-command intercept and journal finalization stay host-side because
 * they differ per profile (conversation id, workbench sink, image tasks).
 */
export const prepareRunPrelude = async (
  deps: PrepareRunPreludeDeps,
  input: PrepareRunPreludeInput,
): Promise<RunPrelude> => {
  const { query } = input;
  const config =
    deps.inputStructuringConfig ??
    resolveInputStructuringConfig(deps.productBehaviorEnabled ? deps.inputStructuring : "off");
  const preflight = deps.productBehaviorEnabled
    ? resolveDeterministicProductBehaviorAnswer(query, {
        ...(input.memories ? { memories: input.memories } : {}),
      })
    : undefined;
  const inputStructure =
    preflight?.guard === "material-clarification"
      ? inputClarificationFromPreflight(query, preflight.answer)
      : !preflight
        ? await structureUserInput({
            input: query,
            llm: input.llm ?? deps.llm,
            ...config,
            hooks: deps.hooks,
            ...(input.history ? { history: input.history } : {}),
            ...(input.signal ? { signal: input.signal } : {}),
          })
        : undefined;
  const executionQuery =
    inputStructure?.action === "proceed" ? inputStructure.executionInput : query;

  const cls =
    !preflight &&
    inputStructure?.action !== "clarify" &&
    !deps.skipClassifierAndPlan &&
    deps.classifier
      ? await deps.classifier.classify(executionQuery)
      : undefined;
  const route = cls ? decideRoute(cls) : undefined;
  const routedTaskType = deps.forcedTaskType ?? cls?.label ?? detectTaskType(executionQuery);

  let structured = inputStructure;
  let planAdviceBlock: string | undefined;
  if (!preflight && structured?.action !== "clarify" && !deps.skipClassifierAndPlan) {
    const decision = resolvePlanAdviceForRun(deps.control, {
      query,
      executionQuery,
      taskType: routedTaskType,
      ...(structured?.structure ? { structure: structured.structure } : {}),
    });
    if (decision.kind === "clarify") {
      structured = inputClarificationFromQuestions(query, decision.questions, "plan-unknowns");
    } else if (decision.kind === "proceed") {
      planAdviceBlock = formatPlanAdviceBlock(decision.advice);
    }
  }

  return {
    ...(preflight ? { preflight } : {}),
    ...(inputStructure ? { inputStructure } : {}),
    ...(structured ? { structured } : {}),
    executionQuery,
    workQuery: executionQuery,
    ...(planAdviceBlock ? { planAdviceBlock } : {}),
    usedPlanAdvice: Boolean(planAdviceBlock),
    ...(cls ? { cls } : {}),
    ...(route ? { route } : {}),
    routedTaskType,
  };
};

/** Re-export so hosts can type their classifier field without a second import path. */
export type { ClassifyResult, ReinforcingClassifier, RouteDecision };
export { TaskType };
