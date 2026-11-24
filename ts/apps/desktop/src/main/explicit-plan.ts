// R1 计划显性化（plan 强化路线 R1）
//
// 现状解剖（2026-11-12）：真正的规划只散在 ReAct 的 Thought 里，
// PlanAdvisor 只给 known/unknown 标注、Planner 模板是通用步骤（太假），
// TaskPool 图来自确定性模板映射。R1 把规划前置为一步显式生成：
// LLM 产出结构化计划（目标 + 3~7 步 + 每步验收标准 + 依赖），
// 注入执行提示调制整个 run，并落盘 journal `plan.created` 事件。
//
// 设计约束：
// - 只对规划/分析类任务与 TaskPool 编排触发，琐碎任务不浪费调用；
// - 生成失败/超时一律 fail-soft：无 plan 照常执行（回退现状行为）；
// - 计划是调制不是枷锁：注入文本明确"与原意冲突以原始要求为准"。
import type { LlmClient } from "@openintj/core";

export interface ExplicitPlanStep {
  id: string;
  title: string;
  acceptance: string;
  dependencies: string[];
}

export interface ExplicitPlan {
  goal: string;
  source: "llm";
  steps: ExplicitPlanStep[];
}

export type ExplicitPlanGate = {
  /** prelude 中的分类任务类型（PLANNING/ANALYSIS 才值得出计划） */
  routedTaskType?: string;
  orchestrationMode: string;
  isPreflight: boolean;
  isClarify: boolean;
  isArtifactRequest: boolean;
  queryLength: number;
};

export const PLAN_WORTHY_TYPES: ReadonlySet<string> = new Set([
  "planning",
  "analysis",
  "research",
  "complex",
  "implementation",
  "debugging",
]);

/** 纯函数门控：是否值得为本次 run 生成显性计划。 */
export const shouldBuildExplicitPlan = (gate: ExplicitPlanGate): boolean => {
  if (gate.isPreflight || gate.isClarify || gate.isArtifactRequest) return false;
  if (gate.queryLength < 12) return false;
  if (gate.orchestrationMode === "taskpool") return true;
  const t = (gate.routedTaskType ?? "").toLowerCase();
  return PLAN_WORTHY_TYPES.has(t);
};

const buildPrompt = (query: string, contextHint?: string): string =>
  [
    "把下面的任务拆成一份可执行的计划。只输出一个 JSON 对象，不要解释、不要 markdown 代码围栏。",
    '结构：{"goal":"一句话目标","steps":[{"title":"步骤标题","acceptance":"完成标准（可观察/可验证）","dependencies":[]}]}',
    "要求：3 到 7 步，每步粒度具体到动作，不要写「分析需求」「设计方案」这类空泛步骤；",
    "acceptance 必须能判真假（文件存在、测试通过、清单完整等）；dependencies 用前面步骤的序号（从 1 起），无依赖则空数组。",
    "",
    `任务：${query}`,
    contextHint ? `补充上下文：${contextHint}` : "",
  ]
    .filter(Boolean)
    .join("\n");

const stripFences = (text: string): string => {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  return (fenced?.[1] ?? trimmed).trim();
};

/** 容错解析模型输出；不合格返回 null（调用方 fail-soft）。 */
export const parseExplicitPlan = (raw: string): ExplicitPlan | null => {
  const text = stripFences(raw);
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text.slice(start, end + 1));
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const goal = (parsed as { goal?: unknown }).goal;
  const stepsRaw = (parsed as { steps?: unknown }).steps;
  if (typeof goal !== "string" || goal.trim().length === 0) return null;
  if (!Array.isArray(stepsRaw) || stepsRaw.length === 0) return null;
  const steps: ExplicitPlanStep[] = [];
  for (const [i, s] of stepsRaw.slice(0, 7).entries()) {
    if (!s || typeof s !== "object") return null;
    const title = (s as { title?: unknown }).title;
    if (typeof title !== "string" || title.trim().length === 0) return null;
    const acceptance = (s as { acceptance?: unknown }).acceptance;
    const depsRaw = (s as { dependencies?: unknown }).dependencies;
    const dependencies: string[] = [];
    if (Array.isArray(depsRaw)) {
      for (const d of depsRaw) {
        const n = typeof d === "number" ? d : Number.parseInt(String(d), 10);
        if (Number.isFinite(n) && n >= 1 && n <= i) dependencies.push(`s${n}`);
      }
    }
    steps.push({
      id: `s${i + 1}`,
      title: title.trim().slice(0, 200),
      acceptance:
        typeof acceptance === "string" && acceptance.trim().length > 0
          ? acceptance.trim().slice(0, 300)
          : "完成且结果可核对",
      dependencies,
    });
  }
  if (steps.length < 2) return null;
  return { goal: goal.trim().slice(0, 300), source: "llm", steps };
};

const PLAN_TIMEOUT_MS = 45_000;

export const buildExplicitPlan = async (args: {
  llm: LlmClient;
  query: string;
  contextHint?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
}): Promise<ExplicitPlan | null> => {
  const timeoutMs = args.timeoutMs ?? PLAN_TIMEOUT_MS;
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  args.signal?.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const text = await args.llm.chat(
      [
        {
          role: "system",
          content: "你是一个任务规划器。输出严格 JSON。",
        },
        { role: "user", content: buildPrompt(args.query, args.contextHint) },
      ],
      { signal: controller.signal, temperature: 0.2, maxTokens: 900 },
    );
    if (!text) return null;
    return parseExplicitPlan(text);
  } catch {
    return null; // fail-soft：计划生成失败不阻断执行
  } finally {
    clearTimeout(timer);
    args.signal?.removeEventListener("abort", onAbort);
  }
};

/** 注入执行提示的计划块（计划是调制，不是枷锁）。 */
export const formatExplicitPlanBlock = (plan: ExplicitPlan): string =>
  [
    "【执行计划（显性化）】",
    `目标：${plan.goal}`,
    ...plan.steps.map(
      (s, i) =>
        `${i + 1}. ${s.title}（验收：${s.acceptance}${s.dependencies.length > 0 ? `；依赖：${s.dependencies.join(",")}` : ""}）`,
    ),
    "按此计划推进；每完成一步对照验收自检。计划与用户原始要求或现场证据冲突时，以原始要求与证据为准，并在回复中说明偏差。",
  ].join("\n");

/** 回读展示用（IPC 透传/聊天区渲染）。 */
export const serializePlanForClient = (plan: {
  goal: string;
  source: string;
  steps: readonly { id: string; title: string; acceptance: string }[];
}): {
  goal: string;
  source: string;
  steps: Array<{ id: string; title: string; acceptance: string }>;
} => ({
  goal: plan.goal,
  source: plan.source,
  steps: plan.steps.map((s) => ({
    id: s.id,
    title: s.title,
    acceptance: s.acceptance,
  })),
});

