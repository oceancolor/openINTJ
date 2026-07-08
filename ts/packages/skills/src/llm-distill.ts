import type {
  CandidateSkillDraft,
  LlmSkillDistiller,
  TrajectorySample,
} from "./learning-runtime.js";

/** 最小 LLM 适配：只要能「给 prompt 出 text」即可，避免 skills 依赖具体 LlmClient 形状。 */
export interface SkillDistillLlm {
  generate(prompt: string): Promise<string>;
}

export interface LlmDistillerOpts {
  /** 送进 prompt 的最多样本数（取最近的）。默认 24。 */
  maxSamples?: number;
  /** 一次蒸馏最多产出的候选数（防 LLM 发散）。默认 3。 */
  maxDrafts?: number;
}

const SYSTEM = [
  "You distill reusable *skills* from an agent's successful task trajectories.",
  "A skill is a concise, reusable capability pack: a name, one-line description, and an instructional body",
  "that would help the agent handle similar future tasks better.",
  "Only propose a skill when several samples share a clear, reusable pattern.",
].join(" ");

const buildPrompt = (samples: readonly TrajectorySample[], maxDrafts: number): string => {
  const lines = samples.map((s, i) => {
    const tt = s.taskType ? ` [${s.taskType}]` : "";
    const tools = s.toolsUsed.length ? ` (tools: ${s.toolsUsed.join(", ")})` : "";
    return `${i + 1}.${tt} ${s.query}${tools}`;
  });
  return [
    SYSTEM,
    "",
    `Here are ${samples.length} successful task samples:`,
    ...lines,
    "",
    `Output at most ${maxDrafts} skill(s) as a JSON array. Each item MUST be:`,
    `{ "id": string(kebab-case), "name": string, "description": string, "triggers": string[], "taskTypes": string[], "body": string }`,
    'Return ONLY the JSON array, no prose. If nothing is reusable, return "[]".',
  ].join("\n");
};

/** 从 LLM 自由文本里抠出第一个 JSON 数组/对象（容错栅栏/前后缀）。 */
const extractJson = (raw: string): unknown => {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  const text = (fenced?.[1] ?? raw).trim();
  const start = text.search(/[[{]/);
  if (start < 0) throw new Error("no JSON found in LLM output");
  const open = text[start];
  const close = open === "[" ? "]" : "}";
  const end = text.lastIndexOf(close);
  if (end <= start) throw new Error("unbalanced JSON in LLM output");
  return JSON.parse(text.slice(start, end + 1));
};

const asStringArray = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];

const toDraft = (o: unknown): CandidateSkillDraft | undefined => {
  if (!o || typeof o !== "object") return undefined;
  const r = o as Record<string, unknown>;
  if (typeof r["name"] !== "string" || typeof r["body"] !== "string") return undefined;
  const draft: CandidateSkillDraft = {
    name: r["name"],
    description: typeof r["description"] === "string" ? r["description"] : r["name"],
    body: r["body"],
    triggers: asStringArray(r["triggers"]),
  };
  if (typeof r["id"] === "string") draft.id = r["id"];
  // taskTypes 宽松透传（runtime/DB 侧按 string 存），下游用作弱信号。
  const taskTypes = asStringArray(r["taskTypes"]);
  if (taskTypes.length > 0) {
    draft.taskTypes = taskTypes as NonNullable<CandidateSkillDraft["taskTypes"]>;
  }
  return draft;
};

/**
 * 用 agent 的 LLM 把成功轨迹蒸馏成候选技能草案。
 * 解析失败会抛错 → {@link SkillLearningRuntime.distill} 自动回退启发式，保证不空手。
 */
export const createLlmSkillDistiller = (
  llm: SkillDistillLlm,
  opts: LlmDistillerOpts = {},
): LlmSkillDistiller => {
  const maxSamples = Math.max(1, opts.maxSamples ?? 24);
  const maxDrafts = Math.max(1, opts.maxDrafts ?? 3);
  return async (samples) => {
    const picked = samples.slice(-maxSamples);
    const raw = await llm.generate(buildPrompt(picked, maxDrafts));
    const parsed = extractJson(raw);
    const arr = Array.isArray(parsed) ? parsed : [parsed];
    const drafts = arr
      .map(toDraft)
      .filter((d): d is CandidateSkillDraft => Boolean(d))
      .slice(0, maxDrafts);
    if (drafts.length === 0) throw new Error("LLM produced no usable skill drafts");
    return drafts;
  };
};
