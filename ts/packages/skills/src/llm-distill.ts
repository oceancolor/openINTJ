import { TaskType, type TaskTypeType, canonicalToolNames } from "@openintj/core";
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
  /** body 最小字符数（低于视为低质量、丢弃）。默认 16。 */
  minBodyChars?: number;
  /** name / description / body / 每草案 triggers|tools 的字符/条数上限（截断防跑飞）。 */
  maxNameChars?: number;
  maxDescriptionChars?: number;
  maxBodyChars?: number;
  maxTriggers?: number;
  maxTools?: number;
}

interface DraftLimits {
  minBodyChars: number;
  maxNameChars: number;
  maxDescriptionChars: number;
  maxBodyChars: number;
  maxTriggers: number;
  maxTools: number;
}

const VALID_TASK_TYPES = new Set<string>(Object.values(TaskType));

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
    `{ "id": string(kebab-case), "name": string, "description": string, "triggers": string[], "taskTypes": string[], "tools": string[], "body": string }`,
    `"taskTypes" must be from: ${[...VALID_TASK_TYPES].join(", ")}. "tools" are ToolHub names (read_file, write_file, execute_command, search).`,
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

/** 归一化字符串数组：转字符串、trim、去空、去重、截断到 max。 */
const normStringArray = (v: unknown, max: number, lower = false): string[] => {
  if (!Array.isArray(v)) return [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const x of v) {
    if (typeof x !== "string") continue;
    const s = (lower ? x.toLowerCase() : x).trim();
    if (s.length === 0 || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
    if (out.length >= max) break;
  }
  return out;
};

const clamp = (s: string, max: number): string => (s.length > max ? s.slice(0, max).trim() : s);

const toDraft = (o: unknown, lim: DraftLimits): CandidateSkillDraft | undefined => {
  if (!o || typeof o !== "object") return undefined;
  const r = o as Record<string, unknown>;
  if (typeof r["name"] !== "string" || typeof r["body"] !== "string") return undefined;
  const name = clamp(r["name"].trim(), lim.maxNameChars);
  const body = clamp(r["body"].trim(), lim.maxBodyChars);
  // 质量门槛：name 非空、body 达最小长度（丢弃 "ok" 这类无信息量草案）。
  if (name.length === 0 || body.length < lim.minBodyChars) return undefined;
  const rawDesc = typeof r["description"] === "string" ? r["description"].trim() : "";
  const draft: CandidateSkillDraft = {
    name,
    description: clamp(rawDesc.length > 0 ? rawDesc : name, lim.maxDescriptionChars),
    body,
    triggers: normStringArray(r["triggers"], lim.maxTriggers, true),
  };
  if (typeof r["id"] === "string" && r["id"].trim().length > 0) draft.id = r["id"].trim();
  // taskTypes 校验到合法枚举（过滤 LLM 幻觉出的类型），空则不带。
  const taskTypes = normStringArray(r["taskTypes"], 8).filter((t) =>
    VALID_TASK_TYPES.has(t),
  ) as TaskTypeType[];
  if (taskTypes.length > 0) draft.taskTypes = taskTypes;
  const tools = canonicalToolNames(normStringArray(r["tools"], lim.maxTools));
  if (tools.length > 0) draft.tools = tools;
  return draft;
};

/** 批内按 id（缺则 name 小写）去重，保留先出现者。 */
const dedupeDrafts = (drafts: readonly CandidateSkillDraft[]): CandidateSkillDraft[] => {
  const out: CandidateSkillDraft[] = [];
  const seen = new Set<string>();
  for (const d of drafts) {
    const key = (d.id ?? d.name.toLowerCase()).trim();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(d);
  }
  return out;
};

/**
 * 用 agent 的 LLM 把成功轨迹蒸馏成候选技能草案。
 * 解析失败会抛错 → {@link SkillLearningRuntime.distill} 自动回退启发式，保证不空手。
 * 校验：name/body 必填 + body 最小长度 + 各字段截断 + triggers/tools 归一去重 + taskTypes 枚举校验 + 批内去重。
 */
export const createLlmSkillDistiller = (
  llm: SkillDistillLlm,
  opts: LlmDistillerOpts = {},
): LlmSkillDistiller => {
  const maxSamples = Math.max(1, opts.maxSamples ?? 24);
  const maxDrafts = Math.max(1, opts.maxDrafts ?? 3);
  const lim: DraftLimits = {
    minBodyChars: Math.max(1, opts.minBodyChars ?? 16),
    maxNameChars: Math.max(1, opts.maxNameChars ?? 80),
    maxDescriptionChars: Math.max(1, opts.maxDescriptionChars ?? 240),
    maxBodyChars: Math.max(1, opts.maxBodyChars ?? 4000),
    maxTriggers: Math.max(1, opts.maxTriggers ?? 8),
    maxTools: Math.max(1, opts.maxTools ?? 8),
  };
  return async (samples) => {
    const picked = samples.slice(-maxSamples);
    const raw = await llm.generate(buildPrompt(picked, maxDrafts));
    const parsed = extractJson(raw);
    const arr = Array.isArray(parsed) ? parsed : [parsed];
    const drafts = dedupeDrafts(
      arr.map((o) => toDraft(o, lim)).filter((d): d is CandidateSkillDraft => Boolean(d)),
    ).slice(0, maxDrafts);
    if (drafts.length === 0) throw new Error("LLM produced no usable skill drafts");
    return drafts;
  };
};
