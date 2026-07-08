import {
  type EmbeddingProvider,
  type TaskTypeType,
  cosineSimilarity,
  estimateTokens,
} from "@openintj/core";
import type { SkillRegistry } from "./registry.js";
import type { SelectedSkill } from "./types.js";

const embed = async (e: EmbeddingProvider, text: string): Promise<number[]> => {
  const r = e.embed(text);
  return r instanceof Promise ? await r : r;
};

export interface SkillSelectOpts {
  /** 本轮任务类型（命中技能 taskTypes 时加成）。 */
  taskType?: TaskTypeType;
}

export interface SkillSelectorOpts {
  registry: SkillRegistry;
  embedder: EmbeddingProvider;
  /** 融合得分下限，低于则不注入。默认 0.35。 */
  minScore?: number;
  /** 最多注入几个技能。默认 2。 */
  topK?: number;
  /** 注入正文合计 token 预算（超出的低分技能被裁掉，至少保留最高分一个）。默认 700。 */
  maxBodyTokens?: number;
  /** 关键词命中加成（query 含 trigger 子串）。默认 0.15。 */
  keywordBoost?: number;
  /** taskType 命中加成（skill.taskTypes 含本轮 taskType）。默认 0.1。 */
  taskTypeBoost?: number;
  /**
   * 可选强化权重供给（Phase 2 自学习）：给「历史越用越准」的技能一点有界偏置。
   * 偏置 = clamp(weight*weightGain, -weightBiasCap, +weightBiasCap)，语义余弦仍主导。
   */
  weightFor?: (id: string) => number;
  /** 权重→偏置的增益。默认 0.05。 */
  weightGain?: number;
  /** 权重偏置绝对值封顶（不让权重压过语义相关度）。默认 0.3。 */
  weightBiasCap?: number;
}

/**
 * 技能选择器（第二级）：embed(query) 与各技能匹配向量做余弦 + 关键词/任务类型加成，
 * 过阈值取 topK，再按正文 token 预算封顶。返回按得分降序的命中技能。
 */
export class SkillSelector {
  private readonly registry: SkillRegistry;
  private readonly embedder: EmbeddingProvider;
  private readonly minScore: number;
  private readonly topK: number;
  private readonly maxBodyTokens: number;
  private readonly keywordBoost: number;
  private readonly taskTypeBoost: number;
  private readonly weightFor?: (id: string) => number;
  private readonly weightGain: number;
  private readonly weightBiasCap: number;

  constructor(opts: SkillSelectorOpts) {
    this.registry = opts.registry;
    this.embedder = opts.embedder;
    this.minScore = opts.minScore ?? 0.35;
    this.topK = opts.topK ?? 2;
    this.maxBodyTokens = opts.maxBodyTokens ?? 700;
    this.keywordBoost = opts.keywordBoost ?? 0.15;
    this.taskTypeBoost = opts.taskTypeBoost ?? 0.1;
    if (opts.weightFor) this.weightFor = opts.weightFor;
    this.weightGain = opts.weightGain ?? 0.05;
    this.weightBiasCap = opts.weightBiasCap ?? 0.3;
  }

  async select(query: string, opts: SkillSelectOpts = {}): Promise<SelectedSkill[]> {
    const skills = this.registry.list();
    if (skills.length === 0 || query.trim().length === 0) return [];
    const qVec = await embed(this.embedder, query);
    const qLower = query.toLowerCase();

    const scored: SelectedSkill[] = [];
    for (const skill of skills) {
      const vec = this.registry.vectorFor(skill.id);
      if (!vec) continue;
      let score = cosineSimilarity(qVec, vec);
      if (skill.triggers.some((t) => t.length > 0 && qLower.includes(t))) {
        score += this.keywordBoost;
      }
      if (opts.taskType && skill.taskTypes.includes(opts.taskType)) {
        score += this.taskTypeBoost;
      }
      if (this.weightFor) {
        const raw = this.weightFor(skill.id) * this.weightGain;
        score += Math.max(-this.weightBiasCap, Math.min(this.weightBiasCap, raw));
      }
      if (score >= this.minScore) {
        scored.push({ skill, score: Math.min(1, score) });
      }
    }

    // 得分降序，并列时 priority 高者优先。
    scored.sort((a, b) => b.score - a.score || b.skill.priority - a.skill.priority);
    const top = scored.slice(0, Math.max(0, this.topK));

    // token 预算封顶：至少保留最高分一个，其余在预算内追加。
    const out: SelectedSkill[] = [];
    let used = 0;
    for (const sel of top) {
      const cost = estimateTokens(sel.skill.body);
      if (out.length === 0 || used + cost <= this.maxBodyTokens) {
        out.push(sel);
        used += cost;
      }
    }
    return out;
  }
}

/**
 * 把命中技能渲染成注入 system prompt 的块（无命中返回 ""）。
 * 位置约定：拼在 persona 之后、`[记忆参考]` 之前。
 */
export const renderSkillPrompt = (skills: readonly SelectedSkill[]): string => {
  if (skills.length === 0) return "";
  const blocks = skills.map((s) => {
    const head = `## ${s.skill.name}`;
    // 工具软绑定：文本协议下提示优先工具子集，引导 Action 选择。
    const toolLine = s.skill.tools.length
      ? `\n（建议优先使用工具：${s.skill.tools.join(", ")}）`
      : "";
    return `${head}${toolLine}\n${s.skill.body.trim()}`;
  });
  return `[技能]（本轮命中的能力包，按需生效）\n${blocks.join("\n\n")}`;
};

/**
 * 命中技能声明的工具子集并集（去重、保序）。装配方可用它做更硬的「工具面收窄」——
 * 例如把 ToolHub 的可用工具与之取交集。空数组表示「本轮命中技能未约束工具」。
 */
export const skillToolAllowlist = (skills: readonly SelectedSkill[]): string[] => {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const s of skills) {
    for (const t of s.skill.tools) {
      if (!seen.has(t)) {
        seen.add(t);
        out.push(t);
      }
    }
  }
  return out;
};
