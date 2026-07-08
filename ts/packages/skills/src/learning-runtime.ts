import type { HookBus, TaskTypeType } from "@openintj/core";
import { InMemorySkillStore, type SkillStore } from "./store.js";
import type { Skill, SkillProposal, SkillWeight } from "./types.js";

/**
 * 从 run 结果推导技能强化信号（与 classifier 的 outcomeSignal 同映射；
 * 本地实现，避免 skills→classifier 反向依赖）：
 *  - completed → +1
 *  - failed/timeout → -0.5
 *  - 其它（max_iter 等）→ +0.2
 */
export const skillOutcomeSignal = (status: string): number => {
  if (status === "completed") return 1;
  if (status === "failed" || status === "timeout") return -0.5;
  return 0.2;
};

/** 一次成功 run 的精简（脱敏后）轨迹，进 buffer 供蒸馏。 */
export interface TrajectorySample {
  query: string;
  taskType?: TaskTypeType;
  finalAnswer: string;
  toolsUsed: string[];
  ts: number;
}

/** 蒸馏产出的候选技能草案（llmDistill 或启发式共用形状）。 */
export interface CandidateSkillDraft {
  /** 稳定 id（省略则由 name slug 兜底，用于跨次去重）。 */
  id?: string;
  name: string;
  description: string;
  triggers?: string[];
  taskTypes?: TaskTypeType[];
  body: string;
}

/** LLM 蒸馏器：把成功轨迹样本压成一个/多个候选技能草案。 */
export type LlmSkillDistiller = (
  samples: readonly TrajectorySample[],
) =>
  | Promise<CandidateSkillDraft[] | CandidateSkillDraft>
  | CandidateSkillDraft[]
  | CandidateSkillDraft;

export interface SkillLearningRuntimeOpts {
  /** 持久化；不传则纯内存（默认/测试）。 */
  store?: SkillStore;
  /** 命中/提案事件总线；用于发 `event.SKILL_PROPOSED`。 */
  hooks?: HookBus;
  /** 时间源（毫秒），测试可注入。默认 Date.now。 */
  clock?: () => number;
  /** 原始权重有界区间（防溢出；选择器另做偏置封顶）。默认 min -2 / max 12。 */
  weightClamp?: { min?: number; max?: number };
  /** 可选 LLM 蒸馏器；不传走启发式。 */
  llmDistill?: LlmSkillDistiller;
  /** 启发式聚类的最小样本数（低于不产候选）。默认 3。 */
  minSamplesToDistill?: number;
  /** 成功轨迹 buffer 上限（超过丢最旧）。默认 500。 */
  maxBufferedTrajectories?: number;
  /** 「本轮选中技能」记忆的 key 上限（超过清空）。默认 256。 */
  maxSelectionMemory?: number;
  /** approve/revoke 后回调（通常 `() => registry.load()` 重嵌入，让新技能立刻可选中）。 */
  onSkillsChanged?: () => void | Promise<void>;
  /** 提案 id 前缀。默认 "prop"。 */
  idPrefix?: string;
  /** 落 buffer 前对 query/finalAnswer 脱敏。默认恒等（不脱敏；agent 装配时可注入）。 */
  redactor?: (text: string) => string;
}

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "to",
  "of",
  "and",
  "or",
  "in",
  "on",
  "for",
  "with",
  "is",
  "are",
  "be",
  "how",
  "what",
  "why",
  "do",
  "does",
  "did",
  "can",
  "i",
  "you",
  "we",
  "it",
  "this",
  "that",
  "my",
  "me",
  "please",
  "help",
  "need",
  "want",
  "give",
  "show",
  "tell",
  "about",
  "use",
  "using",
  "from",
  "by",
  "at",
  "as",
  "if",
  "so",
  "但是",
  "一个",
  "如何",
  "怎么",
  "什么",
  "请",
]);

const slug = (s: string): string =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "skill";

const topKeywords = (queries: readonly string[], n: number): string[] => {
  const freq = new Map<string, number>();
  for (const q of queries) {
    for (const raw of q.toLowerCase().split(/[^a-z0-9\u4e00-\u9fa5]+/)) {
      const w = raw.trim();
      if (w.length < 3 || STOPWORDS.has(w)) continue;
      freq.set(w, (freq.get(w) ?? 0) + 1);
    }
  }
  return [...freq.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, n)
    .map(([w]) => w);
};

/**
 * SkillLearningRuntime —— 技能自学习闭环门面（Phase 2）。
 *
 * 组合三件事，均复用飞轮既有模式：
 *  1. **加权**（classifier reinforce 同构）：`noteSelected` 记本轮命中 → `recordOutcome` 按 outcome
 *     对命中技能累加权重（有界 clamp、写穿 store）；`weightFor` 供选择器做有界偏置。
 *  2. **蒸馏**（dormant mine 同构）：成功轨迹进 buffer；`distill()` 用户触发，`llmDistill` 或启发式
 *     产 `SkillProposal(pending)`（跨次按 candidate id 去重）。
 *  3. **审批**（dormant propose/approve/inject 同构）：`approve` 写 `store.upsertApprovedSkill` +
 *     触发 `onSkillsChanged`（registry 重载）；`reject`/`revoke` 对应。
 *
 * opt-in、默认全关；无 store 时纯内存也完整可用。
 */
export class SkillLearningRuntime {
  readonly store: SkillStore;
  private readonly hooks?: HookBus;
  private readonly clock: () => number;
  private readonly wMin: number;
  private readonly wMax: number;
  private readonly llmDistill?: LlmSkillDistiller;
  private readonly minSamples: number;
  private readonly maxBuffer: number;
  private readonly maxSelMemory: number;
  private readonly onSkillsChanged?: () => void | Promise<void>;
  private readonly idPrefix: string;
  private readonly redactor: (text: string) => string;

  private readonly weights = new Map<string, SkillWeight>();
  private readonly proposals = new Map<string, SkillProposal>();
  private readonly approved = new Map<string, Skill>();
  /** key(query,taskType) → 本轮命中的技能 id（recordOutcome 时消费）。 */
  private readonly selection = new Map<string, string[]>();
  private buffer: TrajectorySample[] = [];
  private seq = 0;

  constructor(opts: SkillLearningRuntimeOpts = {}) {
    this.store = opts.store ?? new InMemorySkillStore();
    if (opts.hooks) this.hooks = opts.hooks;
    this.clock = opts.clock ?? (() => Date.now());
    this.wMin = opts.weightClamp?.min ?? -2;
    this.wMax = opts.weightClamp?.max ?? 12;
    if (opts.llmDistill) this.llmDistill = opts.llmDistill;
    this.minSamples = Math.max(1, opts.minSamplesToDistill ?? 3);
    this.maxBuffer = Math.max(1, opts.maxBufferedTrajectories ?? 500);
    this.maxSelMemory = Math.max(1, opts.maxSelectionMemory ?? 256);
    if (opts.onSkillsChanged) this.onSkillsChanged = opts.onSkillsChanged;
    this.idPrefix = opts.idPrefix ?? "prop";
    this.redactor = opts.redactor ?? ((t) => t);
  }

  /** 从 store 恢复权重 / 提案 / 已批准技能。构造后、首个 record 前调用一次。多次调用安全。 */
  async hydrate(): Promise<void> {
    const snap = await this.store.loadAll();
    this.weights.clear();
    for (const w of snap.weights) this.weights.set(w.skillId, { ...w });
    this.proposals.clear();
    for (const p of snap.proposals) this.proposals.set(p.proposalId, { ...p });
    this.approved.clear();
    for (const s of snap.approvedSkills) this.approved.set(s.id, { ...s });
  }

  private keyOf(query: string, taskType?: TaskTypeType): string {
    return `${taskType ?? ""}\u0000${query}`;
  }

  /** 记住本轮 query 命中的技能 id（由 assembleSkillContext 的 onSelected 回调驱动）。 */
  noteSelected(query: string, taskType: TaskTypeType | undefined, ids: readonly string[]): void {
    if (ids.length === 0) return;
    if (this.selection.size >= this.maxSelMemory) this.selection.clear();
    this.selection.set(this.keyOf(query, taskType), [...ids]);
  }

  /** 已审批生效的学习技能（供 DbSkillSource 读活跃内存状态）。 */
  listApproved(): Skill[] {
    return [...this.approved.values()];
  }

  /** 某技能的当前原始权重（供选择器做有界偏置）。 */
  weightFor(id: string): number {
    return this.weights.get(id)?.weight ?? 0;
  }

  private reinforce(id: string, signal: number): void {
    const prev = this.weights.get(id)?.weight ?? 0;
    const weight = Math.max(this.wMin, Math.min(this.wMax, prev + signal));
    const w: SkillWeight = { skillId: id, weight, lastUsed: Math.floor(this.clock() / 1000) };
    this.weights.set(id, w);
    try {
      this.store.saveWeight(w);
    } catch {
      // 写穿失败不影响内存态。
    }
  }

  /**
   * run 收尾调用：对本轮命中的技能按 outcome 强化，成功轨迹进蒸馏 buffer。
   * `extra` 可带 finalAnswer / toolsUsed 供蒸馏（缺省也能靠 query+taskType 聚类）。
   */
  recordOutcome(
    query: string,
    taskType: TaskTypeType | undefined,
    status: string,
    extra: { finalAnswer?: string; toolsUsed?: readonly string[] } = {},
  ): void {
    const key = this.keyOf(query, taskType);
    const ids = this.selection.get(key);
    if (ids) {
      const signal = skillOutcomeSignal(status);
      for (const id of ids) this.reinforce(id, signal);
      this.selection.delete(key);
    }
    if (status === "completed") {
      const sample: TrajectorySample = {
        query: this.redactor(query),
        ...(taskType ? { taskType } : {}),
        finalAnswer: this.redactor(extra.finalAnswer ?? ""),
        toolsUsed: [...(extra.toolsUsed ?? [])],
        ts: this.clock(),
      };
      this.buffer.push(sample);
      if (this.buffer.length > this.maxBuffer) this.buffer = this.buffer.slice(-this.maxBuffer);
    }
  }

  /** 调试/测试：当前成功轨迹 buffer 条数。 */
  bufferedCount(): number {
    return this.buffer.length;
  }

  private toCandidate(draft: CandidateSkillDraft): Skill {
    return {
      id: draft.id ?? `learned-${slug(draft.name)}`,
      name: draft.name,
      description: draft.description,
      triggers: draft.triggers ?? [],
      taskTypes: draft.taskTypes ?? [],
      priority: 0,
      version: "1.0.0",
      body: draft.body,
      source: "learned:db",
    };
  }

  /** 启发式：按 taskType 聚类，达阈值的簇产一个候选。 */
  private heuristicDrafts(samples: readonly TrajectorySample[]): CandidateSkillDraft[] {
    const byType = new Map<string, TrajectorySample[]>();
    for (const s of samples) {
      const k = s.taskType ?? "general";
      const arr = byType.get(k);
      if (arr) arr.push(s);
      else byType.set(k, [s]);
    }
    const out: CandidateSkillDraft[] = [];
    for (const [k, group] of byType) {
      if (group.length < this.minSamples) continue;
      const keywords = topKeywords(
        group.map((g) => g.query),
        6,
      );
      const tools = [...new Set(group.flatMap((g) => g.toolsUsed))].slice(0, 8);
      const taskType = group[0]?.taskType;
      const body = [
        `You have handled ${group.length} similar ${k} tasks successfully. Reuse this proven approach.`,
        "",
        keywords.length ? `Typical intents: ${keywords.join(", ")}.` : "",
        tools.length ? `Tools that worked well: ${tools.join(", ")}.` : "",
        "",
        "Example requests:",
        ...group.slice(0, 3).map((g) => `- ${g.query}`),
      ]
        .filter((x) => x !== "")
        .join("\n");
      out.push({
        id: `learned-${slug(k)}`,
        name: `Learned: ${k}`,
        description: keywords.length
          ? `Recurring ${k} workflow. Common intents: ${keywords.join(", ")}.`
          : `Recurring ${k} workflow (${group.length} successful runs).`,
        triggers: keywords.slice(0, 4),
        ...(taskType ? { taskTypes: [taskType] } : {}),
        body,
      });
    }
    return out;
  }

  private async draftsFrom(samples: readonly TrajectorySample[]): Promise<CandidateSkillDraft[]> {
    if (this.llmDistill) {
      try {
        const r = await this.llmDistill(samples);
        const arr = Array.isArray(r) ? r : [r];
        return arr.filter((d) => d?.name && d.body);
      } catch {
        // LLM 蒸馏失败回退启发式，保证 distill 不空手。
      }
    }
    return this.heuristicDrafts(samples);
  }

  /**
   * 用户触发一次蒸馏：把 buffer 里的成功轨迹压成候选技能提案（pending）。
   * 跨次按 candidate id 去重（已 pending 更新证据；已 approved 跳过）。消费后清空 buffer。
   * 返回本次新建/更新的 pending 提案。
   */
  async distill(): Promise<SkillProposal[]> {
    if (this.buffer.length === 0) return [];
    const samples = this.buffer;
    this.buffer = [];
    const drafts = await this.draftsFrom(samples);
    const produced: SkillProposal[] = [];
    for (const draft of drafts) {
      const candidate = this.toCandidate(draft);
      // 该 candidate 已生效则跳过（不重复提案）。
      if (this.approved.has(candidate.id)) continue;
      const evGroup = draft.taskTypes?.length
        ? samples.filter((s) => s.taskType && draft.taskTypes?.includes(s.taskType))
        : samples;
      const ev = evGroup.length ? evGroup : samples;
      const queries = [...new Set(ev.map((s) => s.query))].slice(0, 10);
      const evTaskType = draft.taskTypes?.[0] ?? ev[0]?.taskType;
      // 找已存在的同 candidate 的 pending 提案 → 更新证据而非新建。
      const existing = [...this.proposals.values()].find(
        (p) => p.status === "pending" && p.candidate.id === candidate.id,
      );
      const proposal: SkillProposal = {
        proposalId: existing?.proposalId ?? `${this.idPrefix}_${(this.seq += 1)}_${this.clock()}`,
        candidate,
        evidence: { queries, ...(evTaskType ? { taskType: evTaskType } : {}), count: ev.length },
        status: "pending",
        ts: existing?.ts ?? this.clock(),
      };
      this.proposals.set(proposal.proposalId, proposal);
      try {
        this.store.upsertProposal(proposal);
      } catch {
        // 写穿失败不影响内存态。
      }
      produced.push(proposal);
      if (this.hooks) {
        await this.hooks.emit("event.SKILL_PROPOSED", {
          proposalId: proposal.proposalId,
          skillId: candidate.id,
          evidenceCount: proposal.evidence.count,
        });
      }
    }
    return produced;
  }

  listProposals(status?: SkillProposal["status"]): SkillProposal[] {
    const all = [...this.proposals.values()];
    return status ? all.filter((p) => p.status === status) : all;
  }

  private persistProposal(p: SkillProposal): void {
    try {
      this.store.upsertProposal(p);
    } catch {
      // ignore
    }
  }

  /** 批准一条 pending 提案：写入生效技能 + 触发注册表重载。 */
  async approve(proposalId: string): Promise<SkillProposal | undefined> {
    const p = this.proposals.get(proposalId);
    if (!p || p.status !== "pending") return undefined;
    p.status = "approved";
    p.decidedAt = this.clock();
    this.persistProposal(p);
    this.approved.set(p.candidate.id, { ...p.candidate });
    try {
      this.store.upsertApprovedSkill(p.candidate);
    } catch {
      // ignore
    }
    await this.onSkillsChanged?.();
    return p;
  }

  /** 拒绝一条 pending 提案（不写入生效技能）。 */
  reject(proposalId: string): SkillProposal | undefined {
    const p = this.proposals.get(proposalId);
    if (!p || p.status !== "pending") return undefined;
    p.status = "rejected";
    p.decidedAt = this.clock();
    this.persistProposal(p);
    return p;
  }

  /** 撤销一条已批准提案：移除生效技能 + 触发注册表重载。 */
  async revoke(proposalId: string): Promise<SkillProposal | undefined> {
    const p = this.proposals.get(proposalId);
    if (!p || p.status !== "approved") return undefined;
    p.status = "revoked";
    p.decidedAt = this.clock();
    this.persistProposal(p);
    this.approved.delete(p.candidate.id);
    try {
      this.store.removeApprovedSkill(p.candidate.id);
    } catch {
      // ignore
    }
    await this.onSkillsChanged?.();
    return p;
  }

  /** 关停：释放 store 句柄。 */
  async close(): Promise<void> {
    await this.store.close();
  }
}
