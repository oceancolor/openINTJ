/**
 * ReinforcingClassifier —— 前端「可强化」任务分类器。
 *
 * 目标（roadmap 前端分类器）：用一个轻量、本地、零额外 LLM token 的分类器在 agent.run 前
 * 给 query 打 TaskType 标签，用来：
 *  - 降 token：高置信「简单」类路由到单次 LLM 调用（跳过 ReAct 微循环 / 工具描述）。
 *  - 提命中：把标签写进记忆 taskTags，与检索的 taskType ×1.3 加成复利。
 * 并且随使用「越用越好」：每次 run 的 outcome 反馈回来强化 exemplar。
 *
 * 实现：带权 exemplar 的 embedding kNN/质心分类 + 软置信度；低置信或冷启动回退关键词启发式
 * （`detectTaskType`，本地零 token）。与 dormant / memory 共享同一套「使用反馈」哲学。
 */

import {
  type EmbeddingProvider,
  type TaskTypeType,
  cosineSimilarity,
  detectTaskType,
} from "@openintj/core";
import type { ClassifierStore } from "./store.js";

export interface Exemplar {
  vector: number[];
  label: TaskTypeType;
  /** 累计权重（reinforce 加，penalize 减）。 */
  weight: number;
  /** 最近使用时间（秒），用于 LRU 淘汰。 */
  lastUsed: number;
}

export interface SeedExample {
  text: string;
  label: TaskTypeType;
  weight?: number;
}

export interface ClassifyResult {
  label: TaskTypeType;
  /** 0-1 软置信度（最佳标签得分 / 全标签得分）。 */
  confidence: number;
  /** 各标签归一化得分（和为 1，仅含出现过的标签）。 */
  scores: Partial<Record<TaskTypeType, number>>;
  /** 是否走了关键词兜底（无 exemplar 或低于 minConfidence）。 */
  fallback: boolean;
}

export interface ReinforcingClassifierOpts {
  embedder: EmbeddingProvider;
  /** kNN 邻居数，默认 5。 */
  k?: number;
  /** 低于此置信度回退关键词启发式，默认 0.55。 */
  minConfidence?: number;
  /** 同标签 exemplar 合并相似度阈值（高相似则升权而非新增），默认 0.92。 */
  mergeThreshold?: number;
  /** exemplar 上限，默认 500；超出按 weight 升序 + LRU 淘汰。 */
  maxExemplars?: number;
  /** 关键词兜底分类器，默认 core.detectTaskType。 */
  fallbackClassify?: (query: string) => TaskTypeType;
  /** 可选持久化：注入后 hydrate() 载入、reinforce 后自动落盘。默认仅内存。 */
  store?: ClassifierStore;
  clock?: () => number;
}

/** 可序列化状态（CLF.2 持久化用）。 */
export interface ClassifierState {
  exemplars: Exemplar[];
}

export interface ReinforceOpts {
  /** 反馈强度：>0 强化（成功），<0 惩罚（标签判错）。默认 +1。 */
  signal?: number;
}

const embed = async (e: EmbeddingProvider, text: string): Promise<number[]> => {
  const r = e.embed(text);
  return r instanceof Promise ? await r : r;
};

export class ReinforcingClassifier {
  private readonly embedder: EmbeddingProvider;
  private readonly k: number;
  private readonly minConfidence: number;
  private readonly mergeThreshold: number;
  private readonly maxExemplars: number;
  private readonly fallbackClassify: (query: string) => TaskTypeType;
  private readonly store: ClassifierStore | undefined;
  private readonly clock: () => number;
  private exemplars: Exemplar[] = [];

  constructor(opts: ReinforcingClassifierOpts) {
    this.embedder = opts.embedder;
    this.k = opts.k ?? 5;
    this.minConfidence = opts.minConfidence ?? 0.55;
    this.mergeThreshold = opts.mergeThreshold ?? 0.92;
    this.maxExemplars = opts.maxExemplars ?? 500;
    this.fallbackClassify = opts.fallbackClassify ?? detectTaskType;
    this.store = opts.store;
    this.clock = opts.clock ?? (() => Date.now() / 1000);
  }

  /** 从持久化层载入状态（若注入了 store）。装配时调用一次。 */
  async hydrate(): Promise<void> {
    if (!this.store) return;
    const state = await this.store.load();
    if (state) this.loadState(state);
  }

  /** 落盘当前状态（若注入了 store）。reinforce/addSeeds 后自动调用，吞错不阻断主流程。 */
  private persist(): void {
    if (!this.store) return;
    try {
      const r = this.store.save(this.toState());
      if (r instanceof Promise) r.catch(() => {});
    } catch {
      // 持久化抖动不影响分类
    }
  }

  get size(): number {
    return this.exemplars.length;
  }

  /** 批量加入种子 exemplar（embed 后入库）。 */
  async addSeeds(seeds: readonly SeedExample[]): Promise<void> {
    for (const s of seeds) {
      const vector = await embed(this.embedder, s.text);
      this.exemplars.push({
        vector,
        label: s.label,
        weight: s.weight ?? 1,
        lastUsed: this.clock(),
      });
    }
    this.enforceCap();
    this.persist();
  }

  /**
   * 分类：embed → 带权 kNN 聚合 → 软置信度。
   * 无 exemplar 或置信度 < minConfidence 时回退关键词启发式（fallback=true）。
   */
  async classify(query: string): Promise<ClassifyResult> {
    if (this.exemplars.length === 0) {
      return { label: this.fallbackClassify(query), confidence: 0, scores: {}, fallback: true };
    }
    const q = await embed(this.embedder, query);
    const sims = this.exemplars
      .map((e) => ({
        label: e.label,
        sim: Math.max(0, cosineSimilarity(q, e.vector)),
        weight: e.weight,
      }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, this.k);

    const perLabel = new Map<TaskTypeType, number>();
    for (const s of sims) {
      perLabel.set(s.label, (perLabel.get(s.label) ?? 0) + s.sim * Math.max(0, s.weight));
    }
    const total = [...perLabel.values()].reduce((a, b) => a + b, 0);
    const scores: Partial<Record<TaskTypeType, number>> = {};
    let bestLabel: TaskTypeType | undefined;
    let bestScore = -1;
    for (const [label, score] of perLabel) {
      const norm = total > 0 ? score / total : 0;
      scores[label] = norm;
      if (score > bestScore) {
        bestScore = score;
        bestLabel = label;
      }
    }
    const confidence = total > 0 ? bestScore / total : 0;
    if (bestLabel === undefined || confidence < this.minConfidence) {
      return { label: this.fallbackClassify(query), confidence, scores, fallback: true };
    }
    return { label: bestLabel, confidence, scores, fallback: false };
  }

  /**
   * 反馈强化：成功（signal>0）则把 query 当作 label 的正例（合并升权或新增 exemplar）；
   * 判错（signal<0）则衰减该 label 附近的 exemplar。
   */
  async reinforce(query: string, label: TaskTypeType, opts: ReinforceOpts = {}): Promise<void> {
    const signal = opts.signal ?? 1;
    if (signal === 0) return;
    const q = await embed(this.embedder, query);
    const now = this.clock();

    // 找同标签里最相似的 exemplar
    let bestIdx = -1;
    let bestSim = -1;
    for (let i = 0; i < this.exemplars.length; i++) {
      const e = this.exemplars[i]!;
      if (e.label !== label) continue;
      const sim = cosineSimilarity(q, e.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }

    if (signal > 0) {
      if (bestIdx >= 0 && bestSim >= this.mergeThreshold) {
        // 高相似 → 升权 + 向 query EMA 微调质心 + 刷新 LRU
        const e = this.exemplars[bestIdx]!;
        e.weight += signal;
        e.lastUsed = now;
        emaInPlace(e.vector, q, 0.2);
      } else {
        this.exemplars.push({ vector: q, label, weight: Math.max(0.5, signal), lastUsed: now });
      }
    } else {
      // 惩罚：衰减该 label 附近 exemplar（含合并阈值内的），权重归零则移除
      const keep: Exemplar[] = [];
      for (const e of this.exemplars) {
        if (e.label === label && cosineSimilarity(q, e.vector) >= this.mergeThreshold) {
          e.weight += signal; // signal<0
          if (e.weight > 0) keep.push(e);
        } else {
          keep.push(e);
        }
      }
      this.exemplars = keep;
    }
    this.enforceCap();
    this.persist();
  }

  /** 导出状态（持久化）。 */
  toState(): ClassifierState {
    return { exemplars: this.exemplars.map((e) => ({ ...e, vector: [...e.vector] })) };
  }

  /** 载入状态（持久化 hydrate）。覆盖现有 exemplar。 */
  loadState(state: ClassifierState): void {
    this.exemplars = state.exemplars.map((e) => ({ ...e, vector: [...e.vector] }));
    this.enforceCap();
  }

  /** 超过上限时按 weight 升序、再按 lastUsed 升序淘汰最弱/最旧。 */
  private enforceCap(): void {
    if (this.exemplars.length <= this.maxExemplars) return;
    this.exemplars.sort((a, b) => b.weight - a.weight || b.lastUsed - a.lastUsed);
    this.exemplars.length = this.maxExemplars;
  }
}

/** 把 target 向量按 alpha 朝 source EMA 微调（就地）。 */
const emaInPlace = (target: number[], source: readonly number[], alpha: number): void => {
  const n = Math.min(target.length, source.length);
  for (let i = 0; i < n; i++) {
    target[i] = (1 - alpha) * (target[i] ?? 0) + alpha * (source[i] ?? 0);
  }
};
