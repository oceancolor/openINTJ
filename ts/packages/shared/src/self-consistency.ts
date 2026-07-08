/**
 * 自一致性（self-consistency）答案选择——RFC-003 方向一/二接入产品路径的纯函数部分。
 *
 * Agent 并行跑 N 个采样（用 `@openintj/concurrency` 的 forkJoin），拿到 N 个候选答案后，
 * 用这里的纯策略选出最终答案。把"选择策略"与"并行编排"解耦，便于单测与跨入口复用。
 */

export type SelfConsistencyStrategy = "majority" | "longest" | "first";

export interface AnswerCandidate {
  finalAnswer: string;
}

/** 归一化文本用于多数投票的相等性比较（去首尾、压空白、小写）。 */
const normalize = (s: string): string => s.trim().replace(/\s+/g, " ").toLowerCase();

/**
 * 从候选答案里选一个：
 *  - `first`：直接取第一个（等价于关闭自一致性）。
 *  - `longest`：取最长答案（启发式偏向"更完整"）。
 *  - `majority`（默认）：按归一化文本分组取众数；平票时取该组里最长的那条。
 */
export const selectConsistentAnswer = <T extends AnswerCandidate>(
  candidates: readonly T[],
  strategy: SelfConsistencyStrategy = "majority",
): T | undefined => {
  if (candidates.length === 0) return undefined;
  if (candidates.length === 1) return candidates[0];
  if (strategy === "first") return candidates[0];
  if (strategy === "longest") {
    return [...candidates].sort((a, b) => b.finalAnswer.length - a.finalAnswer.length)[0];
  }

  const groups = new Map<string, T[]>();
  for (const c of candidates) {
    const k = normalize(c.finalAnswer);
    const g = groups.get(k);
    if (g) g.push(c);
    else groups.set(k, [c]);
  }
  let best: T[] | undefined;
  for (const g of groups.values()) {
    if (
      !best ||
      g.length > best.length ||
      (g.length === best.length &&
        (g[0]?.finalAnswer.length ?? 0) > (best[0]?.finalAnswer.length ?? 0))
    ) {
      best = g;
    }
  }
  return best?.[0];
};

export interface SelfConsistencyConfig {
  /** 采样次数；<=1 等价关闭。 */
  samples: number;
  /** 选择策略，默认 majority。 */
  strategy: SelfConsistencyStrategy;
  /**
   * 同时在跑的采样上限（透传给 forkJoin 的 `concurrency`，内部用 Semaphore 限流）。
   * 不设 → 全并发（历史行为）。用于给昂贵的多采样设并发上限，避免一次性打满 LLM 配额。
   */
  maxConcurrency?: number;
}

/**
 * 解析自一致性配置：opts > env(OPENINTJ_SELF_CONSISTENCY=采样数, OPENINTJ_SELF_CONSISTENCY_STRATEGY,
 * OPENINTJ_SELF_CONSISTENCY_CONCURRENCY=并发上限) > 关闭。返回 undefined 表示关闭（samples<=1）。
 */
export const resolveSelfConsistency = (opts?: {
  samples?: number;
  strategy?: SelfConsistencyStrategy;
  maxConcurrency?: number;
}): SelfConsistencyConfig | undefined => {
  const envSamples = Number(process.env["OPENINTJ_SELF_CONSISTENCY"] ?? "");
  const samples =
    opts?.samples ?? (Number.isFinite(envSamples) && envSamples > 0 ? Math.floor(envSamples) : 1);
  if (!Number.isFinite(samples) || samples <= 1) return undefined;
  const envStrategy = process.env["OPENINTJ_SELF_CONSISTENCY_STRATEGY"];
  const strategy: SelfConsistencyStrategy =
    opts?.strategy ??
    (envStrategy === "longest" || envStrategy === "first" || envStrategy === "majority"
      ? envStrategy
      : "majority");
  const envConc = Number(process.env["OPENINTJ_SELF_CONSISTENCY_CONCURRENCY"] ?? "");
  const rawConc =
    opts?.maxConcurrency ??
    (Number.isFinite(envConc) && envConc > 0 ? Math.floor(envConc) : undefined);
  const cfg: SelfConsistencyConfig = { samples: Math.min(samples, 8), strategy };
  if (rawConc !== undefined && rawConc > 0) cfg.maxConcurrency = rawConc;
  return cfg;
};
