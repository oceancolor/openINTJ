import { randomUUID } from "node:crypto";
import type { HookBus } from "@openintj/core";

/**
 * ForkJoin —— 把任务拆分为子任务并行执行，最后合并结果。
 *
 * 用法（多 Agent 投票）：
 *   const result = await forkJoin(
 *     [agentA, agentB, agentC],
 *     (agent) => agent.run(query),
 *     { reducer: majorityVote }
 *   );
 *
 * 可观测性：传入 `opts.hooks` 后发出 `forkjoin.beforeFork` / `forkjoin.afterJoin` 事件。
 */
export interface ForkJoinOpts<T, R> {
  /** 子任务的合并器；不提供时返回 [T, ...] 数组。 */
  reducer?: (results: T[]) => R;
  /** 单个子任务超时（ms），超时计入 settledRejected。 */
  timeoutMs?: number;
  /** 最低成功数；不达将抛错。 */
  minSuccess?: number;
  /** 接 HookBus 后发出 forkjoin.* 可观测事件。 */
  hooks?: HookBus;
  /** 分组名（写进事件）。默认随机 uuid。 */
  group?: string;
}

export interface ForkJoinResult<T, R> {
  reduced: R;
  fulfilled: T[];
  rejected: Array<{ index: number; reason: unknown }>;
}

const wait = <T>(p: Promise<T>, ms: number): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`fork-join timeout after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e instanceof Error ? e : new Error(String(e)));
      },
    );
  });

export const forkJoin = async <Item, T, R = T[]>(
  items: readonly Item[],
  fn: (item: Item, index: number) => Promise<T>,
  opts: ForkJoinOpts<T, R> = {},
): Promise<ForkJoinResult<T, R>> => {
  const group = opts.group ?? randomUUID();
  const startedAt = Date.now();
  if (opts.hooks) {
    await opts.hooks.emit("forkjoin.beforeFork", { group, total: items.length });
  }
  const promises = items.map((item, i) => {
    const base = fn(item, i);
    if (opts.timeoutMs !== undefined) return wait(base, opts.timeoutMs);
    return base;
  });
  const settled = await Promise.allSettled(promises);
  const fulfilled: T[] = [];
  const rejected: Array<{ index: number; reason: unknown }> = [];
  for (let i = 0; i < settled.length; i++) {
    const s = settled[i]!;
    if (s.status === "fulfilled") fulfilled.push(s.value);
    else rejected.push({ index: i, reason: s.reason });
  }
  if (opts.hooks) {
    await opts.hooks.emit("forkjoin.afterJoin", {
      group,
      total: items.length,
      fulfilled: fulfilled.length,
      rejected: rejected.length,
      durationMs: Date.now() - startedAt,
    });
  }
  if (opts.minSuccess !== undefined && fulfilled.length < opts.minSuccess) {
    throw new Error(`forkJoin: only ${fulfilled.length} succeeded, required ${opts.minSuccess}`);
  }
  const reduced = (opts.reducer ?? ((arr: T[]) => arr as unknown as R))(fulfilled);
  return { reduced, fulfilled, rejected };
};

/** 多数投票 reducer：返回出现次数最多的元素（用 JSON.stringify 做相等性）。 */
export const majorityVote = <T>(items: T[]): T | undefined => {
  if (items.length === 0) return undefined;
  const counts = new Map<string, { count: number; value: T }>();
  for (const it of items) {
    const k = JSON.stringify(it);
    const r = counts.get(k);
    if (r) r.count++;
    else counts.set(k, { count: 1, value: it });
  }
  let best: { count: number; value: T } | undefined;
  for (const v of counts.values()) {
    if (!best || v.count > best.count) best = v;
  }
  return best?.value;
};
