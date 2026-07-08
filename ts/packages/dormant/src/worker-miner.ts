/**
 * Dormant 挖掘的 worker 下放 + 内联回退编排（RFC-004 §2 utility 蒸馏 worker）。
 *
 * - {@link runMineInWorker}：真实 `worker_threads` 执行器，跑 `dist/mine-worker.js`。
 * - {@link mineWithWorkerFallback}：编排器——先试 worker，任何失败（不支持 / 崩溃 / 超时）
 *   自动回退到主线程内联 `PatternMiner.mine`，保证功能不因 worker 环境问题而丢失。
 *
 * `runner` 可注入，便于单测在不起真实线程的情况下验证「成功透传」与「失败回退」两条路径。
 */
import { Worker } from "node:worker_threads";
import { PatternMiner, type PatternMinerOpts } from "./pattern-miner.js";
import type { DormantPattern, PassiveEvent } from "./types.js";

/** 可跨 worker 边界的 miner 配置（排除函数型 `llmExtract`）。 */
export type SerializableMinerOpts = Omit<Partial<PatternMinerOpts>, "llmExtract">;

/** 把一批事件的 n-gram 挖掘执行掉，产出模式列表。 */
export type OffThreadRunner = (
  events: readonly PassiveEvent[],
  opts: SerializableMinerOpts,
) => Promise<DormantPattern[]>;

interface MineWorkerMessage {
  ok: boolean;
  patterns?: DormantPattern[];
  error?: string;
}

export interface RunMineInWorkerOpts {
  /** worker 超时（毫秒）；超时则 terminate + reject（交由上层回退）。默认 15_000。 */
  timeoutMs?: number;
}

/**
 * 真实 worker 执行器：起一个一次性 worker 跑 `mine-worker.js`，拿到结果即 terminate。
 * 不做回退（把回退语义留给 {@link mineWithWorkerFallback}，职责单一便于测试）。
 */
export const runMineInWorker = (
  events: readonly PassiveEvent[],
  opts: SerializableMinerOpts,
  runOpts: RunMineInWorkerOpts = {},
): Promise<DormantPattern[]> =>
  new Promise<DormantPattern[]>((resolve, reject) => {
    let settled = false;
    const worker = new Worker(new URL("./mine-worker.js", import.meta.url), {
      workerData: { events, opts },
    });
    const finish = (fn: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      void worker.terminate();
      fn();
    };
    const timer = setTimeout(
      () => finish(() => reject(new Error("mine worker timeout"))),
      runOpts.timeoutMs ?? 15_000,
    );
    worker.once("message", (msg: MineWorkerMessage) => {
      if (msg?.ok) finish(() => resolve(msg.patterns ?? []));
      else finish(() => reject(new Error(msg?.error ?? "mine worker failed")));
    });
    worker.once("error", (err) => finish(() => reject(err)));
    worker.once("exit", (code) => {
      if (code !== 0) finish(() => reject(new Error(`mine worker exited with code ${code}`)));
    });
  });

export interface WorkerMinerDeps {
  /** 注入自定义 off-thread 执行器（测试用）；缺省用真实 {@link runMineInWorker}。 */
  runner?: OffThreadRunner;
}

export interface WorkerMineResult {
  patterns: DormantPattern[];
  /** true = worker 成功执行；false = 回退到内联主线程挖掘。 */
  usedWorker: boolean;
}

/**
 * 先试 worker，失败回退内联。返回结果与 `usedWorker` 标记（便于可观测 / 测试）。
 * 语义上与 `new PatternMiner(opts).mine(events)` 等价（worker 只是把 CPU 挪出主线程）。
 */
export const mineWithWorkerFallback = async (
  events: readonly PassiveEvent[],
  opts: SerializableMinerOpts = {},
  deps: WorkerMinerDeps = {},
): Promise<WorkerMineResult> => {
  const runner = deps.runner ?? ((e, o) => runMineInWorker(e, o));
  try {
    const patterns = await runner(events, opts);
    return { patterns, usedWorker: true };
  } catch {
    const patterns = await new PatternMiner(opts).mine(events);
    return { patterns, usedWorker: false };
  }
};
