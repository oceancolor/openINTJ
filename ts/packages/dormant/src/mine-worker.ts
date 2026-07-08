/**
 * Dormant 挖掘 worker 入口（RFC-004 §2 拓扑：utility 蒸馏 worker）。
 *
 * 把 CPU 密集的 n-gram 挖掘放到独立 worker 线程，避免阻塞 Electron main / server 事件循环。
 * 只处理**可序列化**的挖掘（无 `llmExtract`——函数无法跨 worker 边界；带 LLM 抽取的挖掘留在主线程）。
 *
 * 协议：
 *  - 主线程通过 `workerData = { events, opts }` 传入被动事件与 miner 配置
 *  - worker 跑完 `PatternMiner.mine` 后 `postMessage({ ok: true, patterns })`
 *  - 失败则 `postMessage({ ok: false, error })`（主线程据此回退内联挖掘）
 *
 * 该文件被 tsc 编成 `dist/mine-worker.js`；`worker-miner.ts` 用
 * `new URL("./mine-worker.js", import.meta.url)` 相对解析它。dormant 包在打包时被
 * externalize（不进 bundle），因此 dist 会随 node_modules 一起分发，运行时可解析。
 */
import { parentPort, workerData } from "node:worker_threads";
import { PatternMiner, type PatternMinerOpts } from "./pattern-miner.js";
import type { PassiveEvent } from "./types.js";

interface MineWorkerInput {
  events: PassiveEvent[];
  opts: Omit<Partial<PatternMinerOpts>, "llmExtract">;
}

const run = async (): Promise<void> => {
  const { events, opts } = (workerData ?? { events: [], opts: {} }) as MineWorkerInput;
  const patterns = await new PatternMiner(opts).mine(events);
  parentPort?.postMessage({ ok: true, patterns });
};

run().catch((err: unknown) => {
  parentPort?.postMessage({
    ok: false,
    error: err instanceof Error ? err.message : String(err),
  });
});
