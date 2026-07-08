/**
 * `@openintj/concurrency` — RFC-003 方向一（多线程 Agent 模型）原语库。
 *
 * ┌── 集成状态（截至 2026-07-08）────────────────────────────────────────┐
 * │ ✅ 已接入产品路径：                                                     │
 * │    - `RateLimitedLlmClient`：三端在 opts.rateLimit / OPENINTJ_RATE_LIMIT_QPS│
 * │      下包裹 LLM 客户端。                                                │
 * │    - `forkJoin`：自一致性（selfConsistency.samples>1）并行多采样 + 投票。│
 * │    - `Semaphore`（经 forkJoin `concurrency`）：给自一致性采样设并发上限  │
 * │      （selfConsistency.maxConcurrency / OPENINTJ_SELF_CONSISTENCY_CONCURRENCY）。│
 * │ 🧪 实验性（仅库 + 单测，未接入 agent.run() 主路径）：                   │
 * │    Mutex / Channel / ConditionVar / AgentPool / Backpressure           │
 * │    它们经过完整单测，可独立用于并行编排实验。                            │
 * └────────────────────────────────────────────────────────────────────┘
 *
 * 想把实验原语接入产品：见 packages/concurrency/README.md「集成路线」。
 */
export * from "./mutex.js";
export * from "./channel.js";
export * from "./condition.js";
export * from "./agent-pool.js";
export * from "./fork-join.js";
export * from "./backpressure.js";
export * from "./rate-limited-llm.js";
