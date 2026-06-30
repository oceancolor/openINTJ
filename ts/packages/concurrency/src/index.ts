/**
 * `@openintj/concurrency` — RFC-003 方向一（多线程 Agent 模型）原语库。
 *
 * ┌── 集成状态（截至 2026-05-30）────────────────────────────────────────┐
 * │ ✅ 已接入产品路径：`RateLimitedLlmClient`                              │
 * │    cli/server/desktop 的 agent 装配在 `opts.rateLimit` /              │
 * │    env OPENINTJ_RATE_LIMIT_QPS 下用它包裹 LLM 客户端。                  │
 * │ 🧪 实验性（仅库 + 单测，未接入 agent.run() 主路径）：                   │
 * │    Mutex / Channel / ConditionVar / AgentPool / ForkJoin / Backpressure│
 * │    它们经过完整单测，可独立用于并行编排实验；但当前 Agent 主循环          │
 * │    （TAO/ReAct）仍是单线程顺序执行，未消费这些原语。                     │
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
