/**
 * `@openintj/taskpool` — RFC-003 方向二（任务池 + 检索）原语库。
 *
 * ┌── 集成状态（截至 2026-05-30）────────────────────────────────────────┐
 * │ ✅ 已接入产品路径：`HybridRetriever`                                   │
 * │    server/desktop 的 agent 装配在 `opts.retrievalMode='hybrid'` /     │
 * │    env OPENINTJ_RETRIEVAL_MODE=hybrid 下用它做向量+关键词混合检索。      │
 * │ ✅ RFC-007 opt-in：TaskPool / TaskRun / TaskStore                       │
 * │    CLI/server/desktop 的 planning/analysis 可走有界 DAG 编排。           │
 * │ 🧪 独立 opt-in：TaskQueue / ObjectPool / AgentInstancePool / Channel    │
 * └────────────────────────────────────────────────────────────────────┘
 *
 * 启用方式与外部边界见 packages/taskpool/README.md。
 */
export * from "./shared-context.js";
export * from "./hybrid-retriever.js";
export * from "./memory-hybrid-index.js";
export * from "./task-queue.js";
export * from "./object-pool.js";
export * from "./plan-graph-adapter.js";
export * from "./task-pool.js";
export * from "./task-store.js";
export * from "./agent-instance-pool.js";
export * from "./channel.js";
export * from "./synthesizer.js";
