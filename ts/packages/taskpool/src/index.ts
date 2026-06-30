/**
 * `@openintj/taskpool` — RFC-003 方向二（任务池 + 检索）原语库。
 *
 * ┌── 集成状态（截至 2026-05-30）────────────────────────────────────────┐
 * │ ✅ 已接入产品路径：`HybridRetriever`                                   │
 * │    server/desktop 的 agent 装配在 `opts.retrievalMode='hybrid'` /     │
 * │    env OPENINTJ_RETRIEVAL_MODE=hybrid 下用它做向量+关键词混合检索。      │
 * │ 🧪 实验性（仅库 + 单测，未接入 agent.run() 主路径）：                   │
 * │    SharedContext / TaskQueue / ObjectPool                              │
 * │    经过完整单测，可独立用于多任务编排实验；但当前单 Agent 会话           │
 * │    不消费任务队列 / 对象池。                                            │
 * └────────────────────────────────────────────────────────────────────┘
 *
 * 想把实验原语接入产品：见 packages/taskpool/README.md「集成路线」。
 */
export * from "./shared-context.js";
export * from "./hybrid-retriever.js";
export * from "./task-queue.js";
export * from "./object-pool.js";
