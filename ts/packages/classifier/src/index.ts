/**
 * `@openintj/classifier` —— 前端可强化任务分类器。
 *
 * - `ReinforcingClassifier`：embedding kNN/质心分类 + 关键词兜底 + 使用反馈强化 + 封顶。
 * - `DEFAULT_SEEDS`：冷启动种子（零 token）。
 *
 * 与 memory 共享「使用反馈飞轮」：每次 agent.run 的 (query → outcome) 同时喂检索索引与分类器。
 */
export * from "./reinforcing-classifier.js";
export * from "./store.js";
export * from "./routing.js";
export * from "./seeds.js";
