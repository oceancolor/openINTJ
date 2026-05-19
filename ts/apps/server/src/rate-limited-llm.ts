// 兼容层：原实现已迁移到 @openintj/concurrency，从那里 re-export 保持现有 import 路径稳定。
export { RateLimitedLlmClient, type RateLimitOpts } from "@openintj/concurrency";
