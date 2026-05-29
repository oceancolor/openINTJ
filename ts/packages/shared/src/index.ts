/**
 * 跨包工具
 *
 * 当前内容：
 *   - env.ts：仓库根 .env / .env.local 自动加载 + LLM 环境摘要
 *
 * 后续将放置：
 *   - 通用工具函数（深拷贝、debounce、retry 等）
 *   - 通用错误基类
 *   - 通用 logger 接口
 *   - traceId 生成器
 */

export * from "./env.js";
export * from "./agent-prompt.js";

export const __sharedPlaceholder = true;
