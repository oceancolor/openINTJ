import type { ChatMessage, ChatOptions, LlmClient, LlmStatus } from "@openintj/core";
import { TokenBucket } from "./backpressure.js";

/**
 * RFC-003 方向 1：把 TokenBucket 嵌进 LLM 客户端的装饰器。
 *
 * 用于保护 hunyuan / ollama 等下游 API 免被瞬时洪峰打挂：
 *  - chat / visionChat 入口前先 await bucket.acquire(1)
 *  - 不改变接口语义；status 透传
 *  - 调用方完全无感
 *
 * 这个类放在 @openintj/concurrency 里好处是 server / desktop / cli 都能用同一个实现。
 */
export interface RateLimitOpts {
  /** 每秒平均允许的请求数。 */
  qps: number;
  /** 桶容量（瞬时突发上限），默认 = qps。 */
  burst?: number;
}

export class RateLimitedLlmClient implements LlmClient {
  readonly inner: LlmClient;
  readonly bucket: TokenBucket;

  constructor(inner: LlmClient, opts: RateLimitOpts) {
    this.inner = inner;
    this.bucket = new TokenBucket({
      capacity: opts.burst ?? opts.qps,
      refillRate: opts.qps,
    });
  }

  async chat(messages: ChatMessage[], opts?: ChatOptions): Promise<string> {
    await this.bucket.acquire(1);
    return this.inner.chat(messages, opts);
  }

  async visionChat(
    messages: ChatMessage[],
    image: { base64: string; mimeType: string },
    opts?: ChatOptions,
  ): Promise<string> {
    await this.bucket.acquire(1);
    return this.inner.visionChat(messages, image, opts);
  }

  getStatus(): LlmStatus {
    return this.inner.getStatus();
  }

  /** 调试 / 测试：当前可用 token 数与等待中的调用数。 */
  rateLimitStatus(): { availableTokens: number; pending: number } {
    return {
      availableTokens: this.bucket.availableTokens,
      pending: this.bucket.pending,
    };
  }
}
