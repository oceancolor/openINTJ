/**
 * LLM 速率限制装饰器测试（RFC-003 方向 1 + Phase 3.3.C）。
 *
 * 覆盖：
 *  1. 默认不启用速率限制：assembleServerAgent 不包 RateLimitedLlmClient
 *  2. opts.rateLimit = { qps, burst } 启用，agent.llm 是 RateLimitedLlmClient
 *  3. env OPENINTJ_RATE_LIMIT_QPS=N 也能启用
 *  4. 短时间内 burst+1 次调用必须延迟（验证 bucket 真的在控制速率）
 *  5. 装饰器透传 status，对消费侧透明
 */
import { performance } from "node:perf_hooks";
import type { ChatMessage, ChatOptions, LlmClient, LlmStatus } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { assembleServerAgent } from "../src/agent.js";
import { RateLimitedLlmClient } from "../src/rate-limited-llm.js";

class CountingLlm implements LlmClient {
  calls = 0;
  async chat(_messages: ChatMessage[], _opts?: ChatOptions): Promise<string> {
    this.calls += 1;
    return "ok";
  }
  async visionChat(): Promise<string> {
    this.calls += 1;
    return "ok-vision";
  }
  getStatus(): LlmStatus {
    return {
      provider: "counting",
      model: "test",
      configured: true,
      visionSupported: false,
    };
  }
}

describe("RateLimitedLlmClient", () => {
  it("透传 chat 结果 + 计数", async () => {
    const inner = new CountingLlm();
    const limited = new RateLimitedLlmClient(inner, { qps: 100, burst: 100 });
    const r = await limited.chat([{ role: "user", content: "hi" }]);
    expect(r).toBe("ok");
    expect(inner.calls).toBe(1);
  });

  it("透传 getStatus", () => {
    const inner = new CountingLlm();
    const limited = new RateLimitedLlmClient(inner, { qps: 10 });
    expect(limited.getStatus().provider).toBe("counting");
  });

  it("burst+1 次连续调用至少跨一个补桶周期", async () => {
    const inner = new CountingLlm();
    // qps=4, burst=2 → 前 2 调用秒过，第 3 调用必须等 ~250ms
    const limited = new RateLimitedLlmClient(inner, { qps: 4, burst: 2 });
    const start = performance.now();
    await Promise.all([
      limited.chat([{ role: "user", content: "a" }]),
      limited.chat([{ role: "user", content: "b" }]),
      limited.chat([{ role: "user", content: "c" }]),
    ]);
    const elapsed = performance.now() - start;
    expect(inner.calls).toBe(3);
    // 第三次至少补 1 个 token 所需的 1/qps 秒 = 250ms，给点容差
    expect(elapsed).toBeGreaterThanOrEqual(200);
  });

  it("rateLimitStatus 报当前可用 token 数", async () => {
    const inner = new CountingLlm();
    const limited = new RateLimitedLlmClient(inner, { qps: 10, burst: 5 });
    await limited.chat([{ role: "user", content: "x" }]);
    const status = limited.rateLimitStatus();
    expect(status.availableTokens).toBeLessThanOrEqual(5);
    expect(status.pending).toBe(0);
  });
});

describe("assembleServerAgent rate-limit wiring", () => {
  it("默认不启用：agent.llm 不是 RateLimitedLlmClient", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    expect(agent.llm instanceof RateLimitedLlmClient).toBe(false);
    await agent.close();
  });

  it("opts.rateLimit 启用包装", async () => {
    const agent = await assembleServerAgent({
      llmProvider: "mock",
      rateLimit: { qps: 5 },
    });
    expect(agent.llm instanceof RateLimitedLlmClient).toBe(true);
    await agent.close();
  });

  it("OPENINTJ_RATE_LIMIT_QPS 启用包装", async () => {
    const prev = process.env["OPENINTJ_RATE_LIMIT_QPS"];
    process.env["OPENINTJ_RATE_LIMIT_QPS"] = "10";
    try {
      const agent = await assembleServerAgent({ llmProvider: "mock" });
      expect(agent.llm instanceof RateLimitedLlmClient).toBe(true);
      await agent.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_RATE_LIMIT_QPS"];
      else process.env["OPENINTJ_RATE_LIMIT_QPS"] = prev;
    }
  });

  it("OPENINTJ_RATE_LIMIT_QPS=0 / 非数字 不启用", async () => {
    const prev = process.env["OPENINTJ_RATE_LIMIT_QPS"];
    for (const bad of ["0", "-3", "abc", ""]) {
      if (bad) process.env["OPENINTJ_RATE_LIMIT_QPS"] = bad;
      else delete process.env["OPENINTJ_RATE_LIMIT_QPS"];
      const agent = await assembleServerAgent({ llmProvider: "mock" });
      expect(agent.llm instanceof RateLimitedLlmClient).toBe(false);
      await agent.close();
    }
    if (prev === undefined) delete process.env["OPENINTJ_RATE_LIMIT_QPS"];
    else process.env["OPENINTJ_RATE_LIMIT_QPS"] = prev;
  });

  it("opts.rateLimit 覆盖 env（同时设置时以 opts 为准）", async () => {
    const prev = process.env["OPENINTJ_RATE_LIMIT_QPS"];
    process.env["OPENINTJ_RATE_LIMIT_QPS"] = "100";
    try {
      const agent = await assembleServerAgent({
        llmProvider: "mock",
        rateLimit: { qps: 2 },
      });
      const limited = agent.llm as RateLimitedLlmClient;
      expect(limited.bucket.refillRate).toBe(2);
      await agent.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_RATE_LIMIT_QPS"];
      else process.env["OPENINTJ_RATE_LIMIT_QPS"] = prev;
    }
  });
});
