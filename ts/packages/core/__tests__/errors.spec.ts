import { describe, expect, it } from "vitest";
import {
  AgentError,
  ErrorCode,
  isContextOverflowError,
  matchesContextOverflowText,
} from "../src/index.js";

describe("isContextOverflowError (RFC-012)", () => {
  it("recognizes the structured CONTEXT_OVERFLOW code", () => {
    const err = new AgentError({
      code: ErrorCode.CONTEXT_OVERFLOW,
      message: "too big",
      retriable: false,
    });
    expect(isContextOverflowError(err)).toBe(true);
  });

  it("recognizes details.overflow markers", () => {
    expect(isContextOverflowError({ details: { overflow: true } })).toBe(true);
    expect(isContextOverflowError(new Error("details lost"))).toBe(false);
  });

  it("matches known provider phrasings", () => {
    for (const text of [
      "This model's maximum context length is 8192 tokens",
      "context_length_exceeded",
      "prompt is too long: 213462 tokens > 200000 maximum",
      "the request exceeds the available context size, try increasing it",
      "输入内容超过模型上下文长度上限",
      "上下文长度超限，请减少输入",
    ]) {
      expect(matchesContextOverflowText(text), text).toBe(true);
      expect(isContextOverflowError(new Error(text)), text).toBe(true);
    }
  });

  it("does not match generic failures, quotas, or bare status codes", () => {
    for (const text of [
      "HTTP 400 bad request",
      "rate limit: too many requests",
      "token quota exceeded for this hour",
      "Ollama 调用失败 (500): internal error",
      "connection refused",
    ]) {
      expect(matchesContextOverflowText(text), text).toBe(false);
    }
    expect(isContextOverflowError(undefined)).toBe(false);
    expect(isContextOverflowError(42)).toBe(false);
  });
});
