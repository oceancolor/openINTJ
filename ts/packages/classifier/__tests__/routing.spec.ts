import { TaskType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import type { ClassifyResult } from "../src/reinforcing-classifier.js";
import { decideRoute, outcomeSignal } from "../src/routing.js";

const cls = (over: Partial<ClassifyResult> = {}): ClassifyResult => ({
  label: TaskType.QUICK_RESPONSE,
  confidence: 0.9,
  scores: {},
  fallback: false,
  ...over,
});

describe("decideRoute", () => {
  it("高置信 + 简单类 → single=true（触发 enableReact:false 退化）+ 小 topK", () => {
    const r = decideRoute(cls({ label: TaskType.QUICK_RESPONSE, confidence: 0.9 }));
    expect(r.single).toBe(true);
    expect(r.topK).toBe(3);
  });

  it("GENERAL_CHAT 同样视为简单类", () => {
    expect(decideRoute(cls({ label: TaskType.GENERAL_CHAT, confidence: 0.7 })).single).toBe(true);
  });

  it("兜底分类（fallback=true）一律不激进路由 → single=false + 默认 topK", () => {
    const r = decideRoute(cls({ confidence: 0.99, fallback: true }));
    expect(r.single).toBe(false);
    expect(r.topK).toBe(6);
  });

  it("置信度低于阈值 → single=false", () => {
    expect(decideRoute(cls({ confidence: 0.5 })).single).toBe(false);
  });

  it("非简单类（如代码任务）→ single=false + 默认 topK", () => {
    const r = decideRoute(cls({ label: TaskType.CODE_GENERATION, confidence: 0.95 }));
    expect(r.single).toBe(false);
    expect(r.topK).toBe(6);
  });

  it("planning / analysis 复杂类永不 single（RFC-006 护栏）", () => {
    expect(decideRoute(cls({ label: TaskType.PLANNING, confidence: 0.99 })).single).toBe(false);
    expect(decideRoute(cls({ label: TaskType.ANALYSIS, confidence: 0.99 })).single).toBe(false);
  });

  it("policy 可覆盖简单类集合 / 阈值 / topK", () => {
    const r = decideRoute(cls({ label: TaskType.CODE_GENERATION, confidence: 0.65 }), {
      simpleTypes: [TaskType.CODE_GENERATION],
      minConfidence: 0.6,
      simpleTopK: 2,
      defaultTopK: 9,
    });
    expect(r.single).toBe(true);
    expect(r.topK).toBe(2);
  });
});

describe("outcomeSignal", () => {
  it("completed → +1", () => {
    expect(outcomeSignal("completed")).toBe(1);
  });
  it("failed / timeout → -0.5", () => {
    expect(outcomeSignal("failed")).toBe(-0.5);
    expect(outcomeSignal("timeout")).toBe(-0.5);
  });
  it("其它（max_iter 等）→ +0.2 弱正", () => {
    expect(outcomeSignal("max_iterations")).toBe(0.2);
  });
});
