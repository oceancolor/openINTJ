import type { LlmClient } from "@openintj/core";
import { describe, expect, it, vi } from "vitest";
import {
  buildExplicitPlan,
  formatExplicitPlanBlock,
  parseExplicitPlan,
  serializePlanForClient,
  shouldBuildExplicitPlan,
} from "../src/main/explicit-plan.js";

const GOOD_JSON = JSON.stringify({
  goal: "做一份贵州7日自驾游计划",
  steps: [
    { title: "确定逐日路线与驾驶里程", acceptance: "7 天路线表完整且每日里程 ≤ 350km", dependencies: [] },
    { title: "筛选并预订每晚住宿", acceptance: "7 晚住宿清单含价格与取消政策", dependencies: [1] },
    { title: "汇总预算与备选方案", acceptance: "总预算表与雨天备选齐全", dependencies: [1, 2] },
  ],
});

const llmReturning = (text: string): LlmClient =>
  ({
    chat: vi.fn(async () => text),
    visionChat: vi.fn(async () => ""),
    getStatus: () => ({ provider: "test", model: "test", ready: true }),
  }) as unknown as LlmClient;

describe("parseExplicitPlan", () => {
  it("解析合格 JSON，依赖序号映射为 sN", () => {
    const plan = parseExplicitPlan(GOOD_JSON);
    expect(plan).not.toBeNull();
    expect(plan?.goal).toBe("做一份贵州7日自驾游计划");
    expect(plan?.steps).toHaveLength(3);
    expect(plan?.steps[1]?.dependencies).toEqual(["s1"]);
    expect(plan?.steps[2]?.dependencies).toEqual(["s1", "s2"]);
  });

  it("容忍 markdown 围栏与前后噪声", () => {
    const plan = parseExplicitPlan("好的，这是计划：\n```json\n" + GOOD_JSON + "\n```\n请查收");
    expect(plan?.steps).toHaveLength(3);
  });

  it("拒绝空泛/不合格输出（<2 步、无目标、坏 JSON）", () => {
    expect(parseExplicitPlan("not json")).toBeNull();
    expect(parseExplicitPlan(JSON.stringify({ goal: "x", steps: [] }))).toBeNull();
    expect(
      parseExplicitPlan(
        JSON.stringify({ goal: "x", steps: [{ title: "分析需求", acceptance: "好" }] }),
      ),
    ).toBeNull();
    expect(parseExplicitPlan(JSON.stringify({ steps: [{ title: "a" }, { title: "b" }] }))).toBeNull();
  });

  it("缺验收时填默认可核对标准，超 7 步截断", () => {
    const many = JSON.stringify({
      goal: "g",
      steps: Array.from({ length: 9 }, (_, i) => ({ title: `步${i + 1}` })),
    });
    const plan = parseExplicitPlan(many);
    expect(plan?.steps).toHaveLength(7);
    expect(plan?.steps[0]?.acceptance).toBe("完成且结果可核对");
  });
});

describe("shouldBuildExplicitPlan（门控）", () => {
  const base = {
    orchestrationMode: "direct",
    isPreflight: false,
    isClarify: false,
    isArtifactRequest: false,
    queryLength: 40,
  };
  it("planning/analysis 触发，quick/琐碎不触发", () => {
    expect(shouldBuildExplicitPlan({ ...base, routedTaskType: "planning" })).toBe(true);
    expect(shouldBuildExplicitPlan({ ...base, routedTaskType: "analysis" })).toBe(true);
    expect(shouldBuildExplicitPlan({ ...base, routedTaskType: "quick_response" })).toBe(false);
    expect(shouldBuildExplicitPlan({ ...base, routedTaskType: "planning", queryLength: 5 })).toBe(
      false,
    );
  });
  it("taskpool 编排恒触发；preflight/clarify/artifact 恒跳过", () => {
    expect(
      shouldBuildExplicitPlan({ ...base, orchestrationMode: "taskpool", routedTaskType: "other" }),
    ).toBe(true);
    expect(shouldBuildExplicitPlan({ ...base, routedTaskType: "planning", isPreflight: true })).toBe(
      false,
    );
    expect(shouldBuildExplicitPlan({ ...base, routedTaskType: "planning", isClarify: true })).toBe(
      false,
    );
    expect(
      shouldBuildExplicitPlan({ ...base, routedTaskType: "planning", isArtifactRequest: true }),
    ).toBe(false);
  });
});

describe("buildExplicitPlan", () => {
  it("LLM 输出合格 → 返回计划", async () => {
    const plan = await buildExplicitPlan({ llm: llmReturning(GOOD_JSON), query: "做贵州7日游计划" });
    expect(plan?.steps).toHaveLength(3);
  });

  it("LLM 抛错/输出垃圾 → null（fail-soft 不阻断）", async () => {
    const throwing = {
      chat: vi.fn(async () => {
        throw new Error("boom");
      }),
      visionChat: vi.fn(async () => ""),
      getStatus: () => ({ provider: "t", model: "t", ready: true }),
    } as unknown as LlmClient;
    expect(await buildExplicitPlan({ llm: throwing, query: "x".repeat(20) })).toBeNull();
    expect(await buildExplicitPlan({ llm: llmReturning("胡言乱语"), query: "x".repeat(20) })).toBeNull();
  });
});

describe("format/serialize", () => {
  it("注入块含目标、步骤、验收与否决条款", () => {
    const plan = parseExplicitPlan(GOOD_JSON)!;
    const block = formatExplicitPlanBlock(plan);
    expect(block).toContain("【执行计划（显性化）】");
    expect(block).toContain("验收：7 天路线表完整且每日里程 ≤ 350km");
    expect(block).toContain("以原始要求与证据为准");
    const client = serializePlanForClient(plan);
    expect(client.steps[0]).toEqual({
      id: "s1",
      title: "确定逐日路线与驾驶里程",
      acceptance: "7 天路线表完整且每日里程 ≤ 350km",
    });
  });
});
