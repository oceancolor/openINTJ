import { HookBus, TaskType } from "@openintj/core";
import { ControlPlane } from "@openintj/plane-control";
import { describe, expect, it } from "vitest";
import { prepareRunPrelude } from "../src/index.js";

const llm = {
  async chat() {
    return "FINAL: ok";
  },
  async visionChat() {
    return "vision";
  },
  getStatus() {
    return {
      provider: "test",
      model: "x",
      available: true,
      mode: "live" as const,
      status: "connected",
      visionSupported: false,
    };
  },
};

describe("prepareRunPrelude (RFC-011 shared prelude)", () => {
  it("plan advice goes to planAdviceBlock, never into the execution query", async () => {
    const prelude = await prepareRunPrelude(
      {
        hooks: new HookBus(),
        llm,
        control: new ControlPlane(),
        productBehaviorEnabled: true,
        inputStructuring: "off",
      },
      { query: "规划 TypeScript CLI 三阶段迁移方案，并列出每阶段交付物" },
    );
    expect(prelude.routedTaskType).toBe(TaskType.PLANNING);
    expect(prelude.usedPlanAdvice).toBe(true);
    expect(prelude.planAdviceBlock).toContain("【规划】");
    expect(prelude.executionQuery).not.toContain("【规划】");
    expect(prelude.workQuery).toBe(prelude.executionQuery);
  });

  it("non-planning queries get no plan block", async () => {
    const prelude = await prepareRunPrelude(
      {
        hooks: new HookBus(),
        llm,
        control: new ControlPlane(),
        productBehaviorEnabled: true,
        inputStructuring: "off",
      },
      { query: "你好" },
    );
    expect(prelude.usedPlanAdvice).toBe(false);
    expect(prelude.planAdviceBlock).toBeUndefined();
  });
});
