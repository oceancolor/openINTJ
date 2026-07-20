import { afterEach, describe, expect, it } from "vitest";
import { type ServerAgent, assembleServerAgent } from "../src/agent.js";

describe("server RFC-006 Product Behavior regression", () => {
  let agent: ServerAgent | undefined;
  afterEach(async () => {
    await agent?.close();
    agent = undefined;
  });

  it("injects by default, reports treatment, emits once, and preserves prompt order", async () => {
    agent = await assembleServerAgent({
      llmProvider: "mock",
      embedProvider: "simple",
      enableDormant: true,
      enableSkills: true,
      dormantOpts: { initialPersona: { preferences: { style: "偏好阶段计划" } } },
    });
    await agent.persistentStore.addLongTermAsync("TypeScript CLI 迁移计划需要三个阶段");
    const cohorts: boolean[] = [];
    let systemPrompt = "";
    agent.hooks.on("event.PRODUCT_BEHAVIOR", (ctx) => cohorts.push(ctx.payload.enabled));
    agent.hooks.on("react.beforeThought", (ctx) => {
      systemPrompt = ctx.payload.context.systemPrompt;
    });

    await agent.run("规划 TypeScript CLI 三阶段迁移计划");

    expect(cohorts).toEqual([true]);
    expect((await agent.status()).productBehavior).toEqual({
      version: "1.1.0",
      enabled: true,
      cohort: "treatment",
    });
    const positions = ["[Product Behavior", "[用户画像]", "[技能]", "[记忆参考]"].map((marker) =>
      systemPrompt.indexOf(marker),
    );
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });

  it("explicit control omits the prompt and emits exactly one control event", async () => {
    agent = await assembleServerAgent({
      llmProvider: "mock",
      embedProvider: "simple",
      enableProductBehavior: false,
    });
    const cohorts: boolean[] = [];
    let systemPrompt = "";
    agent.hooks.on("event.PRODUCT_BEHAVIOR", (ctx) => cohorts.push(ctx.payload.enabled));
    agent.hooks.on("react.beforeThought", (ctx) => {
      systemPrompt = ctx.payload.context.systemPrompt;
    });

    await agent.run("简洁回答");

    expect(systemPrompt).not.toContain("[Product Behavior");
    expect(cohorts).toEqual([false]);
    expect((await agent.status()).productBehavior.cohort).toBe("control");
  });

  it("preserves OPENINTJ_PRODUCT_BEHAVIOR startup configuration", async () => {
    const previous = process.env["OPENINTJ_PRODUCT_BEHAVIOR"];
    process.env["OPENINTJ_PRODUCT_BEHAVIOR"] = "0";
    try {
      agent = await assembleServerAgent({ llmProvider: "mock", embedProvider: "simple" });
      expect((await agent.status()).productBehavior).toMatchObject({
        enabled: false,
        cohort: "control",
      });
    } finally {
      if (previous === undefined) delete process.env["OPENINTJ_PRODUCT_BEHAVIOR"];
      else process.env["OPENINTJ_PRODUCT_BEHAVIOR"] = previous;
    }
  });

  it("TaskPool opt-in implies its classifier prerequisite and reports activation", async () => {
    agent = await assembleServerAgent({
      llmProvider: "mock",
      embedProvider: "simple",
      enableTaskPool: true,
      enableClassifier: false,
    });

    expect(agent.classifier).toBeDefined();
    expect((await agent.status()).taskPool).toMatchObject({
      requested: true,
      active: true,
      classifierRequired: true,
      classifierEnabled: true,
      reason: "ready",
    });
  });
});
