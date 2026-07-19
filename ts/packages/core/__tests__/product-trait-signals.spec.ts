import { ProductTrait } from "@openintj/shared";
import { describe, expect, it } from "vitest";
import { HookBus, attachProductTraitSignals } from "../src/index.js";

describe("RFC-006 deterministic trait signals", () => {
  it("maps planner, clarification skill, and successful search lifecycle evidence", async () => {
    const bus = new HookBus();
    const detach = attachProductTraitSignals(bus);
    const seen: Array<{ trait: string; signal: string; value: number }> = [];
    bus.on("event.PRODUCT_TRAIT_SIGNAL", (ctx) => {
      seen.push(ctx.payload);
    });

    await bus.emit("tao.afterThink", {
      plan: { planId: "p1", totalSteps: 3 },
      iteration: 1,
    });
    await bus.emit("event.SKILL_SELECTED", {
      query: "部署目标不明确，请澄清",
      skills: [{ id: "clarification", score: 1 }],
    });
    await bus.emit("tool.afterCall", {
      tool: "search",
      result: {
        toolName: "search",
        success: true,
        output: { sources: [] },
        durationMs: 1,
        traceId: "t1",
        callId: "c1",
      },
    });

    expect(seen).toEqual([
      {
        trait: ProductTrait.STRATEGIC_DECOMPOSITION,
        signal: "plan_decomposed",
        value: 3,
        source: "tao.afterThink",
      },
      {
        trait: ProductTrait.CLARIFY_WHEN_NEEDED,
        signal: "clarification_skill_selected",
        value: 1,
        source: "event.SKILL_SELECTED",
      },
      {
        trait: ProductTrait.EVIDENCE_FIRST,
        signal: "search_before_answer",
        value: 1,
        source: "tool.afterCall",
      },
    ]);
    detach();
  });

  it("does not infer signals from text or failed search calls", async () => {
    const bus = new HookBus();
    attachProductTraitSignals(bus);
    let count = 0;
    bus.on("event.PRODUCT_TRAIT_SIGNAL", () => count++);

    await bus.emit("react.afterThought", {
      thought: "I planned and searched, then need clarification",
      reactIter: 1,
      taoIter: 1,
    });
    await bus.emit("tao.afterThink", {
      plan: { planId: "p1", totalSteps: 1 },
      iteration: 1,
    });
    await bus.emit("tool.afterCall", {
      tool: "search",
      result: {
        toolName: "search",
        success: false,
        error: "offline",
        durationMs: 1,
        traceId: "t2",
        callId: "c2",
      },
    });

    expect(count).toBe(0);
  });
});
