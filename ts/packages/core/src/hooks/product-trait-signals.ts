import { ProductTrait } from "@openintj/shared";
import type { HookBus } from "./bus.js";
import type { Unregister } from "./types.js";

/**
 * Translate deterministic lifecycle/tool facts into RFC-006 trait signals.
 *
 * Semantics are deliberately narrow:
 * - T1: the planner produced more than one step.
 * - T5: the clarification skill was selected for the run.
 * - T3: a successful search tool call completed before the final answer.
 *
 * These signals do not claim the model intended, understood, or satisfied a trait.
 */
export const attachProductTraitSignals = (bus: HookBus): Unregister => {
  const offs: Unregister[] = [];

  offs.push(
    bus.on("tao.afterThink", async (ctx) => {
      if (ctx.payload.plan.totalSteps <= 1) return;
      await bus.emit(
        "event.PRODUCT_TRAIT_SIGNAL",
        {
          trait: ProductTrait.STRATEGIC_DECOMPOSITION,
          signal: "plan_decomposed",
          value: ctx.payload.plan.totalSteps,
          source: "tao.afterThink",
        },
        { traceId: ctx.traceId },
      );
    }),
  );

  offs.push(
    bus.on("event.SKILL_SELECTED", async (ctx) => {
      if (!ctx.payload.skills.some((skill) => skill.id === "clarification")) return;
      await bus.emit(
        "event.PRODUCT_TRAIT_SIGNAL",
        {
          trait: ProductTrait.CLARIFY_WHEN_NEEDED,
          signal: "clarification_skill_selected",
          value: 1,
          source: "event.SKILL_SELECTED",
        },
        { traceId: ctx.traceId },
      );
    }),
  );

  offs.push(
    bus.on("tool.afterCall", async (ctx) => {
      if (ctx.payload.tool !== "search" || !ctx.payload.result.success) return;
      await bus.emit(
        "event.PRODUCT_TRAIT_SIGNAL",
        {
          trait: ProductTrait.EVIDENCE_FIRST,
          signal: "search_before_answer",
          value: 1,
          source: "tool.afterCall",
        },
        { traceId: ctx.traceId },
      );
    }),
  );

  return () => {
    for (const off of offs) off();
  };
};
