import { describe, expect, it } from "vitest";
import {
  PROMPT_SECTION_ORDERS,
  createDefaultPromptSections,
  createPromptSectionRegistry,
} from "../src/index.js";

describe("PromptSectionRegistry (RFC-011)", () => {
  it("renders default sections in locked RFC-006 order", async () => {
    const registry = createPromptSectionRegistry(
      createDefaultPromptSections({
        persona: () => "PERSONA",
        skills: () => "SKILLS",
        memory: () => "MEMORY",
      }),
    );
    expect(registry.list().map((s) => s.id)).toEqual([
      "product-behavior",
      "user-persona",
      "skills",
      "plan-advice",
      "memory",
    ]);
    expect(registry.list().map((s) => s.order)).toEqual([
      PROMPT_SECTION_ORDERS.productBehavior,
      PROMPT_SECTION_ORDERS.userPersona,
      PROMPT_SECTION_ORDERS.skills,
      PROMPT_SECTION_ORDERS.planAdvice,
      PROMPT_SECTION_ORDERS.memory,
    ]);
    const rendered = await registry.render({
      query: "q",
      history: [],
      planAdviceBlock: "PLAN",
    });
    const positions = ["[Product Behavior", "PERSONA", "SKILLS", "PLAN", "MEMORY"].map((marker) =>
      rendered.indexOf(marker),
    );
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
  });

  it("product behavior can be disabled for the A/B baseline", async () => {
    const registry = createPromptSectionRegistry(
      createDefaultPromptSections({ productBehaviorEnabled: false }),
    );
    const rendered = await registry.render({ query: "q", history: [] });
    expect(rendered).not.toContain("[Product Behavior");
  });

  it("register replaces by id and render skips empty blocks", async () => {
    const registry = createPromptSectionRegistry(
      createDefaultPromptSections({ persona: () => "old" }),
    );
    registry.register({ id: "user-persona", order: 200, render: () => "new" });
    const rendered = await registry.render({ query: "q", history: [] });
    expect(rendered).toContain("new");
    expect(rendered).not.toContain("old");
  });
});
