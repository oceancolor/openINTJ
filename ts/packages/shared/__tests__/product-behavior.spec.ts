import { describe, expect, it } from "vitest";
import {
  PRODUCT_BEHAVIOR_VERSION,
  ProductTrait,
  assembleSystemPromptPrefix,
  buildProductBehaviorPrompt,
  resolveProductBehaviorEnabled,
} from "../src/product-behavior.js";

describe("Product Behavior contract", () => {
  it("renders a versioned contract containing all eight traits", () => {
    const prompt = buildProductBehaviorPrompt();
    expect(prompt).toContain(`[Product Behavior v${PRODUCT_BEHAVIOR_VERSION}]`);
    expect(Object.values(ProductTrait)).toHaveLength(8);
    expect(prompt.match(/^\d+\./gm) ?? []).toHaveLength(8);
    expect(prompt).not.toContain("你是 INTJ");
  });

  it("assembles Product Behavior before persona and skills", () => {
    const prompt = assembleSystemPromptPrefix({
      base: "[Base]",
      userPersona: "[User Persona]",
      skillBlock: "[Skills]",
    });
    expect(prompt.indexOf("[Product Behavior")).toBeLessThan(prompt.indexOf("[User Persona]"));
    expect(prompt.indexOf("[User Persona]")).toBeLessThan(prompt.indexOf("[Skills]"));
  });

  it("supports explicit and environment A/B controls", () => {
    expect(resolveProductBehaviorEnabled(undefined, {} as NodeJS.ProcessEnv)).toBe(true);
    expect(
      resolveProductBehaviorEnabled(undefined, {
        OPENINTJ_PRODUCT_BEHAVIOR: "0",
      } as NodeJS.ProcessEnv),
    ).toBe(false);
    expect(
      resolveProductBehaviorEnabled(true, {
        OPENINTJ_PRODUCT_BEHAVIOR: "0",
      } as NodeJS.ProcessEnv),
    ).toBe(true);
    expect(buildProductBehaviorPrompt({ enabled: false })).toBe("");
  });
});
