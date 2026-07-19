import { describe, expect, it } from "vitest";
import { ProductTrait } from "../src/product-behavior.js";
import { evaluateTraitAb, evaluateTraits } from "../src/trait-eval.js";
import { TRAIT_SCENARIOS } from "../src/trait-scenarios.js";

describe("RFC-006 trait evaluation", () => {
  it("has at least one scenario for every product trait", () => {
    const covered = new Set(TRAIT_SCENARIOS.map((s) => s.trait));
    expect(covered).toEqual(new Set(Object.values(ProductTrait)));
  });

  it("evaluates contrast cases and aggregates by trait", async () => {
    const scenarios = [
      {
        id: "clarify",
        trait: ProductTrait.CLARIFY_WHEN_NEEDED,
        query: "simple",
        judge: (answer: string) => answer === "direct",
        counterExample: {
          query: "ambiguous",
          judge: (answer: string) => answer === "question",
        },
      },
    ] as const;
    const report = await evaluateTraits(
      async (query) => ({ finalAnswer: query === "simple" ? "direct" : "question" }),
      scenarios,
    );
    expect(report.total).toBe(2);
    expect(report.byTrait[0]).toMatchObject({ passed: 2, total: 2, rate: 1 });
    expect(report.baselineMet).toBe(true);
  });

  it("reports treatment-control completion delta", async () => {
    const scenarios = [
      {
        id: "concise",
        trait: ProductTrait.DIRECT_CONCISE,
        query: "q",
        judge: (answer: string) => answer === "pass",
      },
    ] as const;
    const report = await evaluateTraitAb(
      async () => ({ finalAnswer: "pass" }),
      async () => ({ finalAnswer: "fail" }),
      scenarios,
    );
    expect(report.completionRateDelta).toBe(1);
  });

  it("uses structured search evidence when supplied and keeps final-answer fallback", async () => {
    const t3 = TRAIT_SCENARIOS.find((scenario) => scenario.id === "T3-search-fact")!;
    await expect(
      Promise.resolve(
        t3.judge("查证后回答", { finalAnswer: "查证后回答", evidence: { toolsUsed: [] } }),
      ),
    ).resolves.toBe(false);
    await expect(
      Promise.resolve(
        t3.judge("Node.js LTS", {
          finalAnswer: "Node.js LTS",
          evidence: { toolsUsed: ["search"] },
        }),
      ),
    ).resolves.toBe(true);
    await expect(Promise.resolve(t3.judge("无法联网确认当前版本"))).resolves.toBe(true);
  });

  it("rejects weak T4 verbosity and T7 unchecked arithmetic", async () => {
    const t4 = TRAIT_SCENARIOS.find((scenario) => scenario.id === "T4-concise")!;
    const t7 = TRAIT_SCENARIOS.find((scenario) => scenario.id === "T7-quality-check")!;

    await expect(
      Promise.resolve(t4.judge("当然！很高兴为你解释。REST 是一种架构风格。")),
    ).resolves.toBe(false);
    await expect(
      Promise.resolve(
        t4.judge("REST 是一种以资源和标准 HTTP 方法组织客户端与服务端交互的架构风格。"),
      ),
    ).resolves.toBe(true);
    await expect(Promise.resolve(t7.judge("答案是 4。"))).resolves.toBe(false);
    await expect(Promise.resolve(t7.judge("2+2=4，且 4>3，所以满足 >3。"))).resolves.toBe(true);
  });
});
