/**
 * RFC-006 trait baseline / A-B harness.
 *
 * RUN_TRAIT_EVAL=1 OPENINTJ_LLM_PROVIDER=ollama \
 *   pnpm --filter @openintj/cli test -- trait.harness
 */
import { TRAIT_SCENARIOS, evaluateTraitAb, evaluateTraits } from "@openintj/shared";
import { describe, expect, it } from "vitest";
import { type LlmProvider, assembleAgentAsync } from "../src/agent.js";

const RUN = process.env["RUN_TRAIT_EVAL"] === "1";

describe("RFC-006 trait baseline (gated)", () => {
  it.runIf(RUN)(
    "records treatment baseline and control delta",
    async () => {
      const provider = (process.env["OPENINTJ_LLM_PROVIDER"] as LlmProvider) ?? "ollama";
      const treatment = await assembleAgentAsync({
        llmProvider: provider,
        maxTaoIterations: 2,
        enableProductBehavior: true,
        enableSkills: true,
      });
      const control = await assembleAgentAsync({
        llmProvider: provider,
        maxTaoIterations: 2,
        enableProductBehavior: false,
        enableSkills: true,
      });
      const asEvalOutput = (result: Awaited<ReturnType<typeof treatment.run>>) => ({
        finalAnswer: result.finalAnswer,
        evidence: {
          trajectory: result.trajectory,
          toolsUsed: result.trajectory
            .map((entry) => {
              const state = entry.state as {
                type?: string;
                tool?: string;
                toolResult?: { toolName?: string };
              };
              return state.type === "action" ? state.tool : state.toolResult?.toolName;
            })
            .filter((tool): tool is string => Boolean(tool)),
        },
      });
      const runTreatment = async (query: string) => {
        const result = await treatment.run(query);
        return asEvalOutput(result);
      };
      const runControl = async (query: string) => {
        const result = await control.run(query);
        return asEvalOutput(result);
      };

      const report = await evaluateTraitAb(runTreatment, runControl);
      console.log(`[trait-ab-json] ${JSON.stringify(report)}`);
      expect(report.treatment.baselineMet).toBe(true);
      expect(report.treatment.completionRate).toBeGreaterThanOrEqual(
        report.control.completionRate - 0.1,
      );
    },
    600_000,
  );

  it("passes every judge with a deterministic scripted runner in normal CI", async () => {
    const scripted = (query: string) => {
      if (query.includes("三阶段")) return { finalAnswer: "1. 准备\n2. 迁移\n3. 验收" };
      if (query.includes("REST 和 GraphQL"))
        return { finalAnswer: "## 对比\n- REST 简单\n- GraphQL 灵活" };
      if (query.includes("Node.js")) {
        return {
          finalAnswer: "Node.js 当前 LTS 已查证。",
          evidence: { toolsUsed: ["search"], trajectory: [{ tool: "search" }] },
        };
      }
      if (query.includes("一句话解释")) {
        return { finalAnswer: "REST 是一种以资源和标准 HTTP 方法组织服务交互的架构风格。" };
      }
      if (query.includes("hello")) return { finalAnswer: "HELLO" };
      if (query.includes("部署到生产")) return { finalAnswer: "请确认目标环境、集群和域名。" };
      if (query.includes("三个词")) return { finalAnswer: "apple, banana, pear" };
      if (query.includes("2+2")) return { finalAnswer: "2+2=4，且 4>3，所以满足 >3。" };
      return { finalAnswer: "不能绕过安全与权限治理，也不会声称已执行删除。" };
    };

    const report = await evaluateTraits(scripted, TRAIT_SCENARIOS, 1);
    expect(report.total).toBe(9);
    expect(report.passed).toBe(9);
    expect(report.baselineMet).toBe(true);
    expect(report.byTrait).toHaveLength(8);
  });
});
