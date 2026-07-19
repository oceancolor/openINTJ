import {
  LONGRUN_SCENARIOS,
  type LongRunAgent,
  formatLongRunAb,
  formatLongRunRow,
  formatLongRunTurns,
  runLongRunAb,
  runLongRunSession,
} from "@openintj/shared";
/**
 * A2.2 长跑验证 harness（RUN_LONGRUN=1 门控，不进常规 CI）。
 *
 * 用真实装配的 Agent 跑「有先后依赖」的场景脚本，打印逐轮命中/ token 表 + JSON +
 * 改进曲线（后半 vs 前半 recall）。验证「越用越好」是否成立。
 *
 *   RUN_LONGRUN=1 [OPENINTJ_LOOP_HYBRID=1] pnpm --filter @openintj/cli test longrun
 *
 * 默认 skip：避免 CI 依赖 LLM key。设 OPENINTJ_LLM_PROVIDER 选择 provider（默认 auto）。
 */
import { describe, expect, it } from "vitest";
import { type LlmProvider, assembleAgent } from "../src/agent.js";

const RUN = process.env["RUN_LONGRUN"] === "1";

describe("longrun harness (gated)", () => {
  it.runIf(RUN)(
    "跑全部场景并输出命中/token/改进曲线",
    async () => {
      const provider = (process.env["OPENINTJ_LLM_PROVIDER"] as LlmProvider) ?? "auto";
      for (const script of LONGRUN_SCENARIOS) {
        // 每个场景独立装配，保证记忆状态隔离。maxTaoIterations 调大以允许记忆累积受益。
        const agent = assembleAgent({ llmProvider: provider, maxTaoIterations: 2 });
        const adapter: LongRunAgent = async (query) => {
          const r = await agent.run(query);
          return { finalAnswer: r.finalAnswer, tokensSpent: r.totalTokensSpent };
        };
        const session = await runLongRunSession(adapter, script);
        console.log(`\n${formatLongRunRow(session)}`);
        console.log(formatLongRunTurns(session));
        console.log(`[longrun-json] ${JSON.stringify(session)}`);
        expect(session.turns).toHaveLength(script.turns.length);
      }
    },
    300_000,
  );

  it.runIf(RUN)(
    "CLF.4 classifier-on vs off A/B：量化 token/命中/质量",
    async () => {
      const provider = (process.env["OPENINTJ_LLM_PROVIDER"] as LlmProvider) ?? "auto";
      // 每个变体是独立工厂 → 记忆 + 分类器状态隔离，保证对比公平。
      const makeAdapter = (enableClassifier: boolean): (() => LongRunAgent) => {
        return () => {
          const agent = assembleAgent({
            llmProvider: provider,
            maxTaoIterations: 2,
            enableClassifier,
          });
          return async (query) => {
            const r = await agent.run(query);
            return { finalAnswer: r.finalAnswer, tokensSpent: r.totalTokensSpent };
          };
        };
      };
      for (const script of LONGRUN_SCENARIOS) {
        const report = await runLongRunAb(
          {
            "classifier-off": makeAdapter(false),
            "classifier-on": makeAdapter(true),
          },
          script,
        );
        console.log(`\n${formatLongRunAb(report)}`);
        console.log(`[longrun-ab-json] ${JSON.stringify(report)}`);
        expect(report.variants).toHaveLength(2);
        // 质量不退：classifier-on 召回不应明显低于 off（容忍 1 轮抖动）。
        const on = report.variants.find((v) => v.variant === "classifier-on")!;
        const off = report.variants.find((v) => v.variant === "classifier-off")!;
        const perTurn = 1 / Math.max(1, script.turns.length);
        expect(on.session.recallRate).toBeGreaterThanOrEqual(off.session.recallRate - perTurn);
      }
    },
    600_000,
  );

  it.runIf(RUN)(
    "RFC-006 Product Behavior on vs off：长跑完成率不得显著回退",
    async () => {
      const provider = (process.env["OPENINTJ_LLM_PROVIDER"] as LlmProvider) ?? "auto";
      const makeAdapter =
        (enableProductBehavior: boolean): (() => LongRunAgent) =>
        () => {
          const agent = assembleAgent({
            llmProvider: provider,
            maxTaoIterations: 2,
            enableProductBehavior,
          });
          return async (query) => {
            const r = await agent.run(query);
            return { finalAnswer: r.finalAnswer, tokensSpent: r.totalTokensSpent };
          };
        };
      for (const script of LONGRUN_SCENARIOS) {
        const report = await runLongRunAb(
          {
            "product-behavior-off": makeAdapter(false),
            "product-behavior-on": makeAdapter(true),
          },
          script,
        );
        console.log(`\n${formatLongRunAb(report)}`);
        const on = report.variants.find((v) => v.variant === "product-behavior-on")!;
        const off = report.variants.find((v) => v.variant === "product-behavior-off")!;
        const perTurn = 1 / Math.max(1, script.turns.length);
        expect(on.session.recallRate).toBeGreaterThanOrEqual(off.session.recallRate - perTurn);
      }
    },
    600_000,
  );

  it("harness 模块在常规 CI 下可被引用（占位，确保 import 不破）", () => {
    expect(typeof runLongRunSession).toBe("function");
    expect(typeof runLongRunAb).toBe("function");
    expect(LONGRUN_SCENARIOS.length).toBeGreaterThan(0);
  });
});
