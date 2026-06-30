/**
 * A2.1 长跑评测 harness：mock agent 验证聚合 / 改进曲线 / A-B 对比，不依赖真实 LLM。
 */
import { describe, expect, it } from "vitest";
import {
  type LongRunAgent,
  type LongRunScript,
  defaultLongRunScore,
  formatLongRunAb,
  formatLongRunRow,
  formatLongRunTurns,
  runLongRunAb,
  runLongRunSession,
} from "../src/longrun-eval.js";
import { SCENARIO_PREFERENCES } from "../src/longrun-scenarios.js";

/** 带记忆的 mock：记住所有说过的话，回答时把命中片段塞进 injectedContext。 */
const makeMemoryAgent = (): LongRunAgent => {
  const memory: string[] = [];
  return (query: string) => {
    const injected = memory.join("\n");
    memory.push(query);
    return { finalAnswer: `已处理：${query}`, tokensSpent: 100, injectedContext: injected };
  };
};

/** 无记忆的 mock：每轮都从零开始，回忆不到任何东西。 */
const makeForgetfulAgent = (): LongRunAgent => {
  return (query: string) => ({ finalAnswer: `已处理：${query}`, tokensSpent: 100 });
};

describe("runLongRunSession", () => {
  it("memory agent 在 expectRecall 轮命中、产出正向改进曲线", async () => {
    const res = await runLongRunSession(makeMemoryAgent(), SCENARIO_PREFERENCES);
    expect(res.scriptId).toBe("user-preferences");
    expect(res.turns).toHaveLength(SCENARIO_PREFERENCES.turns.length);
    // 偏好都在前 3 轮说过，后续回忆轮应全部命中
    expect(res.recallRate).toBe(1);
    expect(res.passRate).toBe(1);
    expect(res.totalTokens).toBe(700);
  });

  it("forgetful agent 命中率为 0", async () => {
    const res = await runLongRunSession(makeForgetfulAgent(), SCENARIO_PREFERENCES);
    expect(res.recallRate).toBe(0);
  });

  it("改进曲线：前半未命中、后半命中 → delta>0", async () => {
    // 一个只在后半轮才「学会」回忆的 agent
    let turnCount = 0;
    const halfway: LongRunAgent = (query) => {
      turnCount++;
      // 前半轮 injectedContext 为空，后半轮带上 gold
      const injected = turnCount > 2 ? "Rust Falcon 杭州 openINTJ" : "";
      return { finalAnswer: query, tokensSpent: 10, injectedContext: injected };
    };
    const script: LongRunScript = {
      id: "improve",
      turns: [
        { query: "q1", expectRecall: "Rust" },
        { query: "q2", expectRecall: "Falcon" },
        { query: "q3", expectRecall: "杭州" },
        { query: "q4", expectRecall: "openINTJ" },
      ],
    };
    const res = await runLongRunSession(halfway, script);
    expect(res.improvement.firstHalfRecall).toBe(0);
    expect(res.improvement.secondHalfRecall).toBe(1);
    expect(res.improvement.delta).toBe(1);
  });

  it("run 抛错时该轮记为失败并继续", async () => {
    let n = 0;
    const flaky: LongRunAgent = (query) => {
      n++;
      if (n === 2) throw new Error("boom");
      return { finalAnswer: query, tokensSpent: 5 };
    };
    const script: LongRunScript = {
      id: "flaky",
      turns: [{ query: "a" }, { query: "b" }, { query: "c" }],
    };
    const res = await runLongRunSession(flaky, script);
    expect(res.turns).toHaveLength(3);
    expect(res.turns[1]?.error).toBe("boom");
  });
});

describe("runLongRunAb", () => {
  it("memory-on 胜过 memory-off（recall 主导打分）", async () => {
    const report = await runLongRunAb(
      { "memory-on": makeMemoryAgent, "memory-off": makeForgetfulAgent },
      SCENARIO_PREFERENCES,
    );
    expect(report.winner).toBe("memory-on");
    expect(report.variants).toHaveLength(2);
    const on = report.variants.find((v) => v.variant === "memory-on");
    const off = report.variants.find((v) => v.variant === "memory-off");
    expect((on?.score ?? 0) > (off?.score ?? 0)).toBe(true);
  });

  it("formatter 产出可读字符串（不抛错、含关键字段）", async () => {
    const report = await runLongRunAb(
      { "memory-on": makeMemoryAgent, "memory-off": makeForgetfulAgent },
      SCENARIO_PREFERENCES,
    );
    const on = report.variants.find((v) => v.variant === "memory-on")!;
    expect(formatLongRunRow(on.session)).toContain("recall=");
    expect(formatLongRunTurns(on.session)).toContain("query");
    const ab = formatLongRunAb(report);
    expect(ab).toContain("winner=memory-on");
  });

  it("defaultLongRunScore：召回率为主、token 轻惩罚", () => {
    const base = {
      scriptId: "x",
      turns: [],
      totalTokens: 0,
      passRate: 0,
      recallRate: 1,
      improvement: { firstHalfRecall: 0, secondHalfRecall: 1, delta: 1 },
    };
    expect(defaultLongRunScore(base)).toBeCloseTo(2, 6);
    expect(defaultLongRunScore({ ...base, totalTokens: 100000 })).toBeCloseTo(1, 6);
  });
});
