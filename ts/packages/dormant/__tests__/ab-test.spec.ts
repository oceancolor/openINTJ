import { describe, expect, it } from "vitest";
import { type AbVariant, runAbTest } from "../src/ab-test.js";

describe("runAbTest", () => {
  it("聚合每变体均分并选出 winner（persona-on 越用越好的骨架）", async () => {
    // 两个确定性变体：on 始终给更长、含关键词的答案；off 给短答案。
    const on: AbVariant<undefined, string> = {
      name: "persona-on",
      run: (q) => `详细回答关于「${q}」的内容，并体现用户偏好`,
    };
    const off: AbVariant<undefined, string> = {
      name: "persona-off",
      run: (q) => `回答：${q}`,
    };
    const report = await runAbTest({
      variants: [on, off],
      queries: ["茶", "代码", "天气"],
      // 评分：越长越好（仅作确定性骨架验证）。
      score: (out) => out.length,
    });

    expect(report.queryCount).toBe(3);
    expect(report.winner).toBe("persona-on");
    const onStat = report.perVariant.find((v) => v.name === "persona-on")!;
    const offStat = report.perVariant.find((v) => v.name === "persona-off")!;
    expect(onStat.trials).toBe(3);
    expect(onStat.wins).toBe(3);
    expect(offStat.wins).toBe(0);
    expect(onStat.meanScore).toBeGreaterThan(offStat.meanScore);
  });

  it("并列最高分时双方都记 win", async () => {
    const a: AbVariant<undefined, number> = { name: "a", run: () => 1 };
    const b: AbVariant<undefined, number> = { name: "b", run: () => 1 };
    const report = await runAbTest({
      variants: [a, b],
      queries: ["x", "y"],
      score: (out) => out,
    });
    expect(report.perVariant.find((v) => v.name === "a")!.wins).toBe(2);
    expect(report.perVariant.find((v) => v.name === "b")!.wins).toBe(2);
  });

  it("makeContext 为每个变体提供独立上下文", async () => {
    const seen: string[] = [];
    const v: AbVariant<{ tag: string }, string> = {
      name: "v",
      run: (_q, ctx) => {
        seen.push(ctx.tag);
        return "ok";
      },
    };
    await runAbTest({
      variants: [v],
      queries: ["q1", "q2"],
      score: () => 1,
      makeContext: (name) => ({ tag: `ctx-${name}` }),
    });
    expect(seen).toEqual(["ctx-v", "ctx-v"]);
  });
});
