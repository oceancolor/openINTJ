import { describe, expect, it } from "vitest";
import {
  type TaskCase,
  evaluateTasks,
  judgeContainsAll,
  judgeNonEmpty,
} from "../src/task-eval.js";

describe("evaluateTasks", () => {
  it("聚合完成率：部分通过", async () => {
    const tasks: TaskCase[] = [
      { id: "t1", query: "首都", judge: judgeContainsAll("paris") },
      { id: "t2", query: "问候", judge: judgeNonEmpty },
      { id: "t3", query: "数学", judge: judgeContainsAll("42") },
    ];
    const answers: Record<string, string> = {
      首都: "The capital is Paris",
      问候: "你好",
      数学: "the answer is 41", // 不含 42 → 失败
    };
    const report = await evaluateTasks(tasks, (q) => ({ finalAnswer: answers[q] ?? "" }));
    expect(report.total).toBe(3);
    expect(report.passed).toBe(2);
    expect(report.completionRate).toBeCloseTo(2 / 3, 6);
    expect(report.results.find((r) => r.id === "t3")!.passed).toBe(false);
  });

  it("run 抛错时该任务记为失败但套件继续", async () => {
    const tasks: TaskCase[] = [
      { id: "ok", query: "a", judge: judgeNonEmpty },
      { id: "boom", query: "b", judge: judgeNonEmpty },
    ];
    const report = await evaluateTasks(tasks, (q) => {
      if (q === "b") throw new Error("runner failed");
      return { finalAnswer: "fine" };
    });
    expect(report.total).toBe(2);
    expect(report.passed).toBe(1);
    const boom = report.results.find((r) => r.id === "boom")!;
    expect(boom.passed).toBe(false);
    expect(boom.error).toContain("runner failed");
  });

  it("judgeContainsAll 要求全部关键词命中", () => {
    expect(judgeContainsAll("a", "b")("x a y b")).toBe(true);
    expect(judgeContainsAll("a", "b")("only a")).toBe(false);
  });
});
