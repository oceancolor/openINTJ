import { describe, expect, it } from "vitest";
import type { TaoResult } from "@openintj/core";
import { synthesizeTaskPoolAnswer } from "../src/synthesizer.js";
import type { TaskRunResult } from "../src/task-pool.js";

const failedRun = (errors: ReadonlyMap<string, unknown>): TaskRunResult<TaoResult> => ({
  runId: "run-1",
  planId: "plan-1",
  status: "failed",
  results: new Map(),
  errors,
  states: new Map(),
  attempts: new Map(),
});

describe("synthesizeTaskPoolAnswer", () => {
  it("失败且无任何步骤产出时，兜底文案带上首个节点错误", () => {
    const out = synthesizeTaskPoolAnswer(
      failedRun(new Map([["s1", new Error("s1 timeout after 120000ms")]])),
      "规划自驾游行程",
    );
    expect(out.status).toBe("failed");
    expect(out.finalAnswer).toContain("未能完成");
    expect(out.finalAnswer).toContain("s1 timeout after 120000ms");
    expect(out.failureReason).toBe("taskpool failed");
  });

  it("无错误信息时保持原兜底文案", () => {
    const out = synthesizeTaskPoolAnswer(failedRun(new Map()), "q");
    expect(out.finalAnswer).toBe("[taskpool] 未能完成：q");
  });
});
