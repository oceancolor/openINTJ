import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type StoredTaskRun, TaskPool } from "@openintj/taskpool";
import { describe, expect, it } from "vitest";
import { SqliteTaskStore } from "../src/task-store.js";

describe("SqliteTaskStore recovery", () => {
  it("lists and resumes an incomplete run after restart", async () => {
    const dbPath = join(tmpdir(), `openintj-task-${crypto.randomUUID()}.sqlite`);
    const stored: StoredTaskRun = {
      runId: "recover-run",
      planId: "recover-plan",
      status: "running",
      graph: {
        planId: "recover-plan",
        goalIntent: "test",
        goalInput: "恢复持久化任务",
        nodes: [
          { id: "a", deps: [], action: "x", description: "a" },
          { id: "b", deps: ["a"], action: "x", description: "b" },
        ],
      },
      nodes: [
        { taskId: "a", state: "completed", attempt: 1, result: "A", updatedAt: 1 },
        { taskId: "b", state: "running", attempt: 1, updatedAt: 2 },
      ],
      createdAt: 1,
      updatedAt: 2,
    };
    const first = new SqliteTaskStore(dbPath, false);
    await first.init();
    await first.saveRun(stored);
    await first.close();

    const restarted = new SqliteTaskStore(dbPath, false);
    await restarted.init();
    const [incomplete] = await restarted.listIncompleteRuns();
    expect(incomplete?.runId).toBe("recover-run");
    const calls: string[] = [];
    const result = await new TaskPool({ store: restarted }).recover(
      incomplete!,
      async (node, ctx) => {
        calls.push(node.id);
        expect(ctx.goalInput).toBe("恢复持久化任务");
        expect(ctx.shared.get("task:a:result")).toBe("A");
        return node.id.toUpperCase();
      },
    ).result;
    expect(calls).toEqual(["b"]);
    expect([...result.results.values()]).toEqual(["A", "B"]);
    expect((await restarted.loadRun("recover-run"))?.status).toBe("completed");
    await restarted.close();
    await rm(dbPath, { force: true });
  });
});
