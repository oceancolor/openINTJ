import { describe, expect, it } from "vitest";
import { createDelegateTool, createSubagentSpawner } from "../src/subagent.js";

describe("createSubagentSpawner (RFC-012 §3)", () => {
  it("runs a child and returns its answer with a completed record", async () => {
    const spawner = createSubagentSpawner({
      runChild: async (cid, task) => ({ status: "completed", finalAnswer: `done:${cid}:${task}` }),
    });
    const record = await spawner.spawn({ task: "调研 X", label: "research" });
    expect(record.status).toBe("completed");
    expect(record.childConversationId).toBe(`subagent:${record.id}`);
    expect(record.finalAnswer).toContain("调研 X");
    expect(spawner.list()).toHaveLength(1);
  });

  it("rejects nested delegation beyond maxDepth (ALS propagation)", async () => {
    let nestedError: string | undefined;
    const spawner = createSubagentSpawner({
      runChild: async () => {
        try {
          await spawner.spawn({ task: "嵌套" });
        } catch (e) {
          nestedError = (e as Error).message;
        }
        return { status: "completed", finalAnswer: "outer" };
      },
    });
    await spawner.spawn({ task: "外层" });
    expect(nestedError).toMatch(/depth limit/);
  });

  it("rejects spawns beyond the concurrency cap", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const spawner = createSubagentSpawner({
      maxConcurrent: 1,
      runChild: async () => {
        await gate;
        return { status: "completed", finalAnswer: "ok" };
      },
    });
    const first = spawner.spawn({ task: "一" });
    await expect(spawner.spawn({ task: "二" })).rejects.toThrow(/concurrency limit/);
    release();
    await first;
  });

  it("aborts a child that exceeds the timeout", async () => {
    const spawner = createSubagentSpawner({
      timeoutMs: 20,
      runChild: (_cid, _task, signal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
        }),
    });
    await expect(spawner.spawn({ task: "慢任务" })).rejects.toThrow(/aborted/);
    expect(spawner.list()[0]?.status).toBe("cancelled");
  });

  it("delegate tool validates params and returns the child answer", async () => {
    const spawner = createSubagentSpawner({
      runChild: async () => ({ status: "completed", finalAnswer: "子回答" }),
    });
    const tool = createDelegateTool(spawner);
    expect(tool.descriptor).toMatchObject({
      name: "delegate_task",
      permissions: ["agent.delegate"],
      idempotent: false,
    });
    const out = (await tool.handler({ task: "做事", label: "w" })) as { finalAnswer: string };
    expect(out.finalAnswer).toBe("子回答");
    await expect((async () => tool.handler({}))()).rejects.toThrow(/task/);
  });

  it("effective tools are the spawner ceiling ∩ per-spawn narrowing", async () => {
    const seen: Array<readonly string[] | undefined> = [];
    const spawner = createSubagentSpawner({
      tools: () => ["read_file", "search", "write_file"],
      runChild: async (_cid, _task, _signal, tools) => {
        seen.push(tools);
        return { status: "completed", finalAnswer: "ok" };
      },
    });
    // Model asks for write_file + execute_command: both outside/inside are intersected.
    const record = await spawner.spawn({ task: "t", tools: ["search", "execute_command"] });
    expect(record.tools).toEqual(["search"]);
    expect(seen[0]).toEqual(["search"]);
    // No per-spawn narrowing → the ceiling itself applies.
    const plain = await spawner.spawn({ task: "t2" });
    expect(plain.tools).toEqual(["read_file", "search", "write_file"]);
  });

  it("delegate tool forwards a tools param as narrowing", async () => {
    let received: readonly string[] | undefined;
    const spawner = createSubagentSpawner({
      tools: ["read_file", "search"],
      runChild: async (_cid, _task, _signal, tools) => {
        received = tools;
        return { status: "completed", finalAnswer: "ok" };
      },
    });
    const tool = createDelegateTool(spawner);
    await tool.handler({ task: "x", tools: ["search", "write_file", 42] });
    expect(received).toEqual(["search"]);
  });

  it("without any ceiling, children stay unrestricted (back-compat)", async () => {
    let received: readonly string[] | undefined = ["sentinel"];
    const spawner = createSubagentSpawner({
      runChild: async (_cid, _task, _signal, tools) => {
        received = tools;
        return { status: "completed", finalAnswer: "ok" };
      },
    });
    const record = await spawner.spawn({ task: "t" });
    expect(received).toBeUndefined();
    expect(record.tools).toBeUndefined();
  });
});
