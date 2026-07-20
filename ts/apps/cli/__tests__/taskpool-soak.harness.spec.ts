import { HookBus } from "@openintj/core";
import { resolveLlmClient } from "@openintj/model-runtime";
import { MemoryTaskStore, TaskPool } from "@openintj/taskpool";
import { describe, expect, it } from "vitest";

const RUN = process.env["RUN_TASKPOOL_SOAK"] === "1";

describe("TaskPool provider cancel/recovery soak (gated)", () => {
  it.runIf(RUN)(
    "repeatedly aborts in-flight Ollama requests without waiting for provider timeout",
    async () => {
      const env = {
        ...process.env,
        OLLAMA_MODEL: process.env["OLLAMA_MODEL"] ?? "qwen2.5:0.5b",
        OLLAMA_TIMEOUT_MS: process.env["OLLAMA_TIMEOUT_MS"] ?? "30000",
      };
      const runtime = await resolveLlmClient({ provider: "ollama", env });

      for (let cycle = 0; cycle < 3; cycle++) {
        const hooks = new HookBus();
        let markStarted!: () => void;
        const started = new Promise<void>((resolve) => {
          markStarted = resolve;
        });
        hooks.on("taskpool.task.start", () => markStarted(), { once: true });
        const handle = new TaskPool({ hooks }).submit(
          {
            planId: `provider-cancel-${cycle}`,
            goalIntent: "cancel soak",
            goalInput: "请生成一篇很长的 TypeScript 架构分析，用于验证在途请求取消。",
            nodes: [{ id: "a", deps: [], action: "analyze", description: "long generation" }],
          },
          async (_node, ctx) =>
            runtime.client.chat([{ role: "user", content: ctx.goalInput }], {
              maxTokens: 2048,
              signal: ctx.signal,
            }),
        );

        await started;
        await new Promise((resolve) => setTimeout(resolve, 25));
        const cancelledAt = Date.now();
        handle.cancel(`soak cycle ${cycle}`);
        const result = await handle.result;

        expect(result.status).toBe("cancelled");
        expect(Date.now() - cancelledAt).toBeLessThan(5_000);
      }
    },
    180_000,
  );

  it.runIf(RUN)(
    "repeatedly resumes durable-shaped incomplete snapshots",
    async () => {
      for (let cycle = 0; cycle < 25; cycle++) {
        const store = new MemoryTaskStore();
        await store.saveRun({
          runId: `recovery-${cycle}`,
          planId: `plan-${cycle}`,
          status: "running",
          graph: {
            planId: `plan-${cycle}`,
            goalIntent: "recovery soak",
            goalInput: `original input ${cycle}`,
            nodes: [{ id: "a", deps: [], action: "resume", description: "resume task" }],
          },
          nodes: [{ taskId: "a", state: "running", attempt: 1, updatedAt: cycle + 1 }],
          createdAt: cycle,
          updatedAt: cycle + 1,
        });

        const summary = await new TaskPool({ store }).recoverIncomplete(async (_node, ctx) => {
          expect(ctx.goalInput).toBe(`original input ${cycle}`);
          return `completed-${cycle}`;
        }, "resume");

        expect(summary).toMatchObject({
          found: 1,
          resumed: 1,
          completed: 1,
          cancelled: 0,
          failed: 0,
        });
        expect(await store.listIncompleteRuns()).toEqual([]);
      }
    },
    30_000,
  );
});
