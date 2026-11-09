import { describe, expect, it } from "vitest";
import {
  DEFAULT_REACT_CONFIG,
  DEFAULT_TAO_CONFIG,
  HookBus,
  ReactStateMachine,
  ShaderMode,
  TaoLoop,
  TaskType,
  createAgentHandle,
  createInMemoryJournal,
} from "../src/index.js";
import type { InputStructuredTaoResult, LlmClient } from "../src/index.js";

const silent = { warn: () => {}, error: () => {} };

const makeLlm = (responses: string[]): LlmClient => {
  let i = 0;
  return {
    async chat() {
      return responses[i++] ?? responses.at(-1) ?? "FINAL: done";
    },
    async visionChat() {
      return "vision";
    },
    getStatus() {
      return {
        provider: "test",
        model: "x",
        available: true,
        mode: "live",
        status: "connected",
        visionSupported: false,
      };
    },
  };
};

describe("AgentHandle", () => {
  it("inject does not start a turn; steer while idle follows up", async () => {
    const journal = createInMemoryJournal();
    const hooks = new HookBus({ logger: silent });
    const queries: string[] = [];
    const handle = createAgentHandle({
      id: "c1",
      journal,
      hooks,
      runTurn: async (query) => {
        queries.push(query);
        return {
          traceId: "t",
          status: "completed",
          finalAnswer: query,
          iterations: 0,
          reactTotalSteps: 0,
          totalTokensSpent: 0,
          durationMs: 0,
          trajectory: [],
          taskType: TaskType.QUICK_RESPONSE,
          shaderMode: ShaderMode.LOW_FIDELITY,
          metrics: {},
        };
      },
    });
    handle.inject("file: a.ts");
    expect(handle.status).toBe("idle");
    expect(queries).toEqual([]);
    handle.steer("next");
    await handle.whenIdle();
    expect(queries).toEqual(["next"]);
  });

  it("steer appears in the next ReAct thought messages", async () => {
    const journal = createInMemoryJournal();
    const hooks = new HookBus({ logger: silent });
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    let enteredFirst!: () => void;
    const firstEntered = new Promise<void>((resolve) => {
      enteredFirst = resolve;
    });
    let calls = 0;
    let secondSeen = "";
    const llm = makeLlm([]);
    llm.chat = async (messages) => {
      calls += 1;
      if (calls === 1) {
        enteredFirst();
        await firstGate;
        return 'Thought: looking\nAction: search\nAction-Input: {"q":"a"}';
      }
      secondSeen = JSON.stringify(messages);
      return "Thought: done\nFINAL: ok";
    };
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: { ...DEFAULT_REACT_CONFIG, maxIterations: 4 },
        hooks,
        llm,
        toolRunner: async (name) => ({
          toolName: name,
          success: true,
          output: "ok",
          durationMs: 1,
          traceId: "",
          callId: "c1",
        }),
      }),
      availableTools: () => [
        {
          name: "search",
          description: "搜索",
          inputSchema: {},
          outputSchema: {},
          permissions: [],
          timeoutS: 30,
          idempotent: true,
          errorSemantics: "retry",
        },
      ],
    });
    const handle = createAgentHandle({
      id: "c1",
      journal,
      hooks,
      runTurn: (query, opts) => tao.run(query, opts),
    });
    const running = handle.run("start");
    await firstEntered;
    handle.steer("course-correct");
    releaseFirst();
    await running;
    expect(secondSeen).toContain("course-correct");
    expect(
      journal.list("c1").some((event) => event.type === "user" && event.source === "steer"),
    ).toBe(true);
  });
});

const fakeTurnResult = (query: string): InputStructuredTaoResult => ({
  traceId: "t",
  status: "completed",
  finalAnswer: query,
  iterations: 0,
  reactTotalSteps: 0,
  totalTokensSpent: 0,
  durationMs: 0,
  trajectory: [],
  taskType: TaskType.QUICK_RESPONSE,
  shaderMode: ShaderMode.LOW_FIDELITY,
  metrics: {},
});

describe("RFC-021 F02 run ownership (T03/T04)", () => {
  it("T03: steer/inject are consumed only by the owning run on a shared bus", async () => {
    const hooks = new HookBus({ logger: silent });
    const seen = new Map<string, { steer: unknown[]; inject: unknown[] }>();

    const makeHandle = (
      id: string,
      gate?: Promise<void>,
    ): ReturnType<typeof createAgentHandle> => {
      const record = { steer: [] as unknown[], inject: [] as unknown[] };
      seen.set(id, record);
      let emits = 0;
      return createAgentHandle({
        id,
        journal: createInMemoryJournal(),
        hooks,
        runTurn: async (query, opts) => {
          const emitBoth = async (): Promise<void> => {
            emits++;
            const thought = await hooks.emitOutcome(
              "react.beforeThought",
              { context: { systemPrompt: "" }, reactIter: emits, taoIter: 1 },
              opts?.run ? { runId: opts.run.runId } : undefined,
            );
            record.steer.push(thought.payload.steerMessages ?? null);
            const think = await hooks.emitOutcome(
              "tao.beforeThink",
              { query, iteration: 1 },
              opts?.run ? { runId: opts.run.runId } : undefined,
            );
            record.inject.push(think.payload.injectBlock ?? null);
          };
          await emitBoth();
          if (gate) {
            await gate;
            await emitBoth();
            await emitBoth();
          }
          return fakeTurnResult(query);
        },
      });
    };

    let releaseA!: () => void;
    const gateA = new Promise<void>((resolve) => {
      releaseA = resolve;
    });
    const a = makeHandle("conv-a", gateA);
    const b = makeHandle("conv-b");

    const pA = a.run("a-start");
    // Wait until A's first emit happened (A is running, inbox empty).
    await new Promise((r) => setTimeout(r, 10));
    a.steer("for-a");
    a.inject("inject-a");

    await b.run("b-start");
    expect(seen.get("conv-b")!.steer).toEqual([null]);
    expect(seen.get("conv-b")!.inject).toEqual([null]);

    releaseA();
    await pA;
    // A consumed its own steer/inject exactly once (second emit), third is empty.
    expect(seen.get("conv-a")!.steer).toEqual([null, ["for-a"], null]);
    expect(seen.get("conv-a")!.inject).toEqual([null, "inject-a", null]);
    a.dispose();
    b.dispose();
  });

  it("T04: cancelling while queued prevents the queued run from executing", async () => {
    const hooks = new HookBus({ logger: silent });
    const calls: string[] = [];
    let releaseFirst!: () => void;
    const gate = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    let enteredFirst!: () => void;
    const firstEntered = new Promise<void>((resolve) => {
      enteredFirst = resolve;
    });
    const handle = createAgentHandle({
      id: "conv-q",
      journal: createInMemoryJournal(),
      hooks,
      runTurn: async (query) => {
        calls.push(query);
        if (query === "first") {
          enteredFirst();
          await gate;
        }
        return fakeTurnResult(query);
      },
    });
    const p1 = handle.run("first");
    await firstEntered;
    const p2 = handle.run("second");
    const p2Assertion = expect(p2).rejects.toThrow("cancelled_before_start");
    handle.cancel("user");
    releaseFirst();
    await p1;
    await p2Assertion;
    expect(calls).toEqual(["first"]);
    handle.dispose();
  });

  it("dispose unregisters the handle hooks from the shared bus", () => {
    const hooks = new HookBus({ logger: silent });
    const baseline = hooks.inspect().total;
    const handle = createAgentHandle({
      id: "conv-d",
      journal: createInMemoryJournal(),
      hooks,
      runTurn: async (query) => fakeTurnResult(query),
    });
    expect(hooks.inspect().total).toBe(baseline + 2);
    handle.dispose();
    expect(hooks.inspect().total).toBe(baseline);
    handle.dispose();
    expect(hooks.inspect().total).toBe(baseline);
  });
});

describe("Hook rewrite drives the loop", () => {  it("tao.beforeThink replace() changes the query used to build messages", async () => {
    const hooks = new HookBus({ logger: silent });
    hooks.on("tao.beforeThink", (ctx) => {
      ctx.replace({ ...ctx.payload, query: "rewritten-query" });
    });
    let seen = "";
    const llm = makeLlm(["FINAL: ok"]);
    const orig = llm.chat.bind(llm);
    llm.chat = async (messages, opts) => {
      seen = JSON.stringify(messages);
      return orig(messages, opts);
    };
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: { ...DEFAULT_REACT_CONFIG, maxIterations: 1 },
        hooks,
        llm,
        toolRunner: async (name) => ({
          toolName: name,
          success: true,
          output: "ok",
          durationMs: 1,
          traceId: "",
          callId: "c1",
        }),
      }),
      availableTools: () => [],
    });
    await tao.run("original");
    expect(seen).toContain("rewritten-query");
    expect(seen).not.toContain("original");
  });
});
