import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_REACT_CONFIG,
  HookBus,
  type HookLogger,
  type LlmClient,
  type ReactConfig,
  ReactStateMachine,
  type ToolCallResult,
  type ToolDescriptor,
  __test__,
} from "../src/index.js";

const silentLogger: HookLogger = { warn: () => {}, error: () => {} };

const makeLlm = (responses: string[]): LlmClient => {
  let i = 0;
  return {
    async chat() {
      const r = responses[i] ?? responses.at(-1) ?? "FINAL: done";
      i++;
      return r;
    },
    async visionChat() {
      return makeLlm(responses).chat([], {});
    },
    getStatus() {
      return {
        provider: "test",
        model: "mock",
        available: true,
        mode: "live",
        status: "connected",
        visionSupported: false,
      };
    },
  };
};

const makeToolRunner = (impl: Record<string, (params: unknown) => unknown>) => {
  return async (name: string, params: Record<string, unknown>): Promise<ToolCallResult> => {
    if (!(name in impl)) {
      return {
        toolName: name,
        success: false,
        error: `unknown tool: ${name}`,
        durationMs: 0,
        traceId: "",
        callId: "c1",
      };
    }
    try {
      const out = impl[name]!(params);
      return {
        toolName: name,
        success: true,
        output: out,
        durationMs: 1,
        traceId: "",
        callId: `c-${name}`,
      };
    } catch (err) {
      return {
        toolName: name,
        success: false,
        error: (err as Error).message,
        durationMs: 1,
        traceId: "",
        callId: `c-${name}-err`,
      };
    }
  };
};

const tools: ToolDescriptor[] = [
  {
    name: "search",
    description: "搜索",
    inputSchema: { query: "string" },
    outputSchema: {},
    permissions: ["network.read"],
    timeoutS: 30,
    idempotent: true,
    errorSemantics: "retry",
  },
  {
    name: "calc",
    description: "计算",
    inputSchema: { a: "number", b: "number" },
    outputSchema: {},
    permissions: [],
    timeoutS: 30,
    idempotent: true,
    errorSemantics: "fail_fast",
  },
];

describe("__test__.parseLlmThought", () => {
  const { parseLlmThought } = __test__;

  it("parses Action + Action-Input", () => {
    const r = parseLlmThought(
      `Thought: 我要查一下 cat\nAction: search\nAction-Input: {"query": "cat"}`,
    );
    expect(r.isFinal).toBe(false);
    expect(r.action?.tool).toBe("search");
    expect(r.action?.params).toEqual({ query: "cat" });
  });

  it("parses FINAL marker", () => {
    const r = parseLlmThought(`Thought: 我已知道答案\nFINAL: 42`);
    expect(r.isFinal).toBe(true);
    expect(r.finalAnswer).toBe("42");
  });

  it("treats free text without markers as implicit final", () => {
    const r = parseLlmThought("just a thought without structure");
    expect(r.isFinal).toBe(true);
  });

  it("reports parse error on malformed JSON", () => {
    const r = parseLlmThought(`Thought: x\nAction: search\nAction-Input: {bad json}`);
    expect(r.parseError).toBeDefined();
  });
});

describe("ReactStateMachine.run", () => {
  const baseConfig: ReactConfig = {
    ...DEFAULT_REACT_CONFIG,
    maxIterations: 6,
  };

  it("happy path: tool call → observation → final", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const llm = makeLlm([
      `Thought: I need to search for cats\nAction: search\nAction-Input: {"query":"cat"}`,
      `Thought: Got it.\nFINAL: 猫是一种家养动物`,
    ]);
    const runner = makeToolRunner({
      search: () => "cats are felines",
    });
    const sm = new ReactStateMachine({
      config: baseConfig,
      hooks,
      llm,
      toolRunner: runner,
    });
    const out = await sm.run({
      messages: [{ role: "user", content: "what is a cat?" }],
      availableTools: tools,
      taoIteration: 1,
      systemPrompt: "you are helpful",
    });
    expect(out.status).toBe("ok");
    expect(out.finalAnswer).toContain("家养");
    expect(out.iterations).toBe(2);
    const actions = out.trajectory.filter((t) => t.state.type === "action");
    expect(actions).toHaveLength(1);
  });

  it("emits react.* hooks in order", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const events: string[] = [];
    hooks.on("react.beforeThought", () => void events.push("beforeThought"));
    hooks.on("react.afterThought", () => void events.push("afterThought"));
    hooks.on("react.beforeAction", () => void events.push("beforeAction"));
    hooks.on("react.afterAction", () => void events.push("afterAction"));
    const sm = new ReactStateMachine({
      config: baseConfig,
      hooks,
      llm: makeLlm([
        `Thought: x\nAction: search\nAction-Input: {"query":"a"}`,
        `Thought: done\nFINAL: ok`,
      ]),
      toolRunner: makeToolRunner({ search: () => "result" }),
    });
    await sm.run({
      messages: [{ role: "user", content: "x" }],
      availableTools: tools,
      taoIteration: 1,
      systemPrompt: "",
    });
    expect(events.slice(0, 4)).toEqual([
      "beforeThought",
      "afterThought",
      "beforeAction",
      "afterAction",
    ]);
  });

  it("explicitFinal stop emits onStopCondition", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const stop = vi.fn();
    hooks.on("react.onStopCondition", (ctx) => {
      stop(ctx.payload.kind);
    });
    const sm = new ReactStateMachine({
      config: baseConfig,
      hooks,
      llm: makeLlm([`Thought: easy\nFINAL: 42`]),
      toolRunner: makeToolRunner({}),
    });
    await sm.run({
      messages: [{ role: "user", content: "answer" }],
      availableTools: [],
      taoIteration: 1,
      systemPrompt: "",
    });
    expect(stop).toHaveBeenCalledWith("explicitFinal");
  });

  it("duplicateToolCall stop kicks in", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const llm = makeLlm([
      `Thought: try\nAction: search\nAction-Input: {"q":"x"}`,
      `Thought: again\nAction: search\nAction-Input: {"q":"x"}`,
      `Thought: still\nAction: search\nAction-Input: {"q":"x"}`,
    ]);
    const stop = vi.fn();
    hooks.on("react.onStopCondition", (ctx) => stop(ctx.payload.kind));
    const sm = new ReactStateMachine({
      config: { ...baseConfig, maxIterations: 5 },
      hooks,
      llm,
      toolRunner: makeToolRunner({ search: () => "same result" }),
    });
    const out = await sm.run({
      messages: [{ role: "user", content: "loop?" }],
      availableTools: tools,
      taoIteration: 1,
      systemPrompt: "",
    });
    expect(out.status).toBe("duplicate_loop");
    expect(stop).toHaveBeenCalledWith("duplicateToolCall");
  });

  it("failFast stops on tool error", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const sm = new ReactStateMachine({
      config: baseConfig,
      hooks,
      llm: makeLlm([`Thought: test\nAction: calc\nAction-Input: {"a":1}`]),
      toolRunner: makeToolRunner({
        calc: () => {
          throw new Error("fatal");
        },
      }),
    });
    const out = await sm.run({
      messages: [{ role: "user", content: "do it" }],
      availableTools: tools,
      taoIteration: 1,
      systemPrompt: "",
    });
    expect(out.status).toBe("fail_fast");
    expect(out.failedTool?.tool).toBe("calc");
  });

  it("max_iter stop when budget runs out", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const sm = new ReactStateMachine({
      config: { ...baseConfig, maxIterations: 2 },
      hooks,
      llm: makeLlm([
        `Thought: try1\nAction: search\nAction-Input: {"q":"a"}`,
        `Thought: try2\nAction: search\nAction-Input: {"q":"b"}`,
      ]),
      toolRunner: makeToolRunner({ search: () => "ok" }),
    });
    const out = await sm.run({
      messages: [{ role: "user", content: "x" }],
      availableTools: tools,
      taoIteration: 1,
      systemPrompt: "",
    });
    expect(out.status).toBe("max_iter");
    expect(out.iterations).toBe(2);
  });

  it("recovers from parse error by re-prompting", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const sm = new ReactStateMachine({
      config: baseConfig,
      hooks,
      llm: makeLlm([
        `Thought: x\nAction: search\nAction-Input: {malformed`,
        `Thought: retry properly\nFINAL: 答案`,
      ]),
      toolRunner: makeToolRunner({}),
    });
    const out = await sm.run({
      messages: [{ role: "user", content: "x" }],
      availableTools: tools,
      taoIteration: 1,
      systemPrompt: "",
    });
    expect(out.status).toBe("ok");
    expect(out.finalAnswer).toBe("答案");
  });

  it("hook can mutate action params via beforeAction.replace", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    hooks.on(
      "react.beforeAction",
      (ctx) => {
        ctx.replace({
          ...ctx.payload,
          params: { ...(ctx.payload.params as object), modified: true },
        });
      },
      { priority: 100 },
    );
    let captured: Record<string, unknown> = {};
    const sm = new ReactStateMachine({
      config: baseConfig,
      hooks,
      llm: makeLlm([
        `Thought: t\nAction: search\nAction-Input: {"q":"a"}`,
        `Thought: done\nFINAL: x`,
      ]),
      toolRunner: async (name, params) => {
        captured = params;
        return {
          toolName: name,
          success: true,
          output: "ok",
          durationMs: 0,
          traceId: "",
          callId: "c",
        };
      },
    });
    await sm.run({
      messages: [{ role: "user", content: "x" }],
      availableTools: tools,
      taoIteration: 1,
      systemPrompt: "",
    });
    expect(captured["modified"]).toBe(true);
  });
});
