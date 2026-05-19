import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_REACT_CONFIG,
  DEFAULT_TAO_CONFIG,
  HookBus,
  type HookLogger,
  type LlmClient,
  ReactStateMachine,
  ShaderMode,
  TaoLoop,
  TaskType,
  type ToolCallResult,
  type ToolDescriptor,
  __taoTest__,
} from "../src/index.js";

const silent: HookLogger = { warn: () => {}, error: () => {} };

const tools: ToolDescriptor[] = [
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
];

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

const passingRunner = async (name: string): Promise<ToolCallResult> => ({
  toolName: name,
  success: true,
  output: "ok",
  durationMs: 1,
  traceId: "",
  callId: "c1",
});

describe("__taoTest__.detectTaskType", () => {
  const { detectTaskType } = __taoTest__;
  it("classifies code task", () => {
    expect(detectTaskType("帮我写一个函数实现 fibonacci")).toBe(TaskType.CODE_GENERATION);
  });
  it("classifies analysis", () => {
    expect(detectTaskType("分析以下数据集")).toBe(TaskType.ANALYSIS);
  });
  it("falls back to quick_response for short queries", () => {
    expect(detectTaskType("你好")).toBe(TaskType.QUICK_RESPONSE);
  });
});

describe("TaoLoop.run (single iteration)", () => {
  it("delegates to ReAct and returns final answer", async () => {
    const hooks = new HookBus({ logger: silent });
    const llm = makeLlm([`Thought: 直接回答\nFINAL: 这是答案`]);
    const react = new ReactStateMachine({
      config: DEFAULT_REACT_CONFIG,
      hooks,
      llm,
      toolRunner: passingRunner,
    });
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react,
      availableTools: () => tools,
    });
    const r = await tao.run("你好世界");
    expect(r.status).toBe("completed");
    expect(r.finalAnswer).toBe("这是答案");
    expect(r.iterations).toBe(1);
    expect(r.taskType).toBe(TaskType.QUICK_RESPONSE);
  });

  it("emits all 6 tao.* hooks in order", async () => {
    const hooks = new HookBus({ logger: silent });
    const events: string[] = [];
    for (const e of [
      "tao.beforeThink",
      "tao.afterThink",
      "tao.beforeAct",
      "tao.afterAct",
      "tao.beforeObserve",
      "tao.afterObserve",
    ] as const) {
      hooks.on(e, () => void events.push(e));
    }
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm: makeLlm([`FINAL: hi`]),
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
    });
    await tao.run("你好");
    expect(events).toEqual([
      "tao.beforeThink",
      "tao.afterThink",
      "tao.beforeAct",
      "tao.afterAct",
      "tao.beforeObserve",
      "tao.afterObserve",
    ]);
  });

  it("multi-iter: continues when needsContinue returns true", async () => {
    const hooks = new HookBus({ logger: silent });
    let calls = 0;
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 3 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm: makeLlm([`Thought: t\nFINAL: round-${++calls}`]),
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
      needsContinue: (ctx) => ctx.iteration < 3,
    });
    const r = await tao.run("complex query");
    expect(r.iterations).toBe(3);
    expect(r.finalAnswer).toMatch(/round-/);
  });

  it("propagates ReAct fail_fast as Tao failed", async () => {
    const hooks = new HookBus({ logger: silent });
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 2 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm: makeLlm([`Thought: x\nAction: search\nAction-Input: {"q":"a"}`]),
        toolRunner: async (name) => ({
          toolName: name,
          success: false,
          error: "boom",
          durationMs: 1,
          traceId: "",
          callId: "c",
        }),
      }),
      availableTools: () => tools,
    });
    const r = await tao.run("query");
    expect(r.status).toBe("failed");
    expect(r.failureReason).toContain("boom");
  });

  it("uses shaderSelector to map task type to shader mode", async () => {
    const hooks = new HookBus({ logger: silent });
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm: makeLlm([`FINAL: x`]),
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
      taskClassifier: () => TaskType.CODE_GENERATION,
    });
    const r = await tao.run("write a function");
    expect(r.shaderMode).toBe(ShaderMode.HIGH_FIDELITY);
  });

  it("respects maxTaoIterations and reports max_iter_reached", async () => {
    const hooks = new HookBus({ logger: silent });
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm: makeLlm([`Thought: t\nFINAL: x`]),
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
      needsContinue: () => true,
    });
    const r = await tao.run("loopy");
    expect(r.status).toBe("max_iter_reached");
    expect(r.iterations).toBe(1);
  });

  it("registers hook tag and offByTag clears them", async () => {
    const hooks = new HookBus({ logger: silent });
    const fn = vi.fn();
    hooks.on("tao.beforeThink", fn, { tag: "audit" });
    hooks.on("tao.afterThink", fn, { tag: "audit" });
    expect(hooks.offByTag("audit")).toBe(2);
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm: makeLlm([`FINAL: x`]),
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
    });
    await tao.run("x");
    expect(fn).not.toHaveBeenCalled();
  });
});
