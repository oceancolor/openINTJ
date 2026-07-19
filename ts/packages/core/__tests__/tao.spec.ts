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
  it("fails fast when the caller signal is already aborted", async () => {
    const hooks = new HookBus({ logger: silent });
    const llm = makeLlm(["FINAL: unreachable"]);
    const react = new ReactStateMachine({
      config: DEFAULT_REACT_CONFIG,
      hooks,
      llm,
      toolRunner: passingRunner,
    });
    const tao = new TaoLoop({
      config: DEFAULT_TAO_CONFIG,
      hooks,
      react,
      availableTools: () => tools,
    });
    const controller = new AbortController();
    controller.abort(new Error("cancelled"));
    await expect(tao.run("stop", { signal: controller.signal })).rejects.toThrow("cancelled");
  });

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

  it("enableReact=false 退化为单次 LLM 调用（不跑微循环、不调工具）", async () => {
    const hooks = new HookBus({ logger: silent });
    // 即便 LLM 输出 Action 协议，退化路径也应直接当成答案返回，不解析、不调工具。
    const llm = makeLlm(["Action: search\nAction-Input: {}"]);
    const toolRunner = vi.fn(passingRunner);
    const react = new ReactStateMachine({
      config: DEFAULT_REACT_CONFIG,
      hooks,
      llm,
      toolRunner,
    });
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1, enableReact: false },
      hooks,
      react,
      availableTools: () => tools,
    });
    const r = await tao.run("你好世界");
    expect(r.status).toBe("completed");
    expect(r.finalAnswer).toBe("Action: search\nAction-Input: {}");
    expect(r.iterations).toBe(1);
    expect(toolRunner).not.toHaveBeenCalled();
    // 轨迹是单个 final 节点
    expect(r.trajectory).toHaveLength(1);
    expect(r.trajectory[0]?.state.type).toBe("final");
  });

  it("contextProvider 覆盖静态 systemPrompt 并把记忆注入到 ReAct", async () => {
    const hooks = new HookBus({ logger: silent });
    let capturedSystem = "";
    const llm: LlmClient = {
      async chat(messages) {
        const sys = messages.find((m) => m.role === "system");
        capturedSystem = typeof sys?.content === "string" ? sys.content : "";
        return "FINAL: ok";
      },
      async visionChat() {
        return "v";
      },
      getStatus() {
        return {
          provider: "t",
          model: "x",
          available: true,
          mode: "live",
          status: "connected",
          visionSupported: false,
        };
      },
    };
    const seen: string[] = [];
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm,
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
      systemPrompt: "STATIC_PROMPT",
      contextProvider: ({ query }) => {
        seen.push(query);
        return "BASE_PROMPT\n\n[记忆参考]\n#1 用户喜欢绿茶";
      },
    });
    await tao.run("我喜欢什么？");
    expect(seen).toEqual(["我喜欢什么？"]);
    expect(capturedSystem).toContain("BASE_PROMPT");
    expect(capturedSystem).toContain("用户喜欢绿茶");
    expect(capturedSystem).not.toContain("STATIC_PROMPT");
  });

  it("run(opts.topK) 透传给 contextProvider（外部路由降 token 用），未传则 undefined", async () => {
    const hooks = new HookBus({ logger: silent });
    const llm: LlmClient = {
      async chat() {
        return "FINAL: ok";
      },
      async visionChat() {
        return "v";
      },
      getStatus() {
        return {
          provider: "t",
          model: "x",
          available: true,
          mode: "live",
          status: "connected",
          visionSupported: false,
        };
      },
    };
    const seenTopK: (number | undefined)[] = [];
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm,
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
      systemPrompt: "S",
      contextProvider: ({ topK }) => {
        seenTopK.push(topK);
        return "B";
      },
    });
    await tao.run("q1", { topK: 3 });
    await tao.run("q2");
    expect(seenTopK).toEqual([3, undefined]);
  });

  it("contextProvider 抛错时回退静态 systemPrompt，不阻断主循环", async () => {
    const hooks = new HookBus({ logger: silent });
    let capturedSystem = "";
    const llm: LlmClient = {
      async chat(messages) {
        const sys = messages.find((m) => m.role === "system");
        capturedSystem = typeof sys?.content === "string" ? sys.content : "";
        return "FINAL: ok";
      },
      async visionChat() {
        return "v";
      },
      getStatus() {
        return {
          provider: "t",
          model: "x",
          available: true,
          mode: "live",
          status: "connected",
          visionSupported: false,
        };
      },
    };
    const tao = new TaoLoop({
      config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
      hooks,
      react: new ReactStateMachine({
        config: DEFAULT_REACT_CONFIG,
        hooks,
        llm,
        toolRunner: passingRunner,
      }),
      availableTools: () => [],
      systemPrompt: "FALLBACK_PROMPT",
      contextProvider: () => {
        throw new Error("retrieval boom");
      },
    });
    const r = await tao.run("hi");
    expect(r.status).toBe("completed");
    expect(capturedSystem).toContain("FALLBACK_PROMPT");
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
