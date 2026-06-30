/**
 * RFC-001 §8 性能目标守护（不计 LLM 调用）。
 *
 * 同样是宽松的回归守护：用即时返回的 mock LLM，测量 TAO/ReAct 框架自身的状态机开销。
 * RFC 目标：单轮 TAO 不计 LLM <50ms，ReAct 单步 <5ms。守护阈值放到 ~10x 余量，抗环境抖动。
 */
import { describe, expect, it } from "vitest";
import {
  DEFAULT_REACT_CONFIG,
  DEFAULT_TAO_CONFIG,
  HookBus,
  type HookLogger,
  type LlmClient,
  ReactStateMachine,
  TaoLoop,
  type ToolCallResult,
} from "../../src/index.js";

const silent: HookLogger = { warn: () => {}, error: () => {} };

const instantLlm = (reply: string): LlmClient => ({
  async chat() {
    return reply;
  },
  async visionChat() {
    return reply;
  },
  getStatus() {
    return {
      provider: "perf",
      model: "x",
      available: true,
      mode: "live",
      status: "connected",
      visionSupported: false,
    };
  },
});

const passingRunner = async (name: string): Promise<ToolCallResult> => ({
  toolName: name,
  success: true,
  output: "ok",
  durationMs: 0,
  traceId: "",
  callId: "c",
});

const makeTao = (): TaoLoop => {
  const hooks = new HookBus({ logger: silent });
  return new TaoLoop({
    config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: 1 },
    hooks,
    react: new ReactStateMachine({
      config: DEFAULT_REACT_CONFIG,
      hooks,
      llm: instantLlm("Thought: 直接回答\nFINAL: 答案"),
      toolRunner: passingRunner,
    }),
    availableTools: () => [],
  });
};

describe("perf: TAO single iteration (mock LLM)", () => {
  it("单轮 TAO 框架开销低（RFC-001 §8 目标 <50ms；守护阈值 50ms 平均）", async () => {
    // 预热
    for (let i = 0; i < 20; i++) await makeTao().run("你好");
    const N = 200;
    const tao = makeTao();
    const t0 = performance.now();
    for (let i = 0; i < N; i++) await tao.run("你好世界");
    const perRunMs = (performance.now() - t0) / N;
    console.log(`[perf] TAO single-iter run: ${perRunMs.toFixed(3)} ms/run (N=${N})`);
    // 即时 mock LLM 下，单轮框架开销应远小于 50ms
    expect(perRunMs).toBeLessThan(50);
  });
});
