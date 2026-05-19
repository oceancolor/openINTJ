import { describe, expect, it } from "vitest";
import {
  AgentError,
  ContextBudgetTracker,
  ErrorCode,
  FrameworkConfigSchema,
  LODLevel,
  MemoryFragmentSchema,
  ShaderConfigSchema,
  ShaderMode,
  TASK_SHADER_MAP,
  TaskType,
  ToolDescriptorSchema,
  decayImportance,
  estimateTokens,
  getLodForMode,
  getShaderForTask,
  loadFrameworkConfigFromEnv,
} from "../src/index.js";

describe("AgentError", () => {
  it("formats message with code prefix", () => {
    const err = new AgentError({
      code: ErrorCode.POLICY_BLOCKED,
      message: "blocked by test",
      retriable: false,
    });
    expect(err.message).toBe("[POLICY_BLOCKED] blocked by test");
    expect(err.code).toBe("POLICY_BLOCKED");
    expect(err.retriable).toBe(false);
  });

  it("serializes to JSON cleanly", () => {
    const err = new AgentError({
      code: ErrorCode.VALIDATION_ERROR,
      message: "bad",
      details: { field: "x" },
    });
    expect(err.toJSON()).toMatchObject({
      name: "AgentError",
      code: "VALIDATION_ERROR",
      retriable: false,
      details: { field: "x" },
    });
  });
});

describe("ShaderConfigSchema", () => {
  it("applies defaults aligned with Python framework_core", () => {
    const cfg = ShaderConfigSchema.parse({});
    expect(cfg.mode).toBe(ShaderMode.ADAPTIVE);
    expect(cfg.targetLod).toBe(LODLevel.LOD_1);
    expect(cfg.maxSummaryLength).toBe(200);
    expect(cfg.compactionThreshold).toBe(0.8);
    expect(cfg.recencyHalfLifeHours).toBe(24);
  });

  it("aligns task→shader map with Python TASK_SHADER_MAP", () => {
    expect(TASK_SHADER_MAP[TaskType.CODE_GENERATION]).toBe(ShaderMode.HIGH_FIDELITY);
    expect(TASK_SHADER_MAP[TaskType.GENERAL_CHAT]).toBe(ShaderMode.LOW_FIDELITY);
    expect(TASK_SHADER_MAP[TaskType.ANALYSIS]).toBe(ShaderMode.HYBRID);
    expect(getShaderForTask(TaskType.CODE_GENERATION)).toBe(ShaderMode.HIGH_FIDELITY);
  });
});

describe("getLodForMode", () => {
  it.each([
    [ShaderMode.HIGH_FIDELITY, 0.3, LODLevel.LOD_0],
    [ShaderMode.HIGH_FIDELITY, 0.7, LODLevel.LOD_1],
    [ShaderMode.LOW_FIDELITY, 0.5, LODLevel.LOD_3],
    [ShaderMode.LOW_FIDELITY, 0.95, LODLevel.LOD_4],
    [ShaderMode.HYBRID, 0.3, LODLevel.LOD_1],
    [ShaderMode.HYBRID, 0.6, LODLevel.LOD_2],
    [ShaderMode.HYBRID, 0.9, LODLevel.LOD_3],
    [ShaderMode.ADAPTIVE, 0.0, LODLevel.LOD_0],
    [ShaderMode.ADAPTIVE, 0.45, LODLevel.LOD_1],
    [ShaderMode.ADAPTIVE, 0.6, LODLevel.LOD_2],
    [ShaderMode.ADAPTIVE, 0.85, LODLevel.LOD_3],
    [ShaderMode.ADAPTIVE, 1.0, LODLevel.LOD_4],
  ])("mode=%s ratio=%f → lod=%i", (mode, ratio, expected) => {
    expect(getLodForMode(mode, ratio)).toBe(expected);
  });
});

describe("ContextBudgetTracker", () => {
  it("computes availableTokens correctly", () => {
    const t = new ContextBudgetTracker({
      maxTokens: 10_000,
      reservedTokens: 1000,
      systemPromptTokens: 200,
      conversationTokens: 1000,
      memoryTokens: 500,
      toolTokens: 100,
    });
    expect(t.availableTokens).toBe(10_000 - 1000 - 200 - 1000 - 500 - 100);
  });

  it("usageRatio is bounded [0, 1]", () => {
    const t = new ContextBudgetTracker({
      maxTokens: 1000,
      conversationTokens: 5000,
    });
    expect(t.usageRatio).toBe(1);
  });

  it("triggers compaction at threshold", () => {
    const t = new ContextBudgetTracker({
      maxTokens: 1000,
      reservedTokens: 0,
      conversationTokens: 800,
    });
    expect(t.needsCompaction(0.79)).toBe(true);
    expect(t.needsCompaction(0.81)).toBe(false);
  });

  it("memoryBudget reserves 30% of remaining", () => {
    const t = new ContextBudgetTracker({
      maxTokens: 10_000,
      reservedTokens: 1000,
      systemPromptTokens: 0,
      memoryTokens: 0,
    });
    expect(t.memoryBudget).toBe(Math.floor((10_000 - 1000) * 0.3));
  });
});

describe("decayImportance", () => {
  it("returns importance unchanged at age=0", () => {
    const now = 1_000_000;
    const f = { importance: 0.8, timestamp: now };
    expect(decayImportance(f, 24, now)).toBeCloseTo(0.8, 6);
  });

  it("halves at exactly halfLifeHours", () => {
    const now = 1_000_000;
    const f = { importance: 1, timestamp: now - 24 * 3600 };
    expect(decayImportance(f, 24, now)).toBeCloseTo(0.5, 4);
  });
});

describe("MemoryFragmentSchema", () => {
  it("auto-fills defaults", () => {
    const f = MemoryFragmentSchema.parse({ content: "hello" });
    expect(f.fragmentId).toMatch(/^[0-9a-f-]{36}$/);
    expect(f.content).toBe("hello");
    expect(f.importance).toBe(0.5);
    expect(f.summaries).toEqual({});
    expect(estimateTokens("hello world!")).toBeGreaterThan(0);
  });

  it("coerces summary keys to numbers", () => {
    const f = MemoryFragmentSchema.parse({
      content: "x",
      summaries: { 1: "short", "2": "shorter" },
    });
    expect(f.summaries[1]).toBe("short");
    expect(f.summaries[2]).toBe("shorter");
  });
});

describe("ToolDescriptorSchema", () => {
  it("applies sane defaults", () => {
    const t = ToolDescriptorSchema.parse({ name: "echo", description: "echo back" });
    expect(t.timeoutS).toBe(30);
    expect(t.idempotent).toBe(false);
    expect(t.errorSemantics).toBe("fail_fast");
    expect(t.permissions).toEqual([]);
  });
});

describe("loadFrameworkConfigFromEnv", () => {
  it("rejects when required keys missing", () => {
    expect(() => loadFrameworkConfigFromEnv({})).toThrowError(/缺少必需的配置项/);
  });

  it("parses sane env", () => {
    const cfg = loadFrameworkConfigFromEnv({
      AGENT_ENV: "test",
      AGENT_MAX_RETRY: "5",
      AGENT_DEFAULT_TIMEOUT_S: "60",
      AGENT_GOVERNANCE_STRICT: "false",
      AGENT_MAX_CONTEXT_TOKENS: "200000",
      AGENT_SHADER_MODE: "hybrid",
      AGENT_MEMORY_HALF_LIFE_HOURS: "12.5",
    });
    expect(cfg).toEqual({
      env: "test",
      maxRetry: 5,
      defaultTimeoutS: 60,
      governanceStrict: false,
      maxContextTokens: 200000,
      shaderMode: "hybrid",
      memoryHalfLifeHours: 12.5,
    });
  });

  it("rejects invalid integer", () => {
    expect(() =>
      loadFrameworkConfigFromEnv({
        AGENT_ENV: "dev",
        AGENT_MAX_RETRY: "abc",
        AGENT_DEFAULT_TIMEOUT_S: "30",
      }),
    ).toThrowError(/不是合法整数/);
  });

  it("validates shader mode against schema", () => {
    expect(() =>
      loadFrameworkConfigFromEnv({
        AGENT_ENV: "dev",
        AGENT_MAX_RETRY: "1",
        AGENT_DEFAULT_TIMEOUT_S: "30",
        AGENT_SHADER_MODE: "bogus",
      }),
    ).toThrowError(/配置校验失败/);
  });
});

describe("FrameworkConfigSchema", () => {
  it("rejects max_context_tokens < 1024", () => {
    const r = FrameworkConfigSchema.safeParse({
      env: "dev",
      maxRetry: 0,
      defaultTimeoutS: 30,
      governanceStrict: true,
      maxContextTokens: 100,
    });
    expect(r.success).toBe(false);
  });
});
