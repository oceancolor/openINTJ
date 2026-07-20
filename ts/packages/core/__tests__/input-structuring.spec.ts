import { describe, expect, it, vi } from "vitest";
import { HookBus, type LlmClient, shouldStructureInput, structureUserInput } from "../src/index.js";

const client = (reply: string): LlmClient => ({
  chat: vi.fn(async () => reply),
  visionChat: vi.fn(async () => reply),
  getStatus: () => ({
    provider: "test",
    model: "test",
    available: true,
    mode: "live",
    status: "connected",
    visionSupported: false,
  }),
});

describe("input structuring", () => {
  it("does not spend a model call on simple clear input", async () => {
    const llm = client("{}");
    const result = await structureUserInput({ input: "把 hello 转大写", llm });
    expect(result).toMatchObject({
      action: "proceed",
      mode: "pass-through",
      executionInput: "把 hello 转大写",
      tokensSpent: 0,
    });
    expect(llm.chat).not.toHaveBeenCalled();
  });

  it("structures complex input while preserving explicit constraints", async () => {
    const llm = client(
      JSON.stringify({
        action: "proceed",
        executionInput: "目标：设计迁移方案。约束：三阶段。交付物：每阶段清单。",
        structure: {
          goal: "设计迁移方案",
          context: ["TypeScript CLI"],
          relations: ["三个阶段按顺序执行"],
          constraints: ["必须分为三阶段"],
          deliverables: ["每阶段交付物清单"],
          dependencies: [],
          assumptions: [],
        },
        ambiguityScore: 0.2,
        questions: [],
      }),
    );
    const result = await structureUserInput({
      input: "规划 TypeScript CLI 三阶段迁移方案，并列出每阶段交付物",
      llm,
    });
    expect(result.action).toBe("proceed");
    expect(result.mode).toBe("structured");
    expect(result.structure.constraints).toContain("必须分为三阶段");
    expect(result.tokensSpent).toBeGreaterThan(0);
    expect(llm.chat).toHaveBeenCalledOnce();
  });

  it("pauses for at most three material clarification questions", async () => {
    const hooks = new HookBus();
    const seen = vi.fn();
    hooks.on("event.INPUT_STRUCTURE_CLARIFICATION", (ctx) => seen(ctx.payload));
    const llm = client(
      JSON.stringify({
        action: "clarify",
        executionInput: "",
        structure: {
          goal: "部署系统",
          context: [],
          relations: [],
          constraints: [],
          deliverables: [],
          dependencies: [],
          assumptions: [],
        },
        ambiguityScore: 0.9,
        questions: ["部署到哪个环境？", "目标集群是什么？", "允许的发布时间窗是什么？"],
      }),
    );
    const result = await structureUserInput({
      input: "设计并执行部署方案，然后验证结果",
      llm,
      hooks,
    });
    expect(result.action).toBe("clarify");
    expect(result.questions).toHaveLength(3);
    expect(seen).toHaveBeenCalledWith(expect.objectContaining({ questionCount: 3 }));
  });

  it("falls back to the original input on invalid model output", async () => {
    const result = await structureUserInput({
      input: "分析这个系统并给出完整迁移方案",
      llm: client("not-json"),
    });
    expect(result).toMatchObject({
      action: "proceed",
      mode: "fallback",
      executionInput: "分析这个系统并给出完整迁移方案",
      tokensSpent: 0,
    });
  });

  it("detects complex and ambiguous input adaptively", () => {
    expect(shouldStructureInput("你好")).toBe(false);
    expect(shouldStructureInput("把这个优化一下")).toBe(true);
    expect(shouldStructureInput("设计迁移方案，并分析依赖关系和交付物")).toBe(true);
  });

  it("propagates AbortSignal without falling back to a silent proceed", async () => {
    const controller = new AbortController();
    const llm = client("{}");
    llm.chat = vi.fn(
      async (_messages, opts?: { signal?: AbortSignal }) =>
        new Promise<string>((_resolve, reject) => {
          opts?.signal?.addEventListener(
            "abort",
            () => reject(opts.signal?.reason ?? new Error("aborted")),
            { once: true },
          );
        }),
    );
    const pending = structureUserInput({
      input: "设计并执行完整迁移方案，并列出依赖与交付物",
      llm,
      signal: controller.signal,
    });
    controller.abort(new Error("user_cancelled"));
    await expect(pending).rejects.toThrow("user_cancelled");
  });

  it("honors policy=off and always", () => {
    expect(shouldStructureInput("设计迁移方案并列出交付物", "off")).toBe(false);
    expect(shouldStructureInput("你好", "always")).toBe(true);
  });
});
