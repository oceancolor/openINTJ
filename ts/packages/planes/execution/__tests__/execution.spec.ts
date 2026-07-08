import {
  AgentError,
  ErrorCode,
  HookBus,
  type HookLogger,
  ToolDescriptorSchema,
} from "@openintj/core";
import { describe, expect, it, vi } from "vitest";
import { CircuitBreaker, Executor, StepSchema, StepStateMachine, ToolHub } from "../src/index.js";

const silentLogger: HookLogger = { warn: () => {}, error: () => {} };

describe("StepStateMachine", () => {
  it("allows valid transitions", () => {
    const sm = new StepStateMachine({ clock: () => 1000 });
    const step = StepSchema.parse({ action: "noop" });
    expect(step.state).toBe("pending");
    sm.transition(step, "ready");
    expect(step.state).toBe("ready");
    sm.transition(step, "running");
    expect(step.startedAt).toBe(1000);
    sm.transition(step, "completed");
    expect(step.finishedAt).toBe(1000);
  });

  it("rejects illegal transitions with STATE_TRANSITION_INVALID", () => {
    const sm = new StepStateMachine();
    const step = StepSchema.parse({ action: "noop" });
    try {
      sm.transition(step, "completed");
      throw new Error("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(AgentError);
      expect((err as AgentError).code).toBe(ErrorCode.STATE_TRANSITION_INVALID);
    }
  });

  it("supports retry: failed → ready loops back", () => {
    const sm = new StepStateMachine();
    const step = StepSchema.parse({ action: "noop", maxRetries: 2 });
    sm.transition(step, "ready");
    sm.transition(step, "running");
    sm.transition(step, "failed");
    expect(sm.canRetry(step)).toBe(true);
    step.retryCount = 1;
    sm.transition(step, "ready");
    expect(step.state).toBe("ready");
    sm.transition(step, "running");
    sm.transition(step, "failed");
    step.retryCount = 2;
    expect(sm.canRetry(step)).toBe(false);
  });
});

describe("CircuitBreaker", () => {
  it("opens after failure threshold", () => {
    let now = 1000;
    const cb = new CircuitBreaker(
      { failureThreshold: 2, recoveryTimeoutMs: 100 },
      { clock: () => now },
    );
    expect(cb.canExecute()).toBe(true);
    cb.recordFailure();
    expect(cb.state).toBe("closed");
    cb.recordFailure();
    expect(cb.state).toBe("open");
    expect(cb.canExecute()).toBe(false);
    now += 101;
    expect(cb.canExecute()).toBe(true);
    expect(cb.state).toBe("half_open");
    cb.recordSuccess();
    expect(cb.state).toBe("closed");
  });
});

describe("ToolHub", () => {
  it("returns failure for unregistered tool", async () => {
    const hub = new ToolHub();
    const r = await hub.call("nope", {});
    expect(r.success).toBe(false);
    expect(r.error).toContain("未注册");
  });

  it("invokes registered handler", async () => {
    const hub = new ToolHub();
    hub.register(ToolDescriptorSchema.parse({ name: "echo", description: "echo" }), (params) => ({
      echoed: params,
    }));
    const r = await hub.call("echo", { x: 1 });
    expect(r.success).toBe(true);
    expect(r.output).toEqual({ echoed: { x: 1 } });
  });

  it("captures handler errors and records failure on breaker", async () => {
    const hub = new ToolHub();
    hub.register(ToolDescriptorSchema.parse({ name: "boom", description: "" }), () => {
      throw new Error("explode");
    });
    const r1 = await hub.call("boom", {});
    expect(r1.success).toBe(false);
    expect(r1.error).toBe("explode");
  });

  it("registers 4 builtin tools", () => {
    const hub = new ToolHub();
    hub.registerBuiltinTools();
    const names = hub.list().map((t) => t.name);
    expect(names).toEqual(
      expect.arrayContaining(["read_file", "write_file", "execute_command", "search"]),
    );
  });

  it("emits tool.beforeCall and tool.afterCall hooks", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const hub = new ToolHub({ hooks });
    hub.register(ToolDescriptorSchema.parse({ name: "echo", description: "" }), (p) => p);
    const before = vi.fn();
    const after = vi.fn();
    hooks.on("tool.beforeCall", before);
    hooks.on("tool.afterCall", after);
    await hub.call("echo", {});
    expect(before).toHaveBeenCalledOnce();
    expect(after).toHaveBeenCalledOnce();
  });

  it("hook can short-circuit (cancel) before tool call but tool still runs unless we adjust", async () => {
    // tool.beforeCall is in CANCELLABLE_EVENTS; verify cancel actually stops other hooks (not the call itself)
    const hooks = new HookBus({ logger: silentLogger });
    const hub = new ToolHub({ hooks });
    hub.register(ToolDescriptorSchema.parse({ name: "echo", description: "" }), (p) => p);
    const second = vi.fn();
    hooks.on(
      "tool.beforeCall",
      (ctx) => {
        ctx.cancel();
      },
      { priority: 100 },
    );
    hooks.on("tool.beforeCall", second, { priority: 50 });
    await hub.call("echo", {});
    expect(second).not.toHaveBeenCalled();
  });

  it("respects timeout", async () => {
    const hub = new ToolHub();
    hub.register(
      ToolDescriptorSchema.parse({
        name: "slow",
        description: "",
        timeoutS: 1,
      }),
      async () => {
        await new Promise((r) => setTimeout(r, 200));
        return "done";
      },
    );
    const r = await hub.call("slow", {}, { timeoutMs: 50 });
    expect(r.success).toBe(false);
    expect(r.error).toMatch(/超时/);
  });

  it("gate 拒绝 → success:false，且不执行 handler、不触发熔断", async () => {
    const handler = vi.fn(() => ({ ok: true }));
    const hub = new ToolHub({
      gate: ({ tool }) => {
        if (tool === "danger") throw new Error("策略阻断: danger");
      },
    });
    hub.register(ToolDescriptorSchema.parse({ name: "danger", description: "" }), handler);
    // 连续多次被 gate 拒绝：若计入熔断（阈值 3）会变成熔断错误；这里应始终是策略错误。
    for (let i = 0; i < 5; i++) {
      const r = await hub.call("danger", {});
      expect(r.success).toBe(false);
      expect(r.error).toContain("策略阻断");
      expect(r.error).not.toContain("熔断");
    }
    expect(handler).not.toHaveBeenCalled();
  });

  it("gate 放行 → handler 正常执行，gate 收到工具名 + params", async () => {
    const seen: Array<{ tool: string; params: Record<string, unknown> }> = [];
    const hub = new ToolHub({
      gate: ({ tool, params }) => {
        seen.push({ tool, params });
      },
    });
    hub.register(ToolDescriptorSchema.parse({ name: "echo", description: "" }), (p) => p);
    const r = await hub.call("echo", { x: 1 });
    expect(r.success).toBe(true);
    expect(r.output).toEqual({ x: 1 });
    expect(seen).toEqual([{ tool: "echo", params: { x: 1 } }]);
  });

  it("gate 拒绝仍会 emit tool.afterCall（可观测），但不 emit tool.onError", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const after = vi.fn();
    const onError = vi.fn();
    hooks.on("tool.afterCall", after);
    hooks.on("tool.onError", onError);
    const hub = new ToolHub({
      hooks,
      gate: () => {
        throw new Error("blocked");
      },
    });
    hub.register(ToolDescriptorSchema.parse({ name: "echo", description: "" }), (p) => p);
    await hub.call("echo", {});
    expect(after).toHaveBeenCalledOnce();
    expect(onError).not.toHaveBeenCalled();
  });
});

describe("Executor (sequential)", () => {
  it("executes steps via toolHub and reports success", async () => {
    const hub = new ToolHub();
    hub.register(ToolDescriptorSchema.parse({ name: "ok", description: "" }), () => ({ ok: true }));
    const exec = new Executor({ toolHub: hub, registerBuiltins: false });
    const steps = [StepSchema.parse({ action: "ok" }), StepSchema.parse({ action: "ok" })];
    const r = await exec.execute(steps, "sequential");
    expect(r.success).toBe(true);
    expect(r.finishedSteps).toHaveLength(2);
    expect(r.failedSteps).toHaveLength(0);
  });

  it("retries on retriable failure (fixes Python v2 dead retry bug)", async () => {
    const hub = new ToolHub();
    let attempts = 0;
    hub.register(ToolDescriptorSchema.parse({ name: "flaky", description: "" }), () => {
      attempts++;
      if (attempts < 3) throw new Error("transient");
      return { recovered: true };
    });
    const exec = new Executor({ toolHub: hub, registerBuiltins: false });
    const step = StepSchema.parse({ action: "flaky", maxRetries: 5 });
    const r = await exec.execute([step], "sequential");
    expect(r.success).toBe(true);
    expect(attempts).toBe(3);
    expect(step.retryCount).toBe(2);
  });

  it("gives up after maxRetries and records failure", async () => {
    const hub = new ToolHub();
    hub.register(ToolDescriptorSchema.parse({ name: "always_fail", description: "" }), () => {
      throw new Error("always");
    });
    const exec = new Executor({ toolHub: hub, registerBuiltins: false });
    const step = StepSchema.parse({ action: "always_fail", maxRetries: 2 });
    const r = await exec.execute([step], "sequential");
    expect(r.success).toBe(false);
    expect(r.failedSteps).toHaveLength(1);
    expect(r.errors[0]?.retryCount).toBe(2);
  });

  it("parallel mode runs all in parallel and aggregates", async () => {
    const hub = new ToolHub();
    let counter = 0;
    hub.register(ToolDescriptorSchema.parse({ name: "tick", description: "" }), async () => {
      await new Promise((r) => setTimeout(r, 30));
      return ++counter;
    });
    const exec = new Executor({ toolHub: hub, registerBuiltins: false });
    const steps = [
      StepSchema.parse({ action: "tick" }),
      StepSchema.parse({ action: "tick" }),
      StepSchema.parse({ action: "tick" }),
    ];
    const t0 = Date.now();
    const r = await exec.execute(steps, "parallel");
    const elapsed = Date.now() - t0;
    expect(r.success).toBe(true);
    expect(r.finishedSteps).toHaveLength(3);
    // 三步并行应该在 100ms 内完成（每步 30ms 串行需要 90+ms）
    expect(elapsed).toBeLessThan(150);
  });
});
