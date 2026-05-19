import { afterEach, describe, expect, it, vi } from "vitest";
import { CommandType, ErrorCode, HookBus, type HookLogger } from "../src/index.js";

const silentLogger: HookLogger = {
  warn: () => {
    /* noop */
  },
  error: () => {
    /* noop */
  },
};

describe("HookBus.on/emit", () => {
  it("invokes handlers in priority desc order", async () => {
    const bus = new HookBus({ logger: silentLogger });
    const order: number[] = [];
    bus.on("event.MEMORY_LOADED", () => void order.push(1), { priority: 0 });
    bus.on("event.MEMORY_LOADED", () => void order.push(2), { priority: 100 });
    bus.on("event.MEMORY_LOADED", () => void order.push(3), { priority: 50 });
    await bus.emit("event.MEMORY_LOADED", { count: 3, budgetUsage: 0.1 });
    expect(order).toEqual([2, 3, 1]);
  });

  it("returns payload as-is when no handlers", async () => {
    const bus = new HookBus({ logger: silentLogger });
    const res = await bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 });
    expect(res).toEqual({ count: 0, budgetUsage: 0 });
  });

  it("supports async handlers", async () => {
    const bus = new HookBus({ logger: silentLogger });
    let value = 0;
    bus.on("event.MEMORY_LOADED", async (ctx) => {
      await new Promise((r) => setTimeout(r, 5));
      value = ctx.payload.count;
    });
    await bus.emit("event.MEMORY_LOADED", { count: 42, budgetUsage: 0 });
    expect(value).toBe(42);
  });
});

describe("HookBus.replace", () => {
  it("replaces payload visible to subsequent handlers and final return", async () => {
    const bus = new HookBus({ logger: silentLogger });
    const seen: number[] = [];
    bus.on(
      "event.MEMORY_LOADED",
      (ctx) => {
        seen.push(ctx.payload.count);
        ctx.replace({ count: 99, budgetUsage: 1 });
      },
      { priority: 100 },
    );
    bus.on(
      "event.MEMORY_LOADED",
      (ctx) => {
        seen.push(ctx.payload.count);
      },
      { priority: 50 },
    );
    const res = await bus.emit("event.MEMORY_LOADED", { count: 1, budgetUsage: 0 });
    expect(seen).toEqual([1, 99]);
    expect(res).toEqual({ count: 99, budgetUsage: 1 });
  });
});

describe("HookBus.cancel (short-circuit)", () => {
  it("stops execution after cancel on cancellable event", async () => {
    const bus = new HookBus({ logger: silentLogger });
    const called: number[] = [];
    bus.on(
      "tool.beforeCall",
      (ctx) => {
        called.push(1);
        ctx.cancel();
      },
      { priority: 100 },
    );
    bus.on(
      "tool.beforeCall",
      () => {
        called.push(2);
      },
      { priority: 50 },
    );
    await bus.emit("tool.beforeCall", {
      tool: "read_file",
      params: { path: "x" },
      toolDescriptor: {
        name: "read_file",
        description: "",
        inputSchema: {},
        outputSchema: {},
        permissions: [],
        timeoutS: 30,
        idempotent: true,
        errorSemantics: "fail_fast",
      },
    });
    expect(called).toEqual([1]);
  });

  it("ignores cancel and warns on non-cancellable event in non-strict mode", async () => {
    const warn = vi.fn();
    const bus = new HookBus({ logger: { warn, error: () => {} } });
    let secondRan = false;
    bus.on(
      "event.MEMORY_LOADED",
      (ctx) => {
        ctx.cancel();
      },
      { priority: 100 },
    );
    bus.on("event.MEMORY_LOADED", () => {
      secondRan = true;
    });
    await bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 });
    expect(secondRan).toBe(true);
    expect(warn).toHaveBeenCalledOnce();
  });

  it("throws on non-cancellable event in strict mode", async () => {
    const bus = new HookBus({ strictMode: true, logger: silentLogger });
    bus.on("event.MEMORY_LOADED", (ctx) => {
      ctx.cancel();
    });
    await expect(bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 })).rejects.toThrow(
      /cancel not allowed/,
    );
  });
});

describe("HookBus.once and offByTag", () => {
  it("once handler runs at most once", async () => {
    const bus = new HookBus({ logger: silentLogger });
    const fn = vi.fn();
    bus.on("event.MEMORY_LOADED", fn, { once: true });
    await bus.emit("event.MEMORY_LOADED", { count: 1, budgetUsage: 0 });
    await bus.emit("event.MEMORY_LOADED", { count: 2, budgetUsage: 0 });
    expect(fn).toHaveBeenCalledOnce();
  });

  it("offByTag removes all handlers with matching tag", async () => {
    const bus = new HookBus({ logger: silentLogger });
    const a = vi.fn();
    const b = vi.fn();
    const c = vi.fn();
    bus.on("event.MEMORY_LOADED", a, { tag: "audit" });
    bus.on("event.MEMORY_LOADED", b, { tag: "audit" });
    bus.on("event.MEMORY_LOADED", c);
    expect(bus.offByTag("audit")).toBe(2);
    await bus.emit("event.MEMORY_LOADED", { count: 1, budgetUsage: 0 });
    expect(a).not.toHaveBeenCalled();
    expect(b).not.toHaveBeenCalled();
    expect(c).toHaveBeenCalledOnce();
  });

  it("unregister fn removes specific handler", async () => {
    const bus = new HookBus({ logger: silentLogger });
    const fn = vi.fn();
    const off = bus.on("event.MEMORY_LOADED", fn);
    off();
    await bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 });
    expect(fn).not.toHaveBeenCalled();
  });
});

describe("HookBus error handling", () => {
  it("non-strict mode swallows handler errors and continues", async () => {
    const error = vi.fn();
    const bus = new HookBus({ logger: { warn: () => {}, error } });
    const second = vi.fn();
    bus.on(
      "event.MEMORY_LOADED",
      () => {
        throw new Error("boom");
      },
      { priority: 100 },
    );
    bus.on("event.MEMORY_LOADED", second, { priority: 50 });
    await bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 });
    expect(second).toHaveBeenCalledOnce();
    expect(error).toHaveBeenCalledOnce();
  });

  it("strict mode rethrows handler errors", async () => {
    const bus = new HookBus({ strictMode: true, logger: silentLogger });
    bus.on("event.MEMORY_LOADED", () => {
      throw new Error("strict boom");
    });
    await expect(bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 })).rejects.toThrow(
      /strict boom/,
    );
  });
});

describe("HookBus.inspect", () => {
  it("reports per-event and per-tag counts", () => {
    const bus = new HookBus({ logger: silentLogger });
    bus.on("event.MEMORY_LOADED", () => {}, { tag: "audit" });
    bus.on("event.MEMORY_LOADED", () => {}, { tag: "audit" });
    bus.on("policy.beforeCheck", () => {}, { tag: "policy-stack" });
    const r = bus.inspect();
    expect(r.total).toBe(3);
    expect(r.byEvent).toEqual({
      "event.MEMORY_LOADED": 2,
      "policy.beforeCheck": 1,
    });
    expect(r.byTag).toEqual({ audit: 2, "policy-stack": 1 });
  });
});

describe("HookBus event stack depth", () => {
  it("throws when handler triggers nested emit beyond max depth", async () => {
    const bus = new HookBus({ maxEventStackDepth: 3, logger: silentLogger });
    bus.on("event.MEMORY_LOADED", async () => {
      await bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 }, { traceId: "t" });
    });
    await expect(
      bus.emit("event.MEMORY_LOADED", { count: 0, budgetUsage: 0 }, { traceId: "t" }),
    ).rejects.toMatchObject({ code: ErrorCode.HOOK_ERROR });
  });
});

describe("HookBus type discipline", () => {
  it("preserves payload type through emit chain", async () => {
    const bus = new HookBus({ logger: silentLogger });
    bus.on("policy.beforeCheck", (ctx) => {
      // payload 类型应是 { command: Command }
      expect(ctx.payload.command.commandType).toBe(CommandType.PLAN);
    });
    await bus.emit("policy.beforeCheck", {
      command: {
        commandType: CommandType.PLAN,
        target: "planner",
        payload: {},
        commandId: "cid",
        createdAt: 0,
      },
    });
  });
});

afterEach(() => {
  /* nothing global; each test owns its bus */
});
