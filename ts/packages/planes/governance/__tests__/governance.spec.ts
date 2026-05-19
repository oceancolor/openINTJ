import {
  AgentError,
  CommandSchema,
  CommandType,
  ErrorCode,
  HookBus,
  type HookLogger,
} from "@openintj/core";
import { describe, expect, it, vi } from "vitest";
import { AuditTrail, GovernancePlane, PolicyEngine, QuotaGuard } from "../src/index.js";

const silentLogger: HookLogger = {
  warn: () => {},
  error: () => {},
};

const mkCommand = (target: string, type = CommandType.TOOL_CALL) =>
  CommandSchema.parse({ commandType: type, target });

describe("PolicyEngine", () => {
  it("allows whitelist targets", () => {
    const eng = new PolicyEngine();
    const ev = eng.check(mkCommand("read_file"));
    expect(ev.result).toBe("allowed");
    expect(ev.riskLevel).toBe("low");
  });

  it("blocks blacklisted target in strict mode", () => {
    const eng = new PolicyEngine({ strictMode: true });
    expect(() => eng.check(mkCommand("shell-delete"))).toThrowError(/策略阻断/);
    try {
      eng.check(mkCommand("filesystem-delete-recursive"));
    } catch (err) {
      expect(err).toBeInstanceOf(AgentError);
      expect((err as AgentError).code).toBe(ErrorCode.POLICY_BLOCKED);
      expect((err as AgentError).retriable).toBe(false);
    }
  });

  it("does not block in non-strict mode (acts as warning)", () => {
    const eng = new PolicyEngine({ strictMode: false });
    const ev = eng.check(mkCommand("shell-delete"));
    expect(ev.result).toBe("allowed");
  });

  it("warns for approval-required targets", () => {
    const eng = new PolicyEngine();
    const ev = eng.check(mkCommand("deploy-production"));
    expect(ev.result).toBe("warning");
    expect(ev.riskLevel).toBe("high");
  });

  it("supports runtime block/allow", () => {
    const eng = new PolicyEngine({ strictMode: true });
    eng.block("custom-target");
    expect(() => eng.check(mkCommand("custom-target"))).toThrow();
    eng.allow("custom-target");
    expect(eng.check(mkCommand("custom-target")).result).toBe("allowed");
  });
});

describe("AuditTrail", () => {
  it("records and queries events", () => {
    const trail = new AuditTrail();
    const eng = new PolicyEngine();
    trail.record(eng.check(mkCommand("read_file")));
    trail.record(eng.check(mkCommand("deploy-production")));
    expect(trail.getStats().totalEvents).toBe(2);
    expect(trail.query({ result: "warning" })).toHaveLength(1);
    expect(trail.query({ riskLevel: "high" })).toHaveLength(1);
  });

  it("trims to maxEvents", () => {
    const trail = new AuditTrail({ maxEvents: 5 });
    const eng = new PolicyEngine();
    for (let i = 0; i < 20; i++) {
      trail.record(eng.check(mkCommand("read_file")));
    }
    expect(trail.getStats().totalEvents).toBe(5);
  });
});

describe("QuotaGuard", () => {
  it("api quota accumulates and rejects beyond limit", () => {
    let now = 1000;
    const q = new QuotaGuard({ maxApiCallsPerHour: 3 }, { clock: () => now });
    expect(q.checkApiQuota()).toBe(true);
    q.recordApiCall();
    q.recordApiCall();
    q.recordApiCall();
    expect(q.checkApiQuota()).toBe(false);

    // advance > 1h
    now += 3601;
    expect(q.checkApiQuota()).toBe(true);
  });

  it("token quota sums correctly", () => {
    const now = 1000;
    const q = new QuotaGuard({ maxTokensPerHour: 100 }, { clock: () => now });
    q.recordTokenUsage(40);
    q.recordTokenUsage(40);
    expect(q.checkTokenQuota()).toBe(true);
    q.recordTokenUsage(30);
    expect(q.checkTokenQuota()).toBe(false);
    expect(q.getStats().tokensLastHour).toBe(110);
  });

  it("tool quota uses minute window", () => {
    let now = 1000;
    const q = new QuotaGuard({ maxToolCallsPerMinute: 2 }, { clock: () => now });
    q.recordToolCall();
    q.recordToolCall();
    expect(q.checkToolQuota()).toBe(false);
    now += 61;
    expect(q.checkToolQuota()).toBe(true);
  });
});

describe("GovernancePlane.checkAndRecord", () => {
  it("allows + records when no hooks attached", async () => {
    const plane = new GovernancePlane();
    const ev = await plane.checkAndRecord(mkCommand("read_file"));
    expect(ev.result).toBe("allowed");
    expect(plane.getStats().audit.totalEvents).toBe(1);
    expect(plane.getStats().quota.apiCallsLastHour).toBe(1);
  });

  it("emits hook events around check", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const plane = new GovernancePlane({ hooks });
    const before = vi.fn();
    const after = vi.fn();
    hooks.on("policy.beforeCheck", before);
    hooks.on("policy.afterCheck", after);
    await plane.checkAndRecord(mkCommand("read_file"));
    expect(before).toHaveBeenCalledOnce();
    expect(after).toHaveBeenCalledOnce();
  });

  it("emits onBlock and throws on blacklist", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const plane = new GovernancePlane({ hooks });
    const onBlock = vi.fn();
    hooks.on("policy.onBlock", onBlock);
    await expect(plane.checkAndRecord(mkCommand("shell-delete"))).rejects.toMatchObject({
      code: ErrorCode.POLICY_BLOCKED,
    });
    expect(onBlock).toHaveBeenCalledOnce();
    expect(plane.getStats().audit.blockedCount).toBe(1);
  });

  it("blocks when api quota exhausted (retriable)", async () => {
    const now = 1000;
    const quota = new QuotaGuard({ maxApiCallsPerHour: 1 }, { clock: () => now });
    const plane = new GovernancePlane({ quotaGuard: quota });
    await plane.checkAndRecord(mkCommand("read_file"));
    try {
      await plane.checkAndRecord(mkCommand("read_file"));
      throw new Error("expected to throw");
    } catch (err) {
      expect(err).toBeInstanceOf(AgentError);
      expect((err as AgentError).code).toBe(ErrorCode.POLICY_BLOCKED);
      expect((err as AgentError).retriable).toBe(true);
    }
  });

  it("rejects cancel attempts on policy.beforeCheck if event not cancellable for handler", async () => {
    // policy.beforeCheck IS cancellable by spec; verify cancel path actually short-circuits
    const hooks = new HookBus({ logger: silentLogger });
    const plane = new GovernancePlane({ hooks });
    let extraRan = false;
    hooks.on(
      "policy.beforeCheck",
      (ctx) => {
        ctx.cancel();
      },
      { priority: 100 },
    );
    hooks.on(
      "policy.beforeCheck",
      () => {
        extraRan = true;
      },
      { priority: 50 },
    );
    await plane.checkAndRecord(mkCommand("read_file"));
    expect(extraRan).toBe(false);
  });
});
