/**
 * Dormant 持久化接口单测（Phase 3.4 #9）。
 *
 * 覆盖：
 *  1. InMemoryDormantStore CRUD 与 loadAll 行为
 *  2. DormantRuntime hydrate() 把历史 events / proposals / persona 恢复回内存
 *  3. record / mine / approve / reject / reset / close 五条热路径都写穿 adapter
 *  4. 多次 hydrate 安全（全量覆写）
 */
import { describe, expect, it } from "vitest";
import {
  DormantRuntime,
  InMemoryDormantStore,
  type InternalizationProposal,
  type PassiveEvent,
  type PersonaConfig,
} from "../src/index.js";

describe("InMemoryDormantStore", () => {
  it("空仓 loadAll 返回空数组 + persona undefined", async () => {
    const s = new InMemoryDormantStore();
    const snap = await s.loadAll();
    expect(snap.events).toEqual([]);
    expect(snap.proposals).toEqual([]);
    expect(snap.persona).toBeUndefined();
  });

  it("recordEvent + loadAll 往返", async () => {
    const s = new InMemoryDormantStore();
    const e: PassiveEvent = {
      eventId: "e1",
      ts: 1000,
      source: "user",
      text: "hi",
      metadata: { foo: "bar" },
    };
    s.recordEvent(e);
    const snap = await s.loadAll();
    expect(snap.events).toHaveLength(1);
    expect(snap.events[0]!.eventId).toBe("e1");
    expect(snap.events[0]!.metadata).toEqual({ foo: "bar" });
  });

  it("upsertProposal 按 proposalId 覆盖", async () => {
    const s = new InMemoryDormantStore();
    const p: InternalizationProposal = {
      proposalId: "p1",
      pattern: {
        patternId: "pat1",
        description: "x",
        evidenceIds: ["e1"],
        frequency: 2,
        confidence: 0.5,
        category: "preference",
        ts: 100,
      },
      targetField: "preferences.x",
      value: "y",
      status: "pending",
      ts: 200,
    };
    s.upsertProposal(p);
    s.upsertProposal({ ...p, status: "applied", decidedAt: 300 });
    const snap = await s.loadAll();
    expect(snap.proposals).toHaveLength(1);
    expect(snap.proposals[0]!.status).toBe("applied");
    expect(snap.proposals[0]!.decidedAt).toBe(300);
  });

  it("savePersona 覆写 + clearAll 清空", async () => {
    const s = new InMemoryDormantStore();
    const persona: PersonaConfig = {
      preferences: { tea: "green" },
      phrases: {},
      habits: {},
      context: {},
      meta: { lastUpdated: 100, version: 1 },
    };
    s.savePersona(persona);
    let snap = await s.loadAll();
    expect(snap.persona?.preferences).toEqual({ tea: "green" });

    s.clearAll();
    snap = await s.loadAll();
    expect(snap.persona).toBeUndefined();
    expect(snap.events).toEqual([]);
    expect(snap.proposals).toEqual([]);
  });

  it("loadAll 返回独立副本（修改不影响内部状态）", async () => {
    const s = new InMemoryDormantStore();
    s.recordEvent({
      eventId: "e1",
      ts: 1,
      source: "user",
      text: "x",
      metadata: { k: "v" },
    });
    const snap1 = await s.loadAll();
    (snap1.events[0]!.metadata as Record<string, unknown>)["k"] = "MUTATED";
    const snap2 = await s.loadAll();
    expect(snap2.events[0]!.metadata).toEqual({ k: "v" });
  });

  it("pruneEvents 按时间删除旧事件", async () => {
    const s = new InMemoryDormantStore();
    for (const ts of [100, 200, 300, 400]) {
      s.recordEvent({ eventId: `e${ts}`, ts, source: "user", text: "x", metadata: {} });
    }
    const removed = s.pruneEvents(300);
    expect(removed).toBe(2);
    const snap = await s.loadAll();
    expect(snap.events.map((e) => e.eventId)).toEqual(["e300", "e400"]);
  });

  it("pruneEventsToMax 仅保留最新 N 条（按 ts）", async () => {
    const s = new InMemoryDormantStore();
    for (const ts of [100, 400, 200, 300]) {
      s.recordEvent({ eventId: `e${ts}`, ts, source: "user", text: "x", metadata: {} });
    }
    const removed = s.pruneEventsToMax(2);
    expect(removed).toBe(2);
    const snap = await s.loadAll();
    expect(snap.events.map((e) => e.ts).sort((a, b) => a - b)).toEqual([300, 400]);
  });

  it("pruneEventsToMax 容量足够时不删", () => {
    const s = new InMemoryDormantStore();
    s.recordEvent({ eventId: "e1", ts: 1, source: "user", text: "x", metadata: {} });
    expect(s.pruneEventsToMax(5)).toBe(0);
  });
});

describe("DormantRuntime + adapter（hydrate / write-through）", () => {
  it("无 adapter 时 hydrate 是 no-op", async () => {
    const rt = new DormantRuntime();
    await expect(rt.hydrate()).resolves.toBeUndefined();
    expect(rt.passiveSize()).toBe(0);
  });

  it("hydrate 把历史 events / proposals / persona 拉回内存", async () => {
    const adapter = new InMemoryDormantStore();
    adapter.recordEvent({
      eventId: "old_1",
      ts: 1000,
      source: "user",
      text: "历史输入 1",
      metadata: {},
    });
    adapter.recordEvent({
      eventId: "old_2",
      ts: 2000,
      source: "agent",
      text: "历史回复 1",
      metadata: {},
    });
    adapter.upsertProposal({
      proposalId: "prop_1",
      pattern: {
        patternId: "pat_1",
        description: "老 pattern",
        evidenceIds: ["old_1"],
        frequency: 3,
        confidence: 0.6,
        category: "preference",
        ts: 1500,
      },
      targetField: "preferences.x",
      value: "y",
      status: "applied",
      ts: 1600,
      decidedAt: 1700,
    });
    adapter.savePersona({
      preferences: { x: "y" },
      phrases: {},
      habits: {},
      context: {},
      meta: { lastUpdated: 1700, version: 1 },
    });

    const rt = new DormantRuntime({ adapter });
    await rt.hydrate();

    expect(rt.passiveSize()).toBe(2);
    expect(rt.listProposals().map((p) => p.proposalId)).toEqual(["prop_1"]);
    expect(rt.listProposals("applied")).toHaveLength(1);
    const persona = rt.snapshot();
    expect(persona.preferences).toEqual({ x: "y" });
    expect(persona.meta.version).toBe(1);
  });

  it("record / mine / approve / reject / reset / close 全部写穿 adapter", async () => {
    const adapter = new InMemoryDormantStore();
    const rt = new DormantRuntime({
      adapter,
      minerOpts: {
        ngramSize: 2,
        minFrequency: 2,
        minConfidence: 0.2,
        llmExtract: async (ng) => ({ description: ng, category: "preference" }),
      },
    });
    await rt.hydrate();

    rt.record("绿 茶", "user");
    rt.record("绿 茶", "user");
    rt.record("绿 茶", "user");
    let snap = await adapter.loadAll();
    expect(snap.events).toHaveLength(3);

    const mineRes = await rt.mine();
    expect(mineRes.proposals.length).toBeGreaterThan(0);
    snap = await adapter.loadAll();
    expect(snap.proposals.length).toBe(mineRes.proposals.length);

    const first = mineRes.proposals[0]!;
    rt.approve(first.proposalId);
    snap = await adapter.loadAll();
    expect(snap.proposals.find((p) => p.proposalId === first.proposalId)?.status).toBe("applied");
    expect(snap.persona?.meta.version).toBe(1);

    if (mineRes.proposals.length > 1) {
      const second = mineRes.proposals[1]!;
      rt.reject(second.proposalId);
      snap = await adapter.loadAll();
      expect(snap.proposals.find((p) => p.proposalId === second.proposalId)?.status).toBe(
        "rejected",
      );
    }

    rt.reset();
    snap = await adapter.loadAll();
    expect(snap.events).toEqual([]);
    expect(snap.proposals).toEqual([]);

    await rt.close();
  });

  it("pruneEvents / pruneEventsToMax 同时清理内存与 adapter", async () => {
    const adapter = new InMemoryDormantStore();
    const rt = new DormantRuntime({ adapter });
    await rt.hydrate();
    for (const ts of [1, 2, 3, 4, 5]) {
      adapter.recordEvent({ eventId: `e${ts}`, ts, source: "user", text: "x", metadata: {} });
    }
    await rt.hydrate();
    expect(rt.passiveSize()).toBe(5);

    const removedByTime = rt.pruneEvents(3);
    expect(removedByTime).toBe(2);
    expect(rt.passiveSize()).toBe(3);
    expect((await adapter.loadAll()).events).toHaveLength(3);

    const removedByMax = rt.pruneEventsToMax(1);
    expect(removedByMax).toBe(2);
    expect(rt.passiveSize()).toBe(1);
    expect((await adapter.loadAll()).events).toHaveLength(1);
  });

  it("mine() 末尾按 eventRetentionMs / maxDiskEvents 自动清理", async () => {
    const adapter = new InMemoryDormantStore();
    const rt = new DormantRuntime({ adapter, maxDiskEvents: 2 });
    await rt.hydrate();
    rt.record("a", "user");
    rt.record("b", "user");
    rt.record("c", "user");
    expect(rt.passiveSize()).toBe(3);
    await rt.mine();
    expect(rt.passiveSize()).toBe(2);
    expect((await adapter.loadAll()).events).toHaveLength(2);
  });

  it("hydrate 可以多次调用安全（每次全量覆写）", async () => {
    const adapter = new InMemoryDormantStore();
    adapter.recordEvent({
      eventId: "a",
      ts: 1,
      source: "user",
      text: "1",
      metadata: {},
    });
    const rt = new DormantRuntime({ adapter });
    await rt.hydrate();
    expect(rt.passiveSize()).toBe(1);

    adapter.recordEvent({
      eventId: "b",
      ts: 2,
      source: "user",
      text: "2",
      metadata: {},
    });
    await rt.hydrate();
    expect(rt.passiveSize()).toBe(2);

    adapter.clearAll();
    await rt.hydrate();
    expect(rt.passiveSize()).toBe(0);
  });
});
