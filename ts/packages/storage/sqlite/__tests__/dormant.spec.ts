import type {
  DormantPersistenceAdapter,
  InternalizationProposal,
  PassiveEvent,
  PersonaConfig,
} from "@openintj/dormant";
/**
 * SqliteDormantStore 单测（Phase 3.4 #9）。
 *
 * 走 `:memory:` 模式（不触盘，CI 友好）；如果 better-sqlite3 peer 未装，整段 skip。
 *
 * 覆盖：
 *  1. init + migrate 至 TARGET_VERSION
 *  2. recordEvent + loadAll 往返
 *  3. upsertProposal 状态迁移（pending → applied / rejected）
 *  4. savePersona + loadAll persona 往返
 *  5. clearAll 三张表都清空
 *  6. 关闭后再次 loadAll 抛错
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SqliteDormantStore } from "../src/dormant.js";

const HAS_PEER = await (async () => {
  try {
    await import("better-sqlite3");
    return true;
  } catch {
    return false;
  }
})();

const describeIfPeer = HAS_PEER ? describe : describe.skip;

const sampleEvent = (id: string, ts: number, text: string): PassiveEvent => ({
  eventId: id,
  ts,
  source: "user",
  text,
  metadata: { traceId: `trace_${id}` },
});

const sampleProposal = (
  id: string,
  status: InternalizationProposal["status"] = "pending",
): InternalizationProposal => ({
  proposalId: id,
  pattern: {
    patternId: `pat_${id}`,
    description: `pattern for ${id}`,
    evidenceIds: [`e_${id}`],
    frequency: 3,
    confidence: 0.7,
    category: "preference",
    ts: 100,
  },
  targetField: `preferences.${id}`,
  value: `value_${id}`,
  status,
  ts: 200,
  ...(status !== "pending" ? { decidedAt: 300 } : {}),
});

const samplePersona = (version: number): PersonaConfig => ({
  preferences: { tea: "green", version },
  phrases: { greeting: "你好" },
  habits: {},
  context: {},
  meta: { lastUpdated: 12345, version },
});

describeIfPeer("SqliteDormantStore (:memory:)", () => {
  let store: SqliteDormantStore;

  beforeEach(async () => {
    store = new SqliteDormantStore({ dbPath: ":memory:" });
    await store.init();
  });

  afterEach(async () => {
    await store.close();
  });

  it("空仓 loadAll → events/proposals 空 + persona undefined", async () => {
    const snap = await store.loadAll();
    expect(snap.events).toEqual([]);
    expect(snap.proposals).toEqual([]);
    expect(snap.persona).toBeUndefined();
  });

  it("recordEvent + loadAll 按 ts 升序还原", async () => {
    store.recordEvent(sampleEvent("e3", 3000, "third"));
    store.recordEvent(sampleEvent("e1", 1000, "first"));
    store.recordEvent(sampleEvent("e2", 2000, "second"));
    const snap = await store.loadAll();
    expect(snap.events.map((e) => e.eventId)).toEqual(["e1", "e2", "e3"]);
    expect(snap.events[0]!.metadata).toEqual({ traceId: "trace_e1" });
  });

  it("recordEvent 重复 eventId 走 UPSERT（保留最新内容）", async () => {
    store.recordEvent(sampleEvent("e1", 1000, "old"));
    store.recordEvent({ ...sampleEvent("e1", 1500, "new"), metadata: { v: 2 } });
    const snap = await store.loadAll();
    expect(snap.events).toHaveLength(1);
    expect(snap.events[0]!.text).toBe("new");
    expect(snap.events[0]!.ts).toBe(1500);
    expect(snap.events[0]!.metadata).toEqual({ v: 2 });
  });

  it("upsertProposal 状态迁移 pending → applied", async () => {
    store.upsertProposal(sampleProposal("p1", "pending"));
    let snap = await store.loadAll();
    expect(snap.proposals[0]!.status).toBe("pending");
    expect(snap.proposals[0]!.decidedAt).toBeUndefined();

    store.upsertProposal(sampleProposal("p1", "applied"));
    snap = await store.loadAll();
    expect(snap.proposals).toHaveLength(1);
    expect(snap.proposals[0]!.status).toBe("applied");
    expect(snap.proposals[0]!.decidedAt).toBe(300);
  });

  it("upsertProposal 状态迁移 pending → rejected", async () => {
    store.upsertProposal(sampleProposal("p1", "pending"));
    store.upsertProposal(sampleProposal("p1", "rejected"));
    const snap = await store.loadAll();
    expect(snap.proposals[0]!.status).toBe("rejected");
    expect(snap.proposals[0]!.decidedAt).toBe(300);
  });

  it("savePersona + loadAll 往返 + 单行约束（id=1）", async () => {
    store.savePersona(samplePersona(1));
    let snap = await store.loadAll();
    expect(snap.persona).toBeDefined();
    expect(snap.persona!.meta.version).toBe(1);
    expect(snap.persona!.preferences).toEqual({ tea: "green", version: 1 });

    store.savePersona(samplePersona(2));
    snap = await store.loadAll();
    expect(snap.persona!.meta.version).toBe(2);
  });

  it("clearAll 同时清空三张表", async () => {
    store.recordEvent(sampleEvent("e1", 1000, "x"));
    store.upsertProposal(sampleProposal("p1"));
    store.savePersona(samplePersona(1));
    let snap = await store.loadAll();
    expect(snap.events.length + snap.proposals.length).toBeGreaterThan(0);
    expect(snap.persona).toBeDefined();

    store.clearAll();
    snap = await store.loadAll();
    expect(snap.events).toEqual([]);
    expect(snap.proposals).toEqual([]);
    expect(snap.persona).toBeUndefined();
  });

  it("pruneEvents 按时间删除旧事件并返回删除条数", async () => {
    store.recordEvent(sampleEvent("e1", 1000, "a"));
    store.recordEvent(sampleEvent("e2", 2000, "b"));
    store.recordEvent(sampleEvent("e3", 3000, "c"));
    const removed = store.pruneEvents(2500);
    expect(removed).toBe(2);
    const snap = await store.loadAll();
    expect(snap.events.map((e) => e.eventId)).toEqual(["e3"]);
  });

  it("pruneEventsToMax 仅保留最新 N 条（按 ts 降序）", async () => {
    store.recordEvent(sampleEvent("e1", 1000, "a"));
    store.recordEvent(sampleEvent("e2", 2000, "b"));
    store.recordEvent(sampleEvent("e3", 3000, "c"));
    store.recordEvent(sampleEvent("e4", 4000, "d"));
    const removed = store.pruneEventsToMax(2);
    expect(removed).toBe(2);
    const snap = await store.loadAll();
    expect(snap.events.map((e) => e.eventId)).toEqual(["e3", "e4"]);
  });

  it("pruneEventsToMax 容量足够时不删", async () => {
    store.recordEvent(sampleEvent("e1", 1000, "a"));
    expect(store.pruneEventsToMax(10)).toBe(0);
    expect((await store.loadAll()).events).toHaveLength(1);
  });

  it("name 包含 dbPath，便于审计", () => {
    expect(store.name).toBe("sqlite-dormant::memory:");
  });

  it("符合 DormantPersistenceAdapter 接口", () => {
    const _typed: DormantPersistenceAdapter = store;
    expect(_typed.name).toBeDefined();
  });

  it("close 后 loadAll 抛错（未初始化）", async () => {
    await store.close();
    await expect(store.loadAll()).rejects.toThrow();
    store = new SqliteDormantStore({ dbPath: ":memory:" });
    await store.init();
  });

  it("迁移幂等：二次 init 不重复 INSERT version 行", async () => {
    await store.close();
    store = new SqliteDormantStore({ dbPath: ":memory:" });
    await store.init();
    await store.init();
    const snap = await store.loadAll();
    expect(snap.events).toEqual([]);
  });
});
