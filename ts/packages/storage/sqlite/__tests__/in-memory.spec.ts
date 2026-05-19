import { describe, expect, it } from "vitest";
import { InMemoryMetadataStore } from "../src/index.js";

describe("InMemoryMetadataStore", () => {
  it("init + migrate works", async () => {
    const s = new InMemoryMetadataStore();
    await s.init();
    expect((await s.migrate()).to).toBe(1);
  });

  it("putFragmentMeta + getFragmentMeta", async () => {
    const s = new InMemoryMetadataStore();
    await s.init();
    await s.putFragmentMeta([
      {
        fragmentId: "f1",
        memoryType: "short_term",
        importance: 0.5,
        contentHash: "h",
        taskTagsCsv: "a,b",
        metadataJson: "{}",
        summariesJson: "{}",
        timestamp: 100,
        accessCount: 0,
        lastAccessed: 0,
      },
    ]);
    const r = await s.getFragmentMeta("f1");
    expect(r?.fragmentId).toBe("f1");
    expect(r?.taskTagsCsv).toBe("a,b");
  });

  it("listFragmentMeta filters by memoryType + limit", async () => {
    const s = new InMemoryMetadataStore();
    await s.init();
    await s.putFragmentMeta([
      {
        fragmentId: "a",
        memoryType: "short_term",
        importance: 0.5,
        contentHash: "h",
        taskTagsCsv: "",
        metadataJson: "{}",
        summariesJson: "{}",
        timestamp: 100,
        accessCount: 0,
        lastAccessed: 0,
      },
      {
        fragmentId: "b",
        memoryType: "long_term",
        importance: 0.5,
        contentHash: "h",
        taskTagsCsv: "",
        metadataJson: "{}",
        summariesJson: "{}",
        timestamp: 200,
        accessCount: 0,
        lastAccessed: 0,
      },
    ]);
    const list = await s.listFragmentMeta({ memoryType: "long_term" });
    expect(list).toHaveLength(1);
    expect(list[0]!.fragmentId).toBe("b");
  });

  it("recordAudit + queryAudit", async () => {
    const s = new InMemoryMetadataStore();
    await s.init();
    await s.recordAudit({
      eventId: "e1",
      eventType: "policy.checked",
      command: "read_file",
      riskLevel: "low",
      approved: 1,
      reason: null,
      metadataJson: "{}",
      timestamp: 100,
    });
    await s.recordAudit({
      eventId: "e2",
      eventType: "policy.violated",
      command: "exec",
      riskLevel: "high",
      approved: 0,
      reason: "denied",
      metadataJson: "{}",
      timestamp: 200,
    });
    const all = await s.queryAudit();
    expect(all).toHaveLength(2);
    const violated = await s.queryAudit({ eventType: "policy.violated" });
    expect(violated).toHaveLength(1);
    const recent = await s.queryAudit({ since: 150 });
    expect(recent).toHaveLength(1);
    expect(recent[0]!.eventId).toBe("e2");
  });

  it("pruneAudit removes old events", async () => {
    const s = new InMemoryMetadataStore();
    await s.init();
    for (let i = 0; i < 5; i++) {
      await s.recordAudit({
        eventId: `e${i}`,
        eventType: "x",
        command: null,
        riskLevel: null,
        approved: null,
        reason: null,
        metadataJson: "{}",
        timestamp: i * 100,
      });
    }
    expect(await s.pruneAudit(250)).toBe(3);
    const remain = await s.queryAudit();
    expect(remain).toHaveLength(2);
  });

  it("session crud", async () => {
    const s = new InMemoryMetadataStore();
    await s.init();
    await s.putSession({
      sessionId: "s1",
      startedAt: 100,
      lastActiveAt: 100,
      metadataJson: "{}",
    });
    expect((await s.getSession("s1"))?.startedAt).toBe(100);
    await s.touchSession("s1", 200);
    expect((await s.getSession("s1"))?.lastActiveAt).toBe(200);
  });

  it("deleteFragmentMeta returns deleted count", async () => {
    const s = new InMemoryMetadataStore();
    await s.init();
    await s.putFragmentMeta([
      {
        fragmentId: "a",
        memoryType: "short_term",
        importance: 0.5,
        contentHash: "h",
        taskTagsCsv: "",
        metadataJson: "{}",
        summariesJson: "{}",
        timestamp: 0,
        accessCount: 0,
        lastAccessed: 0,
      },
    ]);
    expect(await s.deleteFragmentMeta(["a", "missing"])).toBe(1);
    expect(await s.getFragmentMeta("a")).toBeUndefined();
  });
});
