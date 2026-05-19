import { describe, expect, it, vi } from "vitest";
import {
  type DormantPattern,
  InternalizationManager,
  type PassiveEvent,
  PassiveStore,
  PatternMiner,
} from "../src/index.js";

const mkEvent = (
  id: string,
  text: string,
  source: PassiveEvent["source"] = "user",
): PassiveEvent => ({
  eventId: id,
  ts: Date.now(),
  source,
  text,
  metadata: {},
});

describe("PassiveStore", () => {
  it("records and queries events", () => {
    const s = new PassiveStore();
    s.record(mkEvent("e1", "hello world"));
    s.record(mkEvent("e2", "foo bar", "agent"));
    expect(s.size()).toBe(2);
    expect(s.query({ source: "user" })).toHaveLength(1);
  });

  it("respects maxSize (FIFO)", () => {
    const s = new PassiveStore(2);
    s.record(mkEvent("a", "1"));
    s.record(mkEvent("b", "2"));
    s.record(mkEvent("c", "3"));
    expect(s.size()).toBe(2);
    const ids = s.query().map((e) => e.eventId);
    expect(ids).toContain("c");
    expect(ids).not.toContain("a");
  });

  it("query.since filters by timestamp", () => {
    const s = new PassiveStore();
    s.record({ ...mkEvent("a", "x"), ts: 100 });
    s.record({ ...mkEvent("b", "y"), ts: 200 });
    expect(s.query({ since: 150 })).toHaveLength(1);
  });
});

describe("PatternMiner", () => {
  it("extracts repeated n-grams above threshold", async () => {
    const events: PassiveEvent[] = [
      mkEvent("1", "我喜欢喝绿茶 早上喝"),
      mkEvent("2", "今天又喜欢喝绿茶"),
      mkEvent("3", "总是喜欢喝绿茶"),
      mkEvent("4", "晚饭后喜欢喝绿茶"),
      mkEvent("5", "我也常喜欢喝绿茶"),
    ];
    const m = new PatternMiner({
      ngramSize: 3,
      minFrequency: 3,
      minConfidence: 0.4,
    });
    const patterns = await m.mine(events);
    expect(patterns.length).toBeGreaterThan(0);
    expect(patterns[0]!.frequency).toBeGreaterThanOrEqual(3);
  });

  it("filters out below minFrequency", async () => {
    const events: PassiveEvent[] = [mkEvent("1", "苹果 香蕉 樱桃")];
    const m = new PatternMiner({
      ngramSize: 2,
      minFrequency: 2,
      minConfidence: 0.1,
    });
    const r = await m.mine(events);
    expect(r).toEqual([]);
  });

  it("uses LLM extractor when injected", async () => {
    const llm = vi.fn(async (ngram: string) => ({
      description: `LLM 解读: ${ngram}`,
      category: "preference" as const,
    }));
    const events = new Array(5).fill(0).map((_, i) => mkEvent(`e${i}`, "用户 喜欢 绿茶 来 一杯"));
    const m = new PatternMiner({
      ngramSize: 3,
      minFrequency: 3,
      minConfidence: 0.4,
      llmExtract: llm,
    });
    const patterns = await m.mine(events);
    expect(patterns.length).toBeGreaterThan(0);
    expect(patterns[0]!.description).toContain("LLM");
    expect(patterns[0]!.category).toBe("preference");
    expect(llm).toHaveBeenCalled();
  });

  it("empty events returns []", async () => {
    const m = new PatternMiner();
    expect(await m.mine([])).toEqual([]);
  });
});

describe("InternalizationManager", () => {
  const mkPattern = (
    category: DormantPattern["category"],
    description = "test pattern",
  ): DormantPattern => ({
    patternId: `pat-${category}-${Math.random().toString(36).slice(2, 8)}`,
    description,
    evidenceIds: ["e1", "e2"],
    frequency: 5,
    confidence: 0.8,
    category,
    ts: Date.now(),
  });

  it("propose creates a pending proposal", () => {
    const im = new InternalizationManager();
    const p = im.propose(mkPattern("preference", "喜欢绿茶"));
    expect(p).toBeDefined();
    expect(p?.status).toBe("pending");
    expect(p?.targetField.startsWith("preferences.")).toBe(true);
  });

  it("approve writes to PersonaConfig", () => {
    const im = new InternalizationManager();
    const p = im.propose(mkPattern("preference", "喜欢绿茶"));
    im.approve(p!.proposalId);
    const cfg = im.snapshot();
    expect(Object.keys(cfg.preferences).length).toBeGreaterThan(0);
    expect(cfg.meta.version).toBe(1);
  });

  it("reject does not write", () => {
    const im = new InternalizationManager();
    const p = im.propose(mkPattern("phrase", "口头禅"));
    im.reject(p!.proposalId);
    const cfg = im.snapshot();
    expect(Object.keys(cfg.phrases).length).toBe(0);
    expect(im.listProposals("rejected")).toHaveLength(1);
  });

  it("approve idempotent: cannot re-approve applied", () => {
    const im = new InternalizationManager();
    const p = im.propose(mkPattern("habit", "每天 8 点写日报"));
    im.approve(p!.proposalId);
    const r = im.approve(p!.proposalId);
    expect(r).toBeUndefined();
  });

  it("category 'other' produces no proposal by default", () => {
    const im = new InternalizationManager();
    const r = im.propose(mkPattern("other"));
    expect(r).toBeUndefined();
  });

  it("custom mapToField overrides default", () => {
    const im = new InternalizationManager(undefined, {
      mapToField: (p) =>
        p.category === "other"
          ? { targetField: "context.fallback", value: p.description }
          : undefined,
    });
    const r = im.propose(mkPattern("other", "未分类信息"));
    expect(r?.targetField).toBe("context.fallback");
  });

  it("proposeBatch processes multiple patterns", () => {
    const im = new InternalizationManager();
    const r = im.proposeBatch([
      mkPattern("preference", "1"),
      mkPattern("phrase", "2"),
      mkPattern("habit", "3"),
      mkPattern("other"), // ignored
    ]);
    expect(r).toHaveLength(3);
  });

  it("snapshot returns deep copy (immutable observation)", () => {
    const im = new InternalizationManager();
    const p = im.propose(mkPattern("preference", "x"));
    im.approve(p!.proposalId);
    const s1 = im.snapshot();
    (s1.preferences as Record<string, unknown>)["mutated"] = "yes";
    const s2 = im.snapshot();
    expect((s2.preferences as Record<string, unknown>)["mutated"]).toBeUndefined();
  });
});
