import { InMemoryMetadataStore } from "@openintj/storage-sqlite";
import { InMemoryVectorStore } from "@openintj/storage-lance";
import { describe, expect, it } from "vitest";
import {
  ContextEngine,
  InMemoryCanonicalMemoryRepository,
  MemoryPlane,
  MemoryRetriever,
  PersistentMemoryStore,
  isVerifiedAssertion,
  validityOf,
} from "../src/index.js";

const DAY = 86_400;

describe("G3 F08 生命周期与有效性 (T16/T17)", () => {
  it("T16a 稳定 long_term 事实不会仅因变旧被硬淘汰", async () => {
    const plane = new MemoryPlane();
    const old = plane.store.addLongTerm("用户的母语是中文", {
      kind: "fact",
      assertion: "user",
    });
    old.timestamp = Date.now() / 1000 - 90 * DAY;
    const hits = await plane.retrieve("母语是什么语言");
    expect(hits.map((h) => h.fragment.fragmentId)).toContain(old.fragmentId);
    // ContextEngine 全链路（retriever+shader）也不淘汰
    const engine = new ContextEngine({ store: plane.store });
    const win = await engine.build({
      query: "母语",
      history: [],
      taskType: "quick_response" as never,
      systemPrompt: "sys",
    });
    expect(win.systemPrompt).toContain("用户的母语是中文");
  });

  it("T16b 过期 intention 在所有入口不可见", async () => {
    const plane = new MemoryPlane();
    plane.store.addLongTerm("本次先讨论，不写代码", {
      kind: "intention",
      assertion: "user",
      validUntil: Date.now() / 1000 - 3600,
    });
    plane.store.addLongTerm("用户偏好先讨论再动手", { kind: "preference", assertion: "user" });
    const retriever = new MemoryRetriever(plane.store);
    const hits = retriever.retrieve("讨论");
    expect(hits.map((h) => h.fragment.content)).not.toContain("本次先讨论，不写代码");
    const engine = new ContextEngine({ store: plane.store });
    const win = await engine.build({
      query: "讨论",
      history: [],
      taskType: "quick_response" as never,
      systemPrompt: "sys",
    });
    expect(win.systemPrompt).not.toContain("本次先讨论，不写代码");
    expect(win.systemPrompt).toContain("用户偏好先讨论再动手");
  });

  it("T17a 纠正 supersede：旧值即刻不可见、新值生效、重启不回潮", async () => {
    const canonical = new InMemoryCanonicalMemoryRepository();
    await canonical.init();
    const make = async () => {
      const s = new PersistentMemoryStore({
        vectorStore: new InMemoryVectorStore(),
        metadataStore: new InMemoryMetadataStore(),
        canonical,
      });
      await s.init();
      return s;
    };
    const first = await make();
    const old = await first.addLongTermAsync("项目截止日期是周五", {
      kind: "fact",
      assertion: "user",
      canonicalKey: "project.deadline",
    });
    const replacement = await first.correctMemory(old.fragmentId, "项目截止日期改为周一", {
      kind: "fact",
      assertion: "user",
      canonicalKey: "project.deadline",
    });
    expect(first.all.map((f) => f.content)).not.toContain("项目截止日期是周五");
    expect(first.all.map((f) => f.content)).toContain("项目截止日期改为周一");
    expect(replacement.canonicalKey).toBe("project.deadline");
    const record = await canonical.getRecord(old.fragmentId);
    expect(record?.state).toBe("superseded");
    expect(record?.supersededBy).toBe(replacement.fragmentId);
    await first.close();

    const second = await make();
    expect(second.all.map((f) => f.content)).not.toContain("项目截止日期是周五");
    expect(second.all.map((f) => f.content)).toContain("项目截止日期改为周一");
    await second.close();
  });

  it("T17b 按 canonicalKey 纠正 + 断言优先级语义", async () => {
    const store = new PersistentMemoryStore({
      vectorStore: new InMemoryVectorStore(),
      metadataStore: new InMemoryMetadataStore(),
      canonical: new InMemoryCanonicalMemoryRepository(),
    });
    await store.init();
    await store.addLongTermAsync("推断：用户喜欢深色模式", {
      kind: "preference",
      assertion: "inferred",
      canonicalKey: "ui.theme",
    });
    const { superseded } = await store.correctByCanonicalKey("ui.theme", "用户明确说：用浅色模式", {
      assertion: "user",
      kind: "preference",
    });
    expect(superseded).toHaveLength(1);
    const current = store.findByCanonicalKey("ui.theme");
    expect(current).toHaveLength(1);
    expect(current[0]?.content).toContain("浅色模式");
    // 用户显式断言才可作已验证事实；推断/引用只作带标签历史陈述
    expect(isVerifiedAssertion(current[0]!)).toBe(true);
    expect(validityOf(current[0]!)).toBe("valid");
    await store.close();
  });
});
