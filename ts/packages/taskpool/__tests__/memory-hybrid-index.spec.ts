/**
 * A1.2 MemoryHybridIndex：session 级增量索引 + change-feed 订阅。
 */
import { HookBus, MemoryFragmentSchema } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { MemoryHybridIndex } from "../src/index.js";

const frag = (id: string, content: string, extra: Record<string, unknown> = {}) =>
  MemoryFragmentSchema.parse({
    fragmentId: id,
    content,
    embedding: [1, 0, 0],
    ...extra,
  });

describe("MemoryHybridIndex", () => {
  it("seed 后可检索，size 正确", () => {
    const idx = new MemoryHybridIndex();
    idx.seed([frag("a", "cats and dogs"), frag("b", "quantum physics")]);
    expect(idx.size).toBe(2);
    const hits = idx.search("cats", [1, 0, 0], { topK: 1 });
    expect(hits[0]?.doc.id).toBe("a");
  });

  it("订阅 change-feed：add → upsert，remove → 删除", async () => {
    const bus = new HookBus();
    const idx = new MemoryHybridIndex();
    idx.subscribe(bus);

    await bus.emit("event.MEMORY_WRITTEN", { fragment: frag("a", "hello world"), op: "add" });
    await bus.emit("event.MEMORY_WRITTEN", { fragment: frag("b", "goodbye moon"), op: "add" });
    expect(idx.size).toBe(2);

    await bus.emit("event.MEMORY_WRITTEN", { fragment: frag("a", "hello world"), op: "remove" });
    expect(idx.size).toBe(1);
    expect(idx.search("hello", [1, 0, 0]).every((h) => h.doc.id !== "a")).toBe(true);
  });

  it("update（晋升）替换文档而非新增", async () => {
    const bus = new HookBus();
    const idx = new MemoryHybridIndex();
    idx.subscribe(bus);
    await bus.emit("event.MEMORY_WRITTEN", {
      fragment: frag("a", "draft", { memoryType: "short_term" }),
      op: "add",
    });
    await bus.emit("event.MEMORY_WRITTEN", {
      fragment: frag("a", "draft", { memoryType: "long_term" }),
      op: "update",
    });
    expect(idx.size).toBe(1);
    const hit = idx.search("draft", [1, 0, 0], { topK: 1 })[0];
    expect(hit?.doc.metadata.memoryType).toBe("long_term");
  });

  it("memoryTypes / taskTags 过滤", () => {
    const idx = new MemoryHybridIndex();
    idx.seed([
      frag("a", "alpha topic", { memoryType: "long_term", taskTags: ["analysis"] }),
      frag("b", "alpha topic", { memoryType: "short_term", taskTags: ["chat"] }),
    ]);
    const onlyLong = idx.search("alpha", [1, 0, 0], { memoryTypes: ["long_term"] });
    expect(onlyLong.map((h) => h.doc.id)).toEqual(["a"]);
    const onlyChat = idx.search("alpha", [1, 0, 0], { taskTags: ["chat"] });
    expect(onlyChat.map((h) => h.doc.id)).toEqual(["b"]);
  });

  it("dispose 后不再接收 change-feed", async () => {
    const bus = new HookBus();
    const idx = new MemoryHybridIndex();
    idx.subscribe(bus);
    idx.dispose();
    await bus.emit("event.MEMORY_WRITTEN", { fragment: frag("a", "x"), op: "add" });
    expect(idx.size).toBe(0);
  });
});
