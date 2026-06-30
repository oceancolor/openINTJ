/**
 * CLF.2 ClassifierStore：内存默认 + 分类器 hydrate/auto-save 接线。
 */
import { SimpleEmbedder, TaskType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { InMemoryClassifierStore, ReinforcingClassifier } from "../src/index.js";

describe("InMemoryClassifierStore + classifier persistence", () => {
  it("reinforce 后自动落盘，hydrate 跨实例恢复", async () => {
    const store = new InMemoryClassifierStore();
    const a = new ReinforcingClassifier({ embedder: new SimpleEmbedder(64), store });
    await a.reinforce("制定上线计划", TaskType.PLANNING, { signal: 1 });
    await a.reinforce("制定回滚计划", TaskType.PLANNING, { signal: 1 });
    expect(a.size).toBeGreaterThan(0);

    // 新实例从同一 store hydrate
    const b = new ReinforcingClassifier({ embedder: new SimpleEmbedder(64), store });
    expect(b.size).toBe(0);
    await b.hydrate();
    expect(b.size).toBe(a.size);
  });

  it("无 store 时 hydrate/persist 是 no-op，不抛错", async () => {
    const c = new ReinforcingClassifier({ embedder: new SimpleEmbedder(64) });
    await c.hydrate();
    await c.reinforce("hi", TaskType.GENERAL_CHAT, { signal: 1 });
    expect(c.size).toBe(1);
  });

  it("store.clear 后 hydrate 不恢复", async () => {
    const store = new InMemoryClassifierStore();
    const a = new ReinforcingClassifier({ embedder: new SimpleEmbedder(64), store });
    await a.reinforce("x", TaskType.ANALYSIS, { signal: 1 });
    store.clear();
    const b = new ReinforcingClassifier({ embedder: new SimpleEmbedder(64), store });
    await b.hydrate();
    expect(b.size).toBe(0);
  });
});
