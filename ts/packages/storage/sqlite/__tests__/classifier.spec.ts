/**
 * SqliteClassifierStore 单测：:memory: 模式；better-sqlite3 未装则整段 skip。
 *
 * 覆盖：init+migrate、save→load 往返、覆盖式 save、clear、关闭后 load 抛错。
 */
import type { ClassifierState } from "@openintj/classifier";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SqliteClassifierStore } from "../src/classifier.js";

const HAS_PEER = await (async () => {
  try {
    await import("better-sqlite3");
    return true;
  } catch {
    return false;
  }
})();

const describeIfPeer = HAS_PEER ? describe : describe.skip;

const state = (n: number): ClassifierState => ({
  exemplars: Array.from({ length: n }, (_, i) => ({
    vector: [i, i + 1, i + 2],
    label: i % 2 === 0 ? "planning" : "analysis",
    weight: 1 + i,
    lastUsed: 1000 + i,
  })),
});

describeIfPeer("SqliteClassifierStore", () => {
  let store: SqliteClassifierStore;

  beforeEach(async () => {
    store = new SqliteClassifierStore({ dbPath: ":memory:" });
    await store.init();
  });

  afterEach(async () => {
    await store.close();
  });

  it("空库 load 返回 undefined", async () => {
    expect(await store.load()).toBeUndefined();
  });

  it("save → load 往返保真", async () => {
    store.save(state(3));
    const loaded = await store.load();
    expect(loaded?.exemplars).toHaveLength(3);
    expect(loaded?.exemplars[0]?.vector).toEqual([0, 1, 2]);
    expect(loaded?.exemplars[1]?.label).toBe("analysis");
    expect(loaded?.exemplars[2]?.weight).toBe(3);
  });

  it("save 覆盖式：第二次 save 全量替换", async () => {
    store.save(state(5));
    store.save(state(2));
    const loaded = await store.load();
    expect(loaded?.exemplars).toHaveLength(2);
  });

  it("clear 清空", async () => {
    store.save(state(3));
    store.clear();
    expect(await store.load()).toBeUndefined();
  });

  it("关闭后 load 抛错", async () => {
    await store.close();
    await expect(store.load()).rejects.toThrow();
  });
});
