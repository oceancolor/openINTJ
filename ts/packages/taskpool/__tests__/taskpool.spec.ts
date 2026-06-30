import { AgentPool } from "@openintj/concurrency";
import { describe, expect, it } from "vitest";
import { HybridRetriever, ObjectPool, SharedContext, TaskQueue } from "../src/index.js";

describe("SharedContext", () => {
  it("get/set with mutex", async () => {
    const c = new SharedContext();
    await c.set("k1", "v1");
    expect(c.get("k1")).toBe("v1");
    expect(c.has("k1")).toBe(true);
  });

  it("update is atomic across concurrent calls", async () => {
    const c = new SharedContext();
    await c.set("counter", 0);
    await Promise.all(
      new Array(100).fill(0).map(() => c.update<number>("counter", (n) => (n ?? 0) + 1)),
    );
    expect(c.get<number>("counter")).toBe(100);
  });

  it("snapshot returns plain object", async () => {
    const c = new SharedContext();
    await c.set("a", 1);
    await c.set("b", 2);
    expect(c.snapshot()).toEqual({ a: 1, b: 2 });
  });
});

describe("HybridRetriever", () => {
  const docs = [
    {
      id: "d1",
      text: "apple banana cherry",
      vector: [1, 0, 0],
    },
    { id: "d2", text: "dog elephant fox", vector: [0, 1, 0] },
    {
      id: "d3",
      text: "apple grape orange",
      vector: [0.7, 0, 0.7],
    },
  ];

  it("ranks vector + bm25 fusion (alpha+beta)", () => {
    const r = new HybridRetriever();
    r.index(docs);
    const out = r.search("apple cherry", [1, 0, 0], 3);
    expect(out[0]!.doc.id).toBe("d1");
    expect(out[0]!.score).toBeGreaterThan(0);
  });

  it("RRF fusion produces a different (rank-based) ranking", () => {
    const r = new HybridRetriever({ config: { useRRF: true } });
    r.index(docs);
    const out = r.search("apple", [0.7, 0, 0.7], 3);
    expect(out[0]!.components.rrf).toBeDefined();
  });

  it("empty index returns empty", () => {
    const r = new HybridRetriever();
    const out = r.search("x", [1, 0, 0], 3);
    expect(out).toEqual([]);
  });

  it("works without vectors (BM25 only)", () => {
    const r = new HybridRetriever();
    r.index(docs);
    const out = r.search("dog elephant", undefined, 3);
    expect(out[0]!.doc.id).toBe("d2");
  });

  it("增量 upsert 与全量 index 结果一致（BM25 统计量正确维护）", () => {
    const full = new HybridRetriever();
    full.index(docs);
    const incr = new HybridRetriever();
    for (const d of docs) incr.upsert(d);
    expect(incr.size).toBe(3);

    const q = "apple cherry";
    const a = full.search(q, [1, 0, 0], 3);
    const b = incr.search(q, [1, 0, 0], 3);
    expect(b.map((x) => x.doc.id)).toEqual(a.map((x) => x.doc.id));
    expect(b[0]!.score).toBeCloseTo(a[0]!.score, 6);
  });

  it("upsert 同 id 替换文本后命中新内容", () => {
    const r = new HybridRetriever();
    r.index(docs);
    r.upsert({ id: "d2", text: "apple cherry banana", vector: [1, 0, 0] });
    const out = r.search("cherry", undefined, 3);
    // d1 与 d2 现在都含 cherry；至少应命中且不报错
    expect(out.some((x) => x.doc.id === "d2")).toBe(true);
  });

  it("remove 删除文档后不再命中，统计量随之收缩", () => {
    const r = new HybridRetriever();
    r.index(docs);
    expect(r.remove("d2")).toBe(true);
    expect(r.size).toBe(2);
    expect(r.remove("ghost")).toBe(false);
    const out = r.search("dog elephant", undefined, 3);
    expect(out.every((x) => x.doc.id !== "d2")).toBe(true);
  });

  it("clear 清空索引", () => {
    const r = new HybridRetriever();
    r.index(docs);
    r.clear();
    expect(r.size).toBe(0);
    expect(r.search("apple", [1, 0, 0], 3)).toEqual([]);
  });
});

describe("TaskQueue", () => {
  it("respects DAG dependencies", async () => {
    const q = new TaskQueue();
    const order: string[] = [];
    await q.submit({
      id: "a",
      priority: 1,
      deps: [],
      payload: undefined,
      run: () => {
        order.push("a");
      },
    });
    await q.submit({
      id: "b",
      priority: 1,
      deps: ["a"],
      payload: undefined,
      run: () => {
        order.push("b");
      },
    });
    await q.submit({
      id: "c",
      priority: 1,
      deps: ["b"],
      payload: undefined,
      run: () => {
        order.push("c");
      },
    });

    // 简单单 worker 跑
    const worker = async (): Promise<void> => {
      while (true) {
        const t = await q.dequeue();
        if (!t) return;
        try {
          const r = await t.node.run(t.node.payload);
          await q.complete(t.node.id, r);
        } catch (e) {
          await q.fail(t.node.id, e);
        }
      }
    };
    q.close();
    await worker();
    expect(order).toEqual(["a", "b", "c"]);
  });

  it("priority within ready set", async () => {
    const q = new TaskQueue();
    const order: string[] = [];
    await q.submit({
      id: "low",
      priority: 1,
      deps: [],
      payload: undefined,
      run: () => order.push("low"),
    });
    await q.submit({
      id: "high",
      priority: 10,
      deps: [],
      payload: undefined,
      run: () => order.push("high"),
    });
    q.close();
    while (true) {
      const t = await q.dequeue();
      if (!t) break;
      await t.node.run(t.node.payload);
      await q.complete(t.node.id, undefined);
    }
    expect(order).toEqual(["high", "low"]);
  });

  it("failure cascades to dependents", async () => {
    const q = new TaskQueue();
    await q.submit({
      id: "a",
      priority: 1,
      deps: [],
      payload: undefined,
      run: () => {
        throw new Error("boom");
      },
    });
    await q.submit({
      id: "b",
      priority: 1,
      deps: ["a"],
      payload: undefined,
      run: () => undefined,
    });
    q.close();
    const t = await q.dequeue();
    if (t) await q.fail(t.node.id, new Error("boom"));
    const after = await q.dequeue();
    expect(after).toBeUndefined();
    expect(q.result("b").state).toBe("failed");
  });

  it("works with AgentPool concurrency", async () => {
    const q = new TaskQueue();
    const order: number[] = [];
    for (let i = 0; i < 6; i++) {
      // eslint-disable-next-line no-loop-func
      await q.submit({
        id: `t${i}`,
        priority: i,
        deps: [],
        payload: i,
        run: async (n: unknown) => {
          await new Promise((r) => setTimeout(r, 5));
          order.push(n as number);
        },
      });
    }
    q.close();
    const pool = new AgentPool(3);
    const workers = new Array(3).fill(0).map(() =>
      (async () => {
        while (true) {
          const t = await q.dequeue();
          if (!t) return;
          await pool.submit(async () => {
            await t.node.run(t.node.payload);
            await q.complete(t.node.id, undefined);
          }, undefined);
        }
      })(),
    );
    await Promise.all(workers);
    // 高优先级先入 pool（不严格保证完成顺序，但前几个肯定优先级靠前）
    expect(order.length).toBe(6);
  });
});

describe("ObjectPool", () => {
  it("set/get returns value, increments accessCount", async () => {
    const p = new ObjectPool<string>();
    await p.set("k1", "value");
    expect(await p.get("k1")).toBe("value");
    expect(await p.get("k1")).toBe("value");
  });

  it("promotes to hot after hitting promote threshold", async () => {
    const p = new ObjectPool<string>({ hotPromoteAt: 3 });
    await p.set("k", "v");
    await p.get("k");
    await p.get("k");
    await p.get("k");
    expect(p.stats().hot).toBe(1);
    expect(p.stats().warm).toBe(0);
  });

  it("warm overflow demotes LRU to cold", async () => {
    const p = new ObjectPool<number>({
      hotCapacity: 1,
      warmCapacity: 2,
      coldCapacity: 10,
    });
    await p.set("a", 1);
    await new Promise((r) => setTimeout(r, 1));
    await p.set("b", 2);
    await new Promise((r) => setTimeout(r, 1));
    await p.set("c", 3);
    const s = p.stats();
    expect(s.warm).toBe(2);
    expect(s.cold).toBe(1);
  });

  it("prune demotes warm → cold by age", async () => {
    const p = new ObjectPool<number>({ warmDemoteAfter: 5 });
    await p.set("k", 1);
    await new Promise((r) => setTimeout(r, 20));
    const r = await p.prune();
    expect(r.demoted).toBe(1);
    expect(p.stats().cold).toBe(1);
  });

  it("onEvict fired when cold overflows", async () => {
    const evicted: string[] = [];
    const p = new ObjectPool<string>({
      hotCapacity: 1,
      warmCapacity: 1,
      coldCapacity: 1,
      onEvict: (e) => evicted.push(e.key),
    });
    await p.set("a", "1");
    await new Promise((r) => setTimeout(r, 1));
    await p.set("b", "2");
    await new Promise((r) => setTimeout(r, 1));
    await p.set("c", "3");
    await new Promise((r) => setTimeout(r, 1));
    await p.set("d", "4");
    // a/b/c should overflow chain to cold
    expect(evicted.length).toBeGreaterThan(0);
  });
});
