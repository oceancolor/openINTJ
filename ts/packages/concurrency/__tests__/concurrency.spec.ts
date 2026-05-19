import { describe, expect, it } from "vitest";
import {
  AgentPool,
  BackpressureGate,
  Channel,
  ConditionVariable,
  Mutex,
  Semaphore,
  TokenBucket,
  forkJoin,
  majorityVote,
} from "../src/index.js";

describe("Mutex", () => {
  it("serializes critical sections", async () => {
    const m = new Mutex();
    const order: number[] = [];
    const tasks = [1, 2, 3].map((n) =>
      m.runExclusive(async () => {
        order.push(n);
        await new Promise((r) => setTimeout(r, 5));
        order.push(n + 10);
      }),
    );
    await Promise.all(tasks);
    expect(order).toEqual([1, 11, 2, 12, 3, 13]);
  });

  it("isLocked tracks state", async () => {
    const m = new Mutex();
    expect(m.isLocked).toBe(false);
    const release = await m.acquire();
    expect(m.isLocked).toBe(true);
    release();
    expect(m.isLocked).toBe(false);
  });
});

describe("Semaphore", () => {
  it("limits concurrency to N", async () => {
    const s = new Semaphore(2);
    let inFlight = 0;
    let maxObserved = 0;
    const work = (): Promise<void> =>
      s.runExclusive(async () => {
        inFlight++;
        maxObserved = Math.max(maxObserved, inFlight);
        await new Promise((r) => setTimeout(r, 10));
        inFlight--;
      });
    await Promise.all([work(), work(), work(), work(), work()]);
    expect(maxObserved).toBe(2);
  });
});

describe("Channel", () => {
  it("send/recv unbuffered handshake", async () => {
    const ch = new Channel<number>();
    const recv = ch.recv();
    await ch.send(42);
    const m = await recv;
    expect(m.value).toBe(42);
    expect(m.done).toBe(false);
  });

  it("buffered channel allows N pending sends", async () => {
    const ch = new Channel<number>(3);
    await ch.send(1);
    await ch.send(2);
    await ch.send(3);
    expect(ch.pendingCount).toBe(3);
    const r1 = await ch.recv();
    expect(r1.value).toBe(1);
  });

  it("close: pending recv resolves with done=true", async () => {
    const ch = new Channel<number>();
    const r = ch.recv();
    ch.close();
    const m = await r;
    expect(m.done).toBe(true);
  });

  it("close: send on closed throws", async () => {
    const ch = new Channel<number>();
    ch.close();
    await expect(ch.send(1)).rejects.toThrow(/closed/);
  });

  it("for-await-of consumes until closed", async () => {
    const ch = new Channel<number>(5);
    for (let i = 0; i < 3; i++) await ch.send(i);
    ch.close();
    const out: number[] = [];
    for await (const v of ch) out.push(v);
    expect(out).toEqual([0, 1, 2]);
  });
});

describe("ConditionVariable", () => {
  it("notify wakes a waiter", async () => {
    const cv = new ConditionVariable();
    const w = cv.wait();
    expect(cv.pendingWaiters).toBe(1);
    cv.notify();
    await expect(w).resolves.toBeUndefined();
  });

  it("notifyAll wakes all", async () => {
    const cv = new ConditionVariable();
    const all = [cv.wait(), cv.wait(), cv.wait()];
    expect(cv.notifyAll()).toBe(3);
    await expect(Promise.all(all)).resolves.toBeDefined();
  });
});

describe("AgentPool", () => {
  it("respects maxConcurrent", async () => {
    const pool = new AgentPool(2);
    let active = 0;
    let maxObserved = 0;
    const job = async (n: number): Promise<number> => {
      active++;
      maxObserved = Math.max(maxObserved, active);
      await new Promise((r) => setTimeout(r, 10));
      active--;
      return n * 2;
    };
    const results = await pool.map([1, 2, 3, 4, 5], job);
    expect(results).toEqual([2, 4, 6, 8, 10]);
    expect(maxObserved).toBe(2);
  });

  it("mapSettled returns mixed success/failure", async () => {
    const pool = new AgentPool(3);
    const r = await pool.mapSettled([1, 2, 3], async (n) => {
      if (n === 2) throw new Error("boom");
      return n;
    });
    expect(r[0]?.status).toBe("fulfilled");
    expect(r[1]?.status).toBe("rejected");
    expect(r[2]?.status).toBe("fulfilled");
  });

  it("stats track active/completed/failed", async () => {
    const pool = new AgentPool(1);
    await pool.submit(async () => 1, 0);
    await pool
      .submit(async () => {
        throw new Error("x");
      }, 0)
      .catch(() => {});
    expect(pool.stats.completed).toBe(1);
    expect(pool.stats.failed).toBe(1);
  });
});

describe("forkJoin", () => {
  it("returns fulfilled + rejected splits", async () => {
    const r = await forkJoin([1, 2, 3], async (n) => {
      if (n === 2) throw new Error("bad");
      return n * 10;
    });
    expect(r.fulfilled).toEqual([10, 30]);
    expect(r.rejected).toHaveLength(1);
    expect(r.reduced).toEqual([10, 30]);
  });

  it("custom reducer (majority vote)", async () => {
    const r = await forkJoin(["A", "B", "C"], async () => "answer-X", {
      reducer: majorityVote<string>,
    });
    expect(r.reduced).toBe("answer-X");
  });

  it("minSuccess enforced", async () => {
    await expect(
      forkJoin(
        [1, 2, 3],
        async () => {
          throw new Error("all fail");
        },
        { minSuccess: 2 },
      ),
    ).rejects.toThrow(/required 2/);
  });

  it("timeout per task", async () => {
    const r = await forkJoin(
      [1, 2],
      async (n) => {
        if (n === 1) await new Promise((res) => setTimeout(res, 100));
        return n;
      },
      { timeoutMs: 20 },
    );
    expect(r.rejected).toHaveLength(1);
  });
});

describe("TokenBucket", () => {
  it("tryAcquire respects current tokens", () => {
    const b = new TokenBucket({ capacity: 3, refillRate: 1 });
    expect(b.tryAcquire(2)).toBe(true);
    expect(b.tryAcquire(2)).toBe(false);
    expect(b.tryAcquire(1)).toBe(true);
  });

  it("acquire waits for refill", async () => {
    const b = new TokenBucket({ capacity: 1, refillRate: 100 });
    expect(b.tryAcquire(1)).toBe(true);
    const t0 = Date.now();
    await b.acquire(1);
    expect(Date.now() - t0).toBeGreaterThanOrEqual(5);
  });

  it("rejects cost > capacity", async () => {
    const b = new TokenBucket({ capacity: 1, refillRate: 1 });
    await expect(b.acquire(2)).rejects.toThrow(/exceeds capacity/);
  });
});

describe("BackpressureGate", () => {
  it("send throttled by token bucket", async () => {
    const gate = new BackpressureGate<number>({
      bufferSize: 100,
      rateLimitPerSec: 100,
      burstCapacity: 2,
    });
    // burst 2 immediately
    await gate.send(1);
    await gate.send(2);
    // 3rd should wait
    const t0 = Date.now();
    await gate.send(3);
    expect(Date.now() - t0).toBeGreaterThanOrEqual(5);
    gate.close();
  });
});
