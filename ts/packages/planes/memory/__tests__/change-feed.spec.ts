/**
 * A1.1 记忆写入 change-feed：验证 MemoryStore 注入 HookBus 后
 * add / 短期溢出晋升 / 工作记忆溢出丢弃 / remove 都会 emit event.MEMORY_WRITTEN。
 */
import { HookBus } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { MemoryStore } from "../src/index.js";

interface Captured {
  fragmentId: string;
  op: "add" | "update" | "remove";
  memoryType: string;
}

const collect = (bus: HookBus): Captured[] => {
  const out: Captured[] = [];
  bus.on("event.MEMORY_WRITTEN", (ctx) => {
    out.push({
      fragmentId: ctx.payload.fragment.fragmentId,
      op: ctx.payload.op,
      memoryType: ctx.payload.fragment.memoryType,
    });
  });
  return out;
};

// emit 是 fire-and-forget，给微任务一拍时间 flush。
const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

describe("memory change-feed (event.MEMORY_WRITTEN)", () => {
  it("不注入 hooks 时零开销、不报错", () => {
    const store = new MemoryStore();
    expect(() => store.addShortTerm("hello")).not.toThrow();
  });

  it("add* 各层都发出 op=add", async () => {
    const bus = new HookBus();
    const events = collect(bus);
    const store = new MemoryStore({}, { hooks: bus });
    const a = store.addShortTerm("s");
    const b = store.addWorking("w");
    const c = store.addLongTerm("l");
    await flush();
    expect(events).toHaveLength(3);
    expect(events.map((e) => e.op)).toEqual(["add", "add", "add"]);
    expect(events.map((e) => e.fragmentId)).toEqual([a.fragmentId, b.fragmentId, c.fragmentId]);
  });

  it("短期溢出晋升发出 op=update（且 memoryType=long_term）", async () => {
    const bus = new HookBus();
    const events = collect(bus);
    const store = new MemoryStore({ maxShortTerm: 1 }, { hooks: bus });
    const first = store.addShortTerm("first");
    store.addShortTerm("second"); // 触发 first 晋升
    await flush();
    const update = events.find((e) => e.op === "update");
    expect(update).toBeDefined();
    expect(update?.fragmentId).toBe(first.fragmentId);
    expect(update?.memoryType).toBe("long_term");
  });

  it("工作记忆溢出丢弃发出 op=remove", async () => {
    const bus = new HookBus();
    const events = collect(bus);
    const store = new MemoryStore({ maxWorking: 1 }, { hooks: bus });
    const first = store.addWorking("first");
    store.addWorking("second"); // 触发 first 丢弃
    await flush();
    const removed = events.find((e) => e.op === "remove");
    expect(removed?.fragmentId).toBe(first.fragmentId);
  });

  it("remove() 发出 op=remove", async () => {
    const bus = new HookBus();
    const events = collect(bus);
    const store = new MemoryStore({}, { hooks: bus });
    const f = store.addLongTerm("x");
    const ok = store.remove(f.fragmentId);
    await flush();
    expect(ok).toBe(true);
    expect(events.filter((e) => e.op === "remove").map((e) => e.fragmentId)).toContain(
      f.fragmentId,
    );
  });
});
