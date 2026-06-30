/**
 * RFC-002 §9 性能基准守护。
 *
 * 注意：这是"回归守护"而非"绝对性能验收"。阈值取得很宽松（相对 RFC 目标有 ~50x 余量），
 * 只为在 CI 上抓住灾难性回退（如不小心给热路径加了 O(n) 同步 IO），不追求复现 RFC 的绝对数字。
 * 实际 µs/op 会打印到 CI 日志，可用于人工对比趋势。
 */
import { describe, expect, it } from "vitest";
import { HookBus, type HookLogger } from "../../src/index.js";

const silent: HookLogger = { warn: () => {}, error: () => {} };

describe("perf: HookBus", () => {
  it("无 handler 的 emit 平均开销很低（RFC-002 §9 目标 <0.5µs；守护阈值 50µs）", async () => {
    const bus = new HookBus({ logger: silent });
    const N = 20_000;
    // 预热
    for (let i = 0; i < 1000; i++) await bus.emit("event.LOOP_ITERATION", { taoIter: i, metrics: {} });
    const t0 = performance.now();
    for (let i = 0; i < N; i++) await bus.emit("event.LOOP_ITERATION", { taoIter: i, metrics: {} });
    const perOpUs = ((performance.now() - t0) * 1000) / N;
    console.log(`[perf] HookBus no-handler emit: ${perOpUs.toFixed(3)} µs/op (N=${N})`);
    expect(perOpUs).toBeLessThan(50);
  });

  it("含 10 个同步 handler 的 emit 开销可控（守护阈值 100µs/op）", async () => {
    const bus = new HookBus({ logger: silent });
    let acc = 0;
    for (let h = 0; h < 10; h++) {
      bus.on("event.LOOP_ITERATION", (ctx) => {
        acc += ctx.payload.taoIter & 1;
      });
    }
    const N = 20_000;
    for (let i = 0; i < 1000; i++) await bus.emit("event.LOOP_ITERATION", { taoIter: i, metrics: {} });
    const t0 = performance.now();
    for (let i = 0; i < N; i++) await bus.emit("event.LOOP_ITERATION", { taoIter: i, metrics: {} });
    const perOpUs = ((performance.now() - t0) * 1000) / N;
    console.log(`[perf] HookBus 10-handler emit: ${perOpUs.toFixed(3)} µs/op (N=${N}, acc=${acc})`);
    expect(perOpUs).toBeLessThan(100);
  });

  it("注册 + offByTag 大量 handler 不退化（守护阈值 150µs/注册）", () => {
    // 注：当前实现是按 priority 插入排序（单事件上 O(n)/insert）。这里用 3000 个 handler
    // 既能反映真实量级，又不会被单事件 O(n²) 放大到误报。阈值留 ~3x 余量抗 CI 抖动。
    const bus = new HookBus({ logger: silent });
    const N = 3_000;
    const t0 = performance.now();
    for (let i = 0; i < N; i++) {
      bus.on("event.LOOP_ITERATION", () => {}, { tag: "bench" });
    }
    const perRegUs = ((performance.now() - t0) * 1000) / N;
    const removed = bus.offByTag("bench");
    console.log(`[perf] HookBus register: ${perRegUs.toFixed(3)} µs/op (N=${N}), offByTag removed=${removed}`);
    expect(removed).toBe(N);
    expect(perRegUs).toBeLessThan(150);
  });
});
