/**
 * 并发 / 多任务 / 多 Agent 可观测性：验证 pool.* / forkjoin.* / task.* 事件
 * 被翻译成独立 span + counter。
 */
import { HookBus } from "@openintj/core";
import { metrics, trace } from "@opentelemetry/api";
import {
  InMemoryMetricExporter,
  MeterProvider,
  PeriodicExportingMetricReader,
} from "@opentelemetry/sdk-metrics";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { attachOtelToHooks } from "../src/index.js";

let spanExporter: InMemorySpanExporter;
let tracerProvider: BasicTracerProvider;
let metricExporter: InMemoryMetricExporter;
let meterProvider: MeterProvider;
let reader: PeriodicExportingMetricReader;

beforeEach(() => {
  spanExporter = new InMemorySpanExporter();
  tracerProvider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(spanExporter)],
  });
  trace.setGlobalTracerProvider(tracerProvider);

  metricExporter = new InMemoryMetricExporter(0);
  reader = new PeriodicExportingMetricReader({
    exporter: metricExporter,
    exportIntervalMillis: 60_000,
  });
  meterProvider = new MeterProvider({ readers: [reader] });
  metrics.setGlobalMeterProvider(meterProvider);
});

afterEach(async () => {
  await tracerProvider.shutdown();
  await meterProvider.shutdown();
  trace.disable();
  metrics.disable();
});

const flushMetrics = async (): Promise<
  Array<{ name: string; sum: number; attrs: Record<string, unknown> }>
> => {
  await reader.forceFlush();
  const out: Array<{ name: string; sum: number; attrs: Record<string, unknown> }> = [];
  for (const rm of metricExporter.getMetrics()) {
    for (const sm of rm.scopeMetrics) {
      for (const m of sm.metrics) {
        for (const dp of m.dataPoints) {
          out.push({ name: m.descriptor.name, sum: dp.value as number, attrs: dp.attributes });
        }
      }
    }
  }
  return out;
};

describe("attachOtelToHooks — concurrency", () => {
  it("AgentPool job → span + openintj.pool.jobs counter", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);

    await bus.emit("pool.beforeJob", { pool: "p", jobId: "j1", active: 1, pending: 0 });
    await bus.emit("pool.afterJob", {
      pool: "p",
      jobId: "j1",
      success: true,
      durationMs: 5,
      active: 0,
      pending: 0,
      completed: 1,
      failed: 0,
    });

    const spans = spanExporter.getFinishedSpans();
    const jobSpan = spans.find((s) => s.name === "openintj.pool.job");
    expect(jobSpan).toBeDefined();
    expect(jobSpan?.attributes["pool.name"]).toBe("p");
    expect(jobSpan?.attributes["pool.success"]).toBe(true);
    expect(jobSpan?.attributes["pool.duration_ms"]).toBe(5);

    const series = await flushMetrics();
    const jobs = series.filter((s) => s.name === "openintj.pool.jobs");
    expect(jobs.reduce((a, b) => a + b.sum, 0)).toBe(1);
    expect(jobs[0]?.attrs["success"]).toBe("true");

    expect(otel.openSpanCount()).toBe(0);
    otel.dispose();
  });

  it("ForkJoin → span + branches/rejected counters", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);

    await bus.emit("forkjoin.beforeFork", { group: "g", total: 3 });
    await bus.emit("forkjoin.afterJoin", {
      group: "g",
      total: 3,
      fulfilled: 2,
      rejected: 1,
      durationMs: 10,
    });

    const span = spanExporter.getFinishedSpans().find((s) => s.name === "openintj.forkjoin");
    expect(span?.attributes["forkjoin.total"]).toBe(3);
    expect(span?.attributes["forkjoin.fulfilled"]).toBe(2);
    expect(span?.attributes["forkjoin.rejected"]).toBe(1);

    const series = await flushMetrics();
    const total = (name: string): number =>
      series.filter((s) => s.name === name).reduce((a, b) => a + b.sum, 0);
    expect(total("openintj.forkjoin.branches")).toBe(3);
    expect(total("openintj.forkjoin.rejected")).toBe(1);

    otel.dispose();
  });

  it("TaskQueue → run span + enqueued/completed counters", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);

    await bus.emit("task.enqueue", {
      queue: "q",
      taskId: "t1",
      priority: 5,
      depCount: 0,
      ready: true,
    });
    await bus.emit("task.beforeRun", { queue: "q", taskId: "t1", priority: 5 });
    await bus.emit("task.afterRun", { queue: "q", taskId: "t1", success: true, durationMs: 2 });

    const span = spanExporter.getFinishedSpans().find((s) => s.name === "openintj.task.run");
    expect(span?.attributes["task.id"]).toBe("t1");
    expect(span?.attributes["task.success"]).toBe(true);

    const series = await flushMetrics();
    const total = (name: string): number =>
      series.filter((s) => s.name === name).reduce((a, b) => a + b.sum, 0);
    expect(total("openintj.task.enqueued")).toBe(1);
    expect(total("openintj.task.completed")).toBe(1);

    otel.dispose();
  });

  it("dispose 兜底结束未完成的并发 span", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    await bus.emit("pool.beforeJob", { pool: "p", jobId: "leak", active: 1, pending: 0 });
    expect(otel.openSpanCount()).toBe(1);
    otel.dispose();
    const span = spanExporter
      .getFinishedSpans()
      .find((s) => s.attributes["pool.job_id"] === "leak");
    expect(span?.attributes["disposed"]).toBe(true);
  });
});
