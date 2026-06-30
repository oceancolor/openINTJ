/**
 * Metric 计数器断言：用 InMemoryMetricExporter + ManualReader，
 * 触发 tool / iteration / policy / memory_loaded 事件，验证 counter 累计正确。
 */
import { HookBus } from "@openintj/core";
import { metrics } from "@opentelemetry/api";
import {
  InMemoryMetricExporter,
  MeterProvider,
  PeriodicExportingMetricReader,
} from "@opentelemetry/sdk-metrics";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { attachOtelToHooks } from "../src/index.js";

let exporter: InMemoryMetricExporter;
let provider: MeterProvider;
let reader: PeriodicExportingMetricReader;

beforeEach(() => {
  // 0 = DELTA aggregation；用 PeriodicExportingMetricReader + 大 interval，手动 forceFlush
  exporter = new InMemoryMetricExporter(0);
  reader = new PeriodicExportingMetricReader({
    exporter,
    exportIntervalMillis: 60_000,
  });
  provider = new MeterProvider({ readers: [reader] });
  metrics.setGlobalMeterProvider(provider);
});

afterEach(async () => {
  await provider.shutdown();
  metrics.disable();
});

const flush = async (): Promise<
  Array<{ name: string; sum: number; attrs: Record<string, unknown> }>
> => {
  await reader.forceFlush();
  const out: Array<{ name: string; sum: number; attrs: Record<string, unknown> }> = [];
  for (const rm of exporter.getMetrics()) {
    for (const sm of rm.scopeMetrics) {
      for (const m of sm.metrics) {
        for (const dp of m.dataPoints) {
          out.push({
            name: m.descriptor.name,
            sum: dp.value as number,
            attrs: dp.attributes,
          });
        }
      }
    }
  }
  return out;
};

describe("attachOtelToHooks — metrics", () => {
  it("counts tao iterations, react actions, tool calls/errors", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const traceId = "trace-metric-1";

    // 走 2 个 iteration，每个 1 个 action / 1 个 tool（第二个失败）
    for (let i = 0; i < 2; i++) {
      const success = i === 0;
      await bus.emit("tao.beforeThink", { query: "x", iteration: i }, { traceId });
      await bus.emit(
        "react.beforeAction",
        { tool: "fake", params: {}, reactIter: 0, taoIter: i },
        { traceId },
      );
      await bus.emit("tool.beforeCall", {
        tool: "fake",
        params: {},
        toolDescriptor: {
          name: "fake",
          description: "",
          inputSchema: {},
          outputSchema: {},
          permissions: [],
          timeoutS: 30,
          idempotent: false,
          errorSemantics: "fail_fast" as const,
        },
      });
      if (!success) {
        await bus.emit(
          "tool.onError",
          { tool: "fake", error: new Error("nope"), willRetry: true },
          { traceId },
        );
      }
      await bus.emit("tool.afterCall", {
        tool: "fake",
        result: {
          toolName: "fake",
          success,
          durationMs: 1,
          traceId,
          callId: `c${i}`,
        },
      });
      await bus.emit(
        "react.afterAction",
        {
          toolResult: {
            toolName: "fake",
            success,
            durationMs: 1,
            traceId,
            callId: `c${i}`,
          },
          reactIter: 0,
          taoIter: i,
        },
        { traceId },
      );
      await bus.emit("tao.afterObserve", { needsContinue: i === 0, iteration: i }, { traceId });
    }

    const series = await flush();
    const total = (name: string): number =>
      series.filter((s) => s.name === name).reduce((a, b) => a + b.sum, 0);

    expect(total("openintj.tao.iterations")).toBe(2);
    expect(total("openintj.react.actions")).toBe(2);
    expect(total("openintj.tool.calls")).toBe(2);
    expect(total("openintj.tool.errors")).toBe(1);

    otel.dispose();
  });

  it("counts openintj.search.sources from search tool hits", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const traceId = "trace-search-metric";
    const result = {
      toolName: "search",
      success: true,
      output: {
        mode: "live",
        sources: [{ url: "https://a" }, { url: "https://b" }, { url: "https://c" }],
      },
      durationMs: 1,
      traceId,
      callId: "s1",
    };

    await bus.emit("tao.beforeThink", { query: "q", iteration: 0 }, { traceId });
    await bus.emit(
      "react.beforeAction",
      { tool: "search", params: {}, reactIter: 0, taoIter: 0 },
      { traceId },
    );
    await bus.emit(
      "react.afterAction",
      { toolResult: result, reactIter: 0, taoIter: 0 },
      { traceId },
    );
    await bus.emit("tao.afterObserve", { needsContinue: false, iteration: 0 }, { traceId });

    const series = await flush();
    const sourceSeries = series.filter((s) => s.name === "openintj.search.sources");
    expect(sourceSeries.reduce((a, b) => a + b.sum, 0)).toBe(3);
    expect(sourceSeries[0]?.attrs["mode"]).toBe("live");

    otel.dispose();
  });

  it("counts policy.onBlock and event.MEMORY_LOADED", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);

    await bus.emit("event.MEMORY_LOADED", { count: 7, budgetUsage: 0.5 });
    await bus.emit("event.MEMORY_LOADED", { count: 3, budgetUsage: 0.8 });

    const block = {
      command: {
        commandId: "cmd-1",
        commandType: "DELETE_FRAGMENT" as const,
        priority: 1,
        timestamp: Date.now(),
      } as unknown as Parameters<typeof bus.emit<"policy.onBlock">>[1]["command"],
      auditEvent: {
        eventId: "e1",
        timestamp: Date.now(),
        action: "block",
        actor: "test",
        target: "x",
        result: "blocked" as const,
        details: {},
        riskLevel: "high" as const,
      },
      reason: "policy denied",
    };
    await bus.emit("policy.onBlock", block);
    await bus.emit("policy.onBlock", { ...block, reason: "another reason" });

    const series = await flush();
    const total = (name: string): number =>
      series.filter((s) => s.name === name).reduce((a, b) => a + b.sum, 0);

    expect(total("openintj.memory.loaded")).toBe(10);
    // 两次命中（count>0）→ retrieval.hit 计 2
    expect(total("openintj.retrieval.hit")).toBe(2);
    expect(total("openintj.policy.blocked")).toBe(2);

    otel.dispose();
  });

  it("counts openintj.tokens.spent from event.LOOP_ITERATION", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);

    await bus.emit(
      "event.LOOP_ITERATION",
      { taoIter: 0, metrics: { totalTokensSpent: 120, totalReactSteps: 2 } },
      { traceId: "t-tok-1" },
    );
    await bus.emit(
      "event.LOOP_ITERATION",
      { taoIter: 0, metrics: { totalTokensSpent: 80 } },
      { traceId: "t-tok-2" },
    );
    // 0 token 不计
    await bus.emit(
      "event.LOOP_ITERATION",
      { taoIter: 0, metrics: { totalTokensSpent: 0 } },
      { traceId: "t-tok-3" },
    );

    const series = await flush();
    const total = (name: string): number =>
      series.filter((s) => s.name === name).reduce((a, b) => a + b.sum, 0);
    expect(total("openintj.tokens.spent")).toBe(200);

    otel.dispose();
  });

  it("respects disableMetrics flag", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus, { disableMetrics: true });
    await bus.emit("event.MEMORY_LOADED", { count: 5, budgetUsage: 0 });
    const series = await flush();
    expect(series).toHaveLength(0);
    otel.dispose();
  });
});
