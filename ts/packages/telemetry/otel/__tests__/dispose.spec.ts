/**
 * dispose() 兜底：handler 在 traceId 上还有未结束 span 时，dispose 必须把它们全部 end 掉，
 * 否则 BatchSpanProcessor.shutdown 会 hang 或丢数据。
 */
import { HookBus } from "@openintj/core";
import { trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { attachOtelToHooks } from "../src/index.js";

let exporter: InMemorySpanExporter;
let provider: BasicTracerProvider;

beforeEach(() => {
  exporter = new InMemorySpanExporter();
  provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  trace.setGlobalTracerProvider(provider);
});

afterEach(async () => {
  await provider.shutdown();
  trace.disable();
});

describe("attachOtelToHooks — dispose()", () => {
  it("ends iteration/action/tool spans that never received their close event", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const traceId = "trace-dispose-1";

    // 只开不关 —— 模拟 agent 在 react 阶段崩了
    await bus.emit("tao.beforeThink", { query: "x", iteration: 0 }, { traceId });
    await bus.emit(
      "react.beforeAction",
      { tool: "fake", params: {}, reactIter: 0, taoIter: 0 },
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

    expect(otel.openSpanCount()).toBe(3);
    otel.dispose();
    expect(otel.openSpanCount()).toBe(0);
    expect(otel.endedSpanCount()).toBeGreaterThanOrEqual(3);

    const spans = exporter.getFinishedSpans();
    expect(spans.find((s) => s.name === "openintj.tao.iteration")?.attributes["disposed"]).toBe(
      true,
    );
    expect(spans.find((s) => s.name === "openintj.react.action")?.attributes["disposed"]).toBe(
      true,
    );
    expect(spans.find((s) => s.name === "openintj.tool.call")?.attributes["disposed"]).toBe(true);
  });

  it("unregisters all hook handlers (offs run)", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const before = bus.inspect().total;
    otel.dispose();
    const after = bus.inspect().total;
    expect(after).toBeLessThan(before);
    expect(after).toBe(0);
  });

  it("starting a new iteration while previous one is unfinished marks the old as unfinished", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const traceId = "trace-dispose-2";

    await bus.emit("tao.beforeThink", { query: "x", iteration: 0 }, { traceId });
    // 不关掉就进下一轮
    await bus.emit("tao.beforeThink", { query: "x", iteration: 1 }, { traceId });
    await bus.emit("tao.afterObserve", { needsContinue: false, iteration: 1 }, { traceId });

    const spans = exporter.getFinishedSpans();
    const iter0 = spans.find((s) => s.attributes["tao.iter"] === 0);
    const iter1 = spans.find((s) => s.attributes["tao.iter"] === 1);
    expect(iter0?.attributes["tao.unfinished"]).toBe(true);
    expect(iter1?.attributes["tao.needs_continue"]).toBe(false);

    otel.dispose();
  });
});
