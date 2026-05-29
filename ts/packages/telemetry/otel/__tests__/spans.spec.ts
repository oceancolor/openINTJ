/**
 * Span 树结构断言：使用 InMemorySpanExporter + BasicTracerProvider，
 * 模拟一个完整 TAO→ReAct→Tool 轨迹，验证 parent/child 关系正确、attribute 写齐。
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

const noopDescriptor = {
  name: "fake",
  description: "",
  inputSchema: {},
  outputSchema: {},
  permissions: [],
  timeoutS: 30,
  idempotent: false,
  errorSemantics: "fail_fast" as const,
};

describe("attachOtelToHooks — span tree", () => {
  it("emits a TAO root → react action → tool call hierarchy", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const traceId = "trace-tree-1";

    await bus.emit("tao.beforeThink", { query: "hi", iteration: 0 }, { traceId });
    await bus.emit(
      "react.beforeAction",
      { tool: "fake", params: {}, reactIter: 0, taoIter: 0 },
      { traceId },
    );
    await bus.emit(
      "tool.beforeCall",
      { tool: "fake", params: {}, toolDescriptor: noopDescriptor },
      { traceId },
    );
    await bus.emit(
      "tool.afterCall",
      {
        tool: "fake",
        result: { toolName: "fake", success: true, durationMs: 1, traceId, callId: "c1" },
      },
      { traceId },
    );
    await bus.emit(
      "react.afterAction",
      {
        toolResult: { toolName: "fake", success: true, durationMs: 1, traceId, callId: "c1" },
        reactIter: 0,
        taoIter: 0,
      },
      { traceId },
    );
    await bus.emit(
      "tao.afterAct",
      { reactOutput: { finalAnswer: "ok", status: "completed", iterations: 1 }, iteration: 0 },
      { traceId },
    );
    await bus.emit("tao.afterObserve", { needsContinue: false, iteration: 0 }, { traceId });

    expect(otel.openSpanCount()).toBe(0);

    const spans = exporter.getFinishedSpans();
    const names = spans.map((s) => s.name).sort();
    expect(names).toEqual([
      "openintj.react.action",
      "openintj.tao.iteration",
      "openintj.tool.call",
    ]);

    const iter = spans.find((s) => s.name === "openintj.tao.iteration")!;
    const action = spans.find((s) => s.name === "openintj.react.action")!;
    const tool = spans.find((s) => s.name === "openintj.tool.call")!;

    // 同一 trace（OTel trace_id 来自 SDK，不等于 HookBus traceId；只断同源）
    expect(action.spanContext().traceId).toBe(iter.spanContext().traceId);
    expect(tool.spanContext().traceId).toBe(iter.spanContext().traceId);

    // parent/child：action.parent === iter；tool.parent === action
    expect(action.parentSpanId).toBe(iter.spanContext().spanId);
    expect(tool.parentSpanId).toBe(action.spanContext().spanId);

    // 关键 attribute
    expect(iter.attributes["tao.iter"]).toBe(0);
    expect(iter.attributes["tao.needs_continue"]).toBe(false);
    expect(iter.attributes["react.status"]).toBe("completed");
    expect(iter.attributes["trace_id"]).toBe(traceId);
    expect(action.attributes["react.tool"]).toBe("fake");
    expect(action.attributes["react.result.success"]).toBe(true);
    expect(tool.attributes["tool.name"]).toBe("fake");
    expect(tool.attributes["tool.success"]).toBe(true);

    otel.dispose();
  });

  it("attaches search source attributes to action + tool spans", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const traceId = "trace-search";
    const searchOutput = {
      ok: true,
      mode: "live",
      query: "今天日期",
      answer: "…",
      sources: [
        { title: "A", url: "https://a.example" },
        { title: "B", url: "https://b.example" },
      ],
    };

    await bus.emit("tao.beforeThink", { query: "今天日期", iteration: 0 }, { traceId });
    await bus.emit(
      "react.beforeAction",
      { tool: "search", params: { query: "今天日期" }, reactIter: 0, taoIter: 0 },
      { traceId },
    );
    await bus.emit(
      "tool.beforeCall",
      { tool: "search", params: {}, toolDescriptor: { ...noopDescriptor, name: "search" } },
      { traceId },
    );
    await bus.emit(
      "tool.afterCall",
      {
        tool: "search",
        result: {
          toolName: "search",
          success: true,
          output: searchOutput,
          durationMs: 1,
          traceId,
          callId: "s1",
        },
      },
      { traceId },
    );
    await bus.emit(
      "react.afterAction",
      {
        toolResult: {
          toolName: "search",
          success: true,
          output: searchOutput,
          durationMs: 1,
          traceId,
          callId: "s1",
        },
        reactIter: 0,
        taoIter: 0,
      },
      { traceId },
    );
    await bus.emit("tao.afterObserve", { needsContinue: false, iteration: 0 }, { traceId });

    const tool = exporter.getFinishedSpans().find((s) => s.name === "openintj.tool.call")!;
    const action = exporter.getFinishedSpans().find((s) => s.name === "openintj.react.action")!;
    expect(tool.attributes["search.sources_count"]).toBe(2);
    expect(tool.attributes["search.mode"]).toBe("live");
    expect(tool.attributes["search.urls"]).toBe("https://a.example,https://b.example");
    expect(action.attributes["search.sources_count"]).toBe(2);

    otel.dispose();
  });

  it("marks failed tool calls as ERROR status with exception", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    const traceId = "trace-tree-err";

    await bus.emit("tao.beforeThink", { query: "x", iteration: 0 }, { traceId });
    await bus.emit(
      "react.beforeAction",
      { tool: "bad", params: {}, reactIter: 0, taoIter: 0 },
      { traceId },
    );
    await bus.emit(
      "tool.beforeCall",
      { tool: "bad", params: {}, toolDescriptor: { ...noopDescriptor, name: "bad" } },
      { traceId },
    );
    await bus.emit(
      "tool.onError",
      { tool: "bad", error: new Error("boom"), willRetry: false },
      { traceId },
    );
    await bus.emit(
      "tool.afterCall",
      {
        tool: "bad",
        result: {
          toolName: "bad",
          success: false,
          error: "boom",
          durationMs: 1,
          traceId,
          callId: "c2",
        },
      },
      { traceId },
    );
    await bus.emit(
      "react.afterAction",
      {
        toolResult: {
          toolName: "bad",
          success: false,
          error: "boom",
          durationMs: 1,
          traceId,
          callId: "c2",
        },
        reactIter: 0,
        taoIter: 0,
      },
      { traceId },
    );
    await bus.emit("tao.afterObserve", { needsContinue: false, iteration: 0 }, { traceId });

    const tool = exporter.getFinishedSpans().find((s) => s.name === "openintj.tool.call")!;
    const action = exporter.getFinishedSpans().find((s) => s.name === "openintj.react.action")!;
    // SpanStatusCode.ERROR = 2
    expect(tool.status.code).toBe(2);
    expect(action.status.code).toBe(2);
    expect(tool.events.find((e) => e.name === "exception")).toBeTruthy();
    expect(tool.attributes["tool.success"]).toBe(false);
    expect(tool.attributes["tool.will_retry"]).toBe(false);

    otel.dispose();
  });
});
