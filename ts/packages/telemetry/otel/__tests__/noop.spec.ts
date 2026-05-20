/**
 * 默认零成本路径：未注册 TracerProvider / MeterProvider 时，
 * attachOtelToHooks 仍能跑完事件序列、不抛错、openSpan=0、endedSpan=0。
 */
import { HookBus } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { attachOtelToHooks } from "../src/index.js";

describe("attachOtelToHooks — no provider (zero cost)", () => {
  it("does not throw across a full tao→react→tool cycle", async () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);

    const traceId = "trace-noop-1";
    await bus.emit("tao.beforeThink", { query: "hi", iteration: 0 }, { traceId });
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
        errorSemantics: "fail_fast",
      },
    });
    await bus.emit("tool.afterCall", {
      tool: "fake",
      result: {
        toolName: "fake",
        success: true,
        durationMs: 1,
        traceId,
        callId: "cid",
      },
    });
    await bus.emit(
      "react.afterAction",
      {
        toolResult: {
          toolName: "fake",
          success: true,
          durationMs: 1,
          traceId,
          callId: "cid",
        },
        reactIter: 0,
        taoIter: 0,
      },
      { traceId },
    );
    await bus.emit(
      "tao.afterAct",
      {
        reactOutput: { finalAnswer: "ok", status: "completed", iterations: 1 },
        iteration: 0,
      },
      { traceId },
    );
    await bus.emit("tao.afterObserve", { needsContinue: false, iteration: 0 }, { traceId });

    // no-op tracer 仍然会"开/关 span"（句柄是 NoopSpan），所以 endedCount 会增长，
    // 但 openSpanCount 必为 0（每个 begin 都有对应 end）。
    expect(otel.openSpanCount()).toBe(0);

    otel.dispose();
    expect(otel.openSpanCount()).toBe(0);
  });

  it("disposes cleanly when no events have been emitted", () => {
    const bus = new HookBus();
    const otel = attachOtelToHooks(bus);
    otel.dispose();
    expect(otel.openSpanCount()).toBe(0);
  });
});
