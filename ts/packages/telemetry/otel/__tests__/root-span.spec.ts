/**
 * withRootSpan：验证根 span 把 agent（HookBus→OTel）span 树挂到自己下面。
 * 必须用 NodeTracerProvider.register()（带 AsyncLocalStorage ContextManager），
 * 否则 context.active() 不跨 await 传播，挂不上。
 */
import { HookBus } from "@openintj/core";
import { context, trace } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { attachOtelToHooks, withRootSpan } from "../src/index.js";

let exporter: InMemorySpanExporter;
let provider: NodeTracerProvider;

beforeEach(() => {
  exporter = new InMemorySpanExporter();
  provider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  provider.register();
});

afterEach(async () => {
  context.disable();
  await provider.shutdown();
  trace.disable();
});

describe("withRootSpan", () => {
  it("把 tao.iteration span 挂到根 span 下", async () => {
    const bus = new HookBus();
    attachOtelToHooks(bus);
    const traceId = "trace-root-1";

    const out = await withRootSpan(
      "openintj.ipc.chat",
      async () => {
        await bus.emit("tao.beforeThink", { query: "hi", iteration: 0 }, { traceId });
        await bus.emit("tao.afterObserve", { needsContinue: false, iteration: 0 }, { traceId });
        return 42;
      },
      { attributes: { "ipc.channel": "openintj://chat" } },
    );
    expect(out).toBe(42);

    const spans = exporter.getFinishedSpans();
    const root = spans.find((s) => s.name === "openintj.ipc.chat");
    const iter = spans.find((s) => s.name === "openintj.tao.iteration");
    expect(root).toBeDefined();
    expect(iter).toBeDefined();
    // 同一 trace，且 iteration 的 parent 是 root
    expect(iter?.spanContext().traceId).toBe(root?.spanContext().traceId);
    expect(iter?.parentSpanId).toBe(root?.spanContext().spanId);
    expect(root?.attributes["ipc.channel"]).toBe("openintj://chat");
  });

  it("fn 抛错时标记 ERROR 并冒泡", async () => {
    const boom = new Error("boom");
    await expect(
      withRootSpan("openintj.ipc.chat", async () => {
        throw boom;
      }),
    ).rejects.toBe(boom);
    const spans = exporter.getFinishedSpans();
    const root = spans.find((s) => s.name === "openintj.ipc.chat");
    expect(root).toBeDefined();
    // SpanStatusCode.ERROR === 2
    expect(root?.status.code).toBe(2);
  });
});
