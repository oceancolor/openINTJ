/**
 * 根 span 包装器（Phase 3.8.1）——把一次 HTTP route / Electron IPC 调用包成一个根 span，
 * 让 `agent.run()` 在它的 active context 内执行。
 *
 * 为什么这样就能接到 agent span 树：
 *  - `attachOtelToHooks` 里的 `startSpan` 在不传 parent 时用 `context.active()` 作父 context；
 *  - `withRootSpan` 用 `tracer.startActiveSpan` 把根 span 设为 active，并在其回调里 await fn；
 *  - 只要进程注册了带 AsyncLocalStorage 的 ContextManager（`bootstrapNodeOtel` / NodeTracerProvider.register()
 *    会做），active context 就能跨 await 传播 → TAO/ReAct/Tool span 自动挂到根 span 下。
 *
 * 零开销：未注册 provider 时 `trace.getTracer` 返回 no-op，startActiveSpan 直接跑回调、不产 span。
 */

import { SpanStatusCode, type Tracer, trace } from "@opentelemetry/api";

const SCOPE_NAME = "@openintj/telemetry-otel";

export interface WithRootSpanOpts {
  /** Tracer 名称（默认 @openintj/telemetry-otel，与 attachOtelToHooks 同 scope）。 */
  tracerName?: string;
  /** 写到根 span 上的属性（如 http.route / ipc.channel）。 */
  attributes?: Record<string, string | number | boolean>;
}

/**
 * 在一个根 span 的 active context 内执行 `fn`。`fn` 内部触发的 agent span 会成为它的子 span。
 * fn 抛错时把 span 标 ERROR 并记录异常后重新抛出。
 */
export const withRootSpan = async <T>(
  name: string,
  fn: () => Promise<T> | T,
  opts: WithRootSpanOpts = {},
): Promise<T> => {
  const tracer: Tracer = trace.getTracer(opts.tracerName ?? SCOPE_NAME);
  return tracer.startActiveSpan(name, async (span) => {
    if (opts.attributes) span.setAttributes(opts.attributes);
    try {
      const result = await fn();
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (e) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: e instanceof Error ? e.message : String(e),
      });
      span.recordException(e instanceof Error ? e : new Error(String(e)));
      throw e;
    } finally {
      span.end();
    }
  });
};
