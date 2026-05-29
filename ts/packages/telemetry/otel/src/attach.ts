/**
 * Hooks → OpenTelemetry 适配（Phase 3.8）
 *
 * 关键设计
 * --------
 * - **零开销默认**：只依赖 `@opentelemetry/api`，consumer 不注册 provider 时
 *   `trace.getTracer()` / `metrics.getMeter()` 都返回 no-op 实现 → 不产 span / 不收 metric。
 * - **traceId 对齐**：HookBus 已经为每次 `run()` 生成 UUID `traceId` 并随 emit 透传。
 *   本适配器维护 Map<traceId, parentSpan>，让 react.action / tool.call span 自动挂到当前 iteration 下，
 *   同 traceId 内的事件天然形成 span 树。
 * - **不在 hook 里 await**：所有 handler 都是同步的，避免在 emit 路径上加额外 await。
 * - **失败软关**：handler 内部抛错不会冒泡到业务（HookBus.strictMode=false 时被 logger 吞）；
 *   handler 实现层也用 try/catch 兜底，确保 telemetry 抖动不影响主循环。
 */

import type { HookBus, HookContext, HookEventMap } from "@openintj/core";
import {
  type Context,
  type Counter,
  type Meter,
  type Span,
  SpanStatusCode,
  type Tracer,
  context,
  metrics,
  trace,
} from "@opentelemetry/api";

const SCOPE_NAME = "@openintj/telemetry-otel";

export interface AttachOtelOpts {
  /** Tracer 名称（默认 `@openintj/telemetry-otel`）。 */
  tracerName?: string;
  /** Meter 名称（默认 `@openintj/telemetry-otel`）。 */
  meterName?: string;
  /** 资源属性附加（写到所有 span 上；常用 `agent.role` / `agent.id` 等）。 */
  defaultAttributes?: Record<string, string | number | boolean>;
  /** 关闭 metric 收集（只保留 trace）。 */
  disableMetrics?: boolean;
  /** 关闭 trace span（只保留 metric）。 */
  disableTraces?: boolean;
}

export interface AttachedOtel {
  /** 取消所有 hook 订阅 + end 所有未结束 span。 */
  dispose(): void;
  /** 当前未结束的 span 数量（debug 用）。 */
  openSpanCount(): number;
  /** 已结束的 span 累计（debug 用）。 */
  endedSpanCount(): number;
}

interface IterationFrame {
  span: Span;
  ctx: Context;
}

interface ActionFrame {
  span: Span;
  ctx: Context;
}

interface ToolFrame {
  span: Span;
  ctx: Context;
}

interface TraceState {
  iteration: IterationFrame | undefined;
  action: ActionFrame | undefined;
  /** 同一 traceId 内可能并发触发多个工具调用（即便当前不是 parallel，未来扩展也允许）。 */
  tools: Map<string, ToolFrame>;
}

/**
 * 把 HookBus 接到 OpenTelemetry。返回 `dispose()` 用于关闭。
 *
 * Span 结构（典型 single-iteration、single-action、single-tool 路径）：
 * ```
 * openintj.tao.iteration  (root per iteration; attribute=tao.iter)
 *   └─ openintj.react.action  (attribute=tool name + react.iter)
 *        └─ openintj.tool.call  (attribute=tool name; status=OK/ERROR)
 * ```
 *
 * Metric 计数器：
 * - `openintj.tao.iterations` (count, attribute=status)
 * - `openintj.react.actions` (count, attribute=tool)
 * - `openintj.tool.calls` (count, attribute=tool, success)
 * - `openintj.tool.errors` (count, attribute=tool, retried)
 * - `openintj.policy.blocked` (count, attribute=reason)
 * - `openintj.memory.loaded` (count, attribute= — 累加 payload.count)
 */
export const attachOtelToHooks = (bus: HookBus, opts: AttachOtelOpts = {}): AttachedOtel => {
  const tracerName = opts.tracerName ?? SCOPE_NAME;
  const meterName = opts.meterName ?? SCOPE_NAME;
  const tracer: Tracer = trace.getTracer(tracerName);
  const meter: Meter = metrics.getMeter(meterName);
  const defaults = opts.defaultAttributes ?? {};
  const enableTraces = !opts.disableTraces;
  const enableMetrics = !opts.disableMetrics;

  const traces = new Map<string, TraceState>();
  let endedCount = 0;

  // ---------- 计数器（即便未注册 MeterProvider，create*Counter 返回 no-op） ----------
  const c = (name: string, description: string): Counter | undefined =>
    enableMetrics ? meter.createCounter(name, { description }) : undefined;

  const cIterations = c("openintj.tao.iterations", "TAO 宏循环迭代次数");
  const cActions = c("openintj.react.actions", "ReAct 微循环 action 次数");
  const cToolCalls = c("openintj.tool.calls", "ToolHub 调用次数");
  const cToolErrors = c("openintj.tool.errors", "ToolHub 错误次数");
  const cPolicyBlocked = c("openintj.policy.blocked", "Governance 拦截次数");
  const cMemoryLoaded = c("openintj.memory.loaded", "Memory 加载的 fragment 总数");
  const cSearchSources = c("openintj.search.sources", "search 工具命中的联网来源数");

  const getTrace = (traceId: string): TraceState => {
    let st = traces.get(traceId);
    if (!st) {
      st = { iteration: undefined, action: undefined, tools: new Map() };
      traces.set(traceId, st);
    }
    return st;
  };

  const clearTraceIfEmpty = (traceId: string): void => {
    const st = traces.get(traceId);
    if (!st) return;
    if (!st.iteration && !st.action && st.tools.size === 0) {
      traces.delete(traceId);
    }
  };

  /**
   * 同步起 span：使用 `tracer.startSpan` 而不是 `startActiveSpan`，
   * 这样我们能拿到 span 引用并稍后在另一个 handler 里 end。
   * parent context 通过 `trace.setSpan(...)` 显式构造。
   */
  // 从 search 工具结果里抽取可观测属性（来源数 / 来源 URL / 联网模式）。
  const SEARCH_URL_ATTR_MAX = 5;
  const extractSearchAttributes = (
    output: unknown,
  ): { attrs: Record<string, string | number | boolean>; count: number; mode: string } => {
    const attrs: Record<string, string | number | boolean> = {};
    let count = 0;
    let mode = "unknown";
    if (output && typeof output === "object") {
      const o = output as { mode?: unknown; sources?: unknown };
      if (typeof o.mode === "string") {
        mode = o.mode;
        attrs["search.mode"] = o.mode;
      }
      if (Array.isArray(o.sources)) {
        count = o.sources.length;
        attrs["search.sources_count"] = count;
        const urls = (o.sources as Array<{ url?: unknown }>)
          .map((s) => (typeof s?.url === "string" ? s.url : undefined))
          .filter((u): u is string => Boolean(u))
          .slice(0, SEARCH_URL_ATTR_MAX);
        if (urls.length > 0) attrs["search.urls"] = urls.join(",");
      }
    }
    return { attrs, count, mode };
  };

  const startSpan = (name: string, parent?: Context): { span: Span; ctx: Context } => {
    const span = parent ? tracer.startSpan(name, undefined, parent) : tracer.startSpan(name);
    for (const [k, v] of Object.entries(defaults)) span.setAttribute(k, v);
    const ctx = trace.setSpan(parent ?? context.active(), span);
    return { span, ctx };
  };

  const safe = <P>(
    handler: (payload: P, ctx: HookContext<P>) => void,
  ): ((ctx: HookContext<P>) => void) => {
    return (ctx) => {
      try {
        handler(ctx.payload, ctx);
      } catch {
        // telemetry 抖动绝不允许影响业务
      }
    };
  };

  const offs: Array<() => void> = [];

  // ---------- TAO 迭代 ----------
  offs.push(
    bus.on(
      "tao.beforeThink",
      safe<HookEventMap["tao.beforeThink"]>((payload, hookCtx) => {
        if (!enableTraces) return;
        const st = getTrace(hookCtx.traceId);
        // 同一 traceId 上一轮 iteration 没收尾就先结一下，避免泄漏
        if (st.iteration) {
          st.iteration.span.setAttribute("tao.unfinished", true);
          st.iteration.span.end();
          endedCount++;
        }
        const { span, ctx } = startSpan("openintj.tao.iteration");
        span.setAttribute("tao.iter", payload.iteration);
        span.setAttribute("trace_id", hookCtx.traceId);
        if (typeof payload.query === "string") {
          span.setAttribute("tao.query.length", payload.query.length);
        }
        st.iteration = { span, ctx };
      }),
    ),
  );

  offs.push(
    bus.on(
      "tao.afterAct",
      safe<HookEventMap["tao.afterAct"]>((payload, hookCtx) => {
        if (!enableTraces) return;
        const st = traces.get(hookCtx.traceId);
        if (!st?.iteration) return;
        st.iteration.span.setAttribute("react.status", payload.reactOutput.status);
        st.iteration.span.setAttribute("react.iterations", payload.reactOutput.iterations);
      }),
    ),
  );

  offs.push(
    bus.on(
      "tao.afterObserve",
      safe<HookEventMap["tao.afterObserve"]>((payload, hookCtx) => {
        const st = traces.get(hookCtx.traceId);
        if (enableTraces && st?.iteration) {
          st.iteration.span.setAttribute("tao.needs_continue", payload.needsContinue);
          st.iteration.span.end();
          endedCount++;
          st.iteration = undefined;
        }
        cIterations?.add(1, {
          needs_continue: payload.needsContinue ? "true" : "false",
        });
        clearTraceIfEmpty(hookCtx.traceId);
      }),
    ),
  );

  // ---------- ReAct action ----------
  offs.push(
    bus.on(
      "react.beforeAction",
      safe<HookEventMap["react.beforeAction"]>((payload, hookCtx) => {
        if (!enableTraces) return;
        const st = getTrace(hookCtx.traceId);
        const parent = st.iteration?.ctx;
        const { span, ctx } = startSpan("openintj.react.action", parent);
        span.setAttribute("react.tool", payload.tool);
        span.setAttribute("react.iter", payload.reactIter);
        span.setAttribute("tao.iter", payload.taoIter);
        if (st.action) {
          st.action.span.setAttribute("react.unfinished", true);
          st.action.span.end();
          endedCount++;
        }
        st.action = { span, ctx };
      }),
    ),
  );

  offs.push(
    bus.on(
      "react.afterAction",
      safe<HookEventMap["react.afterAction"]>((payload, hookCtx) => {
        const st = traces.get(hookCtx.traceId);
        const toolName = payload.toolResult.toolName;
        const isSearchHit = toolName === "search" && payload.toolResult.success;
        const search = isSearchHit
          ? extractSearchAttributes(payload.toolResult.output)
          : undefined;
        if (enableTraces && st?.action) {
          st.action.span.setAttribute("react.result.success", payload.toolResult.success);
          if (search) {
            for (const [k, v] of Object.entries(search.attrs)) st.action.span.setAttribute(k, v);
          }
          if (!payload.toolResult.success) {
            st.action.span.setStatus({ code: SpanStatusCode.ERROR });
          }
          st.action.span.end();
          endedCount++;
          st.action = undefined;
        }
        cActions?.add(1, {
          tool: toolName,
          success: payload.toolResult.success ? "true" : "false",
        });
        if (search) cSearchSources?.add(search.count, { mode: search.mode });
        clearTraceIfEmpty(hookCtx.traceId);
      }),
    ),
  );

  // ---------- Tool 调用 ----------
  offs.push(
    bus.on(
      "tool.beforeCall",
      safe<HookEventMap["tool.beforeCall"]>((payload, hookCtx) => {
        if (!enableTraces) return;
        const st = getTrace(hookCtx.traceId);
        const parent = st.action?.ctx ?? st.iteration?.ctx;
        const { span, ctx } = startSpan("openintj.tool.call", parent);
        span.setAttribute("tool.name", payload.tool);
        st.tools.set(payload.tool, { span, ctx });
      }),
    ),
  );

  offs.push(
    bus.on(
      "tool.afterCall",
      safe<HookEventMap["tool.afterCall"]>((payload, hookCtx) => {
        const st = traces.get(hookCtx.traceId);
        const frame = st?.tools.get(payload.tool);
        if (enableTraces && frame) {
          frame.span.setAttribute("tool.success", payload.result.success);
          if (payload.tool === "search" && payload.result.success) {
            const { attrs } = extractSearchAttributes(payload.result.output);
            for (const [k, v] of Object.entries(attrs)) frame.span.setAttribute(k, v);
          }
          if (!payload.result.success) {
            frame.span.setStatus({ code: SpanStatusCode.ERROR });
          }
          frame.span.end();
          endedCount++;
          st?.tools.delete(payload.tool);
        }
        cToolCalls?.add(1, {
          tool: payload.tool,
          success: payload.result.success ? "true" : "false",
        });
        if (st) clearTraceIfEmpty(hookCtx.traceId);
      }),
    ),
  );

  offs.push(
    bus.on(
      "tool.onError",
      safe<HookEventMap["tool.onError"]>((payload, hookCtx) => {
        const st = traces.get(hookCtx.traceId);
        const frame = st?.tools.get(payload.tool);
        if (enableTraces && frame) {
          frame.span.recordException(payload.error);
          frame.span.setStatus({ code: SpanStatusCode.ERROR, message: payload.error.message });
          frame.span.setAttribute("tool.will_retry", payload.willRetry);
          // 不 end —— 让 afterCall 来 end，保持 happy-path 一致；
          // 但如果不会重试也没人调 afterCall，最终 dispose() 会兜底
        }
        cToolErrors?.add(1, {
          tool: payload.tool,
          will_retry: payload.willRetry ? "true" : "false",
        });
      }),
    ),
  );

  // ---------- 事件计数 ----------
  offs.push(
    bus.on(
      "policy.onBlock",
      safe<HookEventMap["policy.onBlock"]>((payload) => {
        cPolicyBlocked?.add(1, { reason: payload.reason.slice(0, 64) });
      }),
    ),
  );

  offs.push(
    bus.on(
      "event.MEMORY_LOADED",
      safe<HookEventMap["event.MEMORY_LOADED"]>((payload) => {
        if (payload.count > 0) cMemoryLoaded?.add(payload.count);
      }),
    ),
  );

  return {
    dispose: () => {
      for (const off of offs) {
        try {
          off();
        } catch {
          /* ignore */
        }
      }
      // 任何还开着的 span（异常 / 中途 dispose）兜底 end
      for (const [, st] of traces) {
        if (st.iteration) {
          st.iteration.span.setAttribute("disposed", true);
          st.iteration.span.end();
          endedCount++;
        }
        if (st.action) {
          st.action.span.setAttribute("disposed", true);
          st.action.span.end();
          endedCount++;
        }
        for (const [, frame] of st.tools) {
          frame.span.setAttribute("disposed", true);
          frame.span.end();
          endedCount++;
        }
      }
      traces.clear();
    },
    openSpanCount: () => {
      let n = 0;
      for (const st of traces.values()) {
        if (st.iteration) n++;
        if (st.action) n++;
        n += st.tools.size;
      }
      return n;
    },
    endedSpanCount: () => endedCount,
  };
};
