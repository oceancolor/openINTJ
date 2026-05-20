/**
 * 可选 SDK 引导（Node 端）—— 把 `@openintj/telemetry-otel` 接到一个真实 OTLP exporter。
 *
 * 设计要点：
 * - **不强依赖 SDK 包**：consumers 只调 `attachOtelToHooks` 时根本不需要 SDK；
 *   仅当显式调 `bootstrapNodeOtel` 才会触发 `await import(...)`。
 *   SDK 是 `peerDependencies` 且 `meta.optional: true`，CI 装则可用，不装则 import 抛错。
 * - **idempotent**：globalTracerProvider 已注册时不重复注册，返回旧 provider。
 *
 * 典型用法（server/desktop 装配前）：
 * ```ts
 * const otel = await bootstrapNodeOtel({
 *   serviceName: 'openintj-server',
 *   otlpEndpoint: process.env.OTEL_EXPORTER_OTLP_ENDPOINT,
 * });
 * const agent = await assembleServerAgent({ enableOtel: true });
 * // ...
 * await otel.shutdown();
 * ```
 */

import { trace } from "@opentelemetry/api";

export interface BootstrapNodeOtelOpts {
  serviceName?: string;
  /** OTLP HTTP traces endpoint；缺省读 OTEL_EXPORTER_OTLP_ENDPOINT。 */
  otlpEndpoint?: string;
  /** 服务版本/环境等额外 resource 属性。 */
  resourceAttributes?: Record<string, string>;
  /** 是否同时挂 metric reader（默认 false：traces only，保持轻）。 */
  enableMetrics?: boolean;
  /** metric export 间隔（毫秒，默认 10000）。 */
  metricExportIntervalMs?: number;
}

export interface BootstrappedOtel {
  shutdown(): Promise<void>;
  /** 已经存在的 provider —— 反映本地 setGlobalTracerProvider 的实际效果。 */
  alreadyRegistered: boolean;
}

const isPlaceholderProvider = (): boolean => {
  // NoopTracerProvider 的 getTracer 返回 NoopTracer；最稳的探针是看 startSpan 是否 noop
  const t = trace.getTracerProvider().getTracer("__probe__");
  // NoopTracer 上 startSpan 不会抛但返回 NoopSpan；不能直接判别。改用 OTel API 内部：
  // 实测：未注册前 trace.getTracerProvider() === ProxyTracerProvider，其内部 delegate 为 undefined。
  // ProxyTracerProvider 类型不公开，只能用 duck typing：构造的 span 上 spanContext().traceId === '00000000000000000000000000000000'
  const span = t.startSpan("__probe__");
  const traceId = span.spanContext().traceId;
  span.end();
  return /^0+$/.test(traceId);
};

export const bootstrapNodeOtel = async (
  opts: BootstrapNodeOtelOpts = {},
): Promise<BootstrappedOtel> => {
  if (!isPlaceholderProvider()) {
    return {
      shutdown: async () => {
        /* not our provider; leave to caller */
      },
      alreadyRegistered: true,
    };
  }

  // 全部懒 import；SDK 缺包时这里抛错，调用方决定如何处置（fallback no-op 或退出）
  const [
    { NodeTracerProvider },
    { Resource },
    { BatchSpanProcessor },
    { OTLPTraceExporter },
    semconvModule,
    metricsModule,
  ] = await Promise.all([
    import("@opentelemetry/sdk-trace-node"),
    import("@opentelemetry/resources"),
    import("@opentelemetry/sdk-trace-base"),
    import("@opentelemetry/exporter-trace-otlp-http"),
    import("@opentelemetry/semantic-conventions"),
    opts.enableMetrics
      ? import("@opentelemetry/sdk-metrics")
      : Promise.resolve(undefined as unknown as typeof import("@opentelemetry/sdk-metrics")),
  ]);

  const serviceName = opts.serviceName ?? "openintj";
  const SEM = semconvModule.SemanticResourceAttributes;
  const resource = new Resource({
    [SEM.SERVICE_NAME]: serviceName,
    ...(opts.resourceAttributes ?? {}),
  });
  const traceExporter = new OTLPTraceExporter(
    opts.otlpEndpoint ? { url: opts.otlpEndpoint } : undefined,
  );
  const provider = new NodeTracerProvider({
    resource,
    spanProcessors: [new BatchSpanProcessor(traceExporter)],
  });
  provider.register();

  let meterProvider: { shutdown(): Promise<void> } | undefined;
  if (opts.enableMetrics && metricsModule) {
    const { metrics } = await import("@opentelemetry/api");
    const { MeterProvider, PeriodicExportingMetricReader, ConsoleMetricExporter } = metricsModule;
    const reader = new PeriodicExportingMetricReader({
      exporter: new ConsoleMetricExporter(),
      exportIntervalMillis: opts.metricExportIntervalMs ?? 10_000,
    });
    const mp = new MeterProvider({ resource, readers: [reader] });
    metrics.setGlobalMeterProvider(mp);
    meterProvider = mp;
  }

  return {
    alreadyRegistered: false,
    shutdown: async () => {
      await provider.shutdown();
      if (meterProvider) await meterProvider.shutdown();
    },
  };
};
