# Phase 3.8 —— Hooks → OpenTelemetry（#7）

> 状态：**已收官**（2026-05-20）  
> 仓库标签：`v3.0.0-alpha.8`  
> 覆盖路线图：[`phase2-complete.md` §九](./phase2-complete.md#九未完成--后续路线) #7

---

## 一、目标

`HookBus` 自 Phase 2 起就承担 agent 内部的"事件总线"角色——每条
TAO / ReAct / Tool / Policy 关键路径都已发事件，但全是**广播**形态：
desktop renderer 收一份做轨迹面板，audit 收一份记日志，
**没有人在生产路径上把它转成可观测信号**。

Phase 3.8 给 hooks 系统补一条**官方观测出口**：把事件流自动翻译成
OpenTelemetry 的 **span 树** 和 **counter metric**，让你能：

- 在 Jaeger / Tempo / Honeycomb 看到一次 `agent.run()` 完整的
  TAO → ReAct → Tool 树（含 parent / child / duration / status）
- 在 Prometheus / OTLP Collector 看到 iteration / action / tool /
  policy.blocked / memory.loaded 等计数
- **不付任何代价**：不显式启用时整条路径走 OTel API 的 NoopTracer，
  对业务 latency 影响接近 0

## 二、设计要点

### 1. 选型：直接挂在 hook handler，不进改业务代码

替代方案对比：

| 方案 | 优点 | 缺点 | 决定 |
|---|---|---|---|
| OTel Auto-Instrumentation (`@opentelemetry/instrumentation-*`) | 零代码 | 没有 LLM agent loop 的标准 instrumentation | 不适用 |
| 在 `TaoLoop` / `Executor` / `ToolHub` 里加 `tracer.startActiveSpan` | 集成深 | 改业务、强耦合 SDK、未启用时仍要写代码 | 不选 |
| **Hook adapter（本方案）**：`attachOtelToHooks(bus, opts)` | 业务零侵入、可独立测试、未启用零开销 | 需要 hook 已经发了对应事件（已发） | **选** |

业务代码完全不动；新增的 `@openintj/telemetry-otel` 包只读 hook 事件、写 OTel。

### 2. 包结构

```
packages/telemetry/otel/
  src/
    attach.ts       # attachOtelToHooks（核心）
    bootstrap.ts    # bootstrapNodeOtel（可选 SDK 引导）
    index.ts
  __tests__/
    noop.spec.ts    # 未注册 provider 时不抛、不产 span
    spans.spec.ts   # 用 InMemorySpanExporter 断言 span 树 + ERROR 状态
    metrics.spec.ts # 用 InMemoryMetricExporter 断言 counter
    dispose.spec.ts # dispose() 兜底 end 未结束 span / unregister 所有 handler
```

依赖切分：

- **运行时硬依赖**：`@openintj/core` + `@opentelemetry/api`（~50KB）
- **可选 peer**（SDK 包）：`sdk-trace-node` / `sdk-trace-base` / `sdk-metrics` /
  `exporter-trace-otlp-http` / `resources` / `semantic-conventions`
  - 全部标 `peerDependenciesMeta.optional: true`
  - 调用方只调 `attachOtelToHooks` 时**完全不需要**这些 SDK 包
  - 仅 `bootstrapNodeOtel` 通过懒 `await import(...)` 用；缺包就 throw

### 3. Span 树设计

```
openintj.tao.iteration          ← per tao.beforeThink/afterObserve
  attribute: tao.iter, tao.query.length, react.status, react.iterations,
             tao.needs_continue, trace_id (= HookBus traceId)
  └─ openintj.react.action      ← per react.beforeAction/afterAction
       attribute: react.tool, react.iter, tao.iter, react.result.success
       └─ openintj.tool.call    ← per tool.beforeCall/afterCall
            attribute: tool.name, tool.success, [tool.will_retry on error]
            status: ERROR + recordException(err) on tool.onError
```

**parent/child 关系**靠"per-traceId 帧栈"：
adapter 内维护 `Map<traceId, { iteration, action, tools: Map }>`；
新 span 创建时从该 trace 的当前活动帧拿 OTel `Context` 当父；
关 span 时按事件配对 end。

**dispose 兜底**：HookBus.traceId 不一定每条都正常收尾（中途抛错 / desktop 关窗）。
`dispose()` 会强制 end 所有还开着的 span 并打 `disposed=true` 属性。

### 4. Metric counter 设计

| 名称 | 描述 | 属性 |
|---|---|---|
| `openintj.tao.iterations` | TAO 宏循环迭代次数 | `needs_continue` |
| `openintj.react.actions` | ReAct 微循环 action 次数 | `tool`, `success` |
| `openintj.tool.calls` | ToolHub 调用次数 | `tool`, `success` |
| `openintj.tool.errors` | ToolHub 错误次数 | `tool`, `will_retry` |
| `openintj.policy.blocked` | Governance 拦截次数 | `reason`（截 64 字） |
| `openintj.memory.loaded` | Memory 加载的 fragment **总数**（累加 count） | — |

### 5. 启用门槛：opt-in 三通道

- **代码**：`assembleServerAgent({ enableOtel: true })` /
  `assembleServerAgent({ enableOtel: { tracerName: '...', disableMetrics: true } })`
- **env**：`OPENINTJ_OTEL=1`（自动给 `enableOtel: {}`）
- **显式关**：`enableOtel: false`（即便 env 设了也不挂）

未启用时 `agent.otel === undefined`，hook 路径上没有任何额外 handler；
启用时 `agent.close()` 自动调 `otel.dispose()`。

### 6. 真要 export 的话：`bootstrapNodeOtel`

```ts
import { bootstrapNodeOtel, assembleServerAgent } from "...";

const otel = await bootstrapNodeOtel({
  serviceName: "openintj-server",
  otlpEndpoint: process.env.OTEL_EXPORTER_OTLP_ENDPOINT,
  enableMetrics: true,
});
const agent = await assembleServerAgent({ enableOtel: true });
// ... agent.run(...)
await agent.close();
await otel.shutdown();
```

- **idempotent**：底层用 `ProxyTracerProvider` 探针检测全局 provider
  是否已经注册（traceId === all-zeros = 未注册）；已注册时 `bootstrap` 返回
  `alreadyRegistered: true` 而不重复 register
- **懒 import**：所有 SDK 包通过 `await Promise.all([import(...)])` 加载，
  缺包时这一行才抛错，不污染 `attachOtelToHooks` 的零开销路径

## 三、文件清单

### 新增

- `ts/packages/telemetry/otel/package.json`（含 6 个 OTel peerDep 全标 optional）
- `ts/packages/telemetry/otel/tsconfig.json`
- `ts/packages/telemetry/otel/src/attach.ts`（**核心，~290 行**）
- `ts/packages/telemetry/otel/src/bootstrap.ts`（~100 行，懒 import）
- `ts/packages/telemetry/otel/src/index.ts`
- `ts/packages/telemetry/otel/__tests__/noop.spec.ts`（**2 tests**）
- `ts/packages/telemetry/otel/__tests__/spans.spec.ts`（**2 tests**）
- `ts/packages/telemetry/otel/__tests__/metrics.spec.ts`（**3 tests**）
- `ts/packages/telemetry/otel/__tests__/dispose.spec.ts`（**3 tests**）
- `ts/apps/server/__tests__/otel-wiring.spec.ts`（**4 tests**）
- `docs/architecture/phase3-8-otel.md`（本文）

### 改动

- `ts/pnpm-workspace.yaml`：加 `packages/telemetry/*`
- `ts/tsconfig.json`：refs 加 `packages/telemetry/otel`
- `ts/apps/server/{package.json, tsconfig.json, src/agent.ts}`：
  - dep + ref 加 `@openintj/telemetry-otel`
  - devDep 加 `@opentelemetry/{api,sdk-trace-base}`（仅 wiring 测试用）
  - `ServerAgentOpts.enableOtel` + `resolveOtel(...)` + `agent.otel`
  - `close()` 加 `otel?.dispose()`
- `ts/apps/desktop/{package.json, tsconfig.json, src/main/agent.ts}`：
  - dep + ref 加 `@openintj/telemetry-otel`
  - `DesktopAgentOpts.enableOtel` + `resolveDesktopOtel(...)` + `agent.otel`
  - `close()` 加 `otel?.dispose()`

## 四、验证（本地，Windows 11 / Node 22）

```
pnpm lint                                       # exit 0（仍是 2 条 pre-existing useExhaustiveDependencies warn）
pnpm exec turbo run typecheck --concurrency=1   # 35/35 successful（含新 telemetry-otel）
pnpm exec turbo run test --concurrency=1        # 35/35 successful，444 passed + 11 skipped
```

净增 14 个 test：
- telemetry-otel: 10（noop 2 + spans 2 + metrics 3 + dispose 3）
- apps/server otel-wiring: 4

## 五、关键陷阱

1. **HookBus traceId 是 UUID 字符串；OTel traceId 是 128-bit hex**。
   两者**不**相同！本适配器把 HookBus traceId 写到 `trace_id` 属性（带下划线），
   方便在 trace view 里反查 agent log。OTel 自己的 traceId 由 SDK 生成、自动传播。
2. **`tool.beforeCall` / `tool.afterCall` emit 时务必透传 `traceId`**。
   ToolHub 真实代码已经这样做了（参考 `tool-hub.ts:166`）；
   写 hook 单测时若漏传，tool span 会挂在 `'anon'` trace state 上，
   找不到 iteration 父节点 → 测试 parent 断言失败。
3. **`tool.onError` 不会 end span**。设计上让 `tool.afterCall` 来统一 end，
   保持 happy-path 一致；如果业务不会重试也不发 afterCall（理论上不会），
   `dispose()` 兜底。
4. **InMemoryMetricExporter 默认是 DELTA**（构造参数 = `AggregationTemporality.DELTA = 0`）。
   `metrics.spec.ts` 显式传 0 是为了让多个 emit 之间不会 reset 计数；
   切到 CUMULATIVE 会丢失中间增量。
5. **`bootstrapNodeOtel` 的 SDK 是懒 import**。`@opentelemetry/sdk-trace-node`
   等不在 `dependencies` 也不在 `peerDependencies` 的必装项里。生产部署如果
   想用 OTLP 导出，consumer 自己装：
   ```
   pnpm add @opentelemetry/sdk-trace-node @opentelemetry/exporter-trace-otlp-http \
            @opentelemetry/resources @opentelemetry/semantic-conventions
   ```
6. **`attachOtelToHooks` 的 handler 全部同步 + try/catch 兜底**。
   绝不在 emit 路径上 await，绝不让 telemetry 错抖出业务。HookBus 默认非严格
   模式也会吞错，但我们自己也加了 try/catch，是双保险。
7. **`disabled provider 不等于 ProxyTracerProvider`**。`trace.disable()` 会
   把全局回退到一个真正的 NoopTracerProvider；而 `setGlobalTracerProvider(...)` 之前
   是 `ProxyTracerProvider`（delegate=undefined）。两者的 `getTracer().startSpan().spanContext().traceId`
   都是全零，所以 `isPlaceholderProvider` 探针对二者都返回 true，这是预期。

## 六、性能体感

- Hook handler 是同步的 + try/catch，**单次 emit 增量约 5-15μs**（实测桌面/server 装配下）
- 未启用时 `attachOtelToHooks` 不被调用 → handler 链长度 +0
- 启用但未注册 provider 时 OTel API 返回 NoopTracer/NoopMeter → spans 是空对象，setAttribute 是 noop

## 七、下一步候选

- **#7 衍生 / Phase 3.8.1**：自动接入 Hono / Electron IPC instrumentation
  （让 HTTP route 和 IPC channel 自己产 span，再链到 agent 的 hook span 树上）
- **OTel logs**：把 governance audit event / hook `console.warn` 走 OTel Logs API
- **更多 attribute**：当前 attribute 偏少（避免 cardinality 爆炸）；将来按 metric 维度需求逐个补
