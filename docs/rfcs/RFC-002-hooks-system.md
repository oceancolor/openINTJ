# RFC-002：函数钩子系统

| 字段 | 值 |
|---|---|
| 状态 | Draft |
| 起草日期 | 2026-04-29 |
| 决策日期 | TBD（Phase 1 启动前） |
| 作者 | OpenINTJ Core |
| 上游决策 | 路线图 D2（全生命周期钩子，已确认） |
| 影响包 | `@openintj/core` (`packages/core/hooks/`)、所有 plane 与 LLM 适配器 |

---

## 1. 设计目标

提供一套统一、可类型化、可短路、可改写 payload 的钩子机制，使外部代码（用户插件、监控、策略覆盖、A/B 实验、调试工具）能介入 OpenINTJ 的任意生命周期点而不改框架代码。

**关键质量属性**：

- **强类型**：每个事件类型都对应固定的 payload schema（zod），handler 签名编译期约束
- **优先级**：handler 按 `priority` 数值降序执行（数值越大越先执行；与 v2 Python 治理平面的"先执行后阻断"语义对齐）
- **短路**：handler 可调用 `ctx.cancel()` 阻止后续 handler 执行（用于 policy 拦截）
- **改写 payload**：handler 可返回新 payload，覆盖默认值（用于注入、转换、监控埋点）
- **同步与异步双模**：支持 `void` 与 `Promise<void>` handler 混用，最终 `emit` 永远返回 `Promise`
- **零运行时开销**：未注册 handler 的事件 emit 应是 O(1)

## 2. 钩子分类（4 类）

| 类别 | 用途 | 示例事件 | 短路允许？ |
|---|---|---|---|
| **Lifecycle** | TAO/ReAct 循环各阶段 | `tao.beforeThink` / `tao.afterAct` / `react.beforeAction` 等 | 仅 `react.beforeAction` 允许（跳过该工具） |
| **Tool** | 工具调用前后 + 错误 | `tool.beforeCall` / `tool.afterCall` / `tool.onError` | `tool.beforeCall` 允许（取消调用） |
| **Event** | 业务事件订阅（对齐 v2 Python `EventType`） | `event.MEMORY_LOADED` / `event.CONTEXT_COMPACTED` / `event.POLICY_BLOCKED` 等 | ❌ |
| **Policy** | 治理拦截可改写点 | `policy.beforeCheck` / `policy.afterCheck` / `policy.onBlock` | `policy.beforeCheck` 允许（提前放行/拒绝） |

## 3. 核心 API

### 3.1 类型基础

```typescript
// packages/core/src/hooks/types.ts

export type HookCategory = "lifecycle" | "tool" | "event" | "policy";

export interface HookContext<P> {
  /** 当前事件名（含分类前缀，如 "tao.beforeThink"）。 */
  readonly eventName: string;
  /** 当前 trace id（贯穿一次 TAO 调用）。 */
  readonly traceId: string;
  /** 当前 payload（可被 handler 通过 ctx.replace() 改写）。 */
  payload: P;
  /** 已经执行过的 handler 数（按本次 emit 计）。 */
  readonly executedCount: number;
  /** 短路：阻止后续 handler 执行。仅允许在"短路允许"的事件上调用。 */
  cancel(): void;
  /** 是否已被短路。 */
  readonly isCancelled: boolean;
  /** 用 newPayload 替换当前 payload。后续 handler 看到的就是新值。 */
  replace(newPayload: P): void;
  /** 在事件级 metadata 中读取/写入跨 handler 共享数据（不会被 emit 返回）。 */
  meta: Record<string, unknown>;
}

export type HookHandler<P> = (ctx: HookContext<P>) => void | Promise<void>;

export interface HookRegistration {
  /** handler 优先级，越大越先执行；默认 0。 */
  priority?: number;
  /** 是否仅执行一次（首次执行后自动 unregister）。 */
  once?: boolean;
  /** handler 标签，用于按名称批量取消注册。 */
  tag?: string;
  /** 是否允许在 handler 中短路。仅匹配"短路允许"的事件时该参数才生效。 */
  allowCancel?: boolean;
}

/** unregister 函数；调用后该 handler 立即从 bus 中移除。 */
export type Unregister = () => void;
```

### 3.2 HookBus 接口

```typescript
// packages/core/src/hooks/bus.ts

import type { ZodType } from "zod";

export class HookBus {
  /**
   * 注册一个 handler。返回的 unregister 函数可用于卸载。
   *
   * 类型参数 E 是事件名（编译期常量字符串），P 是其 payload 类型。
   * 通过 ts 模板字面量推导 E → P 的映射，避免运行时错配。
   */
  on<E extends keyof HookEventMap>(
    event: E,
    handler: HookHandler<HookEventMap[E]>,
    opts?: HookRegistration,
  ): Unregister;

  /**
   * emit 事件，返回最终（可能被改写过的）payload。
   *
   * 执行顺序：
   *   1. 按 priority 降序排序所有匹配的 handler
   *   2. 顺序执行；每个 handler 可同步或返回 Promise（自动 await）
   *   3. 任一 handler 调用 ctx.cancel() 即停止后续执行
   *   4. 任一 handler 调用 ctx.replace(newPayload) 即更新 ctx.payload
   *   5. 返回最终 ctx.payload
   *
   * handler 抛异常的处理：
   *   - 默认：catch 后写入 audit trail，继续执行后续 handler（fail-soft）
   *   - 严格模式（`bus.strictMode = true`）：异常向上抛，emit 返回 rejected Promise
   */
  emit<E extends keyof HookEventMap>(
    event: E,
    payload: HookEventMap[E],
    opts?: { traceId?: string },
  ): Promise<HookEventMap[E]>;

  /**
   * 按 tag 批量取消注册。
   */
  offByTag(tag: string): number;

  /**
   * 列出当前所有 handler 的统计。
   */
  inspect(): HookInspectResult;

  /** 严格模式开关。 */
  strictMode: boolean;
}
```

### 3.3 事件类型注册表（HookEventMap）

```typescript
// packages/core/src/hooks/event-map.ts

export interface HookEventMap {
  // -------- Lifecycle (TAO 宏循环) --------
  "tao.beforeThink": { query: string; iteration: number };
  "tao.afterThink": { plan: PlanGraph; iteration: number };
  "tao.beforeAct": { plan: PlanGraph; availableTools: ToolDescriptor[]; iteration: number };
  "tao.afterAct": { reactOutput: ReactOutput; iteration: number };
  "tao.beforeObserve": { trajectory: TrajectoryEntry[]; iteration: number };
  "tao.afterObserve": { needsContinue: boolean; iteration: number };

  // -------- Lifecycle (ReAct 微循环) --------
  "react.beforeThought": { context: ContextWindow; reactIter: number; taoIter: number };
  "react.afterThought": { thought: string; reactIter: number; taoIter: number };
  "react.beforeAction": { tool: string; params: unknown; reactIter: number; taoIter: number };
  "react.afterAction": { toolResult: ToolCallResult; reactIter: number; taoIter: number };
  "react.onStopCondition": { reason: ReactStopReason; reactIter: number };

  // -------- Tool --------
  "tool.beforeCall": { tool: string; params: unknown; toolDescriptor: ToolDescriptor };
  "tool.afterCall": { tool: string; result: ToolCallResult };
  "tool.onError": { tool: string; error: Error; willRetry: boolean };

  // -------- Event (对齐 Python EventType) --------
  "event.MEMORY_LOADED": { count: number; budgetUsage: number };
  "event.CONTEXT_COMPACTED": { compactedMessages: number; newBudgetUsage: number };
  "event.SHADER_APPLIED": { mode: ShaderMode; lod: LODLevel };
  "event.POLICY_BLOCKED": { command: Command; reason: string };
  "event.CIRCUIT_OPENED": { tool: string; failureCount: number };
  "event.LOOP_ITERATION": { taoIter: number; metrics: Record<string, number> };

  // -------- Policy --------
  "policy.beforeCheck": { command: Command };
  "policy.afterCheck": { command: Command; auditEvent: AuditEvent };
  "policy.onBlock": { command: Command; auditEvent: AuditEvent; reason: string };
}
```

新事件由各 plane 通过模块扩展（declaration merging）追加：

```typescript
// packages/planes/memory/src/hooks-augmentation.d.ts
declare module "@openintj/core/hooks" {
  interface HookEventMap {
    "memory.fragmentIngested": { fragment: MemoryFragment; memoryType: string };
    "memory.shaderPipelineProcessed": { input: number; output: number; mode: ShaderMode };
  }
}
```

## 4. 短路语义详细规则

| 事件 | 短路允许？ | 短路效果 |
|---|---|---|
| `tao.before*` / `tao.after*` | ❌ | n/a |
| `react.beforeThought` / `afterThought` | ❌ | n/a |
| `react.beforeAction` | ✅ | **跳过该次工具调用**，直接进入下一轮 Thought（视为该步无 observation） |
| `react.afterAction` | ❌ | n/a |
| `tool.beforeCall` | ✅ | **取消工具调用**；返回 ToolCallResult{success: false, error: "cancelled by hook"} |
| `tool.afterCall` / `onError` | ❌ | n/a |
| `event.*` | ❌ | 事件类钩子是"通知"语义，不允许短路 |
| `policy.beforeCheck` | ✅ | **跳过策略检查**；payload 中可放置 `presetResult: AuditEvent` 直接作为结果 |
| `policy.afterCheck` / `onBlock` | ❌ | n/a |

不允许短路的事件上调用 `ctx.cancel()` 时：

- **非严格模式**：忽略调用并打日志 warn
- **严格模式**：抛 `HookError("cancel not allowed for event 'xxx'")`

## 5. 改写 payload 详细规则

任意事件均允许 `ctx.replace(newPayload)`，但不同事件对"改写后是否被框架读取"有不同语义：

| 事件 | 改写效果 |
|---|---|
| `tao.beforeThink` | ✅ 改写后的 query 用于后续的 JIT 加载 |
| `tao.afterThink` | ✅ 改写后的 plan 直接用于 Act 阶段 |
| `tao.beforeAct` | ✅ 改写后的 availableTools 限定 ReAct 可用工具集 |
| `tao.afterAct` | ✅ 改写后的 reactOutput 写入 Observe 阶段 |
| `react.beforeAction` | ✅ 改写 tool / params 替换实际调用对象（可用于做 sandbox） |
| `tool.beforeCall` | ✅ 改写 params（参数注入/脱敏） |
| `tool.afterCall` | ✅ 改写 result（结果改写/脱敏） |
| `event.*` | ⚠️ 改写不会影响业务，仅供后续 handler 看 |
| `policy.beforeCheck` | ✅ 改写 command（治理前的请求修整） |

## 6. 优先级与 handler 排序

- handler 优先级范围：`-1000 ... +1000`，默认 0
- 推荐分层：

| 优先级 | 用途 |
|---|---|
| `+1000` | 系统级安全拦截（治理强制规则、配额硬限） |
| `+500` | 平台级监控（埋点、审计、tracing） |
| `+100` | 用户级 policy 改写 |
| `0` | 默认普通插件 |
| `-100` | 后置统计/日志 |

排序稳定（同优先级按注册顺序）。

## 7. 类型推导与编译期保障

```typescript
// 用例：类型完全推导
hookBus.on("tao.beforeThink", (ctx) => {
  // ctx.payload 自动推导为 { query: string; iteration: number }
  ctx.payload.query.toUpperCase();        // ✅
  ctx.payload.unknownField;                // ❌ 编译错误
  ctx.replace({ query: "...", iteration: 0 }); // ✅ 类型必须匹配
  ctx.replace({ query: "..." });           // ❌ 缺少 iteration 字段
});

// 错误事件名：编译期阻止
hookBus.on("tao.typo", () => {});         // ❌ keyof HookEventMap 不包含
```

## 8. 异常处理策略

| 模式 | handler throw | 后续 handler | emit 返回 |
|---|---|---|---|
| 非严格（默认） | 写 audit trail（risk_level=warning），打日志 | 继续执行 | 成功，返回最后的 payload |
| 严格 | 立即抛出 | 停止执行 | rejected promise |

audit trail 集成由 `@openintj/plane-governance` 自动注册一个 priority=+500 的 `event.*` 监听器实现，详见 RFC-003。

## 9. 性能要求与基准

- 注册一个 handler：O(log n)（按 priority 插入排序）
- emit 一个无 handler 的事件：O(1)，目标 < 0.5 µs
- emit 含 N 个同步 handler：< 5 µs/handler 框架开销（不计 handler 自身）
- emit 含异步 handler：单次 microtask 跳跃 + handler 自身

`packages/core/__tests__/perf/hook-bus-bench.spec.ts` 用 vitest bench API 跑基准。

## 10. 与 v2.0 Python 的对应关系

| Python v2.0 | TS v3 钩子系统等价物 |
|---|---|
| [agent_loop.py:226-235](../../agent_loop.py) `iteration.events.append(Event(...))` | `bus.emit("event.MEMORY_LOADED", {...})` |
| [governance_plane/__init__.py:82-124](../../governance_plane/__init__.py) `PolicyEngine.check` | `bus.emit("policy.beforeCheck", ...)` + 默认 handler 走原 PolicyEngine 逻辑 |
| [execution_plane/__init__.py:184-222](../../execution_plane/__init__.py) `ToolHub.call` | `bus.emit("tool.beforeCall", ...)` → 实际调用 → `bus.emit("tool.afterCall", ...)` |
| [agent_loop.py:367-372](../../agent_loop.py) Reflect 阶段自适应 ShaderMode 调整 | priority=+500 的内置 handler 监听 `tao.afterObserve` |

## 11. 实现里程碑

1. **M1**：`HookContext` + `HookBus.on/emit/offByTag` + 同步 handler + priority 排序
2. **M2**：异步 handler + cancel/replace + strictMode
3. **M3**：完整 `HookEventMap`（lifecycle + tool + event + policy 全部 30+ 事件）
4. **M4**：性能基准 + 与 v2.0 行为对齐测试
5. **M5**：plane 内置 handler（Audit/Governance 监听 `event.*` 和 `policy.*`）

## 12. 风险与未决问题

- **R1**：handler 死锁（A 监听 X 事件、B 监听 Y 事件，A 在处理 X 时触发 Y）→ 通过同 traceId 内的事件栈深度限制（默认 16）防御
- **Q1**：是否需要"事件路径通配符"（如 `tao.*`）？倾向**不支持**，避免歧义；用户用多个明确订阅替代
- **Q2**：跨进程钩子（如 desktop 的 main process 钩子转发到 renderer）需 RFC-004 IPC 协议配合，本 RFC 不展开
