# @openintj/concurrency

RFC-003 **方向一（多线程 Agent 模型）** 的并发原语库。

> ⚠️ **集成状态**：本包大部分是「实验性原语」。除 `RateLimitedLlmClient` 外，其余原语
> 目前**未接入** Agent 主循环（TAO/ReAct 仍是单线程顺序执行）。它们经过完整单测、可独立使用，
> 但请勿误以为产品已经在用多线程编排。

## 集成状态矩阵

| 导出 | 状态 | 说明 |
|---|:-:|---|
| `RateLimitedLlmClient` | ✅ 已接入 | cli/server/desktop agent 在 `opts.rateLimit` 或 env `OPENINTJ_RATE_LIMIT_QPS` 下包裹 LLM 客户端，做令牌桶限速 |
| `Mutex` | 🧪 实验 | 互斥锁；仅库 + 单测 |
| `Channel` | 🧪 实验 | 有界/无界通道；仅库 + 单测 |
| `ConditionVar` | 🧪 实验 | 条件变量；仅库 + 单测 |
| `AgentPool` | 🧪 实验（可观测） | 多 worker Agent 池；未接入主路径，但已支持 HookBus 可观测 |
| `ForkJoin` | 🧪 实验（可观测） | 分叉/合并并行编排；未接入主路径，但已支持 HookBus 可观测 |
| `Backpressure` | 🧪 实验 | 背压控制；仅库 + 单测 |

## 可观测性（多 Agent / 多线程）

`AgentPool` 与 `forkJoin` 支持注入 `HookBus`（来自 `@openintj/core`），发出生命周期事件：

| 原语 | 事件 | OTel 产出（经 `@openintj/telemetry-otel`） |
|---|---|---|
| `AgentPool` | `pool.beforeJob` / `pool.afterJob` | span `openintj.pool.job`（active/pending/success/duration）+ counter `openintj.pool.jobs{pool,success}` |
| `forkJoin` | `forkjoin.beforeFork` / `forkjoin.afterJoin` | span `openintj.forkjoin`（total/fulfilled/rejected/duration）+ counter `openintj.forkjoin.branches{group}` / `openintj.forkjoin.rejected{group}` |

```ts
const pool = new AgentPool(4, { hooks: agent.hooks, name: "vote-pool" });
const res = await forkJoin(agents, (a) => a.run(q), { hooks: agent.hooks, group: "vote" });
```

不传 `hooks` 时零开销（`HookBus.emit` 无订阅者直接返回）。配合 `attachOtelToHooks` 即可在
任意 OTel 后端看到并发/多 Agent 的 span 树与计数。

## 集成路线（若日后要把实验原语接入产品）

1. **并行子任务**：在 `TaoLoop` 把可并行的 ReAct 子目标用 `ForkJoin` 分发，合并结果后再进入下一轮 TAO。
   需要先在 RFC-001 的停机条件上扩展「并行分支的聚合语义」。
2. **多会话并发**：server 端用 `AgentPool` 管理多个独立会话 Agent 实例，`Channel` 做任务投递，
   `Mutex` 保护共享的持久化层写入。
3. **风险**：并行会打乱 trajectory 的线性可观测性（OTel span 树需要改成并行子 span），
   且记忆写入需要加锁避免竞态。接入前应补「并行执行」的端到端测试与 OTel span 断言。

在落地以上任一条之前，本包按「实验库」对待，不计入产品完成度。
