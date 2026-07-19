# @openintj/taskpool

RFC-003 检索原语与 RFC-007 有界任务编排库。TaskPool 已作为三端 opt-in
产品路径接入；默认关闭，因此简单路径保持不变。

## 集成状态矩阵

| 导出 | 状态 | 说明 |
|---|:-:|---|
| `HybridRetriever` | ✅ 已接入 | server/desktop agent 在 `opts.retrievalMode='hybrid'` 或 env `OPENINTJ_RETRIEVAL_MODE=hybrid` 下做向量 + 关键词混合检索 |
| `SharedContext` | 🧪 实验 | 跨任务共享上下文；仅库 + 单测 |
| `TaskQueue` | 🧪 实验（可观测） | 优先级任务队列；未接入主路径，但已支持 HookBus 可观测 |
| `ObjectPool` | 🧪 实验 | 对象复用池；未接入主路径 |
| `TaskPool` / `TaskRun` | ✅ opt-in | 模板 DAG 状态机、有界并发、取消/超时/重试、失败级联 |
| `TaskStore` | ✅ opt-in | 抽象位于本包；SQLite 实现在 `@openintj/storage-sqlite` |
| `AgentInstancePool` / `Channel` | 🧪 opt-in | 角色实例边界与 Zod 约束 reducer；不进入默认 run path |

## TaskPool

启用方式：CLI `chat --task-pool`、`OPENINTJ_TASK_POOL=1`、server/desktop
`enableTaskPool: true`。仅分类为 `planning` / `analysis` 时生效。若同时启用
self-consistency，符合条件的 TaskPool 明确优先；其他任务仍走 self-consistency。

`TaskPool.submit()` 返回可取消的 `TaskRun` handle，`submitRun()` 是等待结果的兼容
便捷方法。节点状态为
`pending → ready → running → completed|failed|timed_out|cancelled`，重试回到
`ready`。拓扑与结果合成顺序稳定；缺失依赖、重复 id、环会在运行前拒绝。

worker 通过 `TaskWorkerContext.signal` 接收 cooperative cancellation。每节点可用
`timeoutMs` 覆盖默认 watchdog。成功的 partial result 同时写入
`SharedContext` 的 `task:<id>:result`。

真实 data dir 且 TaskPool 开启时，server/desktop 使用 SQLite 快照；默认关闭或
memory 模式不会创建数据库。`listIncompleteRuns()` 与 `TaskPool.recover()` 用于重启恢复。

## 可观测性

`TaskQueue` 支持注入 `HookBus`，发出任务生命周期事件：

| 事件 | 时机 | OTel 产出 |
|---|---|---|
| `task.enqueue` | submit | counter `openintj.task.enqueued{queue,ready}` |
| `task.beforeRun` | dequeue 取出（state→running） | span `openintj.task.run` 开始 |
| `task.afterRun` | complete / fail | span 结束（success/duration）+ counter `openintj.task.completed{queue,success}` |

```ts
const q = new TaskQueue({ hooks: agent.hooks, name: "dag" });
```

事件在 mutex 临界区**外**发出，避免 handler 再入队列导致死锁；不传 `hooks` 时零开销。

TaskPool 另发出 `run.submit/complete` 与
`task.ready/start/retry/timeout/cancel/complete`。OTel 产出 run/task spans、runId
关联属性及 run/task/retry/timeout/cancellation counters。

## 边界

仍不包含 LLM 动态拆图、跨进程/分布式调度、Kubernetes、worker_threads Agent
承载。多 Agent 原语保持显式 opt-in，等待真实角色策略和安全模型后再接默认路径。
