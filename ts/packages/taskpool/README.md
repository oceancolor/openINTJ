# @openintj/taskpool

RFC-003 **方向二（任务池 + 检索）** 的原语库。

> ⚠️ **集成状态**：除 `HybridRetriever` 外，本包其余原语目前**未接入** Agent 主循环
> （单 Agent 会话不消费任务队列 / 对象池）。它们经过完整单测、可独立使用，但请勿误以为
> 产品已经在用任务池编排。

## 集成状态矩阵

| 导出 | 状态 | 说明 |
|---|:-:|---|
| `HybridRetriever` | ✅ 已接入 | server/desktop agent 在 `opts.retrievalMode='hybrid'` 或 env `OPENINTJ_RETRIEVAL_MODE=hybrid` 下做向量 + 关键词混合检索 |
| `SharedContext` | 🧪 实验 | 跨任务共享上下文；仅库 + 单测 |
| `TaskQueue` | 🧪 实验（可观测） | 优先级任务队列；未接入主路径，但已支持 HookBus 可观测 |
| `ObjectPool` | 🧪 实验 | 对象复用池；未接入主路径 |

## 可观测性（多任务）

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

## 集成路线（若日后要把实验原语接入产品）

1. **任务编排**：把用户的一次复杂请求拆成多个子任务投进 `TaskQueue`，由 worker 消费；
   `SharedContext` 承载子任务间的中间结果。需要先定义子任务的依赖图与失败传播策略。
2. **对象复用**：用 `ObjectPool` 复用昂贵资源（如 embedder / DB 连接）。
   当前这些资源在 agent 装配期单例创建，复用收益有限，优先级低。
3. **检索质量**：`HybridRetriever` 已接入，但目前每次检索重建索引；
   下一步应让索引随 `PersistentMemoryStore` 增量维护（见 next-session §8 检索基准）。

在落地以上任一条之前，`SharedContext / TaskQueue / ObjectPool` 按「实验库」对待，不计入产品完成度。
