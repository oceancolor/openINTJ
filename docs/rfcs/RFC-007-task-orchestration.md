# RFC-007: TaskPool 与并发执行

| 字段 | 值 |
|------|-----|
| 状态 | **Implemented（opt-in）** |
| 作者 | OpenINTJ |
| 关联 | RFC-003 方向一/二、ControlPlane 模板 DAG |
| 包 | `@openintj/taskpool` |

## 摘要

在现有 `TaskQueue` / `SharedContext` / `Semaphore` 原语之上，新增 **TaskPool** 门面，将 ControlPlane **模板** `PlanGraph` 转为 `TaskGraph` 并按依赖**有界并行**执行多个 Tao worker。

**首期不做**：LLM 动态拆图、跨进程调度、TaoLoop 内部并行、worker_threads 承载 Agent。

## 领域模型

### 状态机

`pending → ready → running → completed | failed | timed_out | cancelled`

重试回到 `ready`；上游失败级联下游（沿用 `TaskQueue.fail` 语义）。

### 计划来源

仅 **ControlPlane 模板 DAG**（`Planner` + `GoalParser`），经 `planGraphToTaskGraph()` 适配。不允许 LLM 动态生成 DAG（后续 RFC）。

### 并行边界

- 每个可运行 task 委托一次 `tao.run()`（线性 Tao/ReAct）
- TaskPool 调度器限制并发（默认 3），测试锁定实际 peak
- 结果经 `synthesizeTaskPoolAnswer()` 合成单答案

## Opt-in

- `OPENINTJ_TASK_POOL=1` 或 `enableTaskPool: true`
- 仅 `planning` / `analysis` 分类且分类器启用时走 TaskPool
- 关闭时行为零变化

## 可观测性

Hook 事件：

- `taskpool.run.submit` / `taskpool.run.complete`
- `taskpool.task.ready` / `start` / `retry` / `timeout` / `cancel` / `complete`

并行 Tao 共享 `runId`；OTel 创建 run/task spans，并以 `taskpool.run_id` 显式关联
各 worker trace，同时记录 run/task/retry/timeout/cancellation counters。

## 已实现阶段

1. **MVD hardening**：`TaskRun` handle、完整节点状态机、稳定拓扑/合成、partial
   `SharedContext`、图验证、失败/取消级联与并发 peak 测试。
2. **可靠性**：`AbortSignal` 从 TaskPool 贯通 Tao/ReAct/tool 及 `LlmClient.chat` 调用边界，
   Ollama/Hunyuan 会中止在途 provider fetch；调用方取消与 provider timeout 分离；
   per-task timeout、watchdog、有界指数 backoff 重试。
3. **三端 parity**：CLI `--task-pool`、server env/config/status、desktop AppConfig/Settings；
   TaskPool opt-in 自动启用其必需的 classifier，符合条件时明确优先于 self-consistency，
   默认简单路径不变。状态统一暴露 requested/active/reason、classifier prerequisite 以及
   persistence/recovery capability；CLI 明确不提供 SQLite restart recovery。
4. **生产持久化**：`TaskStore` 抽象与 `@openintj/storage-sqlite` 的 `SqliteTaskStore`；
   server/desktop 在 opt-in + real data dir 下启动扫描 incomplete runs。快照持久化原始
   `goalInput`，completed partial results 可跨重启复用。默认安全取消遗留 run；
   仅显式 `OPENINTJ_TASK_POOL_RECOVERY=resume` 时自动重跑未完成节点，旧版缺输入快照拒绝 resume。
5. **多 Agent 基础**：role-based `AgentInstancePool` 与 Zod `Channel`/typed reducer；
   保持 opt-in，不复用 `ObjectPool`，不接默认路径。

## 外部/后续边界

- LLM 动态生成/修订 DAG 需要独立安全与验证 RFC。
- 跨进程、分布式或 Kubernetes 调度不在本 RFC 范围。
- 内置 Ollama/Hunyuan adapter 已消费 `ChatOptions.signal` 并中止在途 fetch；后续新增
  provider 必须遵守同一取消契约。TaskPool watchdog 继续作为不遵守 signal 的第三方实现兜底。
- 自动 resume 不提供 exactly-once 外部副作用保证，因此默认策略为 cancel。只有调用方确认
  worker 可安全重放时才应启用 resume。
- 多 Agent 默认启用仍需真实角色策略、prompt 隔离、权限与成本基准。

## 验收

- 钻石 DAG 测试：依赖顺序与并行峰值正确
- TaskPool 关闭时零行为变化
- 开启时并发不超过配置上限
- timeout/cancel/retry、失败级联、SQLite restart recovery 均有确定性测试
- TaskPool cancel 可中止 ReAct 内在途 LLM 请求，Ollama/Hunyuan adapter 有 fetch abort 回归测试
- server 显式 resume 与 desktop 默认 cancel 均有真实 SQLite 应用级 E2E
