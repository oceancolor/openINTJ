# RFC-015：Scheduler + 持久任务 + 通知

| 字段 | 值 |
|---|---|
| 状态 | **Accepted** |
| 起草日期 | 2026-11-05 |
| 决策日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 依赖 | RFC-009 Journal、RFC-010 AgentHandle、RFC-013 授权、RFC-014 档案 |
| 后继 | 连接器定时拉取（RFC-016 予定）、按档案时区/目标排程 |

---

## 摘要

系统目前是纯反应式的：用户不说话，agent 什么都不做。Personal Agent 的
分水岭是**主动性**——到点提醒、定期汇总、离线期间攒下的任务在回来时补跑。

本 RFC 加一个 `@openintj/scheduler` 包：持久化的定时任务（job），到点经
`AgentHandle.run()` 把 prompt 作为新一轮对话投递，运行结果回写任务状态，
并通过宿主通知面（桌面事件 / server 查询）让用户看见发生了什么。

## 1. 数据模型

```typescript
type ScheduleTrigger =
  | { kind: "once"; at: number }                        // 一次性（提醒）
  | { kind: "interval"; everyMs: number; startAt?: number }
  | { kind: "daily"; time: "09:00" };                   // 本地时间 HH:mm

interface ScheduledJob {
  id: string;
  title: string;
  prompt: string;              // 到点投递给 agent 的指令
  conversationId?: string;     // 缺省宿主默认会话（"scheduler"）
  trigger: ScheduleTrigger;
  enabled: boolean;
  maxRuns?: number;            // 成功运行上限，达到后自动停用
  createdAt: number;
  lastRunAt?: number;
  nextRunAt?: number;
  runCount: number;            // 成功次数
  lastStatus?: "completed" | "failed";
  lastError?: string;
}
```

v1 不做 cron 表达式、不做每任务时区（用宿主本地时间；按 RFC-014 档案时区
排程留给后续）。`once` 任务触发后自动停用。

## 2. 引擎语义

- **Tick 引擎**：可注入时钟 `now()`；`start()/stop()` + 手动 `tick()`（测试用）。
  同一时刻多个到期任务串行执行；同一任务不重入。
- **错过补跑（catch-up）**：进程离线期间错过的触发点**只补跑一次**
  （coalesce），不按错过次数连发；补跑事件带 `lateMs` 供 UI 标注「迟到」。
- **安静时段（quiet hours）**：可配 `{start,end}`（本地时间，可跨午夜）；
  命中时任务顺延到安静时段结束，发 `deferred` 事件，不丢弃。
- **失败重试**：失败后按 `retryDelayMs × 2^n` 退避重试（默认最多 2 次、
  60s 起）；重试用尽后按触发器算下一次。重试计数只在内存，重启清零，
  任务本体不丢。
- **治理**：定时运行就是普通 `run()`——走 Journal、治理配额与 RFC-013
  审批。无人值守时需审批的工具调用会被 fail-closed 拒绝，任务记 `failed`，
  这是预期行为（用户回来处理待审批列表）。
- 事件：`fired / completed / failed / deferred`，经 `scheduler.on(fn)`
  订阅；宿主负责转发到通知面。

## 3. 存储

`ScheduleStore` 接口 + `InMemoryScheduleStore`（CLI/测试）+
`JsonScheduleStore`（server/desktop：`<dataDir>/schedules.json`，整体落盘；
任务量级是几十条，够用）。

## 4. 宿主面

| 宿主 | 存储 | 入口 |
|---|---|---|
| Desktop | `schedules.json` | IPC `schedule.list/add/update/remove` + `EVT_SCHEDULE` + 「定时」页；main 启动时 `scheduler.start()` |
| Server | `schedules.json` / 内存 | `GET/POST /api/schedules`、`PATCH/DELETE /api/schedules/:id`；进程启动时 start |
| CLI | 内存 | `agent.scheduler` 编程面 |

三端 agent 统一暴露 `scheduler`（`SchedulerService`），与 `profile`/`grants`
并列。默认启用；`OPENINTJ_SCHEDULER=0` 时 `start()` 为 no-op（CRUD 仍可用）。
Runner 统一经 handle：`handle.run(job.prompt)`，结果 `status !== completed`
记 failed。任务投递的对话轮次进 Journal（conversationId = 任务指定会话）。

## 5. 验收

- 单测（假时钟 + 手动 tick）：once 到点触发并停用；interval 周期推进；
  daily 跨天计算；错过补跑只跑一次且带 lateMs；安静时段顺延；失败退避
  重试并在用尽后记 failed；maxRuns 停用；JSON store 往返。
- 接线：agent 暴露 scheduler；runner 经 handle.run 投递且状态回写。
- 桌面契约：schedule IPC CRUD + 事件转发；server 路由 CRUD。

## 6. 不做

- 不做 cron 表达式 / 每任务时区 / 随机抖动（v1）。
- 不做系统级唤醒：桌面应用关闭后任务不跑（托盘常驻是后续桌面议题）；
  重启后靠 catch-up 补跑一次。
- 不做任务间依赖编排（那是 TaskPool 的领域）。
- 通知 v1 只做宿主内事件面（桌面事件 + 面板、server 轮询），
  推送/邮件/消息通道归 P2 渠道层。

## 7. 落地记录（2026-11-05）

- 新包 `@openintj/scheduler`（零运行时依赖）：`types` / `store`
  （内存 + `JsonScheduleStore` 写 `<dataDir>/schedules.json`）/ `engine`
  （`computeNextRunAt`、安静时段纯函数）/ `service`（tick 引擎：catch-up
  合并、安静时段顺延、退避重试、maxRuns、事件订阅）。10 个单测全过
  （含假时钟补跑/顺延/重试场景）。
- 三端接线：agent 统一暴露 `scheduler`，runner 经 `handle.run(job.prompt)`
  投递（server 单 handle；desktop `handleFor(conversationId ?? "scheduler")`；
  结果非 completed 记 failed）。server `close()` / desktop `close()` 停引擎；
  `OPENINTJ_SCHEDULER=0` 时 `start()` no-op。
- 宿主：server 启动时 `scheduler.start()` 并把 fired/failed 打日志，提供
  `/api/schedules` CRUD；desktop main 在 IPC 注册后 `start()`（热重装时旧
  agent 的 close 已停、新 agent 再 start），IPC `schedule.list/add/update/
  remove` + `EVT_SCHEDULE` 转发，renderer 新增「定时」页（新建/启停/删除 +
  事件实时刷新）。registerIpcHandlers 会把 runner 换成 workbench journal
  感知版，定时轮次直接落目标会话。
- 验证：scheduler 10 / CLI 45 / server 69 / desktop 100 测试全过；
  五包 `tsc --noEmit` 全过；biome check 已跑。
- 已知取舍：重试计数只在内存（重启清零，任务本体不丢）；桌面应用关闭后
  不唤醒，靠重启 catch-up 补跑一次；定时运行需审批的工具会被 fail-closed
  拒绝并记 failed，等用户处理待审批列表（RFC-013 预期行为）。
