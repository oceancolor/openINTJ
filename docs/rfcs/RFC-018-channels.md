# RFC-018：消息通道（出站通知）与移动接入

| 字段 | 值 |
|---|---|
| 状态 | **Accepted**（2026-11-05 落地） |
| 起草日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 上游决策 | RFC-015 Scheduler（事件源）、RFC-013 治理（通道不绕过工具面） |
| 本文目的 | 给主动式系统一条出站通道：定时任务/长任务的结果能离开桌面到达人 |

---

## 1. 定位

「移动通道」在当前形态下不是再做一个 App，而是**出站通知通道**：手机通过用户自选的推送网关（ntfy、Server 酱、企业微信机器人等，本质都是一个 webhook）收到消息。通道只负责**发通知**，不接收指令（入站指令面仍是桌面/server API，避免把命令面暴露到公网）。

## 2. 设计

- 新包 `@openintj/channels`（零运行时依赖，`fetch` 直发）：
  - `Channel { id, kind, send(event), status() }`；`ChannelEvent { kind, title, body, at, data? }`。
  - 内置 `webhook`（POST JSON，可选 Bearer token、自定义 header、10s 超时）与 `memory`（测试/观测）两种通道。
  - `ChannelDispatcher`：扇出分发，单通道失败隔离（记录 `lastError`，不影响其他通道与主流程），支持按 `kinds` 过滤、按通道开关。
- 事件源（v1）：Scheduler 事件（fired/completed/failed/deferred → 同名 kind）。后续可加 TaskPool 完成、审批待决等。
- 配置只走 env：`OPENINTJ_CHANNELS` = JSON `[{id, kind:"webhook", url, token?, kinds?}]`。URL/token 不落盘到仓库数据目录（env 即配置，与 RFC-016 同哲学）。
- 宿主面：三端统一暴露 `channels`；server `GET /api/channels` + `POST /api/channels/:id/test`；desktop `channels.list` IPC（暂无面板）。CLI 无观测面（dispatcher 仍工作，失败只记状态）。

## 3. 安全

- 通道只能**发**、不能收；webhook 目标由部署方 env 显式配置，无默认外发地址。
- 事件正文只含任务标题/状态/错误摘要，不带对话全文与凭据。
- 分发失败不阻塞调度主流程（fire-and-forget + 状态可观测）。

## 4. 落地（2026-11-05）

`@openintj/channels` + 三端装配（scheduler 事件订阅在 agent 层，桌面原有 EVT_SCHEDULE UI 推送不变）+ server 路由 + desktop IPC/preload + 包级测试（webhook 分发/过滤/失败隔离，本地 HTTP server 验证）。
