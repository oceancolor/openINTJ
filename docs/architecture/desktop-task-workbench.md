# Desktop 多模型任务工作台

> ⚠️ 2026-11-06：内嵌 VSCodium IDE 已整体剥离（基于使用体验与场景判断，IDE 非
> 优先设施）。左区回到自研 `WorkspaceWorkbench`（文件树 + 预览）；本文中涉及
> VSCodium reh-web 的描述均已失效，仅余任务/对话容器模型仍有效。

## 目标

桌面端以“任务”为一级容器：一个任务属于一个工作区，包含多个对话，并可记录一个
TaskPool run。对话独立保存模型 Profile 与消息历史，切换模型不修改全局 Agent 默认值。

## 运行结构

```text
Workspace(rootPath, dataDir?)
  └─ Task(status, taskPoolRunId?)
       └─ Conversation(modelProfileId)
            └─ Conversation journal（`journal_events`，追加写）
                 投影 → ChatPanel / Tao history / Trajectory
            └─ Message 缓存（双写期仍 INSERT；`listMessages` 以 journal 投影为准）
```

- `workbench.sqlite` 位于 Electron `userData`；开发用
  `OPENINTJ_DESKTOP_NO_PERSIST=1` 时使用内存库。
- 首次启动从当前 `workspaceDir` 创建“默认工作区 / Inbox / 新对话”，不移动旧 memory
  与 TaskPool 数据。
- RFC-009：会话事实源是 `journal_events`。`listMessages` 从 `deriveChatMessages` 投影；
  下一轮 Tao / ContextEngine 共用 `deriveHistory(journal)`。`/clear` 追加
  `session.cleared`，兼容期同时 `DELETE FROM messages`。
- 消息查询有界；旧库无 journal 时从 `messages` 一次性 backfill 成 turn 事件。
- memory preflight 通过 `workspace:* / task:* / conversation:*` 标签限定上下文。
- RFC-008：CHAT 在 classifier 前做自适应输入结构化。消息可带
  `messageKind=clarification|answer` 与 `inputStructure`（任务理解卡）；原始用户文本不改写。
  schema `user_version=4`：`journal_events` 表；此前 `user_version=3` 含
  `message_kind` / `input_structure_json` / `workspaces.closed_at`。

## 模型 Profile

`ModelRegistry` 合并内置 Profile 与用户配置，用 Profile ID 缓存客户端。API Key 不进入
`config.json` 或 renderer 状态，而由 `safeStorage` 加密后写入
`model-credentials.json`。保存或删除凭据、修改 Profile 时清除对应缓存。

云模型统一走 OpenAI-compatible adapter；显式 provider 缺凭据或调用失败时 fail closed。
默认 Profile 包含 Hy3、Kimi K3、MiniMax M3、GLM 5.2、Ollama、auto 与开发用 mock。

## 工作区切换和关闭

工作区 root 影响工具权限边界，因此 UI 保存新 root 后执行受控重启来重新装配 Agent。
重启顺序为：拒绝新对话请求、abort 在途请求、注销 IPC/文件监听、关闭 Agent 与 SQLite、
`app.relaunch()`、`app.quit()`。这避免旧工具继续访问前一个 root。

## IPC 信任边界

renderer 仅能通过 Zod 校验的 IPC 请求读写任务数据；凭据接口只返回
`hasCredential`，永不返回明文。任务归档不级联删除对话或消息。
