# Desktop 多模型任务工作台

## 目标

桌面端以“任务”为一级容器：一个任务属于一个工作区，包含多个对话，并可记录一个
TaskPool run。对话独立保存模型 Profile 与消息历史，切换模型不修改全局 Agent 默认值。

## 运行结构

```text
Workspace(rootPath, dataDir?)
  └─ Task(status, taskPoolRunId?)
       └─ Conversation(modelProfileId)
            └─ Message(role, content, messageKind?, inputStructure?, traceId?, tokens?, status)
```

- `workbench.sqlite` 位于 Electron `userData`；开发用
  `OPENINTJ_DESKTOP_NO_PERSIST=1` 时使用内存库。
- 首次启动从当前 `workspaceDir` 创建“默认工作区 / Inbox / 新对话”，不移动旧 memory
  与 TaskPool 数据。
- 消息查询有界；执行时把既有 user/assistant 消息作为 Tao/ReAct history。
- memory preflight 通过 `workspace:* / task:* / conversation:*` 标签限定上下文。
- RFC-008：CHAT 在 classifier 前做自适应输入结构化。消息可带
  `messageKind=clarification|answer` 与 `inputStructure`（任务理解卡）；原始用户文本不改写。
  schema `user_version=2` 对旧库做 `message_kind` / `input_structure_json` 列迁移。

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
