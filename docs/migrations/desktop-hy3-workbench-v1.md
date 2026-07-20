# Desktop Hy3 与任务工作台迁移

## 模型配置

- Hunyuan 默认端点为 `https://tokenhub.tencentmaas.com/v1`，默认模型为 `hy3`。
- 仅已知退役默认值会自动迁移：
  - `hy3-preview`、`hunyuan-turbos-latest` → `hy3`
  - `https://api.hunyuan.cloud.tencent.com/v1` → TokenHub
- 自定义模型 ID 和自定义端点保持不变。映射发生时会输出脱敏 warning。
- Kimi K3、MiniMax M3、GLM 5.2 的 Key 请在桌面设置中保存；Key 经 Electron
  `safeStorage` 加密，不应继续写入 `config.json`。

## 桌面数据

升级后首次启动会创建 `userData/workbench.sqlite`，并把当前 `workspaceDir` 注册为默认
工作区，同时建立 Inbox 和一个空对话。旧 memory、embedding 与 TaskPool SQLite 文件
不改名、不复制、不删除。

旧版没有持久化聊天历史，因此升级前的 renderer 临时消息无法迁移。之后的消息会保存
role、trace ID、token 数和完成状态。

## 回退

回退旧版本前先退出应用并备份 `workbench.sqlite` 与
`model-credentials.json`。旧版本会忽略这两个新文件；不要删除原有 memory/TaskPool
数据。若需恢复旧 Hunyuan 自定义配置，请显式填写原值，迁移器不会改写未知值。
