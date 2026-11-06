# RFC-019：语音输入（STT）与跨端同步（Bundle v1）

| 字段 | 值 |
|---|---|
| 状态 | **Accepted**（2026-11-05 落地） |
| 起草日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 上游决策 | RFC-014 个人档案、RFC-015 Scheduler、RFC-018 通道 |
| 本文目的 | 补两条输入/状态面：语音转文字的提供者抽象；多设备间可控的状态迁移 |

---

## 1. 语音输入（STT）

- 新包 `@openintj/voice`（零依赖）：`SpeechToTextProvider { transcribe({audio, mimeType, language?}), status() }`。
- 实现：`http`（OpenAI 兼容 `POST <base>/audio/transcriptions`，multipart，`whisper-1` 默认模型）与 `mock`（固定文本，开发/测试）。
- 配置只走 env：`OPENINTJ_STT_URL` / `OPENINTJ_STT_KEY` / `OPENINTJ_STT_MODEL`，或 `OPENINTJ_STT=mock`。未配置时宿主面返回 503/`stt_not_configured`，不假装可用。
- 宿主面：三端暴露 `stt`（可为 undefined）；server `POST /api/voice/transcribe`（base64 音频，上限约 25MB）；desktop `voice.transcribe` IPC + preload。【2026-11-06 补：桌面渲染层录音 UI 已落地——ChatPanel 麦克风按钮（点击切换录音、120s 上限、转写回填输入框待用户确认发送，不自动发送）；主进程权限策略只放行主窗口的 media 请求，其余权限全拒；未配置 STT 时给内联配置提示。】

## 2. 跨端同步（Bundle v1）

- 新包 `@openintj/sync`（依赖 profile/scheduler）：`SyncBundle { version: 1, exportedAt, profile: {entries, dismissed}, schedules }`。
- 合并语义（v1，刻意保守）：
  - **档案**：按 (section, key) 比较 `updatedAt`，后写胜；本地 user 条目不被远端 dormant/import 覆盖（沿 RFC-014 规则）；dismissed 墓碑取并集，且墓碑会删本地对应 live 条目（删除可传播）。
  - **定时任务**：按 id 并集，新 id 直接落库，已有 id **保留本地**（v1 不传播任务编辑/删除，文档明示，避免无时钟基准的误删）。
  - 导入的定时任务按引擎既有语义恢复（过期 once 任务会合并补跑一次）。
- 宿主面：三端 `exportSyncBundle()` / `importSyncBundle(bundle)`；server `GET /api/sync/export`、`POST /api/sync/import`；desktop `sync.export` / `sync.import` IPC + preload。传输与存储介质（文件、局域网、云盘）由用户自选，bundle 是纯 JSON。
- 不做：实时同步、CRDT、冲突 UI、记忆片段（memory fragments）同步——记忆涉及向量与隐私面，另行评审。

## 3. 安全

- STT 音频只发部署方 env 指定的端点；key 不落盘。
- SyncBundle 含个人档案明文：导出/导入都是用户显式动作，宿主不自动外发；导入只接受 `version: 1` 结构，非法 bundle 抛错不部分写入（先校验后应用）。
