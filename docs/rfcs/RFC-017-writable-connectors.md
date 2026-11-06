# RFC-017：可写连接器（本地文件后端，第一批）

| 字段 | 值 |
|---|---|
| 状态 | **Accepted**（2026-11-05 落地） |
| 起草日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 上游决策 | RFC-016 连接器框架（只读）、RFC-013 审批与能力授权、RFC-004 沙箱边界 |
| 本文目的 | 在只读连接器之上加写操作；写默认不存在，显式 opt-in 后出现且全部走审批 |

---

## 1. 范围

第一批只写**本地文件后端**（与 RFC-016 的只读面同一批数据源）：

| 连接器 | 写工具 | 语义 |
|---|---|---|
| notes | `notes_create` / `notes_append` / `notes_delete` | 新建（拒覆盖）、追加（须存在）、删除移入 `.trash/`（可恢复，不 unlink） |
| calendar | `calendar_create_event` / `calendar_delete_event` | 向 ICS 追加 VEVENT / 按 UID 删除整个事件（周期事件删整列，文档明示） |
| mail | `mail_compose_draft` | 只往**独立的草稿 mbox** 追加一封邮件；**不发送**（真发送是未来的 external 连接器） |

不做：远程写 API（Google/Outlook/IMAP 发送）、ICS 周期事件单次例外（EXDATE）、笔记改名/移动。

## 2. 启用与授权

- 写工具**默认不存在**：notes/calendar 需构造期 `writable: true`；mail 需显式 `draftMboxPath`。
- env 面：`OPENINTJ_CONNECTORS_WRITE=1` 才给 env 装配的 notes/calendar 开写；mail 草稿还需 `OPENINTJ_MAIL_DRAFT_MBOX` 指明草稿文件。两个条件缺一不可（mail）。
- 写工具 descriptor：`permissions: ["filesystem.write"]`、`idempotent: false`。经 RFC-013 `classifyToolRisk` 归类为 **write** 风险；桌面默认 `approvalRisks: ["write","execute"]`，即**第一次写调用先弹审批**，可按工具发 scoped grant 免重复打扰。server 同理取决于部署方配置。
- MCP 工具维持 RFC-016 语义（未知即 write 风险），本 RFC 不改变。

## 3. 安全不变量

- **沙箱**：notes 全部写路径经 `resolveInside`（拒 `..` 逃逸），扩展名白名单与读面一致；正文大小上限 `maxWriteChars`（默认 200_000）。
- **原子性**：ICS 写走「同目录临时文件 + rename」，写前留 `.bak` 副本；并发外部修改不做合并（后写覆盖，`.bak` 可回滚）——单用户本地文件的可接受语义，文档明示。
- **可恢复**：notes 删除 = 移入 `<root>/.trash/<ISO 时间>-<文件名>`；`.trash` 以点开头，列表/搜索天然跳过。
- **不落凭据**：与 RFC-016 一致，连接器只认路径与开关，不读任何账号凭据。
- **失败响亮**：路径逃逸、覆盖已存在笔记、删不存在的日程等一律抛错进 ToolHub 观察，不静默成功。

## 4. 落地（2026-11-05）

- `notes.ts`：`writable` / `maxWriteChars` 选项 + 3 个写工具；`calendar.ts`：`writable` 选项 + 2 个写工具（UID `openintj-<uuid>@openintj.local`，DTSTAMP UTC，原子重写 + `.bak`）；`mail.ts`：`draftMboxPath` 选项 + `mail_compose_draft`（标准 mbox 追加，可被读面 `parseMbox` 读回）。
- `env.ts`：`OPENINTJ_CONNECTORS_WRITE=1` 总开关 + `OPENINTJ_MAIL_DRAFT_MBOX`。
- 读面行为不变；未启用写时 `tools()` 与 RFC-016 完全一致（status 的 toolCount 随写工具增长）。
- 测试：连接器包覆盖写后读回、沙箱逃逸拒绝、拒覆盖、trash 删除、ICS 原子写 + `.bak`、按 UID 删除、草稿追加可解析。
