# RFC-009：Session Journal（会话日志为唯一事实源）

| 字段 | 值 |
|---|---|
| 状态 | **Accepted** |
| 起草日期 | 2026-11-02 |
| 决策日期 | 2026-11-03 |
| 作者 | OpenINTJ Core |
| 上游决策 | 从 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 吸收「model-visible 必须入日志 + `deriveMessages` 投影」；**不**引入 Cordis / JSONL 多代迁移 |
| 影响包 | `@openintj/core`、`@openintj/storage-sqlite`、三端 `agent.run`、Desktop workbench |
| 依赖 | RFC-001 TAO/ReAct、ADR-001 文本协议、RFC-004 IPC、桌面任务工作台 |
| 后继 | RFC-010 Agent Handle / 斜杠命令、RFC-011 统一装配 |

---

## 摘要

今天一次对话的「事实」散落在三处：桌面 SQLite `messages`、进程内 ReAct `trajectory`、易失的 TAO/ReAct IPC 推送。`/clear`、切会话、崩溃恢复对这三套对不齐。

本 RFC 规定：**每个 conversation 有一份追加写的 Session Journal**。TAO/ReAct 每一步把模型可见的事实 append 进去；下一轮 history **只**从 journal 投影；Chat / 轨迹面板 / 审计都是投影，不是第二份权威存储。

循环语义不变（仍是 RFC-001 + ADR-001）。本 RFC 不换 loop、不引入 function-calling。

---

## 1. 问题

| 现状 | 后果 |
|---|---|
| `TaoLoop.run({ history })` 与 `ContextEngine.build({ history })` 各收一份 `ChatMessage[]` | 两处投影会漂 |
| Desktop `messages` 只存 user/assistant/system，不含 thought / tool | TrajectoryPanel 刷新即空 |
| `clearMessages` 只删 SQLite 行 | 内存轨迹、记忆侧栏、journal 未来事件无法统一「清空」 |
| 无 turn/step 边界 | 无法对齐 dsh 的「一轮用户输入 = turn，一次 LLM+工具 = step」 |

---

## 2. 词汇映射（不改循环）

```text
OpenINTJ                         Journal
────────────────────────────────────────────────
一次 agent.run() / 一次用户发送    turn
ReAct 一轮（一次 LLM ± 一个 Action） step
Thought: / Action: / FINAL:      事件（见 §3）
观察（tool 结果）                  tool.result
```

外层 TAO 多轮时：同一 `turnId` 下多个 step；`turn.end` 在整次 `run()` 结束（含 RFC-008 clarify 短路）时写入。

---

## 3. 事件模型

权威类型放在 `@openintj/core`（zod）。追加写，**禁止 update / 改写已提交事件**。清空不是 delete-in-place，而是 append `session.cleared`；投影从该序号之后重算（物理 GC 见 §7）。

```typescript
type JournalEventType =
  | "turn.start"
  | "user"
  | "react.thought"
  | "tool.call"
  | "tool.result"
  | "assistant"
  | "turn.end"
  | "session.cleared";

interface JournalEventBase {
  seq: number;                 // 会话内单调递增
  id: string;
  conversationId: string;
  turnId?: string;
  step?: number;               // ReAct iteration，从 1 计
  traceId?: string;
  createdAt: number;
}

// 载荷（与 base 合并）
// turn.start    { queryPreview?: string }
// user          { content: string; originalInput?: string; inputStructure?: InputStructure }
// react.thought { content: string }
// tool.call     { tool: string; params: unknown }
// tool.result   { tool: string; ok: boolean; observation: string; recoverable?: boolean }
// assistant     { content: string; messageKind: "message" | "answer" | "clarification"; status?: string }
// turn.end      { status: "completed" | "failed" | "cancelled" | "clarify" | "max_iter" | "timeout" }
// session.cleared { by: "command" | "system"; command?: "/clear" }
```

**model-visible 必须入日志。** 新的会进下一轮 LLM 的内容（用户句、FINAL、工具观察、注入的系统提示变更）都要有对应事件。仅 UI 用的流式增量不属于本 RFC（见 RFC-010 `ASSISTANT_DELTA`）；delta 在 turn 结算后折叠进 `assistant` / `react.thought`。

---

## 4. 投影

### 4.1 `deriveHistory(conversationId, opts?)`

ReAct / Tao 的**唯一**历史入口。默认：

- 从最后一次 `session.cleared`（若无则从 seq=1）之后折叠
- 每个已结束 turn：一条 `user` + 一条 `assistant`（clarify 也算 assistant，`messageKind=clarification`）
- 进行中的 turn：包含已提交的 `user`，以及本 turn 内已完成的 `tool.result` 对应的 `{ role: "tool" }`（若 ReAct 仍在跑，由 RFC-010 的句柄使用；本 RFC 的 `run()` 在 turn 结束后才给下一 `run()` 用）
- **不**把 `react.thought` / `tool.call` 编进下一轮 history（避免 Thought 标记污染下一 turn；轨迹走 4.2）

`ContextEngine.build` 与 `TaoLoop.run` 必须调用同一函数，禁止再各传一份手拼 `ChatMessage[]`。

### 4.2 轨迹投影

`react.thought` + `tool.call` + `tool.result` + `assistant` 按 seq 排序 → TrajectoryPanel / `onTaoEvent` / `onReactEvent` 的持久化源。进程重启后轨迹可从 journal 重建。

### 4.3 工作台消息投影

Desktop ChatPanel 继续消费 `{ role, content, messageKind, inputStructure }`。实现上：

1. **迁移期双写**：append journal 且仍 `INSERT INTO messages`（现表）。
2. **切投影**：`listMessages` 改为 `deriveChatMessages(journal)`；`messages` 表可保留作缓存，以 journal `seq` 为水位。
3. `/clear`（RFC-010 命令）只 append `session.cleared`；投影变空。兼容期可同时 `DELETE FROM messages`。

Workbench `user_version` 升一档（落地为 4，见 [desktop-task-workbench.md](../architecture/desktop-task-workbench.md)），新增 `journal_events` 表。旧库无 journal 时，从 `messages` 一次性 backfill：每条 user/assistant 合成 `user`/`assistant` 事件，包在合成的 `turn.*` 里。

---

## 5. 存储

| 宿主 | 后端 |
|---|---|
| CLI | 内存 `InMemoryJournal`（进程内；测试可注入） |
| server / desktop | SQLite。Desktop 复用 `workbench.sqlite` 新表；server 放在既有 dataDir，**不**与 TaskPool 表混主键 |

```sql
CREATE TABLE journal_events (
  conversation_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  id TEXT NOT NULL,
  type TEXT NOT NULL,
  turn_id TEXT,
  step INTEGER,
  trace_id TEXT,
  payload_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (conversation_id, seq)
);
```

**不做：** JSONL 文件、zstd、按 generation 的 `session.vN` 迁移链、fork-at-`inheritedEventCount`。需要 fork 会话时另开 conversation，而不是切事件流。

---

## 6. 写入点（循环侧）

在 [tao.ts](../../ts/packages/core/src/loop/tao.ts) / [react.ts](../../ts/packages/core/src/loop/react.ts) 用可选 `JournalSink`（默认 no-op，保证单测零行为变化）：

| 时机 | 事件 |
|---|---|
| `run()` 开始（preflight 之后、结构化之后） | `turn.start` + `user`（`content` = executionInput，`originalInput` 另存） |
| RFC-008 `clarify` 短路 | `assistant` (clarification) + `turn.end` status=clarify |
| `parseLlmThought` 得到 thought | `react.thought` |
| 即将 `toolHub.call` | `tool.call` |
| 工具返回 | `tool.result`（observation 已按 `observationMaxChars` 截断） |
| FINAL / 隐式终止 / fail_fast 答案 | `assistant` |
| `run()` finally | `turn.end` |

Sink 失败：fail-soft 记审计，**不**中断循环（与 RFC-002 fail-soft 一致）。恢复路径以已提交前缀为准。

---

## 7. 清空与 GC

- 逻辑清空：append `session.cleared`。`derive*` 全部从其后开始。
- 物理 GC（可选、桌面空闲时）：删除该 `seq` 及之前的行。CLI 内存 journal 可直接 drop 前缀。
- 禁止只删 `messages` 而不写 `session.cleared`（RFC-010 落地后 `/clear` 必须走命令 → journal）。

---

## 8. 与既有决策

- RFC-001 / ADR-001：Thought/Action/FINAL 文本协议不变；journal 只是把解析结果落盘。
- RFC-006：记忆参考仍由 ContextEngine 注入 system prompt，不是 journal 事件（除非将来把「提示词段快照」做成 `system` 事件；本 RFC 不做）。
- RFC-007 TaskPool：每个 worker 的 `tao.run` 仍是一次 turn；`runId` 写入 `turn.start.traceId` 或 payload，**不**把 DAG 节点塞进 journal。
- RFC-008：`originalInput` / `executionInput` / `inputStructure` 落在 `user` 事件上。

---

## 9. 验收

- core 单测：append 后 `deriveHistory` 顺序稳定；`session.cleared` 截断；thought/tool 不泄漏进下一 turn 的 ChatMessage
- 双调用方：Tao 与 ContextEngine 对同一 fixture 得到同一 history
- workbench：旧 `messages` backfill 后 ChatPanel 观感不变
- `/clear` 在双写期：投影为空且下一 `run()` history 为空（命令接线属 RFC-010，本 RFC 提供 `cleared` 语义）
- CLI 默认内存 journal，不新建 SQLite
- **不**出现 JSONL / 多代文件格式

---

## 10. 落地顺序

1. `JournalEvent` zod + `InMemoryJournal` + `deriveHistory` / `deriveTrajectory` / `deriveChatMessages`
2. Tao/ReAct 可选 sink（测试注入）
3. Desktop `journal_events` + backfill + 双写
4. 切 `listMessages` 为投影；保留 messages 缓存或删除（实现时二选一，默认保留缓存一版）
5. RFC-010 把 `/clear` 接到 `session.cleared`
