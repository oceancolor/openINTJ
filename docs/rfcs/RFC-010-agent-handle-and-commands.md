# RFC-010：斜杠命令、Agent Handle 与流式钩子

| 字段 | 值 |
|---|---|
| 状态 | **Accepted** |
| 起草日期 | 2026-11-02 |
| 决策日期 | 2026-11-03 |
| 作者 | OpenINTJ Core |
| 上游决策 | 吸收 DeepSeek Harness 的 `ctx.commands`、`Agent.followup/steer/inject/cancel`、流式 `assistant-stream`；**不**换 loop、不改 ADR-001 |
| 影响包 | `@openintj/core`、三端 `agent.run`、RFC-002 HookBus、RFC-004 IPC、Desktop ChatPanel |
| 依赖 | RFC-009 Session Journal（`/clear` 写 `session.cleared`）；RFC-002（补齐「改写驱动循环」） |

---

## 摘要

人发的 `/clear` 不应进模型。进行中的 turn 应能被取消、在下一 ReAct step 被 steer、或被 inject 上下文（例如「刚打开的文件」）而不另开一轮。流式 token 是桌面体验缺口，但 **delta 不是事实源**（事实源是 RFC-009 结算事件）。

本 RFC 三件事：

1. **斜杠命令注册表**（不经 LLM）
2. **活 `AgentHandle`**（run / followup / steer / inject / cancel / whenIdle）
3. **Hook 瀑布真正驱动循环** + 可选 `event.ASSISTANT_DELTA`

---

## 1. 斜杠命令

### 1.1 契约

```typescript
interface SlashCommand {
  name: string;                 // 含前导 "/"，如 "/clear"
  aliases?: string[];           // 如 "/clear history"
  description: string;
  /** 返回 true 则该文本是本命令（默认：整行 trim 后精确匹配 name/aliases，大小写不敏感） */
  match?(text: string): boolean;
  run(ctx: SlashCommandContext): Promise<SlashCommandResult>;
}

interface SlashCommandContext {
  conversationId?: string;
  journal: Journal;             // RFC-009
  handle: AgentHandle;
  raw: string;
}

type SlashCommandResult =
  | { kind: "ok"; message?: string }      // 给 UI 的系统句，默认不入下一轮 history
  | { kind: "error"; message: string };
```

`agent.run` / Desktop CHAT **在 RFC-008 结构化之前** 先 `matchCommand(text)`。命中则执行命令并返回，不调用 classifier / Tao。

首批内置：

| 命令 | 行为 |
|---|---|
| `/clear`（`/clear history`） | journal append `session.cleared`；投影变空；若 turn 在跑则 `cancel("command")` |
| `/compact` | **本 RFC 只注册占位**，真正压缩见 RFC-012；未实现时返回明确错误，不进模型 |

后续（不在首期必须）：`/plan` 进入 plan-only（governance 禁 `write_file` / `execute_command`，**不改 Planner 模板**）。

### 1.2 兼容

现有 Desktop `isChatClearCommand` + CHAT 拦截、`clearWorkbenchMessages` IPC 在命令表落地后改为 `commands.invoke` 的实现细节。renderer 不再特判字符串。

CLI：`openintj chat` 若输入以 `/` 开头且命中注册表，同样不跑模型。

---

## 2. AgentHandle

今天只有 `run(query)` + 传入的 `AbortSignal`。没有「活着的会话对象」。

```typescript
type AgentStatus = "idle" | "running" | "cancelling";

interface AgentHandle {
  readonly id: string;                 // conversationId 或 "ephemeral"
  readonly status: AgentStatus;
  readonly journal: Journal;

  /** 打开一个 turn（今日 agent.run 的语义） */
  run(query: string, opts?: RunOpts): Promise<InputStructuredTaoResult>;

  /** 排队下一 turn，并在 idle 后唤醒。与 run 串行 */
  followup(message: string): void;

  /**
   * 插入到最近的 ReAct step 边界（下一轮 Thought 之前）。
   * idle 时等价于 followup。
   */
  steer(message: string): void;

  /**
   * 下一轮 pre-Think 注入模型可见上下文，不单独唤醒。
   * 例：桌面打开文件 → inject 路径，等用户下一句才跑。
   */
  inject(message: string): void;

  cancel(cause: "user" | "command" | "dispose", opts?: { keepInbox?: boolean }): void;

  whenIdle(): Promise<void>;
}
```

实现约束：

- 一个 conversation **同时只有一个** running turn（与今日 Desktop CHAT 串行一致）。
- `steer` / `inject` 入队到 inbox；`react.ts` 在每轮 Thought **之前** drain steer（拼进 user 或独立 `user` 事件，`source: "steer"`）；inject 在 `tao.beforeThink` 拼进 system 或 user 附件，并写 journal（model-visible）。
- `cancel` 必须贯通既有 `ChatOptions.signal`（RFC-007 已打通 provider abort）。
- `dispose`（宿主卸载）= `cancel("dispose")` + `whenIdle`。Desktop 热重装已有类似顺序，应复用。

**不做：** dsh 式多 inbox 边界枚举、`runMaintenance`、跨进程 initiator 归因。

---

## 3. 补齐 RFC-002：改写必须影响下一步

RFC-002 已规定 `emit` 返回改写后的 payload、`cancel()` 短路。循环侧多数只 `await emit(...)` **丢掉返回值**。

本 RFC 要求：

| 钩子 | 循环必须消费 |
|---|---|
| `tao.beforeThink` | 采用返回 payload 的 `query` / 注入块；`cancel` → 跳过本 TAO 轮（记 `turn.end` cancelled 或无 step） |
| `react.beforeAction` | `cancel` → 跳过该工具（已允许）；改写后的 `tool` / `params` 必须交给 `toolHub.call` |
| 新增 `react.beforeThought`（可选） | 可把 inbox.steer 合入 messages |

不是新的 Cordis waterfall，只是 **把已有 HookBus 返回值接进 Tao/ReAct**。不必单开 RFC。

新增事件：

```typescript
"event.ASSISTANT_DELTA": {
  traceId: string;
  turnId: string;
  step: number;
  text: string;          // 增量
  channel: "thought" | "answer";
};
```

Provider 暂不支持真正 token 流时：可在 `parseLlmThought` 之后发 **一条** 全量 delta（退化），UI 仍订阅同一事件。不改 ADR-001。完整 OpenAI-compatible streaming 仍是 next-session 独立项，本 RFC 只定钩子名与 IPC。

---

## 4. RFC-004 IPC 增量

在现有 `openintj:*` 上增加（名称可微调，落地时写入 ipc-protocol）：

```text
openintj:commands.invoke          { conversationId?, text } → SlashCommandResult
openintj:agent.steer              { conversationId, message }
openintj:agent.inject             { conversationId, message }
openintj:agent.cancel             { conversationId, cause }
openintj:journal.subscribe        → EVT_JOURNAL（seq 水位推送）
EVT_ASSISTANT_DELTA               推送 ASSISTANT_DELTA
```

`WORKBENCH_MESSAGES_CLEAR` 保留为 `commands.invoke("/clear")` 的别名一版，随后废弃。

ChatPanel：订阅 delta 更新气泡；turn.end / journal 投影为权威终态（防 HMR 丢 preload 一类问题）。

---

## 5. 与既有决策

- ADR-001：不引入 function-calling；流式是文本增量。
- RFC-009：命令与句柄的持久副作用只通过 journal。
- RFC-007：TaskPool worker 各持临时 handle；steer 默认只对 Desktop 前台 conversation。
- RFC-008：命令匹配 **先于** `structureUserInput`，避免 `/clear` 被当成任务。

---

## 6. 验收

- `/clear` 三端不调用 LLM；journal 有 `session.cleared`；下一 run history 为空
- 未注册的 `/foo` **仍进模型**（避免吞掉用户字面量）
- `cancel` 能中止在途 Hunyuan/Ollama fetch（复用 RFC-007 测试）
- `steer` 在下一 ReAct thought 前出现在 messages；idle 时不丢（或文档化为丢弃并测试锁定）
- `inject` 不单独开 turn
- `tao.beforeThink` 的 `replace()` 改变实际 query（新回归）
- Desktop：delta 钩子有契约测试；无 provider 流时至少终态正确

---

## 7. 落地顺序

1. SlashCommand 注册表 + 三端 `run()` 前置匹配；`/clear` 接 RFC-009
2. 循环消费 HookBus 返回值
3. AgentHandle 包一层现有 `run` + cancel；再加 inbox（steer/inject/followup）
4. IPC + ChatPanel；ASSISTANT_DELTA 可空实现
