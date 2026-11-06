# RFC-012：压缩驱动、能力缝与进程内 Subagent（§1 已实现；§2/§3 延期）

| 字段 | 值 |
|---|---|
| 状态 | **§1 压缩驱动 + 任务检查点已实现**（2026-11-05）；**§3 Subagent 已实现**（2026-11-05，评审见 §3）；§2 Fs/Shell 缝仍 **Deferred**（确认前 **禁止实现**） |
| 起草日期 | 2026-11-02 |
| 决策日期 | 2026-11-05（§1 打开条件 1–3 已满足，先行落地；§2/§3 维持延期） |
| 作者 | OpenINTJ Core |
| 上游决策 | 吸收 DeepSeek Harness 的 `ctx.compaction`、fs/subprocess seam、`ctx.subagents`；**不**接 ACP/Codex/Claude 后端，**不**替代 RFC-007 TaskPool |
| 依赖 | RFC-009 Journal、RFC-010 Handle、RFC-011 Runtime |
| 本文目的 | 把「以后要做」写进仓库，避免再次从 dsh 重新发现同一缺口 |

---

## 为什么延期

- 压缩没有 journal + 单一 history 投影会压错对象。
- Fs/Shell 缝牵涉 RFC-004「VSCodium 直写磁盘 vs Agent `write_file`」产品边界，需要单独安全评审。
- 进程内子 Agent 对接 next-session「默认多 Agent」，但 TaskPool 已覆盖模板 DAG；先做会双轨混乱。

**确认前不动手**包括：不新建包、不拆 `createWorkspaceTools`、不在 TaoLoop 里自动 compact、不暴露 subagent 工具。

---

## 1. 压缩驱动

### 现状

[`ContextEngine.build`](../../ts/packages/planes/memory/src/context-engine.ts) 在超 budget 时 emit `event.CONTEXT_COMPACTED`，**调用方不自动再跑 pipeline**。桌面上下文环只是展示。

### 目标（以后）

在 `tao.beforeThink`（RFC-010 已要求消费该钩子返回值）调用：

```typescript
compactIfNeeded(ctx, trigger: "pressure" | "overflow" | "command"): Promise<void>
```

- `pressure`：budget ≥ 0.8（与今日 emit 阈值对齐）时 ShaderPipeline 切 LOW_FIDELITY / Summarizer，**写回**下一轮 `[记忆参考]`
- `overflow`：provider 报上下文溢出时（需 RFC-010 的 request-error 通道）重试一次 compact【2026-11-06 已实现，形态见 §1 落地记录】
- `command`：`/compact`（RFC-010 占位）显式触发

压缩是投影，**不是**改写 RFC-009 已提交事件。可选：compact 后 append 一条非 model-visible 的审计事件（若加，类型名另定，不占用 `session.cleared`）。

### 落地（2026-11-05，第一段）

- Core journal 新增 `session.compacted { summary, coveredTurns, upToTurnId, trigger }` 投影事件；`deriveHistory` 把 `upToTurnId` 之前已完成的轮次折叠为一条 `[历史摘要]` 消息，`deriveChatMessages`（UI 视图）不受影响，已提交事件永不改写。
- `@openintj/runtime` 新增 `createCompactor`：按轮对齐切分，保留最近 `keepRecentTurns`（默认 3）轮原文，其余交给 LLM 摘要；失败 fail-soft（`llm_failed` / `empty_summary` / `too_short` / `below_pressure`），绝不阻塞 run。
- 触发面：`pressure`（每轮 prelude 估算 token 超 `pressureTokens`，默认 24k，三端已接，成功时 `metrics.compacted = 1`）与 `command`（真实 `/compact`，经 RFC-010 `SlashCommandContext.compact` 由宿主绑定）。`overflow` 于 2026-11-06 补齐（见下条）。
- 任务检查点：`TaskStore.listRuns(limit)`（Memory + SQLite 实现）；`TaskPool.listCheckpoints() / resumeRun(runId, worker) / cancelRun(runId)`（活跃 run 注册表 + 持久快照取消；legacy 无 `goalInput` 快照按既有语义取消而非重建）。Server 暴露 `GET /api/taskpool/checkpoints`、`POST /api/taskpool/runs/:id/resume|cancel`；Desktop 暴露 `taskpool.checkpoints / taskpool.resume / taskpool.cancel` IPC + preload（暂无渲染面板）。三端共用同一 worker 抽取，恢复与正常编排走同一节点执行路径。

### 落地（2026-11-06，overflow 触发补齐）

经评审定型（循环内回调 / 双层分类 / 重试一次 / 仅主路径）：

- **分类**：Core 新增 `isContextOverflowError` / `matchesContextOverflowText`——结构化信号优先（`AgentError.code === CONTEXT_OVERFLOW` 或 `details.overflow === true`），消息正则兜底（`context_length_exceeded`、`maximum context length`、`prompt is too long`、`exceeds the available context size`、中文超限句式等，从严防误报）。Hunyuan/Ollama 适配器在上游拒绝时抛 `CONTEXT_OVERFLOW` 并附 `details.overflow`。
- **恢复**：ReAct（`run` 与 `runSingle`）在 `llm.chat` 命中 overflow 时调用宿主经 Tao 透传的 `onContextOverflow(historySegment)` 回调；拿到压缩后历史后就地 splice 替换 conversation 的历史段（system、当前 query、本轮已追加的工具步骤全保留，无副作用重跑），重试这一次 chat。每轮 run 最多恢复一次，二次 overflow 或压缩失败抛原始错误。
- **宿主**：三端主对话路径组装回调（`trigger: "overflow"`，摘要输入临时收紧到 16k 字符保证小窗口模型压得动）；成功置 `metrics.compacted = 1` 与 `metrics.overflowRecovered = 1`。TaskPool 节点与自洽采样 v1 不覆盖（各自重试语义待单独评估）。

---

## 2. 能力缝：Fs / Shell 与工具声明分离

### 现状

[`createWorkspaceTools`](../../ts/packages/planes/execution/src/workspace-tools.ts) 同时是沙箱后端和模型可见工具。VSCodium reh-web **直写工作区磁盘**，不经 ToolHub / governance（RFC-004 外部产品边界）。

### 目标（以后）

```text
FsProvider / ShellProvider     ← 路径根、大小上限、命令白名单、cwd URI
        ↑
ToolHub read_file write_file execute_command   ← 仅模型 schema + execute 委托
        ↑
GovernancePlane.checkToolCall
```

换本地/远程后端时工具名不变（dsh seam 的要点）。

长期：**IDE 直写仍允许**，但可走同一 `FsProvider` 记审计（不强迫 IDE 调用 `write_file`）。未评审前保持今日边界。

MCP：作为 `toolHub.register` 的又一个来源（与技能并列）。优先级低于缝本身。

---

## 3. 进程内 Subagent

### 与 TaskPool 的分工

| | RFC-007 TaskPool | 本 RFC Subagent |
|---|---|---|
| 是什么 | 模板 DAG 上的并行 `tao.run` worker | **委托对话**：子 `AgentHandle` + 独立 journal |
| 工具面 | 与父相同（除非技能 allowlist） | 默认技能式 ALS 收窄 |
| 何时用 | planning/analysis 图可拆成步骤 | 需要隔离上下文的子任务（调研、测一处代码） |

**不替代** TaskPool。**不**接入 dsh 的 ACP / Codex / Claude Code / 外部 SDK 后端。仅 `in-process`。

草图（确认后才细化）：

- `runtime.spawnChild({ parent, label, allowlist, depthLimit })` → `AgentHandle`
- 父模型可见工具 `delegate`（名称待定）把任务写成子 journal 的 `user`，子 `run` 结束后一条 `tool.result` 回到父
- 深度默认 1，禁止递归爆炸
- 子会话出现在 Desktop 工作台为只读或折叠节点（UI 另议）

### 评审与落地（2026-11-05，§5 条件 4 单独评审结论）

- **分工确认**：TaskPool 管「计划内 DAG 步骤」（共享上下文、模板拆分）；Subagent 管「计划外委托对话」（隔离 journal、结果回父）。两者不互相替代，编排模式判定（`resolveOrchestrationMode`）不变。
- **成本护栏**：默认**关闭**（`OPENINTJ_SUBAGENTS=1` 或 opts 显式开）；深度上限 1（AsyncLocalStorage 传播深度，子内再 delegate 直接拒绝）；并发上限 2（超限拒绝不排队）；单次超时 120s（AbortSignal 注入子 run）。
- **安全**：`delegate_task` 工具 `permissions: ["agent.delegate"]`、非幂等 → RFC-013 归类 write 风险，桌面默认先审批；子 run 与父共用同一 ToolHub/治理/审批面，无旁路。子级工具面 v1 不做技能 allowlist 收窄（与父同面），收窄留待后续（已知取舍，文档明示）。【2026-11-06 已收窄，见下】
- **形态**：`@openintj/runtime` 的 `createSubagentSpawner({ runChild, maxDepth, maxConcurrent, timeoutMs })` + `createDelegateTool(spawner)`；宿主用各自 handle 体系提供 `runChild`（CLI/server 走 `handle.run` + 子 conversationId 的 journal ctx，desktop 走 `handleFor`）。子会话 id 形如 `subagent:<id>`，与父同 journal sink、独立 conversation。`list()` 供宿主观测（server `GET /api/subagents`、desktop `subagents.list` IPC）；不提供宿主直调 spawn（只经模型工具，避免绕过审批语义）。Desktop 工作台子会话节点 UI 未做（另议）。【子智能体记录已并入桌面「系统」页，2026-11-06】

### 落地（2026-11-06，子工具面收窄）

- **上限模型**：宿主按风险定上限（`classifyToolRisk` 分级；默认 `read`，opts `subagentMaxRisk` / env `OPENINTJ_SUBAGENT_MAX_RISK` 调整），`delegate_task` 本身恒排除；`delegate_task` 新增可选 `tools` 参数，模型可在上限内再收窄（与上限取交集，越界名字直接丢弃）。有效名单记入 `SubagentRecord.tools` 供观测。
- **执行机制**：ToolHub 新增 `setRunCap` 硬上限（有效名单 = 技能名单 ∩ cap，子会话内选技能也不能突破）与 scope `inherit` 语义（run 包装层复用外层盒子）；宿主 `runChild` 为子 run 开新 scope 并置 cap，上限按 spawn 时刻解析，后注册的连接器只读工具也能进子会话。治理审批面不变（上限内若含 write 工具，调用仍照常走审批）。

---

## 4. 明确不做

- Cordis、YAML profile、Creator mode 写插件
- 用 dsh `agent-loop` 替换 TAO/ReAct
- function-calling 作为默认协议（仍受 ADR-001 约束）
- Agent Teams 实验协调器
- JSONL 多代 session 文件

---

## 5. 打开条件（满足再改状态为 Draft 可实现）

1. RFC-009 `deriveHistory` 已是 Tao 与 ContextEngine 的唯一入口
2. RFC-010 `/clear` 与 Handle.cancel 可用
3. RFC-011 三端共用 assemble，避免在三份 `agent.ts` 上各拆一遍 FsProvider
4. 单独评审：IDE 直写 vs 审计缝、子 Agent 成本与深度

打开后的建议顺序：压缩驱动（与 `/compact` 闭合）→ Fs/Shell 缝 → 进程内 subagent。

> 2026-11-05：条件 1–3 已满足，§1（压缩驱动 + `/compact` + 任务检查点）已落地（见 §1「落地」）。§2/§3 打开前仍需满足条件 4 的单独评审。
