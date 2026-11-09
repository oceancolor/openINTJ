# RFC-021 可靠记忆闭环（Reliable Memory Loop）实施记录

设计规格：[`docs/openintj-reliable-memory-design.md`](../openintj-reliable-memory-design.md)（v0.1，2026-11-09）
设计基线：`main@2326b4b02ad578363fd470317ce733a039c775e0`
状态：**实施中**（G0 已完成，G1 进行中）

本文件是实施记录，不是设计规格。每个阶段记录：核实结论、实际改动、测试证据、
未解决问题与门控确认。阶段顺序与门控以设计文档 §7 为准；每阶段结束暂停，
等待用户确认后进入下一阶段。

已确认的推进决策（2026-11-09，用户拍板）：

1. **分阶段门控**：只先行当前阶段，汇报暂停，不连跑后续阶段。
2. **桌面优先**：可靠闭环先保证桌面端；CLI/server 维持 `durability=memory`
   并明确标注，三端统一持久化另行决策。
3. **止损先提案**：自动升权／蒸馏晋升的止损开关只出方案，用户确认后才实施。

---

## G0 基线与止损（2026-11-09 完成）

### G0.1 E01–E14 证据复核（只读，未改代码）

复核时 HEAD 与设计基线完全一致，工作区仅设计文档本身未提交。

| 编号 | 状态 | 实锤定位 |
| --- | --- | --- |
| E01 原文直存 + 固定 importance | 仍存在 | `planes/memory/src/memory-plane.ts:55-87`，user 0.6 / assistant 0.5 |
| E02 finalize 见事件即跳过 | 仍存在 | `core/src/journal/sink.ts:161`，turnId 有任意事件即 return；`safeJournalAppend` 吞异常 |
| E03 TAO 先写 final、宿主后修订 | 仍存在 | `core/src/loop/tao.ts:438-439` 先写 assistant + turn.end |
| E04 正文只在 Lance | 仍存在 | `planes/memory/src/persistent-store.ts:124-132` 从 `vectorStore.scanAll()` hydrate |
| E05 内存先变、pending 链可毒化 | 仍存在 | `persistent-store.ts` `scheduleWrite` 无 catch 的 `.then` 链 |
| E06 tier/clear 无持久化 mutation | 仍存在 | `planes/memory/src/store.ts:149-151` `clearWorking` 只清内存 |
| E07 ContextEngine 无 scope | 仍存在 | `planes/memory/src/context-engine.ts` 全文无 scope/eligibility |
| E08 embedder 装配不一致 | 仍存在 | `apps/desktop/src/main/agent.ts:717-723` `Object.defineProperty` 换 store，retriever embedder 未换 |
| E09 24h 半衰期硬淘汰 | 仍存在 | `core/src/types/shader.ts:60,70`（0.3 阈值 + 24h，全 tier） |
| E10 compact 虚报覆盖 | 仍存在 | `runtime/src/compact.ts:182-199`，输入截断却写全量 `coveredTurns` |
| E11 无 handler 仍 success | 仍存在 | `planes/execution/src/tool-hub.ts:253-265`，`no_handler` 也 `success: true` |
| E12 skills 按 query 归因 + 默认 +0.2 | 仍存在 | `packages/skills/src/learning-runtime.ts:12-16,200-201,234-236` |
| E13 revoke 直接删字段 | 仍存在 | `packages/dormant/src/internalization-manager.ts:148-157`；`packages/profile/src/service.ts:97-117` 只 upsert |
| E14 跨会话控制污染 | 部分确认 | 桌面实锤：`apps/desktop/src/main/ipc-handlers.ts:369-374` 指定会话 cancel 仍 abort 全部 activeChats；handle 侧归属校验待 G1 用 T03 复现钉死 |

结论：设计文档的判断在当前 HEAD 上基本全部成立，无需重复修已修项。

### G0.2 Baseline（与新增回归分开记账，2026-11-09 实测）

- `@openintj/core`：168 passed
- `@openintj/runtime`：18 passed
- `@openintj/plane-memory`：83 passed / 3 skipped
- `@openintj/skills`：59 passed
- `@openintj/profile`：6 passed
- `@openintj/desktop`：109 passed / 2 skipped，typecheck 全绿（随 2326b4b 提交验证）
- 既有失败：目标目录 `biome check` 有 25 个存量诊断（以 format/行尾为主），
  与设计文档引用的 CI lint 红灯吻合。后续阶段**不顺手全库 `lint:fix`**。

### G0.3 止损方案（仅提案，未实施）

给 skills outcome 升权与蒸馏晋升加可回退开关：建议新增
`OPENINTJ_SKILLS_LEARN_MODE=observe|active`。observe 时只记 selection/outcome
观察，不写权重、不进蒸馏 buffer；默认 active（不改变现有行为），待 G3 的
F10 归因修好后再讨论默认值翻转。是否实施及何时翻转默认值，待用户确认。

### G0.4 类型契约草案（G1 落地时定型）

- `RunContext`：principalId / workspaceId / taskId / conversationId / turnId /
  runId / parentRunId? / branchId? / traceId / memoryEpoch / signal。身份由宿主
  验证，不得从模型文本取得（设计 §4.1）。
- `ToolOutcome`：executionStatus（not_executed/blocked/completed/error/
  cancelled/timeout）、businessStatus（success/failure/unknown）、
  evidenceStatus、callId（执行前生成）、runId、origin、outputRef、
  sideEffectStatus（设计 §F03）。旧 `success` 仅短期派生保留，禁止继续
  决定学习晋升。

---

## G1 可靠事件入口（2026-11-09 启动）

范围（用户确认的边界）：F02 → F03 → F01 → F04，验收挂 T01–T09 中已存在
入口的部分；先失败测试再改实现。**不自动生成正式语义记忆。**

- F02：RunContext 贯通 + 控制/取消隔离（T03、T04）
- F03：ToolOutcome 三分（T05、T06）
- F01：TurnCoordinator + host-owned final，桌面优先（T01、T02）
- F04：运行时 scope/eligibility 过滤先于截断（T07–T09 入口部分）

实施明细与测试结果随推进追加于此节。

### G1 实施明细（2026-11-09）

**F02 运行身份与控制隔离**
- 新增 `core/src/run-context.ts`：`RunContext`（principal/workspace/task/
  conversation/turn/run/trace/epoch）+ `createRunContext` /
  `runContextFromJournal` / `childRunContext` / `runScopeKey`。
- HookBus：`HookContext.runId` + emit opts 贯通；TAO/ReAct 发射带 runId，
  ReAct 工具调用 opts 带 runId。
- AgentHandle：steer/inject 只被所属 run 消费（T03）；queued 取消不启动
  （T04）；followup 生成新 turnId/runId；新增 `dispose()` 退订 hook。
- 审批：`ApprovalRequest.signal` 贯通，parked 请求 abort 即 deny，迟到
  `decide` 返回 false（governance T04）。
- 桌面：AGENT_CANCEL 只 abort 目标会话的流（新增 per-conversation 注册表，
  全局 abort 仅保留给 restart/reload）；`DesktopRunOptions.run` 贯通至 TAO。

**F03 工具结果三分**
- `ToolCallResult` 新增 executionStatus / businessStatus / evidenceStatus /
  origin / runId（旧 `success` 为派生视图）。
- ToolHub：callId 执行前生成；无 handler = `not_executed`（不再伪 success、
  不记 breaker success）；exitCode≠0 = business failure；gate/allowlist =
  blocked；超时区分 timeout。
- 桌面产物兜底改走 `toolHub.call(..., { origin: "host_fallback" })`，且仅
  `result.status === "completed"` 时触发，失败时答案明确写失败而非"已写入"。

**F01 最终提交所有权**
- core journal 新增 `commitHostTurn`：部分过程事件不阻止提交；同 final
  重提 = replayed；异 final = conflict 且零写入；sink 抛错上抛（T01/T02）。
- TAO 新增 `completionOwner: "loop" | "host"`；host 模式只记过程事件。
- runtime 新增 `TurnCoordinator`（`createTurnCoordinator` /
  `taoStatusToTurnEnd`）；桌面 IPC 三处收尾（主回答/preview/失败）统一走
  coordinator，endStatus 按真实 status 映射，不再"非 clarify 全 completed"。
- CLI/server 维持 loop-owned 兼容模式（其 journal final 视为 draft）。

**F04 scope 与 eligibility（入口部分）**
- memory 新增 `scope.ts`：scope tag（`workspace:`/`task:`/`conversation:`）
  fail-closed 可见性 + `scopeFromContextTags`。
- Retriever 在评分前过滤（不占 topK、不涨 accessCount）；ContextEngine
  接收 `scope` 并对 candidateRetrieve 结果再做权威过滤（T07/T08 入口级）。
- 桌面：taoOpts 用 contextTags 覆盖 run 的 scope 字段，contextProvider
  把 run scope 传给 ContextEngine.build。

**验证（2026-11-09 实测）**
- core 176 passed（基线 168，+8）；plane-memory 87 passed / 3 skipped
  （+4）；plane-execution 65 passed / 1 skipped（+4）；governance 41 passed
  （+1）；runtime 18 passed；skills 59 passed；server 77 passed / 11
  skipped；cli 54 passed / 6 skipped；desktop typecheck + 109 passed /
  2 skipped + 生产构建全绿。
- 过程中修复：编码回写损坏 `ipc-handlers.ts` 后已从 HEAD 恢复并以正规
  编辑重做（该文件当时无其他未提交改动）。

**已知遗留（不阻塞 G1 门控）**
- 子代理（subagent）run 尚未带 scope，contextProvider 在该路径无 scope
  过滤；G2/G3 随 RunContext 全量贯通处理。
- candidateRetrieve（hybrid）路径为召回后过滤，topK 槽位仍可能被不可见
  候选占用；G2 索引重放时改为召回前过滤（设计 F04.3）。
- server/cli 的 finalize 仍为 loop-owned + `finalizeJournalTurn` 跳过
  语义；三端统一 host-owned 待桌面稳定后推广。

---

## G2 实施明细（2026-11-09）

**F05 canonical memory（SQLite 权威源）**
- 新增 `CanonicalMemoryRepository` 契约（plane-memory `canonical.ts`）：
  记录含正文/scope 标签/来源/状态（active|tombstoned|quarantined）/修订号；
  唯一 mutation 边界 `applyMutation`，写键幂等（同键同 hash replayed、
  异 hash conflict）、CAS（expectedRevision）、单事务 = 记录变更 + 收据 +
  index outbox。附 `InMemoryCanonicalMemoryRepository` 同语义测试实现。
- storage-sqlite 新增 `SqliteCanonicalMemoryStore`（`canonical-memory.ts`）：
  表 memory_records/mutations/candidates/index_outbox/index_generations/
  control/suppressions。
- `PersistentMemoryStore` 注入 canonical 后：hydrate 只读 canonical（向量
  层全丢正文仍在，T10）；晋升/删除/clearWorking/forget 全走 mutation
  （T11，顺修 E06）；pending 链逐项隔离失败（顺修 E05）。

**F06 outbox 与接收边界**
- mutation 同事务 enqueue index job；`projectIndex()` 逐 job 隔离：坏 job
  只累计 attempts 不毒化队列（T13）；过期 upsert（已 tombstone/修订更新）
  跳过防复活（T14）；索引落后只降级状态，正文/词法检索不中断。
- 候选批次接收：键 `(sourceJournalId, sourceCommitId, consumer,
  policyVersion, extractionRevision)`，幂等 applied/replayed、异 hash
  conflict；只存待审 `status=received`，绝不激活（固定 fixture 验证）。
- legacy 迁移 `auditLegacyMemory` / `migrateLegacyMemory`：Lance 与 SQLite
  meta 逐 ID 对账；dry-run 默认且零写入；正式迁移时 scope 不明/无正文进
  quarantined 不静默激活；不向旧库反向双写。

**F07 embedding 空间身份**
- `embedding-space.ts`：指纹 = provider/model/dimension/transformVersion，
  随 index generation 记录；`assertEmbeddingCompatible` 维度硬冲突显式
  报错（T15：投影失败但正文持久、检索不中断）；`rebuildIndex(fp)` 开新
  代次从 canonical 全量重建投影。
- `MemoryPlaneOpts.store` 构造注入：桌面/server 删除 `Object.defineProperty`
  反射替换 store/retriever.store（E08 装配侧修复）。

**F12 P0 地基**
- control ledger：principal 级 `memoryEpoch`（bump 接口就绪）、suppression
  表 + hydrate 读检查；`forgetFragment` = suppression + tombstone 同走
  mutation。管理面完整语义（epoch 失效派生缓存等）留 G4。

**验证（2026-11-09 实测）**
- 新增测试：plane-memory `canonical-store.spec.ts` 8 例（T10–T15 + 候选
  接收 + forget/epoch）、`legacy-migration.spec.ts` 1 例；storage-sqlite
  `canonical-memory.spec.ts` 4 例（SQLite 侧 T12/outbox/候选/control）。
- 结果：storage-sqlite 37 passed；plane-memory 96 passed / 3 skipped；
  server 77 passed / 11 skipped；desktop typecheck + 109 passed / 2 skipped
  + 生产构建全绿。零基线回归。

**G2 遗留 / 门控说明**
- canonical 已在桌面启用（2026-11-09 用户确认后执行）：真实数据目录
  `%APPDATA%\@openintj\desktop\memory-store` 先备份至
  `memory-store-backup-g2-20261109`，dry-run 对账 16 向量行 / 16 meta /
  全 matched / 无空正文 / scope 不明 0；正式迁移 16 条全部 active、
  0 quarantined；canonical 模式实测 hydrate 16 条、待投影 job 启动自动
  补齐、768 维（ollama nomic-embed-text）兼容。桌面 real 模式传
  `useCanonical: true`；server/cli 未启用（維持桌面优先）。
- 同步 add* 路径仍为 fire-and-forget 写 canonical（与旧行为一致）；async
  路径已 await 到收据。晋升（溢出自动迁移）未单独发 reassign mutation，
  其 memoryType 漂移在下次 upsert 时收敛。
- embedding 空间的 model 名目前取 embedder.name（真实模型名在 llm/runtime
  指纹侧已有完整记录），G3 接 writer 前统一。

---

## 待决策 / 未解决问题

- 止损开关（G0.3）是否实施、何时翻转默认值。
- CLI/server 持久 journal adapter 的排期（当前决策：桌面优先，两端标注
  `durability=memory`）。
- 物理 purge 的完整度承诺（F12/G4 决策，基础版先交付有效撤回 + barrier）。
- 真实用户数据迁移（G2 前必须单独走备份 + dry-run 确认流程）。
