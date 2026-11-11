# RFC-021 可靠记忆闭环（Reliable Memory Loop）实施记录

设计规格：[`docs/openintj-reliable-memory-design.md`](../openintj-reliable-memory-design.md)（v0.1，2026-11-09）
设计基线：`main@2326b4b02ad578363fd470317ce733a039c775e0`
状态：**实施中**（G0–G4 已完成，G5 代码子阶段 G5.0–G5.5 已完成，Hy-Memory 第二轮 drop-in 对照已补齐；剩余激活决策门，为数据/决策事项而非代码任务）

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

## G0–G2 复核收口（2026-11-09，G3 启动前）

逐条核对三阶段全部遗留，分级处理：本阶段可修的已修，后续阶段归属的
确认不属本轮欠账，有意保留的注明决策依据。

| 遗留条目 | 结论 |
| --- | --- |
| G2 晋升/丢弃 tier 漂移 | **已修**：add 路径前后 tier 快照对账——晋升补写、工作记忆溢出丢弃补删（T11b）。复核中另发现真 bug：mutation `inputHash` 只哈希正文，tier 变更会被误判 replayed 静默丢弃；已改为全记录哈希 |
| G0.3 止损开关 | **已实施**：`OPENINTJ_SKILLS_LEARN_MODE=observe\|active`，observe 只记观察不升权/不进蒸馏 buffer；默认 active 不改变现行为（skills 61 passed，+2） |
| G1 子代理 run 未带 scope | **已修**（桌面）：父 run 身份经 AsyncLocalStorage 传播，spawn 时 `childRunContext` 派生（同 workspace/task/principal/epoch、新会话与 runId）。CLI 侧子代理维持现状（durability=memory 决策内） |
| G1 candidateRetrieve 召回后过滤 | **已修**：`MemoryHybridIndex.search` 新增 `filter` 谓词（过量取回→过滤→截断），ContextEngine 把 scope 透传给候选召回器，桌面/server 在召回前做 scope 过滤（taskpool 49 passed，+1） |
| G2 embedding model 名取 embedder.name | **已修**：`embeddingSpaceInfo` 贯通 factory→store，`store.embeddingSpace()` 产真实 provider/model 指纹；桌面传 llm/runtime 指纹 |
| G2 同步 add* fire-and-forget 写 canonical | **保留并注明**：同步 API 契约使然；桌面/server 的落盘路径均走 Async API（已 await 到收据），同步路径仅 CLI/测试便利层使用 |
| E09 shader 24h 半衰期 / E10 compact coveredTurns | **非欠账**：经设计 §7 核对属 F08/F09 → **G3 范围**，已列入 G3 输入 |
| E01 原文直存 | **非欠账**：属 F13 writer → G5（G3 先做生命周期 F08–F10） |
| E13 dormant/profile revoke、物理 purge | **非欠账**：F12 管理面 → G4，G2 只要求 P0 地基（已交付） |
| server/cli loop-owned finalize 与持久 journal | **有意保留**：用户已定桌面优先决策，两端标注 durability=memory；三端统一另行排期 |

复核回归（2026-11-09 实测）：core 176、runtime 18、plane-memory 97/3、
plane-execution 65/1、plane-governance 41、storage-sqlite 37、skills 61、
taskpool 49、profile 6、server 77/11、cli 54/6、desktop typecheck +
109/2 + 生产构建——全部通过，零基线回归。

---

## G3 实施明细（2026-11-09）

范围：F08 生命周期与有效性、F09 压缩精确覆盖、F10 反馈归因。验收 T16–T21
（T18/T19 归 F09、T20/T21 归 F10）。不接真实 writer（G5），不把 completed
当验证通过。

**F08 语义生命周期（T16/T17）**
- core `MemoryFragment` 新增生命周期字段（默认值兼容旧数据）：kind
  （fact/preference/intention/episode/state_reference/procedure/note）、
  assertion（user/assistant/inferred/quoted/tool）、validUntil、canonicalKey、
  supersededBy。
- 新增 `validity.ts`：validity 与 recency 分离；expired/superseded 在
  Retriever、ContextEngine（候选路径）、fragmentsToRanked 全部入口预过滤，
  不是只靠 shader 不显示；`isVerifiedAssertion` 区分已验证断言与带标签
  历史陈述。
- shader geometry 与 retriever 评分：stable long_term 不再被 24h 半衰期
  硬淘汰，时间只参与排名；短期/工作记忆保留衰减门（T16a：90 天前的
  long_term 用户事实仍可召回；T16b：过期 intention 全入口不可见）。
- 纠正语义：`PersistentMemoryStore.correctMemory` /
  `correctByCanonicalKey`——新值生效 + 旧值 `mark_superseded` mutation
  （canonical 保留 superseded 审计态 + 索引删除 job），重启不回潮（T17）。
- canonical 记录增 kind/assertion/validUntil/canonicalKey/supersededBy
  五列；SQLite 对 G2 已建库做 PRAGMA 增量列迁移（幂等）。

**F09 压缩精确覆盖（T18/T19）**
- journal `session.compacted` 新增 `includedTurnIds` + `inputHash`；
  deriveHistory 优先按精确覆盖集隐藏，未覆盖轮次保留原文（旧前缀假设
  仅作无 includedTurnIds 的兼容路径）。
- runtime compactor：按完整 turn 打包（不再单条 3000 字截断、不跨轮
  截断）；cap 只容部分 fold 时只宣称覆盖完整进入输入的前缀，
  `uncoveredTurns` 如实报告；单轮超 cap → `turn_too_large` 不压缩（T18/T19）。
- ContextEngine 预算：conversationTokens 改为当前完整消息集量（与既有
  基线取大），不再逐轮重复累加同一历史。

**F10 反馈归因（T20/T21）**
- retrieved ≠ used：Retriever/fragmentsToRanked 不再 bump accessCount；
  `MemoryStore.recordUsage` 只在真正注入 prompt 后调用，ContextEngine
  以 `usageEventId`（桌面传 run 级 id）幂等记使用并发出
  `event.MEMORY_INJECTED`。
- 技能归因修 E12：新增 `beginSelection → selectionId` 与
  `recordOutcomeForSelection`——outcome 按 selectionId/runId 归属（同
  query 不串奖、重复结算幂等拒绝）；cancelled/clarify/preview 等中性；
  **未验证的 completed 只记观察，不自动升权、不进蒸馏 buffer**；verified
  才走正向信号。桌面接线已切 selectionId（暂无独立验证器，传
  verified=false，待验证器接线后开启正向学习）；server/cli 保留
  recordOutcome 宿主断言路径（显式调用语义不变）。

**验证（2026-11-09 实测）**
- 新增测试：plane-memory lifecycle.spec 4 例（T16/T17）、runtime compact
  +2（T18/T19）、skills F10 归因 4 例；既有 2 例 accessCount 旧语义测试
  按 F10 改写。
- 结果：core 176、runtime 20、plane-memory 101/3、skills 65、
  storage-sqlite 37、taskpool 49、plane-execution 65/1、governance 41、
  profile 6、server 77/11、cli 54/6、desktop typecheck + 109/2 + 生产
  构建全绿，零基线回归。

**G3 遗留 / 说明**
- 桌面技能正向学习待独立验证器（工具证据核验或用户反馈信号）接线后
  以 verified=true 开启；当前行为是 F10 规定的"基础阶段关闭自动正向
  升权"，不是故障。止损开关 observe 与此叠加生效。
  **【2026-11-09 用户决策：验证器放 G5 与 writer 一起做】**——G5 范围
  因此包含：writer 候选接口 + 影子验证 + **技能正向学习的验证器**
  （verified 信号来源：工具证据核验/用户反馈），在此之前桌面技能只记
  观察与负信号。
- assertion/kind 目前由写入方（宿主/AddOptions）标注；自动分类标注
  属于 writer（G5）职责，本阶段不猜。
- 第二次压缩的"前摘要+新增轮"继承覆盖（inheritedCoverage）未做：
  当前 derive 按最近一条 session.compacted 的精确集隐藏，重复压缩的
  覆盖继承留待 G4 与管理面一并处理。

---

## G4 实施明细（2026-11-09）

范围：F11 Profile/Persona 撤销恢复、F12 有效遗忘传播、F14 记忆状态面。
验收 T22–T25。口径：重启/旧任务/导入不回潮；purge 只承诺"canonical
本体 + 未投影 outbox 清除 + 抑制屏障保留"，**不宣称**备份/导出副本或
已发送上下文已物理清理。

**F11 Profile/Persona assertion ledger（T22/T23）**
- `ProfileService` 从"单值覆盖"改造成断言账本的投影：user / plugin /
  dormant（internalized）多层断言并存，`project()` 按 user 层优先、
  否则应用序（`appliedSeq`）最新胜出，且 revoked/expired/dismissed 的
  非 user 层不参与投影。
- 撤销语义：`revokeAssertion` 按来源撤销单层，撤销 user 层后下层
  （如已学到的 persona）自动复活（T22）；撤销层不会再被旧 upsert
  覆盖式"复活"——非 user 写入在同 key 有 live user 层时直接拒绝。
- 防回潮：`dismissInferredKey` 抑制键抑制后续同 key 推断复活（T23）；
  `removeUserOverride` 默认 `suppress` 模式（移除用户覆盖但保留抑制，
  下层不自动回潮），`restore` 模式才显式允许回潮。
- 导出/导入：`exportJSON` 升至 version 2，导出**全部断言层** +
  dismissed 抑制键，导入 v2 原样恢复账本（T23b：导出导入不回潮）；
  v1 旧格式仍走 upsert 兼容路径。
- 桌面 `DesktopAgent` 暴露 `memoryStatus()` 已就绪（见 F14），profile
  服务本身经既有 RPC 面使用，本阶段未改 dormant 包内部撤销逻辑
  （dormant 走 profile service 的 `remove`/dismiss 语义入口）。

**F12 有效遗忘传播（T24/T25）**
- canonical 契约新增 `applyForget({fragmentId, principalId, reason})`：
  suppression + tombstone + epoch+1 在同一事务提交，返回新 epoch；
  InMemory 与 SQLite 两实现一致。`PersistentMemoryStore.forgetFragment`
  改走 applyForget 并同步移除内存投影。
- epoch 门：`assertMemoryEpoch(runEpoch)` / `MemoryEpochStaleError`；
  桌面 contextProvider 在 `run.memoryEpoch` 落后时整表重播种
  `MemoryHybridIndex`（forget 后旧 run 不再读到已遗忘记忆）。
- purge 基础版：`purgeMemory(fragmentId, principalId)` 先立 "purge"
  抑制屏障，再 `purgeFragment` 删除 canonical 记录本体与未投影
  outbox job，清理向量/metadata 投影；抑制屏障**保留**以防旧备份/
  导入重建同 id。报告明确返回 `backups: "unsupported"`（T25：
  本体消失、barrier 保留、备份未承诺）。
- T24（forget 原子性 + epoch 门 + 重启不可见）、T25 在
  `planes/memory/__tests__/control.spec.ts`，用真实 SQLite 临时库验证。

**F14 记忆状态面**
- `PersistentMemoryStore.memoryStatus()`：聚合 canonical 状态计数
  （active/tombstoned/quarantined/superseded）、suppression 计数、
  memory epoch、indexStatus；无 canonical 的降级部署返回降级形状，
  不伪造 canonical 健康。
- 桌面 `DesktopAgent.memoryStatus()` 透传该状态面。**IPC/UI 未接**，
  留待 UI 工作接线（本阶段只交付数据面，不新增渲染面）。

**验证（2026-11-09 实测）**
- 新增测试：profile +4（T22 恢复矩阵、T22b、T23、T23b），
  plane-memory control.spec 2 例（T24/T25）；G2 forget 用例按新
  epoch 语义更新。
- 结果：core 176、runtime 20、plane-memory 103/3、skills 65、
  storage-sqlite 37、taskpool 49、plane-execution 65/1、governance 41、
  profile 10、server 77/11、cli 54/6、desktop typecheck + 109/2 +
  生产构建全绿，零基线回归。注：批量回归首轮 core 曾出现 1 次
  perf 守护阈值抖动失败（hook-bus-bench，机器并发负载下），单跑
  176 全绿，与 G4 改动无关。

**G4 遗留 / 说明**
- purge 不覆盖：备份目录、导出 JSON、已发送给 LLM 的上下文副本；
  报告以 `backups: "unsupported"` 明示，需物理清理备份时另行决策。
- F14 状态面尚未接 IPC/UI；memoryStatus 仅桌面主进程方法就绪。
- 第二次压缩 inheritedCoverage 仍未做（G3 遗留延续）：derive 仍按
  最近一条 compacted 精确集隐藏，留待后续与管理面一并处理。
- dormant 包内部撤销仍走 profile service 语义入口，未深入改造其
  internalization-manager 内部状态机（E13 的 dormant 侧）。

**G4 遗留决策（2026-11-10，用户拍板）**
1. **purge 边界：维持声明制。** `backups: "unsupported"` 保持不变；
   桌面当前无应用内自动备份机制，副本（手工备份目录、用户导出件）
   不在应用掌控内。新增约束：**今后任何应用内产生的记忆副本
   （备份/导出/同步包），创建时必须登记位置**，使 purge 传播有据
   可循；功能立项时同步实现，不预先建空登记表。
2. **F14 状态面并入 G5。** memoryStatus 与 G5 的 writer 候选审核面
   共用一个 IPC 通道与面板（状态面作审核台页眉），不单独先行。
3. **inheritedCoverage：G5 前先行修复（已完成，见下）。**
4. **dormant 深方案并入 G5。** dormant 降为 profile ledger 的
   dormant 层写入方，PersonaConfig 变投影；内化提议并入 G5 候选
   审核通道（提议→批准流与 writer 候选同构），撤销/恢复统一走
   G4 ledger 语义。G5 启动时与 writer 候选接口一并设计。

**inheritedCoverage 先行修复（2026-11-10，G5 前）**
- 背景核实：compactor 每次从原始 journal 事件重折叠全部轮次，
  当前配置下覆盖集"碰巧"单调递增，derive 只看最后一条 compacted
  事件才未出故障；该不变量隐含且无强制（改 keepRecentTurns/cap、
  将来 compactor 不再重序列化已覆盖轮次、或外来 journal 都会让
  已压缩轮次以原文回潮）。
- 修复：`core/src/journal/derive.ts` 改为按 seq 顺序遍历全部
  `session.compacted` 事件、并集覆盖集（F09 精确集与 legacy 前缀
  路径可混合），只展示最新一条摘要，计数为并集总数。旧 journal
  无需迁移。
- 测试：core journal.spec +2（覆盖非单调、legacy/新格式混合），
  core 178、runtime 20、server 77/11、cli 54/6、desktop
  typecheck + 109/2 全绿。

---

## G5 启动：范围与子阶段（2026-11-10，用户拍板）

范围 = 设计 F13（writer 影子闭环）+ 后续并入四项：技能正向学习
验证器（G3 决策）、Hy-Memory-30B writer 候选对照（§10)、F14 审核台
（G4 遗留决策 2)、dormant 深方案（G4 遗留决策 4）。

**子阶段划分与门控（逐个暂停汇报）**

| 子阶段 | 内容 | 验收 |
| --- | --- | --- |
| G5.0 契约与校验器 | MemoryWriter/MemoryCandidate 严格 schema、host 生成 candidateId、校验管线（来源 hash/span、scope、epoch、revision、用户值优先、敏感项、大小/未知字段）、fixture writer | T26 全拒绝/隔离有原因，不进活跃投影 |
| G5.1 影子管线 | derivation writer 任务（单 worker、预算、重试上限、死信）、validated → shadow_only（不 apply、不建索引投影）、shadow ACK 只确认完整接收 | T27（fixture 部分）；影子零影响（回答/Profile/技能/权限逐项不变） |
| G5.2 审核台（F14） | IPC + 面板：memoryStatus 页眉、待审候选（来源/证据）列表，approve 才由 host 执行 mutation | T28（桌面部分）；模型只能建议，mutation 由代码/用户批准 |
| G5.3 技能验证器 | verified 信号源接线桌面（现恒传 false） | 证据通过才升权；completed 照旧中性 |
| G5.4 真实 writer 对照 | 通用 writer 首轮 + 原文基线（固定 reader/retriever/预算、§10 十类样本人工复核）→ 激活建议报告；Hy-Memory 第二轮补入 | §10 指标；产出激活决策依据，不在 G5 内激活 |
| G5.5 dormant 并入 | dormant 降为 profile ledger 的 dormant 层写入方；PersonaConfig 变投影；内化提议复用审核台通道 | 撤销/恢复走 G4 ledger 语义；提议经审核不直写 |

**已确认的 G5 边界决策（2026-11-10）**
- 门控：子阶段逐个暂停汇报，不连跑。
- 验证器信号源：**工具证据为主 + 用户反馈可覆盖**——ToolOutcome 的
  business/evidence 核验通过才 verified；用户显式否定可撤销已记的
  正向信号。
- Hy-Memory 时机：**第二轮补对照**。协议预留同接口，首轮只做原文
  基线 vs 通用 writer，避免 8–25GB GGUF 部署阻塞主线。
- 激活边界：**G5 只到激活决策**（影子 + 对照 + 报告）。实际开放
  自动 apply（哪怕低风险白名单）另开门控；人格/程序性策略始终
  需审批（设计 §10）。

### G5.0 实施明细（2026-11-10 完成）

范围：候选契约与校验器（W1）。纯确定性、零模型、零既有路径改动
（新增 `planes/memory/src/writer.ts` + 导出一行）。

- **契约**：`MemoryWriter.propose(input) → { candidates: unknown[] }`
  （输出按未受信处理）；`WriterInput` 只含 committed 证据 span、
  scoped 记录快照、allowedKinds、policyVersion、inputHash、
  memoryEpoch、extractionRevision，scope 由 host 固定、不入候选。
- **严格 schema**：`MemoryCandidateSchema`（zod strict）——operation/
  kind 枚举、content ≤2000、sourceRefs ≤8、未知字段（包括试图自带
  `scope`）一律 `SCHEMA_INVALID`。
- **host 身份**：`hostCandidateId = cand_ + sha256(extractionRevision:
  inputHash:ordinal)[..24]`，模型自报 candidateId 被覆盖。
- **校验管线**（顺序确定，稳定原因码）：noop 短路 → NO_SOURCE /
  EVIDENCE_NOT_FOUND / EVIDENCE_SPAN_INVALID / EVIDENCE_HASH_MISMATCH
  （以 host 存储重算 span 哈希，writer 自带文本不作数）→
  KIND_NOT_ALLOWED → SENSITIVE_CONTENT（保守秘密模式，拒绝而非脱敏
  通过）→ VALIDITY_INVALID → supersede/retract 目标链（MISSING_TARGET
  / TARGET_NOT_FOUND / TARGET_SUPPRESSED / SCOPE_VIOLATION /
  MISSING_EXPECTED_REVISION，revision 不符为 **conflicted** 可重提）
  → USER_VALUE_PROTECTED（user 占位 key、user 断言目标）。批次层先查
  epoch，过期整批 STALE_EPOCH（T24 迟到 writer 口径）。候选内容只作
  数据保存，本子阶段不存在任何 apply/执行路径。
- **fixture writer**：只认显式标记（记住：→fact，我偏好→preference），
  无标记零候选，同输入同输出、host id 稳定可重放。
- **验收 T26**：`__tests__/writer.spec.ts` 10 例覆盖无来源、hash/span
  错、越 scope（字段注入 + 跨 workspace 目标）、过期 epoch、错
  revision、未知字段、敏感内容、用户值保护、指令内容惰性。
  plane-memory 113/3 全绿，零基线回归。

**G5.0 遗留 / 说明**
- 校验器是纯函数，尚未与 G2 `receiveCandidates` 串联（影子管线
  G5.1 接线：validated → shadow_only 存储，不建索引投影）。
- 敏感检测为保守正则（密钥/口令模式），不做语义级敏感分类；
  真实 writer 接入（G5.4）时以人工复核样本校准。

### G5.1 实施明细（2026-11-10 完成）

范围：影子管线（W2）。新增 `planes/memory/src/writer-pipeline.ts`；
canonical 契约扩展候选结局存储（InMemory + SQLite 两实现，
`memory_candidate_outcomes` 表，重启可读）。

- **管线**：`ShadowWriterPipeline.enqueue(job)` 串行链单 worker；
  job = 五元接收键 + WriterInput。流程：证据预算预检 → writer 提名
  （异常以同一 extractionRevision/inputHash 重试，默认共 3 次）→
  G5.0 批次校验 → G2 `receiveCandidates` 持久化批次 → 结局逐条
  upsert（validated→`shadow_only`、rejected/conflicted 原样）。
- **ACK 语义**：`shadow_recorded` 只确认"批次完整接收 + 结局已
  记录"；被拒候选与空批次同样 ACK，不代表生效。重放同键同 hash
  返回 `replayed` 不重复记结局；同键异 hash 死信 `RECEIVE_CONFLICT`。
- **死信**：`WRITER_FAILED`（重试耗尽，零落盘）/ `OVER_BUDGET`
  （证据超长不调 writer；候选数超限整批死信不截断）/ 上述冲突；
  死信只记录不自动重投，"不无限自动修复生成"。
- **影子零影响（实测断言）**：管线全程不调 applyMutation、不建
  索引投影、不碰 epoch；测试以种子记录对账 records/indexStats/
  epoch 前后完全一致。
- **验收 T27（fixture 部分）**：`writer-pipeline.spec.ts` 9 例——
  零影响、ACK≠生效、重放幂等、接收冲突、同版本重试、双死信、
  过期 epoch 整批拒绝（T24 联动）、单 worker 顺序、真实 SQLite
  重启后结局可读。plane-memory 122/3、storage-sqlite 37、desktop
  typecheck + 109/2 全绿，零基线回归。

**G5.1 遗留 / 说明**
- 管线尚未接宿主触发：桌面"turn committed → 构造 WriterInput →
  enqueue"接线与 G5.2 审核台同批落地（无查看面时先跑影子只会
  攒不可见数据）。`buildContext` 的真实实现（从 PersistentMemoryStore
  快照 records/suppression/user keys）届时一并提供。
- 通用/Hy writer 的同协议对照（T27 后半）归 G5.4。

### G5.2 实施明细（2026-11-10 完成）

范围：审核台（W4）+ 桌面宿主触发接线。兑现 G4 遗留决策 2
（F14 状态面与审核面共用 IPC 通道，状态面作页眉）。

- **store 审核 API**（plane-memory，`PersistentMemoryStore`）：
  `listMemoryCandidates`（无 canonical 返回空列表不伪造）+
  `reviewMemoryCandidate(candidateId, approve|reject, principalId)`。
  只有 `shadow_only` 可审核；approve 由 host 代码执行 mutation——
  add → `addLongTermAsync`（带 kind/canonicalKey/validUntil 与
  viaCandidate 来源 metadata），supersede → apply 时复查目标
  revision 后 `correctMemory`（漂移报 `candidate_revision_conflict`
  且保持 shadow_only），retract → `forgetFragment`（走 G4 有效
  遗忘），noop 直接转 applied。reject 记新终态 `dismissed`（与
  校验 rejected 区分）。结局状态联合扩展 applied/dismissed
  （canonical 契约 + SQLite 本地类型同步）。
- **管线小改**：`WriterShadowJob` 加 host 固定 `scope`；
  `buildContext(input, job)`；新增标准上下文构造
  `buildCanonicalValidationContext`（canonical 全量快照 + user
  占位 key + suppression + 当前 epoch）。
- **桌面接线**：canonical 启用且 `OPENINTJ_MEMORY_WRITER != off`
  时装配影子管线（fixture writer）；每次 `rememberTurn` 落盘后以
  用户原文为证据 fire-and-forget 入队（commitId 为内容哈希，
  同文重放幂等），失败只 warn 不影响回答。新增 IPC 四通道
  （memory.status / memory.candidates / memory.candidate.approve /
  memory.candidate.reject）+ preload 四方法 + DesktopAgent 三方法。
- **审核台 UI**：`MemoryPanel` 顶部加状态面页眉（canonical 启用
  态、各状态计数、epoch、索引待投影/失败/代次），中部待审候选
  区（kind/operation/证据数/正文/理由 + 通过并写入/驳回），已决
  候选只读展示状态与原因；原语义检索保留在下方。
- **验收**：新增 `review.spec.ts` 5 例（真实 SQLite：approve add
  生效/重复审核被拒、reject 零落盘、supersede 生效 + revision
  漂移冲突、retract 走有效遗忘 epoch+1、无 canonical 降级）。
  结果：plane-memory 127/3、storage-sqlite 37、server 77/11、
  desktop typecheck + 109/2 + 生产构建全绿，零基线回归。

**G5.2 遗留 / 说明**
- 影子触发只覆盖 `rememberTurn` 路径（主对话/生图后处理）；证据
  span 只含用户原文（助手文本不作证据，防助手回声自证——与 §10
  "助手回声"样本口径一致）。
- 审核 apply 的 add 路径 assertion 默认 assistant 层：候选证据
  虽来自用户原文，但经 writer 转述后不直接标 user 断言；用户在
  审核台点"通过"即人工背书，来源记在 metadata.viaCandidate。
- 审核台为桌面单端；server/cli 无候选审核面（两端本就无持久
  canonical 桌面式部署）。

### G5.3 实施明细（2026-11-10 完成）

范围：技能正向学习验证器（W3），兑现 G3 决策（与 writer 同批做）
与 G5 启动决策（工具证据为主 + 用户反馈可覆盖）。

- **验证器**（skills 新增 `src/verifier.ts`，纯函数）：`verifySkillOutcome
  (status, toolOutcomes, feedback?)`。规则：用户负反馈一票否决 →
  非 completed 不验证 → 任一 business=failure 在场不验证 → 存在
  business=success + evidence=verified 的工具证据则 verified
  （tool_evidence）→ 否则用户正反馈可独立构成 verified
  （user_feedback）→ 其余未验证。原因码稳定可诊断。
- **反馈追认**（learning-runtime）：结算时把 selection 保留进有界
  `settledById`；`recordFeedbackForSelection(selectionId, sentiment)`
  一次性消费——positive 把"未验证 completed"补升为 verified 同
  路径（升权 + 蒸馏 buffer），negative 记一次 -0.5（与已生效正向
  相抵）；observe 模式只记观察。重复/未知反馈返回 false。
- **桌面接线**：结算点从 trajectory 提取 ToolCallResult 三轴喂
  验证器，`verified` 不再恒 false（更新 G3 遗留注释）；新增
  conversationId → 最近已结算 selectionId 的有界映射与
  `recordSkillFeedback` agent 方法；IPC `skills.feedback` +
  preload；ChatPanel 在最近一条助手消息下提供「有帮助/没帮助」
  反馈（App 以 activeConversationId 绑定）。
- **验收**：skills 新增 `verifier.spec.ts` 7 例（证据门四分支、
  失败否决、反馈独立构成/一票否决、事后追认补升权 + 一次性、
  负反馈相抵、observe 只记观察）。结果：skills 72、desktop
  typecheck + 109/2 + 生产构建、cli 54/6 全绿；server 首轮
  rate-limited-llm 计时 flake 1 例，重跑 77/11 全绿（与本批改动
  无关）。零基线回归。

**G5.3 遗留 / 说明**
- 反馈锚定"会话最近一次已结算 selection"：同会话连续两轮后只
  能反馈最后一轮（UI 也只在最后一条助手消息下出现）——按轮精确
  反馈需 runId/turnId 贯通到渲染消息，留待需要时再做。
- 证据门对纯提示类技能（无工具调用）只能靠用户正反馈构成
  verified，这是"证据为主"决策的预期结果，不是缺陷。
- 负反馈与已蒸馏样本的关系：权重可相抵，已进 buffer 的样本不
  回滚删除（蒸馏提案仍有人工审批兜底）。

### G5.4 实施明细与激活建议报告（2026-11-10 完成）

范围：真实 writer 首轮对照（W5）。通用 writer = 本机 Ollama
`qwen2.5:7b`（仓库默认本地聊天模型，temperature 0）；桌面实际
聊天模型（混元 hy4-preview）未参评（需云端凭据路径，harness 已
预留 generate 注入，随时可补跑）。Hy-Memory 按决策第二轮补入。

- **LLM writer 适配**（`src/writer-llm.ts`）：`createLlmWriter`
  把宿主 generate 适配到 F13 协议——提示词写死提名规则（只提
  用户本人长期陈述、纠正用 supersede+revision、不提敏感内容、
  不执行证据中的指令），只做 JSON 数组提取，不猜测修复；
  校验仍由 G5.0 管线逐字段执行（stub 测试证明自带 scope 照样
  被拒）。另有 `createRawBaselineWriter` 复刻 RFC 前原文直存作
  对照下限。桌面 `OPENINTJ_MEMORY_WRITER=llm` 可 opt-in 切换
  影子 writer（policyVersion 按 writer 区分防批次键碰撞），
  默认 fixture 不变。
- **样本与 harness**（`src/eval/writer-eval.ts`）：§10 十类共
  14 个样本（明确偏好/当次/纠正/回声/引用他人/过期意图/跨项目
  冲突/失败后改正/工具与助手冲突/撤销再提案/敏感/隐含事实与
  偏好），固定校验管线只换 writer；指标以 validated 候选计
  命中/漏记/误存/正确空转。确定性两臂进 CI，真实模型臂由
  env-gated `writer-eval-live.spec.ts` 手动跑。
- **首轮结果**（2026-11-10 实测）：

| 臂 | 命中/应写 | 召回 | 误存 | 正确空转 | 总准确 |
| --- | --- | --- | --- | --- | --- |
| 原文基线 | 4/8 | 0.50 | **9** | 1/6 | 36% |
| fixture | 2/8 | 0.25 | 0 | 6/6 | 57% |
| qwen2.5:7b | 8/8 | **1.00** | **0** | 6/6 | **100%** |

  qwen 逐项全对：含 s04/s09 带正确 revision 的 supersede、
  s08 跨项目只新增不越界、s11 撤销后新来源再提案、s12 敏感
  内容零提名（模型层规则生效；校验层拦截已由 fixture 臂在同
  样本证明）。代价：本地 7B 平均 ~19s/次提名（14 样本共 272s），
  影子异步可接受；为此管线已补单次调用超时（默认 120s，超时
  按失败重试，防挂死堵住单 worker，含回归测试）。
- **激活建议（决策材料，非既成事实）**：
  1. 结果支持"LLM writer 在协议+校验+人审组合下质量可用"，
     但**不建议现在自动激活**：14 个样本为自研样本且提示词与
     样本同源（有 teach-to-test 风险）、只测了一个模型、没有
     真实对话分布数据。§10 的质量阈值应看真实分布后定，不编造。
  2. 建议路径：桌面默认保持 fixture；真实使用中 opt-in
     `OPENINTJ_MEMORY_WRITER=llm` 影子积累真实候选，用审核台
     人审统计真实误存率；Hy 第二轮补齐三方对照；三者齐了再开
     激活决策门（届时最早只白名单"带原文证据的低风险事实+
     明确偏好"，人格/程序性策略仍审批——设计 §10 原口径）。
  3. 本子阶段不改任何默认行为，激活与否等用户在决策门拍板。
  **【2026-11-10 用户决策：采纳上述建议】**——默认保持 fixture，
  真实使用 opt-in llm 影子积累 + 人审统计，Hy 第二轮后再开激活
  决策门。本子阶段不改任何默认行为即为执行该决策。
- **验证**：plane-memory 133（+评估/超时测试，live spec 默认
  跳过）、storage-sqlite 37、server 77/11、cli 54/6、desktop
  typecheck + 109/2 + 生产构建全绿，零基线回归。

**G5.4 遗留 / 说明**
- 混元（桌面实际聊天模型）未做 writer 参评；harness 支持以
  宿主 generate 注入补跑，结论可能与本地 7B 不同。
- 样本集 14 个且自研，后续应从真实影子候选人审结果中沉淀
  留出样本（时间顺序留出，§10 原要求），避免与提示词同源。
- Hy-Memory-30B 第二轮：需先完成 GGUF 经 Ollama 导入部署
  （8–25GB），协议与 harness 已就绪，届时同一命令补跑。

### G5.5 实施明细（2026-11-10 完成）

范围：dormant 并入（G4 遗留决策 4 的深方案）。兑现方式说明：
内化提议的对象是 persona 断言而非记忆 fragment，故"并入候选
审核通道"落实为**统一账本 + 统一提议→批准→撤销语义**（提议
入口仍为 DormantPanel，与记忆候选在审核台分开呈现——对象与
生效路径不同，混排反而有害）。

- **dormant 降为写入方**：`InternalizationManager` 新增可选
  `PersonaLedgerPort`（结构端口，dormant 不新增包依赖）。注入
  后 approve 写穿账本 dormant 层；PersonaConfig 中账本托管字段
  成为账本 live 断言的**投影**（每次 approve/revoke/restore 后
  以 live 集重投影）；revoke 按 targetField 从 live 集定位断言
  只撤 dormant 层 + 登记字段抑制，重启后同样有效（不依赖内存
  映射）。被用户显式层遮蔽时账本拒绝写入，投影不显示该字段
  （用户值优先）。无 ledger 注入时保持旧行为不变。
- **防白撤销**：propose 前查 `isFieldDismissed`——撤销过的字段
  （或用户在档案侧 dismiss 过的 key）不再被重新挖掘提议，修复
  E13 dormant 侧"撤销无抑制记忆"的问题。
- **桥工厂**：`createProfileLedgerPort`（dormant 包内）封装字段
  映射（preferences→preferences、phrases/habits→habits、
  context→notes，与既有 importPersona 对齐）与 provenance 无损
  还原（`dormant:<targetField>`）。桌面装配处只剩 6 行结构适配
  （profile 构造已提前到 dormant 之前；OPENINTJ_PROFILE=0 时
  桥不启用，回退 dormant 本地权威）。
- **验收**：dormant 新增 6 例（假端口语义 5 + 工厂映射 1）；
  桌面新增真 ProfileService 集成 2 例（approve 后档案投影
  可见、user 值优先不被覆盖、revoke+抑制后同 pattern 不再
  提议、导出 v2 导入不回潮）。结果：dormant 65、profile 10、
  core 178、runtime 20、plane-memory 132/4（live spec 跳过）、
  skills 72、storage-sqlite 37、server 77/11、cli 54/6、
  desktop typecheck + 111/2 + 生产构建全绿，零基线回归。

**G5.5 遗留 / 说明**
- dormant 自身的 SQLite adapter 仍存 PersonaConfig 快照（与
  账本投影双写）：账本是权威，恢复时以账本重投影托管字段；
  adapter 快照保留作 persona 的非托管字段与降级路径。
- 审核台（MemoryPanel）未混排 dormant 提议；persona 提议仍在
  DormantPanel 审核。若未来要统一审核入口，需先统一两类对象
  的呈现模型，不在本阶段强行合并。

### Hy-Memory 第二轮对照（2026-11-10 完成，drop-in 臂）

部署（本机 i7-6700 / 40GB RAM / GT 630 基本不可用，纯 CPU）：
- 前置：Ollama 模型目录自 C 盘迁至 `F:\ollama-models`
  （`OLLAMA_MODELS` 用户环境变量；C 盘仅剩 11.2GB，原副本
  保留作备份），迁移后 embedding（768 维）与 qwen 对话验证
  可用。
- 模型：官方 `tencent/Hy-Memory-30B-A3B-GGUF` 的
  `Q8_0-expsIQ1_M`（8.03GB，路由专家 IQ1_M + 其余 Q8_0 的
  官方混合精度；SHA256 与官方值一致），GGUF 存
  `F:\models\hy-memory\`，经 Modelfile（num_ctx 8192）导入
  Ollama 为 `hy-memory:8g`。`hy_v3` 架构被 Ollama 0.40.13
  直接支持，未走 llama.cpp 回退。冒烟：中英抽取、更新
  （delete+add）、无记忆返回 `[]` 均正确；冷加载约 100s，
  生成约 4 tok/s。
- 跑法：与首轮同一提示词 drop-in、temperature 0、同一
  `evaluateWriter` + G5.0 校验管线。因单样本约 36s、写死
  600s 的 live spec 放不下 14 样本，改用同口径分片 runner
  （复用 dist 的 evaluateWriter，额外留存原始输出），口径
  与首轮一致。

结果（§10 固定 14 样本，应写 8）：

| 臂 | 命中/应写 | 召回 | 误存 | 总准确 | 耗时 |
|---|---:|---:|---:|---:|---:|
| 原文直存基线 | 4/8 | 0.50 | 9 | 36% | — |
| fixture writer | 2/8 | 0.25 | 0 | 57% | — |
| qwen2.5:7b（通用） | 8/8 | 1.00 | 0 | 100% | 272s |
| Hy-Memory 8g（drop-in） | 6/8 | 0.75 | 4 | 57% | 499s |

失败定性（原始输出留存于 `F:\models\hy-memory\eval-raw.json`）：
- **格式零障碍**：14/14 输出可解析 JSON 数组，drop-in 方言
  本身成立；两个 supersede 漏记（s04/s09）为同一系统性
  原因——`expectedRevision` 输出为字符串 `"3"`/`"2"`，被
  strict schema 以 SCHEMA_INVALID 拦下。语义与目标均正确，
  差在类型合规；若原生臂/提示词明确数字类型，预计可挽回
  至 8/8 召回（待验，不预支）。
- **判断误存 4**：s03 一次性提醒仍以 episode 提名（且自知
  "仅这一次"）；s06 把同事偏好存为记忆；s07 已取消的意图
  仍提名；s10 按量规判误存（其"以日志为准"抽取本身有争议
  空间）。s12 敏感内容被模型提名、由校验层 SENSITIVE_CONTENT
  拦下（校验网兜底的又一实证）。
- **重复提名倾向**：多个样本同一内容以两种 kind 各提一条
  （s02/s06/s10 等 validatedCount=2），现有指标不惩罚，但
  真实影子下会挤占审核队列，原生臂需关注去重。
- **成本**：约 36s/样本（qwen 约 19s），纯 CPU 可接受于
  离线对照；作为桌面影子 writer 仍偏慢且需常驻约 9GB。

结论：drop-in 口径下 Hy 未胜过通用 qwen2.5:7b（质量、速度
均然），且其优势场景本在原生框架提示词下——本结果只否定
"拿来即当通用 writer 换装"，不否定其原生上限。原生提示词
适配臂（需翻译层代码）是否立项，等激活决策门与影子数据
一并议；本轮不接桌面、不改任何默认行为。

## G5 收口状态（2026-11-10）

代码子阶段 G5.0–G5.5 全部完成，Hy-Memory 第二轮 drop-in
对照已补齐。剩余为非代码事项：
1. **激活决策门**：按已采纳的建议——真实使用中 opt-in
   `OPENINTJ_MEMORY_WRITER=llm` 影子积累真实候选、审核台人审
   统计真实误存率后，再决定是否开放低风险白名单自动 apply。
2. Hy 原生提示词适配臂与混元 writer 参评：可选诊断项，
   与激活决策门一并议，不单独排期。

### 影子积累期开启（2026-11-11）

按激活建议路径进入影子积累（决策：writer=混元云端、先修
盲区、补轻量可观测、双门复盘）：

- **修盲区**：新增 `buildScopedRecordSnapshots`（作用域内
  active 记录、fail-closed 过滤、时间倒序、上限 32），桌面
  入队时注入 writer 输入——writer 从此能提 supersede（此前
  `scopedCurrentRecords` 写死为空，纠正类只能以 add 重复
  提名）。校验上下文口径不变。
- **轻量可观测**：canonical 新增按 policyVersion 聚合的
  writer 计数（processedTurns/emptyTurns/deadLetters/
  candidatesNominated/startedAt），管线每次结算落盘
  （replayed 不重复计；计数失败不影响管线），SQLite 新表
  `memory_writer_stats`；F14 状态面透出并在记忆面板展示。
  分母就此成立（提名率/空转率/死信率可算；漏记率仍靠人审
  抽查，空转≠漏记）。
- **开启口径**：用户环境变量 `OPENINTJ_MEMORY_WRITER=llm`，
  writer 跟随桌面聊天模型（policyVersion `g5-llm-v1`）；
  影子期内切换聊天模型会混臂，需避免或记录。起始日
  2026-11-11。注：当日 Hy4 Preview 下线，全系统默认模型
  迁移至 Hy4 正式版（桌面配置档 `hunyuan-hy4`），影子 writer
  随桌面重启切换为 hy4——积累期第一天以此为界，量小可忽略，
  复盘时按 hy4 臂计。
- **复盘双门**：累计已审（applied+dismissed）≥50 条，或
  2026-12-02 届满，先到先开激活决策门复盘；已审候选沉淀
  为留出样本。全程影子，零生效路径变更，随时可 `off` 停。
- **验证**：plane-memory 134/4、storage-sqlite 37、desktop
  typecheck + 111/2 + 生产构建全绿（新增计数落盘、作用域
  快照、SQLite 重启后计数可读等用例）。

---

## 待决策 / 未解决问题

- 技能正向学习的验证器：已决策放 **G5** 与 writer 一起做（2026-11-09）。
- 止损开关默认值何时翻转 observe（建议 G3 的 F10 归因修好后评估）。
- CLI/server 持久 journal adapter 的排期（当前决策：桌面优先，两端标注
  `durability=memory`）。
- 物理 purge 的完整度承诺：已决策（2026-11-10）维持声明制——
  canonical 本体 + outbox 清除 + barrier 保留，备份/导出副本明确
  `unsupported`；未来应用内副本创建时必须登记位置（见 G4 遗留决策 1）。
- ~~真实用户数据迁移~~ 已于 2026-11-09 经备份 + dry-run 确认后完成
  （16/16 激活，见 G2 门控说明）。
