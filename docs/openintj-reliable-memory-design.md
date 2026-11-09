# openINTJ 可靠记忆闭环：基础修正详细设计

版本：设计稿 v0.1｜日期：2026-11-09｜用途：交给 CodeBuddy 或其他开发 agent 分阶段实施。

本文是设计规格，不构成执行授权。本次仅交付设计；开发 agent 的文件修改、生产行为调整和数据迁移，必须以用户随后明确授权的具体阶段为边界。

设计基线：[openINTJ](https://github.com/oceancolor/openINTJ) 的 main@2326b4b02ad578363fd470317ce733a039c775e0。本文基于只读源码审查，未修改实现、运行项目测试或部署记忆模型。接口、表名和测试名中标注“建议”的内容均为待实现规格，不是现有能力。

## 0. 目标、范围与交付原则

用户确认的目标是“可靠记忆闭环”；期望收益是“增加语义 memory writer 的架构收益”。本阶段不追求个人神经模型权重训练，不以 Hy-Memory 特定模型的基准优势作为项目收益证明。

可靠闭环定义：

**有来源的交互与结果 → 可靠最终提交 → 可重放派生任务 → 受约束的记忆候选 → 可解释的生效／更新／撤回 → 作用域内召回与使用 → 可归因反馈。**

本阶段要保证：系统知道自己记住了什么、依据是什么、在哪里有效、何时失效、如何纠正，以及失败时能否恢复。不能仅保证“又向数据库写了一行”。

### 0.1 范围内

- 最终 turn 提交边界与所有入口的一致性。
- scope、来源、有效性、修订与幂等契约。
- SQLite 权威记忆源、可重建向量索引、可靠 outbox。
- 工具结果与学习标签的拆分。
- 精确压缩覆盖、上下文预算与实际使用反馈。
- Profile／Persona 的来源撤销和覆盖恢复。
- 独立语义 writer 候选接口及影子运行。
- 数据迁移、故障降级、确定性验收和开发阶段门控。

### 0.2 明确不做

- 不整体迁移到 Python Hy-Memory SDK，不替换主聊天模型。
- 不重做完整文档知识库、分布式多租户平台、全量 agent 断点续执行。
- 不承诺工具副作用 exactly-once，不把 TaskPool 节点快照称为完整恢复。
- 不自动训练个人权重，不自动批准人格／认知策略／工具权限变更。
- 不把本地推理与完整私有化等同：外部模型、日志、导出与同步仍是独立的数据出口。

### 0.3 架构取舍

保留 TS runtime、TAO／ReAct、工具治理、现有包边界与 LanceDB。只新增必要的提交与派生层。逻辑权威源可以分属多个 SQLite 文件，但每个可靠消息边界必须在所属数据库中事务提交；不能伪称跨文件事务。

基础版本优先正确性：索引落后或不兼容时，可以退回基于权威正文的作用域内词法／内存检索。不能为保持向量功能而忽略 scope、使用旧向量空间或关闭正文持久化。

## 1. 已核实的现状与设计前提

以下为源码事实；其实际用户可见症状多数未动态复现。

| 编号 | 已核实现状 | 对闭环的影响 | 核心位置 |
| --- | --- | --- | --- |
| E01 | recordUserInputAsync／recordAssistantOutputAsync 直接存原文，importance 分别固定为 0.6／0.5 | 没有独立语义事实与纠正链 | planes/memory/src/memory-plane.ts:52–87 |
| E02 | finalizeJournalTurn 在 turnId 已有任意事件时返回；safeJournalAppend 吞异常 | finalize 不保证完整最终提交 | core/src/journal/sink.ts:10–19,145–177 |
| E03 | TAO 先写 assistant/end，宿主随后可能修订答案；Desktop IPC 又可能补产物 | Journal、Memory 与即时回答可能是不同版本 | core/src/loop/tao.ts:424–447；三端 agent 收尾；desktop/main/ipc-handlers.ts:710–739 |
| E04 | PersistentMemoryStore 从 Lance scanAll hydrate；正文不在 SQLite FragmentMeta 中 | 向量库损坏不能仅靠 SQLite metadata 重建正文 | planes/memory/src/persistent-store.ts:119–133；storage/sqlite/src/types.ts |
| E05 | 内存先变更／发 hook，随后 Lance 和 SQLite 分步写；同步 pending 链可能被 rejected 阻塞后续写入 | hook 不是可靠持久化回执，双写无恢复协议 | persistent-store.ts:154–175,222–238 |
| E06 | 自动 tier 晋升、工作溢出、clearWorking 没有统一持久化 mutation | 重启可能恢复旧 tier 或复活已清除工作记录 | planes/memory/src/store.ts；persistent-store.ts |
| E07 | ContextEngine 主链无 read scope；desktop contextTags 只贯穿部分写入／preflight | 面板／preflight 隔离不等于最终提示词隔离 | context-engine.ts；desktop/main/agent.ts:734–750,988–1005,1191–1216 |
| E08 | MemoryPlane 用默认配置构造后反射替换 store，未同步其 retriever.embedder | 公开检索与主 ContextEngine 可能使用不同向量空间 | desktop/main/agent.ts:695–750；server 同构装配 |
| E09 | geometryShader 对所有 tier 使用 24h 半衰期和 0.3 阈值 | long_term 存储不等于长期可用；旧稳定事实会被时间硬过滤 | core/src/types/shader.ts；shader/geometry.ts:22–35 |
| E10 | compact 摘要输入可能被字符 cap 截断，coveredTurns 却覆盖整个 fold | 未被摘要的内容可能从 history 投影隐藏 | runtime/src/compact.ts:178–204 |
| E11 | ToolHub 无 handler 时仍 success=true；非零命令退出可正常返回对象 | handler 正常返回不等于业务成功／真实执行 | execution/src/tool-hub.ts:247–266；workspace-tools.ts:123–144 |
| E12 | skills 用 query/taskType 归因；缓存命中不重放选择／allowlist 副作用；非 completed/failed/timeout 默认 +0.2 | 改写、并发、取消及分支易错归因 | skills/src/agent-helper.ts；learning-runtime.ts |
| E13 | Dormant revoke 直接 deleteNested；Profile importPersona 只 upsert | 无按来源恢复，撤销后 Profile 镜像可能残留 | dormant/src/internalization-manager.ts:124–156；profile/src/service.ts:97–117 |
| E14 | 共享 handle hook 未校验运行归属；desktop 指定会话取消仍 abort 所有 activeChats | 跨会话控制消费与取消污染风险 | core/src/handle/create.ts；desktop/main/ipc-handlers.ts:369–374 |

限定：内存 MemoryHybridIndex 在有 tags/type 过滤时会取至少全索引数量，再过滤后 topK；不能误报成它现在只取 4×topK。它仍缺少 scope，且排序统计来自全库。Lance taskTags 在返回候选后过滤；原生 where 的实际 prefilter 语义应实测，不能只根据方法调用顺序推断。

基线 CI run 42216257564 在 lint 阶段失败，后续测试 skipped；本文未读完整 lint 日志或修复。开发前应记录当前分支的独立基线，不声称本设计已通过现有测试。

## 2. 十二条闭环不变量

| 编号 | 不变量 | 验收含义 |
| --- | --- | --- |
| I01 | 每个已结束的新 turn 只有一个权威 final 和一个规范 terminal | 宿主后处理完成后提交；重复收尾不重复，部分过程事件不阻止收尾 |
| I02 | committed 仅表示可靠提交，不表示事实真实 | 返回明确 durability／commit receipt；未提交不得声称“已记住” |
| I03 | 派生记忆必须指向存在、可访问且未撤回的来源 | 用户、助手、工具、摘要与审批来源分开，摘要不增加证据次数 |
| I04 | 记忆可见性先于任何候选截断 | 所有入口、向量与词法两路、preflight 和面板使用同一 eligibility |
| I05 | 存储 tier 不扩大 scope，不提升事实信任 | short_term 晋升 long_term 不自动共享或变成确认事实 |
| I06 | SQLite 正文／状态是权威，向量索引可丢弃重建 | 索引故障只降低检索能力，不丢事实、不复活已删除记录 |
| I07 | 业务写入与派生应用可重复执行而不重复生效 | 稳定操作身份、payload hash、事务 receipt、版本检查 |
| I08 | 所有变更通过同一 mutation 边界 | 晋升、纠正、清除、撤销、共享都落盘且可恢复 |
| I09 | 稳定长期事实不会仅因变旧而失效 | validity 与 recency 分离，长期 active 事实保持候选资格 |
| I10 | 权限、反馈与取消按 run/branch 归属 | 文本不是身份，另一个会话的事件不能消费本会话控制 |
| I11 | 压缩只隐藏实际完整被覆盖的内容 | 保存 source coverage/hash；模型失败或覆盖不完整时保留原文 |
| I12 | 撤回先阻止生效，再重放清理各投影 | 已撤销值、被遗忘来源不能因旧任务、导入或缓存回潮 |

这些是不依赖 Hy-Memory 的系统性质。writer 必须适应它们，而不能以更强模型替代它们。

## 3. 目标架构与权威边界

```text
宿主验证 RunContext
  → TAO/ReAct/编排执行，记录可验证过程事件
  → 受治理后处理：答案修订、来源、产物兜底
  → commitTurn（所属 Journal 数据库事务）
      ├ final assistant + terminal
      └ derivation outbox
  → 后台 consumer（至少一次投递）
      → 固定 evidence envelope + writer input hash
      → writer 候选结果持久化
      → 事务接收 candidate batch / incoming receipt
      → 经校验／审批的 canonical mutation / index outbox
  → SQLite canonical memory + control ledger
      ├ Profile/Persona 派生投影
      └ LanceDB / 内存检索投影
  → eligibility → scope 内召回 → 预算装配 → 实际注入记录
  → 验证/用户反馈，按 run/selection 归因
```

### 3.1 最小职责拆分

- TurnCoordinator：放在 runtime，负责宿主生命周期和最终提交，不负责记忆语义。
- ReliableTurnJournal：core 定义端口，宿主／storage 提供持久实现；过程 append 与最终 commit 分开。
- MemoryRepository：完整记录与 mutation 的权威接口，不能继续把只有 metadata 的 store 当完整记忆源。
- MemoryDerivationWorker：消费 committed turn，生成并保存候选，不直接改变人格／技能。
- MemoryEligibility：统一 ownership、状态、有效期、来源撤回与适用条件判断。
- IndexProjector：可重放投影，不作为权威状态恢复源。
- MemoryControlLedger：审批、撤销、dismiss、forget 的权威变更记录。可与 canonical memory 同库，以便控制与生效事务一致。

不要求一次拆出全部新 package。先在现有 runtime/memory/storage 包建立边界；避免 core 直接 import SQLite 或 runtime 形成反向循环依赖。

### 3.2 两类 outbox，不混成一个“写入成功”

1. Journal derivation outbox：与 final/terminal 在 Journal 所属数据库同事务。Desktop 可扩展现有 workbench SQLite；不必先搬走所有工作台表。
2. Memory index outbox：与 canonical mutation/incoming receipt 在 memory SQLite 同事务。

跨两个库的接收模式：目标库先幂等接收完整抽取批次，再 ACK 源 outbox。接收键统一为 (sourceJournalId, sourceCommitId, consumer, policyVersion, extractionRevision)；sourceJournalId 是稳定逻辑命名空间，不是可变化的文件路径。同键不同 inputHash 报冲突。

ACK 仅表示候选全集（或明确的空结果／拒绝结果）与 incoming receipt 已在目标库同事务保存，不表示候选已生效。影子／待审批候选不产生 index outbox，只有 canonical mutation 才产生。ACK 前崩溃导致重投，复用已保存批次；尚未保存结果的推理可重算，但只能一次原子接收批次。禁止声称两个库同事务，也不需要跨库两阶段提交。

CLI/Server 若仍使用内存 Journal，只能报告 durability=memory；要验收跨重启可靠闭环，应配置持久 adapter。三端协议一致不等于三端默认持久化程度相同。

## 4. 数据契约：scope、记录、来源与操作

以下 TypeScript 仅为接口规格示意，不是实现代码；实施时应与现有 Zod／strict／exactOptionalPropertyTypes 规范对齐。

### 4.1 RunContext 与 MemoryScope

```ts
type MemoryScope =
  | { level: "user"; principalId: string }
  | { level: "workspace"; principalId: string; workspaceId: string }
  | { level: "task"; principalId: string; workspaceId: string; taskId: string }
  | {
      level: "conversation";
      principalId: string;
      workspaceId: string;
      taskId: string;
      conversationId: string;
    };

interface RunContext {
  principalId: string;
  workspaceId: string;
  taskId: string;
  conversationId: string;
  turnId: string;
  runId: string;
  parentRunId?: string;
  branchId?: string;
  traceId: string;
  memoryEpoch: number;
  signal: AbortSignal;
}
```

- 单用户本地版 principalId 可以来自固定 installation namespace；不需要引入登录系统，但不得从模型文本取得。
- ID 和层级关系由宿主验证；未知 conversation 不能变成 unscoped 全库运行。
- 无工作台的 CLI 显式创建 local workspace/task/conversation namespace，不把 undefined 解释成全局权限。
- 读范围是当前 conversation + 同 task 显式共享项 + 当前 workspace 显式共享项 + 用户确认的 user 项。
- 普通交互默认 conversation scope。模型不能自行扩大范围；scope 晋升需要显式 UI/API 操作或经授权的确定规则。
- taskTags 只表示语义标签，不表示授权。
- scopeKey 用规范结构编码／哈希生成，禁止简单拼接造成 ID 分隔符歧义。
- 空授权集合表示没有候选，不表示不加过滤；与可选的非授权 tag 过滤分开定义。
- 基础版 memoryEpoch 统一为 principal 级、由 canonical control ledger 维护的单调计数，不按各 scope 分别计数。一次有效遗忘可使该 principal 的旧运行重新授权，牺牲部分并发效率以简化安全正确性。

### 4.2 MemoryRecord：五个维度必须独立

| 维度 | 建议值／字段 | 规则 |
| --- | --- | --- |
| kind | episode / fact / preference / intention / procedure / state_reference | 表示语义用途，不能用 tier 代替 |
| tier | short_term / working / long_term | 缓存与组织策略，不是可信度或 ownership |
| state | active / superseded / revoked / archived / deleted | 派生候选审批另有状态，不与 record state 混用 |
| verification | user_stated / tool_observed / independently_verified / inferred / unverified | 来源与核验范围是显式事实；不能把模型 confidence 当校准概率 |
| applicability | scope + validFrom/validUntil + conditions + sourceStatus | 决定本次能否使用；recency 只参与排序 |

建议完整字段：recordId、revision、kind、tier、state、content、canonicalKey、scope、sourceRefs、verification、validFromMs、validUntilMs、supersedes、derivedFrom、policyVersion、createdAtMs、updatedAtMs、summaries、metadata。

canonicalKey 是 scope 内的语义键，用于识别同一项，例如 tool.preference.editor；模型只能建议键，最终由受控命名和冲突检查确定。不同项目中同名键不自动互相替换。一个候选包含多个可分离主张时应拆开，不能用一句长摘要同时承载互相独立的有效期。

现有 timestamp 使用秒，其他模块存在毫秒字段。本设计新字段显式 Ms／Sec 后缀；迁移适配旧字段，禁止混用无单位 number。顺序／并发裁决用提交 seq/revision，不用墙钟判断“最新”。

### 4.3 来源契约

```ts
interface EvidenceRef {
  sourceJournalId: string;
  sourceEventId: string;
  sourceTerminalSeq: number;
  sourceRole: "user" | "assistant" | "tool" | "approval";
  conversationId: string;
  turnId: string;
  runId: string;
  contentHash: string;
  span?: { start: number; end: number };
  callId?: string;
  observedAtMs: number;
}
```

- 用户陈述证明“用户这样说过”，不当然证明外部客观事实。
- 助手输出可保留 episode，但不得自动视为用户偏好或核验成功。
- 工具证据限定到实际读写对象、时间、版本和验证 predicate。
- 失败／取消 turn 中的用户明确事实仍可生成候选；不能因任务失败丢掉用户纠正。
- 引用文字、第三方材料和 steer/inject 的来源／意图需要标记，不能当本人长期偏好。
- 原始事件通常保留；不要求存模型内部推理作为学习证据。优先存可观测决策、动作、结果和用户纠正。
- span 在基础版约定为原始事件 content 的 UTF-16 code-unit 半开区间 [start, end)，与 TypeScript 字符串切片一致；hash 对原始文本 UTF-8 编码求值，不在验证前静默正规化。中英混合、emoji／代理对都要测试，避免错引用。
- span/hash 不匹配、来源已撤回、scope 无权限，候选必须拒绝／隔离。
- 派生摘要保存 derivedFrom，不成为新的独立 evidence；助手复述用户话不能增加“用户独立说了几次”的计数。

### 4.4 幂等与版本契约

建议 writeKey 由宿主稳定生成：namespace + conversationId + turnId + sourceEventId/segmentId + operationKind。用 SHA-256 对规范 payload 求 requestHash。

- 同 scope/writeKey/hash：返回同一 receipt，不再次生成 record、outbox 或奖励。
- 同 scope/writeKey、不同 hash：IDEMPOTENCY_CONFLICT；修订要新 mutationId。
- 相同正文、不同来源事件：可保留两个 episode，语义层去重由后续候选协调完成。
- 所有修订带 expectedRevision；冲突后重新读取当前有效记录，不盲写旧值。重读不能自动授权旧抽取结果覆盖新值。
- 来源序位为 (sourceJournalId, conversationId, terminalSeq)，只在同一 Journal、同一 conversation 内可比较；当前宿主 seq 可能按 conversation 分配，不能跨会话按数字大小推断新旧。不同 Journal／conversation 的来源默认不可比较，除非存在经验证的显式纠正／替代关系。
- 同 scope/canonicalKey 已有有效值时，add 不隐式覆盖；supersede 必须指向 targetRecordId/expectedRevision，且旧来源不得覆盖较新的纠正。不可比较、语义有冲突或跨时间效力不清楚时进入 conflicted 待确认；不能按 writer 完成时间判断来源更新。
- 字段共享／撤回也是 mutation，不允许直接修改内存对象绕过版本。
- mutation 收据必须能表示 committed、replayed、index pending/ready 与 durability。

## 5. 基础闭环修正清单（F01–F14）

优先级：P0 为正式闭环前必须成立；P1 为语义 writer 正式激活前必须成立；P2 为性能与更深经验优化，不阻塞影子验证。每项均包含改动、边界、验收和依赖。

### F01｜P0：统一最终提交所有者

**目标**：所有入口共享一个 host-owned finalization，保证用户最终可见答案等于已提交 final。

改动：

1. TAO 增加 completion owner 契约。宿主模式只记录过程／执行结束，不提前提交宿主 final/terminal。独立 loop 使用者可保留 loop-owned 兼容模式，但必须声明能力。
2. runtime 建立 TurnCoordinator，覆盖 prelude、clarify、preflight、image、preview、普通 TAO、TaskPool、self-consistency、scheduler／直接 handle 调用。
3. 所有答案修订、来源附加、受治理产物兜底发生在 commitTurn 之前。Desktop IPC 不再承担唯一的可靠收尾职责。
4. commitTurn 返回 receipt；final assistant + host terminal + derivation outbox 在所属数据库同事务。部分 start/user 不阻止提交，重复 finalization 比较 payload hash，内容不一致报冲突。
5. 规范 status 映射，不采用“非 clarify 全 completed”。失败、取消、超时、max_iter 分开；已有未闭合过程可标 interrupted，不恢复成成功。
6. 对原有事件不做静默覆写。旧 loop final 作为 legacy/draft 处理；新投影明确使用 host final。迁移不能凭事后猜测生成已验证成功。

位置：core/journal/types.ts、sink.ts、core/loop/tao.ts、runtime/run、三端 agent.ts、desktop ipc-handlers/workbench-store。

验收：T01、T02。依赖：F02 的身份契约、F03 的结果标签。不承诺外部工具副作用与最终提交同事务。

### F02｜P0：运行身份、控制和取消隔离

**目标**：run/turn/branch 是身份，query 不是；共享 bus 不产生跨会话控制。

改动：

- 不可变 RunContext 从入口贯通 TAO、ReAct、ToolHub、approval、contextProvider、TaskPool、subagent、skills 和 outbox。
- handle 的 steer/inject 只消费所属 scope；增加 dispose 退订。SSE／桌面推送也按 scope 过滤。
- controller 在入队时建立。queued 取消不会在 execute 时重新 running；followup 创建新 turnId，child 使用独立队列，避免父 run 等 child、child 又排父 tail。
- 当前会话取消只取消目标 run；全局 cancelAll 是不同 API。
- hook 取消链、阻止业务操作、取消整个 run 三种语义明确区分；消费者检查对应 outcome，isCancelled 不使用过期快照。
- signal 贯通审批等待；取消后迟到 approve 无效。副作用前、provider 返回后、宿主后处理前重检 signal。
- terminal commit 是线性化点：提交前的有效取消写 cancelled；提交后不再改 terminal。已发生或不确定副作用原样记录，不假装回滚。

位置：core/handle、core/hooks、core/loop、governance/approval、desktop ipc、server routes、TaskPool/subagent adapter。

验收：T03、T04。范围只做到记忆与控制正确归属，不重做整个并发架构。

### F03｜P0：工具、业务、任务和验证结果拆分

建议 ToolOutcome 字段：executionStatus（not_executed/blocked/completed/error/cancelled/timeout）、businessStatus（success/failure/unknown）、evidenceStatus、callId、runId、origin、outputRef、sideEffectStatus。

规则：

- callId 在执行前建立；无 handler = not_executed，不记 breaker success，不给正向学习。
- handler 正常返回不是业务成功。命令非零退出为 failure；未知协议为 unknown，不能默认为 success。
- 成功写文件的 receipt 包含路径、bytes、contentHash 或实际版本；必要时读回核验。验证只证明对应 predicate，不证明用户整体任务满意。
- Desktop 自动补写通过受治理执行门面，origin=host_fallback；保留路径沙箱和 gate，写真实工具／产物回执。
- fallback 成功不能让 failed/cancelled 变 completed，也不能自动给原 skill 功劳。
- 带 claim 的经验只可使用匹配的核验结果。run_completed、journal_committed、evaluation_passed、user_accepted、candidate_approved 均不是同一标签。

位置：execution/tool-hub.ts、workspace-tools.ts、shared/product-behavior.ts、desktop ipc、core/journal tool.result。

验收：T05、T06。兼容字段 success 可短期保留为派生值，旧消费者逐个迁移；禁止旧布尔继续决定学习晋升。

### F04｜P0：scope 与 eligibility 全链一致

改动：

1. BuildContextInput／TaoContextInput／run options／RetrieveOptions 都接受经过宿主验证的 readContext。
2. 默认检索先过滤；hybrid 在可见集上评分、BM25 统计和 RRF，再 topK。
3. Lance 向量与 FTS 在每路 candidateK 前使用相同授权条件。未验证原生 prefilter 的路径关闭或改为作用域内内存检索，不能降级全库。
4. 外部 candidateRetrieve 返回结果后再次权威校验；这只是防御，不代替召回前过滤。
5. preflight、memory panel、最近记录、搜索 API、summary、writer 输入与 export 都走同一 eligibility。全局管理查看需要显式管理上下文，不能偷偷复用普通查询。
6. 临时容量按 scope 管理，B 的写入不能丢弃 A 的 working records。有效长期事实不因全局临时容量淘汰。

位置：memory/context-engine.ts、retriever.ts、shader、taskpool/memory-hybrid-index.ts、hybrid-retriever.ts、storage/lance、三端入口／面板。

验收：T07–T09。未知 scope fail closed；tier 晋升不共享。基础版本不要求全局多租户认证系统。

### F05｜P0：完整 canonical memory 与统一 mutation

选择：SQLite 保存完整正文、来源、scope、状态、修订、摘要与控制；Lance 只是检索投影。

建议新增逻辑表：memory_records、memory_sources、memory_mutations/write_receipts、memory_candidates、memory_index_outbox、memory_index_generations、memory_control/assertions。

最小要求：

- MemoryRepository 以 async mutation 为持久提交边界。禁止“先改内存再证明写盘”的语义。
- 事务写完整记录／expectedRevision、mutation write receipt 和 index outbox；提交后才发布 committed change-feed。候选接收回执与正式应用回执是不同状态。
- 自动晋升、共享、工作溢出／清除、归档、纠正、删除全部调用 mutation。scope 内 clearWorking 不能全库清空。
- 缓存使用提交后的不可变快照；重启从 SQLite hydrate。访问统计可独立表聚合，不能因查询次数修改语义正文 revision 并无限重建向量。
- 旧 add* 同步接口只作为明确 memory-only compatibility；正式持久路径不可依赖它返回即可靠。
- canonical 写失败可不阻断主业务，但返回 memoryCommitStatus=failed，不能宣布“已记住”；失败不污染后续队列。
- tags 使用 JSON/关系表，不能继续把 CSV 当无损存储。

位置：memory/store.ts、persistent-store.ts、persistence-factory.ts、storage/sqlite types/sqlite、新 repository adapter。

验收：T10、T11、T12。不必迁移所有现有审计／任务表，只补权威记忆接口。

### F06｜P0：outbox、索引重放与降级

规则：

- source final + derivation outbox 同事务；destination candidate batch + incoming receipt 同事务；后续正式 canonical mutation + application receipt + index outbox 同事务。影子／待审批接收不建 index outbox。
- consumer key 使用 §3.2 的完整接收键。source 在目标完整批次接收提交后 ACK，允许至少一次投递；ACK 与候选应用状态分开。
- LLM 输出作为完整抽取批次先保存；同批次候选、空结果或拒绝结果都可形成接收回执。已保存批次的重投不重新抽取。保存前崩溃可再次推理，但以批次唯一键保证只接收一次；显式重新抽取是新 extraction revision，不是普通 retry。
- index job 携带 recordId/revision/generation。旧任务处理时读取最新 canonical state，不能旧 add 覆盖新 delete。
- 基础版同 generation 单 worker 或按 record 串行；不要先建设无界并行投影。
- Lance 成功但 ACK 前崩溃允许幂等重放；失败保留 attempts/lastError，其他任务继续，支持隔离坏任务。
- 投影落后、不可用或 fingerprint 不兼容时，使用 canonical 可见正文路径；不让 stale hit 占满 topK，也不遗漏新记录。可先全量 scoped scan，后续再做 delta overlay。
- GC 不删除未消费来源；或 outbox 保存足够的脱敏、不可变 envelope。forget 是另一个有明确优先级的控制操作，不能被 GC 保留规则阻止。

验收：T13、T14。不保证跨库原子提交，不保证外部副作用 exactly-once；保证应用幂等与可恢复投影。

### F07｜P0：embedding 空间身份与装配一致性

改动：

- MemoryPlane 支持注入同一个 store/embedder，取消 Object.defineProperty 反射替换。
- dimension 由 embedder 取得；显式维度冲突直接报错，不返回 relevance=0 装作检索成功。
- embedding fingerprint 包含 provider/model/revision（可得时）、dimension、transformVersion、normalization、query/document 编码约定。模型 revision 不可得时记录“不确定”，不要虚构。
- generation 另存 metric、projection schema；向量有限值／长度／内容 hash 和空间兼容性校验。
- 公共 retrieve、ContextEngine、hybrid、直接 vectorSearch 和工厂都使用同一 EmbeddingSpace，不能只在宿主启动校验。
- fingerprint 变化进入 rebuild，不混用同维旧向量；不可用时正文继续持久化和词法检索。
- Simple SHA 伪向量仅 mock/degraded；生产语义评测不得把它当真实 embedding。显示 memory durable、lexical available、vector unavailable 等独立状态。

验收：T15。不要求本阶段选择最终 embedding 模型；先保证不混用空间。

### F08｜P1：语义生命周期与冲突纠正

建议有效性：active 且在有效期内、scope 合法、来源未撤回、适用条件满足。未核验内容可作为“带标签的历史陈述”召回，但不能作为执行权限或成功经验。

- fact/preference：明确 user assertion 或批准后生效；时间只降排序，不默认 TTL。
- intention：带 validUntil；“本次先讨论”默认本次／当前 task，不推断为永久禁止实现。
- episode：保留过去发生的事件，不能投影为现在的状态。
- state_reference：含对象／版本／observedAt；行动前用工具重检。
- procedure：含适用前提、结果和验证条件；候选不自动成为可执行 skill。
- supersede：同 scope/canonicalKey 的明确纠正，事务激活新值并 supersede 旧值；不同有效时间段可共存。
- inferred 与用户显式值冲突：用户优先，推断留候选／冲突记录，不覆盖。
- inference 之间冲突：pending conflict，不按 importance 或模型 confidence 强行选最新。
- 被引用陈述与本人断言区分；跨项目偏好没有显式提升则不共享。

去掉 geometryShader 对稳定 long_term 的统一硬时间淘汰；recency 可参与排名。设置 TTL 的记录到期仍必须在所有入口不可见，不能只让 shader 不显示。

验收：T16、T17。24h 默认只可用于短期缓存策略，不是长期事实有效性规则。

### F09｜P1：压缩精确覆盖与上下文预算

最小实现：按完整 turn 打包；只压缩连续闭合前缀，不跨未闭合洞。

- coverage 保存实际 includedTurnIds、sourceEventIds、inputHash、sourceWatermark、clearEpoch、inheritedCoverage；upToTurnId 仅为连续覆盖的兼容字段。
- 单条截断、半轮输入、cap 未容纳的尾部都不能宣称完整覆盖。单轮超 cap 时分块完整处理或跳过并报告。
- 第二次压缩使用前摘要+新增长完整轮，继承来源覆盖，不无条件反复总结旧原文。
- 明确摘要是否覆盖 user/assistant、steer/inject、工具证据；未纳入的事件仍有独立来源路径。摘要不是工具事实的权威副本。
- LLM 失败、空摘要、clearEpoch 变化、异步水位不匹配则不提交新投影，保留原 history。
- ContextEngine 的 token snapshot 按当前完整消息集计算，不能每轮 build 将同一历史重复累加；按 run/conversation 隔离。
- 预算在最终拼装点硬校验。估算器不应宣称精确 token；有 tokenizer 用真实 tokenizer，无则保守估算+余量。自定义 summarizer 超长时再截断／移除，不越预算。

位置：runtime/compact.ts、core/journal/derive.ts、memory/context-engine.ts、shader/fragment.ts。

验收：T18、T19。不能用换更强摘要模型修复覆盖范围算法。

### F10｜P1：实际记忆使用与技能反馈归因

把 retrieved、injected、used、evaluated 分开：

- retrieval 不立即增加使用次数；实际写入最终 prompt 后记录 MemoryInjectedEvent。
- 事件包含 run/turn/iteration、recordId/revision、query hash、retrieverVersion、prompt segment 与 budget。重复同事件幂等。
- “回答引用该记忆”是模型／解析器给出的 usage claim，不等于因果贡献；需另存 claim 与验证结果。
- skills 缓存只缓存纯 selection 数据；每次在当前 scope 重放 allowlist、实际注入和 selection 记录。
- 缓存 key 包含 registry/weight 版本，不以原 query/taskType 作为 outcome 身份。
- selectionId/outcomeId/runId/branchId 归因，TaskPool 节点与根汇总、赢家与输家不混奖。
- cancelled/clarify/preview/preflight/unknown 默认中性；completed 不自动进入“验证成功经验”。
- 基础阶段关闭自动正向升权与自动蒸馏晋升，仅记录观察。指定判据或用户反馈通过后才有限评估；负面反馈也不自动证明某 skill 造成错误。
- 未来蒸馏输入至少包括任务、适用前提、实际动作／结果、验证范围、失败修正和用户反馈，而非仅 query+工具名。

验收：T20、T21。保留审批和权重 clamp，但它们不能替代归因正确性。

### F11｜P1：Profile／Persona 按来源撤销与恢复

禁止只补一个 oldValue 字段作为完整回滚方案。采用 assertion ledger：assertionId、section/key、value、sourceType/sourceRef、sourceField、evidenceRefs、appliedSeq、revokedSeq、validUntil。

投影：先过滤 revoked/expired/suppressed；有效用户覆盖优先；非用户有效层按确定提交顺序裁决。保留被覆盖层，撤销只撤相应来源。

三种 API 必须不同：

- revokeProposal：撤一个 proposal 来源，不永久禁止整个 key。
- dismissInferredKey：用户不希望再使用该 key 推断，持久 suppression；普通导入不可绕过。
- removeUserOverride：明确是恢复仍有效推断，还是“忘记此项并抑制回潮”；默认不要静默复活旧推断。

恢复矩阵：baseline X，P1=A，P2=B。撤 P1 仍 B；撤 P2 且 P1 有效则 A；都撤则 X/无值；用户 U 不受 proposal 撤销影响；过期 A 不恢复。phrases.x 与 habits.x 保留来源 namespace。

ledger 提交后，Profile/Persona JSON 可以异步重建，但渲染必须校验 ledger 水位，不显示已撤销值。ledger 失败不能返回“可靠撤销成功”。export/import/sync 包含 revocations/suppressions/版本，不能只导出 live entries。

位置：profile/service/store、dormant/internalization-manager/runtime/persistence、storage/sqlite/dormant、sync bundle。

验收：T22、T23。旧数据已覆盖掉的前值无法凭空恢复，必须在迁移报告中说明。

### F12｜P0/P1：撤回、遗忘与隐私删除不回潮

三个层次：

1. archive：不常规召回，但原始证据保留。
2. retract/forget-effective：停止在后续 prompt／writer／Profile／skill 使用相关内容，保留不含敏感正文的 tombstone/control barrier 防回潮。
3. purge：用户要求物理删除；清理原文、派生记录、摘要、索引、缓存、outbox payload、导出／同步副本，并报告不可控副本与备份保留范围。

不能把 tombstone 当“物理删除完成”。conversation clear 默认是历史投影清除，不等于全部个人记忆遗忘；产品要明确提示并提供独立范围选项。

控制：forget 的 source suppression 与 principal memoryEpoch 在 canonical control ledger 同事务提交；来源撤回以该 ledger 为准。后台候选应用在同一目标事务中检查当前 epoch 与来源状态；迟到 writer、旧 outbox、旧索引不能复活。

每次 prompt／writer 输入在调用派发的授权点复查 epoch；过期则重建或拒绝。基础版用 principal 级串行授权门与控制提交协调，定义“撤回提交之后的新授权不会包含被撤回内容”。已经授权或已发送给模型的调用不承诺追溯撤回，可尽力取消；返回结果仍须过当前控制检查后才能应用。新证据是否可再提案，由 source revoke 和 key dismissal 的不同规则决定。

F12 的 control 表、epoch 与 suppression 读写／检查在 G2 作为 P0 先建立，G4 再完成管理操作、各投影传播与删除语义。

脱敏：writer 输入、候选、telemetry 默认去凭证／密钥；原始日志按用户设置与最小权限保存。自动推断敏感个人画像默认拒绝或要求明确许可。外部 writer 发送前需要明确数据出口策略。

验收：T24、T25。基础版本至少交付有效撤回与 barrier；物理 purge 未完整实现时必须显示 unsupported/pending，不宣称已彻底删除。

### F13｜P1：独立 writer 候选接口与影子验证

只接语义 writer，不让其获得文件写入／工具执行或权限配置能力。

```ts
interface MemoryWriter {
  propose(input: WriterInput): Promise<WriterResult>;
}

interface WriterInput {
  committedTurnRef: string;
  evidence: readonly EvidenceRef[];
  scopedCurrentRecords: readonly RecordSnapshot[];
  allowedKinds: readonly string[];
  policyVersion: string;
  inputHash: string;
  memoryEpoch: number;
}

interface MemoryCandidate {
  candidateId: string;
  operation: "add" | "supersede" | "retract" | "noop";
  kind: string;
  content: string;
  sourceRefs: readonly EvidenceRef[];
  proposedKey?: string;
  targetRecordId?: string;
  expectedRevision?: number;
  proposedValidity?: { validFromMs?: number; validUntilMs?: number };
  rationale: string;
}
```

实际 schema 使用严格 enum、大小上限、未知字段拒绝。scope 由 host 固定；候选不能扩大 scope。candidateId 由 host/extraction revision/ordinal 生成，不能相信模型自报身份。

候选状态：proposed → validated → pending_approval / shadow_only → applied；旁路 rejected/conflicted/superseded/revoked。JSON 可解析只是第一步，不能等于 validated。

校验：来源/hash/span、授权、epoch、target revision、用户显式值优先、秘密／敏感项、操作边界、大小限制和 no-op 处理。模型可建议撤回；真正控制 ledger mutation 由代码／用户批准执行。

影子模式：输出与验证结果可查看，不改变普通回答、Profile、Persona、技能或工具权限。故障只影响派生任务。预算限额、并发单 worker、重试上限与明确死信原因；不无限自动修复生成。

后续激活：最先开放带原文证据的低风险事实与明确用户偏好，procedures/cognitive strategy 继续审批。Hy-Memory、通用 writer、确定性 fixture writer 使用同协议；固定 reader/retriever/budget 做对照，不把架构变化归功于特定模型。

验收：T26、T27。此项是基础闭环收口后的 writer 入口，不是提前全面接入模型。

### F14｜P1/P2：可观察性与管理面

最低状态：journalDurability、canonicalStatus、derivationPending/failed、indexLag、vectorSpaceStatus、scope、active/revoked count、controlWatermark。

最低用户操作：查看这条记忆的来源／有效范围／类型；纠正／撤销／共享；查看待审批项；确认未持久化／索引降级；知道正在使用哪些记忆。

最低诊断：按 commitId/runId 查 final→candidate→record→injected→feedback 链，脱敏日志不存全段敏感 prompt。稳定 error codes 如 INVALID_SCOPE、COMMIT_FAILED、IDEMPOTENCY_CONFLICT、REVISION_CONFLICT、EVIDENCE_INVALID、SOURCE_REVOKED、EMBEDDING_SPACE_MISMATCH、INDEX_DEGRADED。

统计不能用“总记忆数增加”证明进步。记录误存、冲突、撤回传播、无来源候选、跨 scope 泄漏、实际注入率、错误记忆使用和成本。

验收：T28。全文检索、图谱和更复杂 dashboard 放 P2，不阻塞正确性。

## 6. 建议存储契约与事务边界

### 6.1 Journal 端

建议 FinalizedTurn 包含：完整 run identity、original/execution input、最终答案、terminal、入口／execution mode、actual skill injection refs、工具回执 refs、验证状态、memoryEpoch、policyVersion。

commitTurn 最小事务：

- 幂等检查 (conversationId, turnId, hostFinalKind)。
- 补齐必要 user/start 或引用已有过程事件。
- 插入最终 assistant、terminal 与不可变 FinalizedTurn envelope。
- 插入 derivation outbox。
- 返回 final hash／terminal seq／durability。

不能默默吞提交失败。过程事件缺失必须在 evidence completeness 中显式标记；不存在的 tool evidence 不能由 finalAnswer 补造。

### 6.2 canonical 端

applyMutation 最小事务：

- 校验 scope、control epoch、来源状态、幂等 requestHash。
- CAS 检查当前 revision。
- 写正文与来源；替代旧记录／撤回来源。
- 写 mutation receipt 与 index outbox；incoming batch receipt 已在独立接收事务中保存，canonical 应用还需独立的 candidate/mutation 幂等键。
- 更新控制 assertion 或 suppression（相关操作时）。

所有读从 canonical 取完整业务记录。向量 hit 只提供 ID/score/revision/generation；再次核验版本／状态后取正文。索引 dirty 时切 scoped fallback，不能靠固定倍数 overfetch 声称完全一致。

### 6.3 不建议的捷径

- 给 metadata 随意塞 scope，底层接口仍无校验。
- 用 MEMORY_WRITTEN hook 作为可靠消息队列。
- 重启从 Lance 把已删记录重新导回。
- 删除失败后用全量 sync 再次 upsert 旧记录。
- 以 contentHash 合并不同用户／不同项目所有相同文本。
- 在 query 文本上找“上一轮选了哪个 skill”。
- 用户一句“这次这样”直接变成全局永久策略。
- 只在 UI 面板隐藏撤销项，后台 writer 仍可读。

## 7. 六个开发阶段与门控

阶段顺序按证据→提交→恢复→使用→控制→语义推进；阶段边界是验收条件，不是时间估计。

| 阶段 | 本阶段范围 | 必须交付与门控 | 明确禁止 |
| --- | --- | --- | --- |
| G0 基线与止损 | 复核当前代码；经阶段授权新增失败复现；提出关闭未验证自动升权／蒸馏晋升方案；确定 RunContext/ToolOutcome 类型契约 | 记录 baseline、待修风险、target tests 与最小 G1 改动方案 | 不改生产学习行为，不全库 lint:fix，不改依赖／provider，不接模型 |
| G1 可靠事件入口 | F02/F03 + F01 + F04 现有入口的接口和运行时过滤；三端 host-owned final | T01–T09 的现有入口部分；最终答案一致、scope 正确、取消不跨会话 | 不自动生成正式语义记忆 |
| G2 canonical 与恢复 | F05/F06/F07 + F12 的 control/epoch/suppression P0；legacy dry-run、outbox、index generation；固定候选 fixture 验接收 | T10–T15；重启/故障/重放无丢失、无复活、向量失效正文仍存 | 不调用真实 writer，不向旧库反向双写，不静默激活归属不明数据 |
| G3 生命周期与使用 | F08/F09/F10；精确 coverage、预算、幂等使用事件 | T16–T21；稳定长期事实可用、过期失效、反馈不串奖 | 不把 completed 当验证通过 |
| G4 撤销控制与管理 | F11/F12/F14；assertion/control ledger、有效遗忘与导出控制 | T22–T25/T28；重启/旧任务/导入都不回潮 | 不声称尚未完成的物理 purge 已完成 |
| G5 writer 影子闭环 | F13；fixture writer 先验证，真实 writer 后替换；人审样本 | T26–T27；影子零影响、候选可追溯；固定变量对照后决定激活 | 不全面替换聊天模型，不自动批准人格／技能 |

每阶段结束暂停，提供 diff、测试结果、未解决问题和下一阶段计划，等待用户确认。G0 的类型和止损项只提交方案，用户确认后才在 G1 修改生产代码。类型先行不等于全部业务同时重构；共享文件由主开发 agent 顺序修改，独立测试可并行准备。

测试范围逐步扩展：G1 的 T07 只验已存在入口；summary、export、writer 在 G3/G4/G5 引入时必须加入同一 scope 回归套件。G2 用 fixture 验接收与控制 barrier，不要求先建真实 writer。

核心依赖：F02/F03 → F01；F04/F05 → F06；F05/F07 → 索引恢复；F05/F12 → F11 控制投影；F01/F04/F05/F06/F08/F12 → F13。F09/F10 可在 G2 后并行，但不能跳过身份／来源基础。

## 8. 迁移与回滚设计

### 8.1 原有记忆迁移

1. 开发 agent 先探测真实持久化配置／目录／schema，并记录 dry-run；当前只读快照不是用户实际数据源。
2. 用户确认后进入排他维护模式，备份真实应用数据。备份成功才迁移；禁止顺手扫描／清理个人目录。
3. 新增 canonical 表／schemaVersion，不覆盖旧表。为 Lance 补真正全表分页扫描，不能用零向量 topK=count 作为保证完整迁移。
4. Lance 完整行与 SQLite metadata 逐 ID 对账：一致则导入；差异无 revision 不能猜谁最新；Lance-only、SQLite-only／无正文、scope 冲突项隔离并报告。
5. 原文记录迁移为 legacy episode，不自动升格为事实／偏好。解析旧 scope tags，缺失／多值矛盾不设全局 user scope。
6. 保留可证明兼容的 embedding fingerprint；无指纹即正文保留、vector pending，不能按维度猜模型。
7. old Profile 用户显式值作为 baseline；来源不明 dormant 镜像不猜 proposal；已丢失前值明确标 unavailable。
8. 对正文 hash、ID 数量、scope、状态、来源、摘要做对账，输出 active/quarantined/missing_content 等分类。
9. 从 canonical 构建新 generation；验证后切 active pointer，旧库只读保留。
10. 新路径正式启用后不再从旧向量库反向 hydrate；历史自动学习关闭，需单独人工确认。

### 8.2 回滚

- 代码回滚不应让旧客户端重新读取已撤销的旧表。存在新 mutation 后，回滚到只读安全模式或提供兼容 adapter；不要直接启用旧写入逻辑。
- 保存 migration manifest、source hashes、active generation 与 control watermark。
- writer 可独立禁用，不停止 canonical／正文检索。索引可重新构建，不回退控制 ledger。
- 对迁移后的新数据，禁止用旧备份直接覆盖；先导出规范 bundle，并包含撤销／suppression。
- 若无法保证安全回滚，应在切换前明确说明，而不是留下一个恢复旧库会复活事实的开关。

## 9. 确定性测试矩阵（T01–T28）

所有测试先使用固定时钟／ID、fixture writer、stub reader、确定性向量、fake stores、barrier/deferred Promise 和故障注入。不依赖真实模型、网络、sleep 或墙钟等待。另设 real SQLite/LanceDB 临时目录集成测试；fake 测试不替代真实事务／SDK prefilter 验证。

| 测试 | 固定构造 | 必须断言 |
| --- | --- | --- |
| T01 最终提交覆盖 | 普通 TAO、clarify、preflight、image、preview、编排、scheduler；先有部分过程事件 | 每 turn 一次 host final/terminal/outbox；返回答案与刷新后 Journal 投影一致；失败不变 completed |
| T02 提交故障与幂等 | final 事务提交前失败、提交后返回前崩溃、重复 finalize、不同 final hash | 前失败无提交；后重试同 receipt；不同 hash 明确冲突；无重复 |
| T03 共享控制隔离 | A/B 共用 bus，A 有 steer/inject，先触发 B；dispose | B 不消费 A；A 消费一次；监听器恢复基线 |
| T04 取消/审批/队列 | 入队即取消；审批取消后迟到 approve；provider 无视 signal 返回；父等待子 | 不重入 running、不执行迟到副作用、不跨会话取消；child 不自等待；terminal cancelled |
| T05 工具标签 | descriptor 无 handler、exitCode=1、有效 write receipt、hook 阻止 | not_executed/business failure/verified predicate 分开；无伪 success；callId 执行前生成 |
| T06 宿主兜底治理 | gate 拒绝、兜底前取消、成功兜底 | 拒绝/取消 writer 调用为零；成功有 host_fallback 回执；不冒充 skill 成功 |
| T07 全入口 scope | 两 workspace/task/conversation 的 canary；product behavior/hybrid 开关各组合 | preflight、prompt、面板、查询 API、writer 输入都不泄漏 |
| T08 topK 前过滤 | 100 条不可见高分，6 条可见低分，topK=6 | 每路完整返回6条可见；本地 scoped ranker 不受不可见语料统计影响；真实 Lance 路径验证 |
| T09 范围与容量 | 未知/缺失/冲突 scope；tier 晋升；显式共享；B 写爆 working | 无全库降级；晋升不共享；共享只在指定范围；A working 不受 B 清除 |
| T10 权威恢复 | SQLite 正文完整，Lance 缺失/错误正文；重启 | canonical 不丢，索引不得覆盖正文；重建 hash/revision 一致 |
| T11 mutation 覆盖 | 每次写后提交，触发晋升／溢出／scope clear，不额外 sync | 重启保持 tier/清除状态；无幽灵；commit hook 不早于提交 |
| T12 写幂等/CAS | 同 key/hash 并发重试；同 key 不同 hash；相同正文不同 event；旧 revision | 同 receipt、无重复；冲突显式；独立事件不跨 scope 合并 |
| T13 outbox 崩溃 | source committed未消费；candidate保存未ACK；index成功未ACK；坏任务 | 最终一次应用；重放不重复 record/reward；坏任务不毒化后续队列 |
| T14 stale index | delete/supersede 后延迟投影；旧 add 晚到；新记录还未索引 | 即时读取无已删值、可读新正文；旧任务不能复活；dirty 使用正确 fallback |
| T15 向量空间 | 非64维 embedder，公共/主循环/hybrid；同维不同fingerprint；模型 unavailable | 统一空间；冲突明确拒绝；正文仍持久化，词法仍可用 |
| T16 长期/过期 | 固定时钟推进30天；有效长期fact、TTL已过、archived/revoked | 长期有效项仍候选；失效项在所有入口不可见 |
| T17 纠正与来源 | A→B显式纠正；旧A的writer迟到；跨Journal/会话不可比来源；助手复述A；引用别人偏好；不同项目同键 | B生效A superseded；旧A不能通过add或CAS重读覆盖B；不可比冲突待确认；复述不增证据；跨项目不替换 |
| T18 压缩覆盖 | cap仅够完整前两轮；半轮／超长单条／未闭合洞；反复compact | 只覆盖实际完整输入；未摘要原文保留；不跨洞；继承coverage正确 |
| T19 预算与异步 | 同history重复build；summarizer超长；摘要期间clear | 预算不累加不越硬界；clear后旧摘要不提交；另一会话不受污染 |
| T20 注入与cache | 召回2条预算只注入1；相同query两轮；缓存命中 | 只记真实injected且幂等；每轮allowlist/selection生效；版本变化失效 |
| T21 反馈归因 | 改写query、分类器关闭、并行同文本、TaskPool节点、赢家输家、cancel | 按run/branch/selection归因；不串奖不重复；未知/取消中性 |
| T22 覆盖撤销 | baseline X、P1=A、P2=B、用户U，撤销顺序/过期各组合 | 撤P1不删B；撤P2恢复有效A；用户U保留；过期不复活 |
| T23 控制投影故障 | ledger revoke成功、JSON失败、重启、dismiss后import/export/sync | prompt/Persona/Profile不显示旧值；重放可恢复；suppression保留 |
| T24 遗忘与迟到任务 | writer运行中forget；旧outbox/index/candidate返回；scope epoch改变 | 旧来源不生效；不回潮；新来源再提案遵守规则 |
| T25 删除语义与隐私 | clear、effective forget、purge；秘密文本进入writer/日志；备份副本 | 三种承诺明确；输入脱敏；未清副本不宣称彻底删除 |
| T26 候选校验 | 无来源、hash/span错、越scope、过期epoch、错revision、未知字段 | 全拒绝/隔离且有原因；不进入活跃投影；不执行模型指令 |
| T27 影子与协议 | fixture/通用/Hy writer同输入；异常/超预算/普通重试；空/拒绝批次与shadow ACK | 影子不改正式行为且不建索引投影；ACK只确认完整接收；重试同抽取版本；模型可替换而契约一致 |
| T28 可观察性 | commit失败、index落后、vector禁用、撤销传播失败 | 明确显示durability/lag/control状态；可追踪链；无秘密telemetry |

测试组织建议：core 契约与 Journal；runtime coordinator/compact；memory eligibility/mutation/worker；storage real 故障与迁移；三端宿主契约 E2E。用共享行为测试套件检查各 adapter，不只测单个类。

安全／正确性硬门槛：该测试集中的跨 scope 泄漏、无来源生效、重复应用、撤销回潮、虚假提交、未摘要内容隐藏应为零容忍；“本轮没触发”不能证明长期绝不会发生。

## 10. writer 质量验收：基础架构验收之后

本阶段确定性测试证明的是系统契约，不证明语义 writer 理解正确。真实模型质量另设时间顺序的留出样本，并由人工确认主张／来源／scope／有效期。

样本至少覆盖：明确长期偏好、仅当次偏好、用户纠正、助手回声、引用他人、过期意图、项目间冲突、失败后改正、工具返回与助手结论冲突、撤销再提案。

对照组：原文记录基线、通用 writer、Hy-Memory writer。固定 reader、retriever、预算、候选协议和评估任务。先比较正确写入／更新／误存，再比较后续问答与任务表现，不只数记忆条数。

指标：有来源精确率、scope准确性、长期误存率、纠正正确率、重复主张率、遗忘生效率、错误记忆使用率、后续回答正确性／偏好遵守、任务验证率、每turn成本与延迟。

默认正式激活前规则：所有确定性门槛通过；影子中无越权写入；风险类型人工复核；低风险类型单独白名单激活；人格／程序性策略仍需审批。具体质量阈值在看过真实样本分布后设置，不编造百分比或使用论文成绩代替。

## 11. 开发 agent 交接规则

### 11.1 首轮必须先做什么

- 读取当前项目 AGENTS／开发约定、package scripts 与现有 RFC；核实 HEAD 与本文基线差异。
- 保留所有用户未提交改动，不 reset、checkout 或全库格式化。
- 对 E01–E14 分成“仍存在／已修正／接口已变化／待复现”。已修正项更新设计，不重复修。
- 建立 baseline：环境、目标包 typecheck/lint/test、真实存储测试能力。基线既有失败与新增回归分开。
- 第一轮只做 G0 和最小复现／止损；不一次实施 F01–F14。

### 11.2 可直接粘贴的启动任务

```text
目标：为 openINTJ 建立可靠记忆闭环。请以本文设计稿为规格，只启动 G0，不实施后续全量改造。
先只读核对当前 HEAD、项目开发规则、未提交改动和 E01–E14 的真实状态。
优先核对最终提交边界：TAO final、宿主答案修订、Desktop产物兜底、finalize已有事件即跳过，以及各入口status映射。
输出证据定位、当前行为矩阵、最小失败复现计划和 G1 修改边界。
允许在确认现有测试结构后增加目标测试，不修改用户数据、不迁移真实数据库、不接入模型、不更新依赖、不全库格式化。
对未核验的completed自动升权先提出可回退止损开关方案；不要悄悄改变全部学习行为。
完成后汇总diff、实际执行测试与基线失败，暂停等待我确认，再进入G1。
```

### 11.3 每个后续任务的固定格式

- 本次阶段与目标不变量。
- 已确认现状与真实代码定位。
- 允许修改的包／接口，禁止事项。
- 设计决定、兼容行为与数据影响。
- 要先失败再通过的测试编号。
- 迁移／回滚／降级条件。
- 实际验证命令与结果，未运行项明确标记。
- 本阶段结束后的剩余问题与确认点。

提交遵守项目现有流程与 hooks；需要提交时先汇总，不绕过 lint/signing。不要为避免基线失败而悄悄改 unrelated 文件。无法在资源约束下运行测试时暂停报告，不硬跑。

## 12. 最终验收与落地判断

可称“基础可靠闭环已完成”必须满足：

1. 每个新 turn 的最终内容可可靠提交，源证据可定位，失败有明确状态。
2. 全入口 scope 一致，缺失权限不退全库。
3. canonical 正文／状态跨重启保持，索引可独立重建。
4. 重试不重复应用，旧任务不复活纠正／撤回内容。
5. 稳定长期事实保持资格，过期内容正确失效。
6. 压缩没有虚假覆盖，预算与使用记录正确。
7. Profile／Persona 撤销按来源传播并恢复合法覆盖层。
8. 反馈按实际运行归因，未知成功不自动学习。
9. writer 只能提出有来源候选，影子验证先于正式激活。
10. 用户可知道记住了什么、为何生效、如何纠正与停止使用。

最重要的实现顺序是：**先建立可信输入与最终提交，再建立可恢复存储和有效性控制，最后让语义 writer 参与。**

这不是把记忆模型接到一个空接口，而是把它接到一个能够约束、解释、纠正和验证学习的系统。

## 附录：核心证据链接

下列链接固定到本次审查 commit，开发时仍须比较当前 HEAD。

- [MemoryPlane 原文记录](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/planes/memory/src/memory-plane.ts#L52-L87)
- [Journal finalize 边界](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/core/src/journal/sink.ts#L145-L177)
- [PersistentMemoryStore 恢复与双写](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/planes/memory/src/persistent-store.ts)
- [Desktop store/embedder 装配](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/apps/desktop/src/main/agent.ts#L695-L750)
- [记忆几何过滤](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/planes/memory/src/shader/geometry.ts#L22-L35)
- [Compactor 覆盖](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/runtime/src/compact.ts#L178-L204)
- [ToolHub 结果构造](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/planes/execution/src/tool-hub.ts#L247-L266)
- [Skills 学习归因](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/skills/src/learning-runtime.ts#L234-L311)
- [Profile import/revoke 镜像边界](https://github.com/oceancolor/openINTJ/blob/2326b4b02ad578363fd470317ce733a039c775e0/ts/packages/profile/src/service.ts#L82-L117)
- [基线 CI run](https://github.com/oceancolor/openINTJ/actions/runs/42216257564)
