# OpenINTJ Changelog

本文件追踪 OpenINTJ 的对外可观察变更。
版本号沿用 [SemVer](https://semver.org/lang/zh-CN/) 与
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格。

## [3.0.0-alpha.6] —— Phase 3.6 Python v2 ↔ TS 行为对齐测试 (2026-05-20)

> 给 TS 实现盖一层"行为级回归网"——把冻结的 Python v2.0 当语义参考，
> 在固定输入上断言 TS 输出等价。详见
> [`docs/architecture/phase3-6-parity-tests.md`](./docs/architecture/phase3-6-parity-tests.md)。

### Added

- **`scripts/python-parity/generate_fixtures.py`** —— Python 端取证脚本
  - 加载仓库根冻结的 `framework_core` / `memory_plane` / `control_plane` / `execution_plane`
  - 在预设输入上跑 → 把可观察输出固化为 4 份 JSON fixture（每个 TS 包一份）
  - **只读**：绝不修改 Python 代码；Python v2 已冻结
- **`scripts/python-parity/README.md`** —— 工具使用说明 + 已知偏差速查表
- **4 个 TS parity spec**（共 **64 个新 tests**）：
  - `ts/packages/core/__tests__/parity/python-v2.spec.ts` —— **23 tests**：
    SimpleEmbedder (SHA-256) / cosineSimilarity / decayImportance
  - `ts/packages/planes/control/__tests__/parity/python-v2.spec.ts` —— **21 tests**：
    GoalParser.parse 中英文意图 + 引号实体 + 优先级；Planner.createPlan 5 个公共 intent
  - `ts/packages/planes/execution/__tests__/parity/python-v2.spec.ts` —— **17 tests**：
    StepStateMachine 合法/非法转换表；Executor sequential / parallel 事件轨迹
  - `ts/packages/planes/memory/__tests__/parity/python-v2.spec.ts` —— **3 tests**：
    MemoryStore overflow；MemoryRetriever 评分组件 + 排序
- **4 份 fixture JSON**（`__tests__/parity/fixtures/python-v2.json`）：
  - 每份带 `schemaVersion` + `generatedFrom` + 关键设计 `notes`
  - 由 Python 端脚本统一生成，commit-in，CI 无需 Python
- **`docs/architecture/phase3-6-parity-tests.md`** —— 阶段记录 + 已知偏差矩阵 + 容差策略

### Changed

- **`ts/biome.json`**：`files.ignore` 加 `**/__tests__/parity/fixtures/**`
  （fixture 是 Python 产物，不参与 biome formatter）

### Testing

- CI 模式（`pnpm exec turbo run test --concurrency=1`）：
  - 33/33 packages successful
  - **430 passed + 11 skipped**（净增 64 个 parity 测试；previously 366 + 11）
- E2E 模式（`OPENINTJ_E2E=1`）：
  - 33/33 packages successful
  - **441 passed + 0 skipped**（previously 377 + 0）

### Notes

- 容差策略（详见 phase3-6 文档）：
  - SHA-256 向量 / cosineSimilarity：`1e-12`（bit-identical）
  - `decayImportance`：`1e-4`（Python 用 `0.693` 近似 `Math.LN2`；TS 更精确）
  - MemoryRetriever 评分：分量 `1e-12`（纯位运算）/ recency + 最终 score `1e-4`
- 5 类**已知偏差**已显式记录在 phase3-6 文档"已知偏差矩阵"：
  1. `decayImportance` 0.693 vs `Math.LN2` —— TS 精度更高
  2. MemoryRetriever 半衰期口径（Python 写死 `max_summary_length/10` 是 v2 bug；
     fixture 把 `max_summary_length=240` 让两边都跑 24h 半衰期，严格可比）
  3. Planner `delete`/`execute` 模板 —— TS 扩展；parity 只跑公共 5 个 intent
  4. Executor 死重试 bug —— TS 已修复；fixture 只跑全成功路径
  5. StepStateMachine 错误码命名 —— TS spec 接受两者之一
- fixture 一次生成、长期复用；只有 Python 端"延寿活动"（极少）或 `generate_fixtures.py`
  自身改动时才需要重跑

## [3.0.0-alpha.5] —— Phase 3.5 Dormant 审批 UI (2026-05-19)

> Phase 3.4 把 Dormant 持久化的模型/装配/IPC 都做完了，但桌面端 renderer 没接。
> 这一版把最后一公里补上：preload 暴露 5 个 dormant API + 桌面端审批面板 +
> StatusBar 暴露 dormant pending 角标。详见
> [`docs/architecture/phase3-5-dormant-approval-ui.md`](./docs/architecture/phase3-5-dormant-approval-ui.md)。

### Added

- **`apps/desktop/src/shared/ipc-protocol.ts`** —— 把 IPC 协议补成"所见即所得"
  - `StatusResponseSchema` 补 `persistence` / `retrievalMode` / `dormant` 三个 optional
    字段（与 main 进程 `agent.status()` 实际返回对齐；之前 renderer 端类型早就过期）
  - 新增响应 DTO：`DormantMineResponseSchema` / `DormantListResponseSchema` /
    `DormantDecisionResponseSchema` / `DormantPersonaResponseSchema` /
    `DormantProposalDtoSchema` / `DormantPatternDtoSchema`
  - 新增错误 schema：`DormantErrorSchema` / `DormantDecisionErrorSchema`
- **`apps/desktop/src/preload/index.ts`** —— 5 个新 API：
  - `dormantMine()` / `dormantList(req?)` / `dormantApprove({ proposalId })` /
    `dormantReject({ proposalId })` / `dormantPersona()`
  - 返回类型是联合类型（success | error），renderer 必须 narrow 才能用，无法把错误当数据
- **`apps/desktop/src/renderer/components/DormantPanel.tsx`** —— 新组件
  - 顶栏 [Mine] 按钮 + 状态 filter（pending/applied/rejected/all）
  - proposal 列表卡片：状态徽章 + 频次 + 置信度 + 描述 + `targetField ← value` +
    [✓ 应用] [✗ 拒绝] 按钮
  - 底部折叠区：当前 Persona JSON
  - 未启用时显示居中提示 + 启用方法
- **`apps/desktop/src/renderer/App.tsx`** —— 右侧栏从单 panel 改成 tab 布局
  - tab 标题：[推理轨迹] [Dormant + pending 数字角标]
  - tab 角标：`status.dormant.pendingProposals > 0` 时显示黄色徽章
- **`apps/desktop/src/renderer/components/StatusBar.tsx`**
  - 新增条目：检索模式 / 持久化模式 / Dormant 状态（passive 事件数 + 待审 proposal 数）
  - 类型从本地 `StatusSnapshot` interface 改为 protocol 中的 `StatusResponse` re-export

### Changed

- `TrajectoryPanel.tsx` 和 `DormantPanel.tsx` 去掉外层 border/bg/header 装饰
  （这些 chrome 现在由 App.tsx 的 tab 容器统一提供，避免双层 border）

### Testing

- `apps/desktop/__tests__/ipc-handlers.spec.ts`：12 → **18 tests passed**（+6）
  - STATUS 用 StatusResponseSchema 全字段校验（含 dormant + retrievalMode + persistence）
  - DORMANT_MINE 用 DormantMineResponseSchema 校验返回值结构
  - DORMANT_LIST 默认（无 status）返回所有 proposals + 字段校验
  - DORMANT_REJECT 返回 status=rejected + 不污染 persona
  - APPROVE/REJECT 不存在 proposalId 时返回 `not_found_or_already_decided`
  - DORMANT_PERSONA 未启用时返回 `dormant_not_enabled`

### Notes

- 当前 desktop 工作区只有 main-process 测试，**没有 renderer React 测试**
  （DormantPanel 的逻辑分支已被 IPC 层契约测试覆盖；UI 留给手动 / Playwright e2e #4）
- Mine 任务由用户主动触发；后台 mine 推送（`EVT_DORMANT`）留给未来需要时再做
- 用户唯一的 persona 写入路径仍然是审批 proposals，UI 不提供字段级直接编辑

---

## [3.0.0-alpha.4] —— Phase 3.4 Dormant 持久化 (2026-05-19)

> Phase 3.3 留下的最重的尾巴：PassiveStore / PersonaConfig 进程一断电就丢。
> 这一版给 Dormant 子系统接上 SQLite 真盘适配器，
> 把"用户审批过的偏好/习惯"留下来。详见
> [`docs/architecture/phase3-4-dormant-persistence.md`](./docs/architecture/phase3-4-dormant-persistence.md)。

### Added

- **`@openintj/dormant`**
  - 新增 `DormantPersistenceAdapter` 接口（`persistence.ts`）：`loadAll` / `recordEvent` /
    `upsertProposal` / `savePersona` / `clearAll` / `close`，热路径同步、不抛错
  - 新增 `InMemoryDormantStore`：参考实现 + 测试用
  - 新增 `DormantSnapshot` 类型
  - `DormantRuntime`：新增 `adapter` 槽 + `hydrate()` 方法；`record / mine / approve / reject /
    reset / close` 全部写穿 adapter
  - `PassiveStore`：新增 `recordBulk(events)` 批量回填
  - `InternalizationManager`：新增 `restoreState(proposals, persona?)` 不触发
    `lastUpdated` / `version` 自增
- **`@openintj/storage-sqlite`**
  - 新增 `SqliteDormantStore`：实现 `DormantPersistenceAdapter`，独立的 `dormant.sqlite`
    文件，schema v1（`dormant_events` / `dormant_proposals` / `dormant_persona`）+ WAL +
    prepared statements
  - 新增 `createSqliteDormantStore` 工厂；输入类型为 `SqliteDormantConfigInput`（`wal` 可选）
- **apps/server**
  - `ServerAgentOpts` 新增 `dormantPersistence: 'auto' | 'memory' | 'real'`（默认 `auto`）+
    `dormantDbPath`（覆盖默认 `${dataDir}/dormant.sqlite`）
  - `ServerAgent` 新增字段 `dormantPersistenceInfo: { adapter, dbPath? }`
  - `status().dormant.persistence` 暴露 adapter 名 / 路径
  - `assembleServerAgent`：在 `enableDormant + dataDir` 时自动挂 SqliteDormantStore，
    构造后 `await dormant.hydrate()`；`close()` 先 `await dormant.close()`
- **apps/desktop**
  - `DesktopAgent` 镜像同上：`dormantPersistence` / `dormantDbPath` / `dormantPersistenceInfo`
    / `status().dormant.persistence`
- **环境变量**
  - `OPENINTJ_DORMANT_DB_PATH` 可覆盖默认 SQLite 文件路径

### Changed

- `@openintj/storage-sqlite/index.ts` 修复重复 `export * from "./dormant.js"`
- biome formatter 一次性整理 4 个文件（`packages/storage/sqlite/tsconfig.json` 等）

### Testing

- **CI 模式**：
  - `packages/dormant/__tests__/persistence.spec.ts`：9 个（InMemoryDormantStore CRUD +
    hydrate + write-through）
  - `packages/storage/sqlite/__tests__/dormant.spec.ts`：11 个（`:memory:` 路径走真
    better-sqlite3）
  - `apps/server/__tests__/dormant-persistence-e2e.spec.ts`：2 个 memory 模式（4 个 e2e skip）
- **E2E 模式（`OPENINTJ_E2E=1`）**：上述 e2e 6 个全部跑通，含 `record → mine → approve →
  close → 重装配 → hydrate → 验证状态恢复` 的完整往返

### Notes

- 桌面端审批 UI 仍未接（#9.B 留给下一个 phase）
- `dormant_events` 表未做自动清理；当前 PassiveStore 仅内存层有 `maxPassiveEvents` 环形上限
- `dormant.sqlite` 是明文；用户敏感偏好不应通过 dormant 路径学习

---

## [3.0.0-alpha.3] —— Phase 3.3 RFC-003 装配进主 Agent (2026-05-11)

> RFC-003 的三个孤岛包（@openintj/concurrency / @openintj/dormant / @openintj/taskpool）
> 全部接进 apps/server 与 apps/desktop 主装配点，三条线均提供环境变量 / 代码 opt-in，
> 默认零开销，启用后能直接通过 HTTP / IPC 使用。

### Added

- **方向 1 — LLM 速率限制**（`@openintj/concurrency`）
  - 新增 `RateLimitedLlmClient`：TokenBucket 装饰 `LlmClient.chat / visionChat`
  - server / desktop opt-in：`opts.rateLimit = { qps, burst? }` 或 env `OPENINTJ_RATE_LIMIT_QPS` / `OPENINTJ_RATE_LIMIT_BURST`
- **方向 2 — HybridRetriever 混合检索**（`@openintj/taskpool`）
  - server: `retrieveHybrid()` 顶层函数 + 路由 `GET /api/memory?mode=hybrid[&rrf=true]`
  - desktop: `agent.retrieveHybrid()` + IPC `MEMORY_QUERY` 支持 `{ mode: 'hybrid', rrf }`
  - 默认检索模式 opt-in：`opts.retrievalMode = 'hybrid'` 或 env `OPENINTJ_RETRIEVAL_MODE=hybrid`
- **方向 3 — Dormant Memory Learning**（`@openintj/dormant`）
  - 新增 `DormantRuntime`：PassiveStore + PatternMiner + InternalizationManager 三件套门面
  - server 路由：`POST /api/dormant/mine` / `GET /api/dormant/proposals` / `POST /api/dormant/proposals/:id/approve|reject` / `GET /api/dormant/persona`
  - desktop IPC：`DORMANT_MINE / DORMANT_LIST / DORMANT_APPROVE / DORMANT_REJECT / DORMANT_PERSONA`
  - `agent.run()` 自动把用户输入和 final answer 喂进 PassiveStore（启用后才生效）
  - opt-in：`opts.enableDormant = true` 或 env `OPENINTJ_DORMANT=1`；未启用时所有 API 一律 503 / `dormant_not_enabled`

### Changed

- `apps/server/src/agent.ts` `ServerAgent` 新增字段：`retrievalMode` / 可选 `dormant` / `status().dormant` / `status().retrievalMode`
- `apps/desktop/src/main/agent.ts` `DesktopAgent` 镜像 server 端字段
- `apps/server/package.json` / `apps/desktop/package.json` 新增 workspace 依赖：`@openintj/concurrency` / `@openintj/dormant` / `@openintj/taskpool`
- IPC 协议 `apps/desktop/src/shared/ipc-protocol.ts` 扩展：
  - `MemoryQueryRequestSchema` 加 `mode` / `rrf`
  - 新增 `DormantListRequestSchema` / `DormantProposalDecisionSchema`
  - `IPC` 常量增加 5 个 Dormant channel

### Testing

- 新增测试（CI 模式）：
  - `@openintj/dormant`：`__tests__/dormant-runtime.spec.ts` 6 个
  - `@openintj/server`：`__tests__/dormant.spec.ts` 12 个 + `__tests__/hybrid-retrieve.spec.ts` 14 个 + `__tests__/rate-limited-llm.spec.ts` 9 个
  - `@openintj/desktop`：`__tests__/ipc-handlers.spec.ts` 扩展 5 个（hybrid + Dormant IPC）
- CI 跑分：默认 mode 312 passed / 7 skipped，E2E mode（`OPENINTJ_E2E=1`）全部跑通

### Design 备忘

- HybridRetriever 装配是"每次查询临时建索引"——适合中等规模（≤几千 fragments）；大规模建议改用 LanceDB FTS
- DormantRuntime 默认不持久化 PassiveStore 与 PersonaConfig；持久化层等下一个 phase 接入
- `RateLimitedLlmClient` 实现已经迁移到 `@openintj/concurrency` 包，`apps/server/src/rate-limited-llm.ts` 仅做兼容 re-export

---

## [3.0.0-alpha.2] —— Phase 3.2 GitHub Actions CI (2026-05-09)

> 把本地已经能跑通的 lint / typecheck / test (CI + E2E) 锁进 GitHub Actions，
> 给后续所有改动兜底。

### Added

- `.github/workflows/ci.yml`（仓库根，旧的错放在 `ts/.github/` 下从未触发，已删除）
  - **lint-and-typecheck**：matrix 跑 Node 20 + Node 22；先 biome lint，再 turbo typecheck
  - **test**：matrix 跑 ubuntu / windows / macos × Node 20；先 turbo build 再 turbo test（CI 模式）
  - **e2e-persistence**：仅 ubuntu，设 `OPENINTJ_E2E=1` 跑 LanceDB + SQLite 真盘端到端
  - 加 `concurrency.cancel-in-progress` 减少同分支重复跑
  - 全局 `NODE_OPTIONS=--max-old-space-size=6144` 防 tsc OOM
  - 全部 turbo 调用都带 `--concurrency=1`，统一跨 OS 的策略

### Changed

- `ts/turbo.json`：`test` 任务的 cache key 加入 `OPENINTJ_E2E` / `OPENINTJ_DATA_DIR` / `OPENINTJ_DESKTOP_NO_PERSIST` / `OPENINTJ_LANCE_DEBUG`
  - **关键修复**：之前 turbo 不感知这些 env 的变化，e2e job 会命中常规 test 的缓存、e2e 测试被默默跳过
- `ts/biome.json`：放宽与历史代码冲突的规则（`useLiteralKeys` / `noNonNullAssertion` / `noUnusedTemplateLiteral` / `noDelete` / `noArrayIndexKey` 等共 13 条）
  - 这些是**风格偏好**而非 bug；保留 `useImportType` / `noUnusedVariables` / `noUnusedImports` 等真正的正确性规则
  - 现状：`pnpm lint` exit 0，2 条 React `useExhaustiveDependencies` 警告（已知，不阻塞）
- biome formatter 一次性格式化 107 个 tsconfig.json / package.json（多行 references 数组改单行）

### Tooling

- 现在三条线都能本地一把跑通（也是 CI 跑的命令）：
  - `pnpm lint`
  - `pnpm exec turbo run typecheck --concurrency=1`
  - `pnpm exec turbo run test --concurrency=1`（默认 292，`OPENINTJ_E2E=1` 时 299）
- turbo cache key 修了之后：
  - 同 env 的二次运行：33/33 cache hit，full turbo ~500ms
  - 切换 `OPENINTJ_E2E` 取值：所有 test 任务 cache miss，重新执行

---

## [3.0.0-alpha.1] —— Phase 3.1 真实持久化 e2e (2026-05-09)

> Phase 3 第 1 步：把 `apps/server` / `apps/desktop` 从 in-memory 兜底切到真实磁盘
> （LanceDB + SQLite），并补端到端"写入 → 关闭 → 重启 → 读回"测试。
> CI 默认 292/292 绿（Phase 2 286 + 6 新增 in-mem）；`OPENINTJ_E2E=1` 全量 299/299 绿。

### Added

- **持久化工厂** `createPersistentMemoryStore`（`@openintj/plane-memory`）
  - 根据 `dataDir` / `mode` 自动选择 LanceDB+SQLite 真盘或 in-memory 兜底
  - 真盘模式自动建 `lancedb/` 子目录与 `metadata.db` 文件
  - 缺 `dataDir` 但 `mode='real'` 时显式抛错（fail-fast）
- **服务端入口** `assembleServerAgent({ dataDir?, persistenceMode? })`
  - 支持 env `OPENINTJ_DATA_DIR` 启用真盘
  - 新增 `agent.close()`（关 LanceDB / SQLite）与 `persistentInfo`
  - `/api/status` 暴露当前持久化模式与数据目录
- **桌面端入口** `assembleDesktopAgent({ dataDir?, persistenceMode? })`
  - Electron 主进程默认用 `app.getPath('userData')` 作 dataDir
  - `app.on('before-quit')` 钩 `agent.close()` 防止数据库句柄泄漏
  - env `OPENINTJ_DESKTOP_NO_PERSIST=1` 可强制走 in-memory（CI 友好）
- **e2e 测试**（`OPENINTJ_E2E=1` 启用）
  - `plane-memory/__tests__/persistence-factory.spec.ts`：工厂自身的真盘往返
  - `apps/server/__tests__/persistence-e2e.spec.ts`：装配 → 写 → close → 重装配 → hydrate → 检索 + 审计读回
  - `apps/desktop/__tests__/agent-persistence.spec.ts`：desktop agent 真盘往返与 NO_PERSIST 短路

### Changed

- `@openintj/storage-lance`：`apache-arrow` 从 `peerDependencies` 移到 `dependencies`（`init()` 必用，不是可选）
- `LanceDBVectorStore.init()`：从"靠 seed-row 推断 schema"改为用 `apache-arrow` 显式声明 `FixedSizeList<Float32, N>` + `List<Utf8>` schema；旧版 LanceDB 无 `createEmptyTable` 时回落到 seed-row + delete 路径
- `LanceDBVectorStore` 的 `delete` / `search` SQL：camelCase 列名一律双引号（LanceDB 大小写敏感，否则报 "No field named fragmentid"）
- `LanceDBVectorStore.search()`：新增 `normalizeEmbedding` / `normalizeStringArray`，把 LanceDB 返回的 TypedArray / Arrow Vector 规范化成 plain `number[]` / `string[]` 后再 `VectorRowSchema.parse`，修复"`count()` 返 N 但 `scanAll()` / `search()` 返空"的静默丢行 bug
- e2e suite 全部带 30 秒超时（`describe(..., { timeout: 30_000 }, ...)`），LanceDB 首次建表 + 重新打开偏慢

### Fixed

- 真实持久化模式下 vector search 返空数组（zod parse 因 TypedArray / Arrow Vector 静默失败）
- LanceDB SQL 过滤器对 camelCase 字段名报 "No field named fragmentid"
- `apache-arrow` 静态导入失败（peer 解析路径不一致）
- e2e 测试在重启第二个进程时因 5s 默认超时被误判为失败

### Tooling

- 调试用：设置 `OPENINTJ_LANCE_DEBUG=1` 时，`LanceDBVectorStore.search()` 会把 zod 解析失败的行打印到 stderr
- 本地真盘自检命令：`$env:OPENINTJ_E2E="1"; pnpm -r --workspace-concurrency=1 test`

---

## [3.0.0-alpha.0] —— Phase 2 完成 (2026-04-29)

> Phase 2 收尾：TS 端在 `v2.0-python-reference` 之上完成"装配 + 持久化 + 客户端 + RFC-003 三方向"四个纵深方向。
> typecheck 全绿；workspace 内 17 个测试包共 **286 个用例全部通过**（详见
> [`docs/architecture/phase2-complete.md`](./docs/architecture/phase2-complete.md)）。

### Added

- **Memory Shader Pipeline**（`@openintj/plane-memory`）
  - `vertexShader` / `geometryShader` / `fragmentShader` 三阶段对齐 Python `memory_plane.ShaderPipeline`
  - `ShaderPipeline` 主类 + `ContextEngine` 上下文构建器
  - 钩子事件：`event.SHADER_APPLIED`、`event.CONTEXT_COMPACTED`
- **EmbeddingProvider 抽象**（`@openintj/core`）
  - 统一 `EmbeddingProvider` 接口（同步 / 异步），保留 `SimpleEmbedder` 兜底
  - `MemoryStore` 与 `MemoryRetriever` 改造为可注入 provider
- **嵌入实现**
  - `@openintj/embed-ollama`：通过 Ollama `/api/embeddings` 端点
  - `@openintj/embed-xenova`：本地 `@xenova/transformers`（peer dependency）
- **持久化**
  - `@openintj/storage-lance`：`VectorStore` 接口 + `InMemoryVectorStore` + `LanceDBVectorStore`（peer 依赖 `@lancedb/lancedb`）
  - `@openintj/storage-sqlite`：`MetadataStore` 接口 + 内存兜底 + `SqliteMetadataStore`（peer 依赖 `better-sqlite3`），含 fragments_meta / audit / sessions 三张表与迁移
  - `PersistentMemoryStore`：包装内存层 + LanceDB + SQLite，启动 hydrate、写入 dual-write、`reassignMemoryType` 升级 short→long
- **`MemoryFragment.memoryType`**：显式区分 `short_term | working | long_term`
- **应用形态**
  - `apps/server`：Hono HTTP + SSE 流式 chat、`/api/status`、`/api/memory`、`/api/audit`，请求体由 zod 校验
  - `apps/desktop`：Electron 主进程 IPC（RFC-004 协议）+ preload `contextBridge` + Renderer（React 18 + Vite + Tailwind）三栏布局
- **RFC-003 三方向原型**
  - `@openintj/concurrency`：Mutex / Semaphore / Channel / ConditionVariable / AgentPool / ForkJoin / TokenBucket / BackpressureGate
  - `@openintj/taskpool`：SharedContext / HybridRetriever（vector + BM25 + RRF）/ TaskQueue（DAG 优先级）/ ObjectPool（hot/warm/cold + LRU）
  - `@openintj/dormant`：PassiveStore / PatternMiner（n-gram + 可注入 LLM 抽取，CJK 字符级分词）/ InternalizationManager（用户审批写入 PersonaConfig）
- **集成测试**：`apps/cli/__tests__/rfc3-integration.spec.ts` 覆盖三方向端到端流程
- **文档**：[`docs/architecture/phase2-complete.md`](./docs/architecture/phase2-complete.md) 收尾报告

### Changed

- `MemoryStore` / `MemoryRetriever` 现在以构造时注入的 `EmbeddingProvider` 为准；同步 API 在异步 provider 下会显式抛错
- `ContextEngine` 的预算追踪修正：`conversationTokens` 现在是累加而非覆盖，`CONTEXT_COMPACTED` 钩子触发条件更准确
- TAO/ReAct 与 4 平面默认在 `apps/server` / `apps/desktop` 通过 `assembleAgent`-pattern 统一装配

### Fixed

- `Executor` 重试路径：替换原 Python 端的"伪重试"，落地真正的指数退避 + 状态机合法转换
- `ShaderConfig` 拆出独立的 `recencyHalfLifeHours`，纠正 Python 端把"摘要最大长度"误用为"半衰期小时数"的 bug
- 多处 TypeScript `exactOptionalPropertyTypes: true` 严格模式下的类型问题（AgentPool 泛型 / BackpressureGate 定时器 / persistent-store 属性删除等）
- PatternMiner CJK 分词：从"按空白切词"改为"CJK 字符级 + Latin 词级"混合分词，能正确从中文流水中挖掘 n-gram

### Tooling

- 新增工作区目录：`packages/embed/*`、`packages/concurrency`、`packages/taskpool`、`packages/dormant`
- `pnpm-workspace.yaml` / `tsconfig.json` 引用同步更新
- 验证命令：`pnpm -r typecheck` 与 `pnpm -r --workspace-concurrency=1 test`（Windows 下并行 esbuild 偶发 "service was stopped" 时使用串行模式）

---

## [2.0.0-python-reference] —— Python 实现冻结 (2026-04-29)

- Python v2.0 在仓库根目录冻结为"语义参考实现"
- 不再接收新功能；仅修复严重安全 / 文档 / 行为对齐问题
- 详见 [`docs/architecture/python-reference.md`](./docs/architecture/python-reference.md)
