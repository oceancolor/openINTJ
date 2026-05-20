# 下一次工作交接备忘

> 本文件用于工作中断 / 多日离开后快速恢复上下文。
> 上次更新：2026-05-20（Phase 3.8 Hooks → OpenTelemetry 收官当日）

---

## 一、当前停在哪里

- **Phase 3.1（真实持久化 e2e）已收官**，仓库标签：`v3.0.0-alpha.1`
- **Phase 3.2（GitHub Actions CI）已收官**，仓库标签：`v3.0.0-alpha.2`
- **Phase 3.3（RFC-003 装配进主 Agent）已收官**，仓库标签：`v3.0.0-alpha.3`
- **Phase 3.4（Dormant 持久化 #9.A）已收官**，仓库标签：`v3.0.0-alpha.4`
- **Phase 3.5（Dormant 审批 UI #9.B）已收官**，仓库标签：`v3.0.0-alpha.5`
- **Phase 3.6（Python v2 ↔ TS 行为对齐测试 #1）已收官**，仓库标签：`v3.0.0-alpha.6`
- **Phase 3.7（Desktop E2E / Playwright + Electron #4）已收官**，仓库标签：`v3.0.0-alpha.7`
- **Phase 3.8（Hooks → OpenTelemetry #7）已收官**，仓库标签：`v3.0.0-alpha.8` ⭐ 本轮新增
- **CI 状态**（本地与 GitHub Actions 同口径）：
  - `pnpm lint` exit 0（2 条 React useExhaustiveDependencies 警告，不阻塞）
  - `pnpm exec turbo run typecheck --concurrency=1` → 35/35 successful（新增 `@openintj/telemetry-otel`）
  - `pnpm exec turbo run test --concurrency=1`（CI 模式）→ 35/35 successful，**444 passed + 11 skipped**
  - `OPENINTJ_E2E=1 pnpm exec turbo run test --concurrency=1`（真盘模式）→ 35/35 successful，**455 passed，0 skipped**
  - **`OPENINTJ_PLAYWRIGHT=1 pnpm --filter @openintj/desktop run e2e`（Desktop E2E 模式）→ 7/7 passed（约 35s）**
  - turbo cache 已经把 `OPENINTJ_E2E` 等 env 算进 cache key，env 切换会强制 invalidate 测试任务
- **本轮主要产出（Phase 3.8 / #7 Hooks → OpenTelemetry）**：
  - **新包 `@openintj/telemetry-otel`**（packages/telemetry/otel/）：
    - `attachOtelToHooks(bus, opts)` —— 订阅 hook 事件、per-traceId 维护
      iteration / action / tool span 帧栈、产 6 个 counter；返回 `dispose()`
    - `bootstrapNodeOtel(opts)` —— 可选 SDK 引导（懒 import；缺包才抛错）
    - 10 个新 spec：noop 2 / spans 2 / metrics 3 / dispose 3
  - server / desktop agent 装配端：`enableOtel?: boolean | AttachOtelOpts` + `OPENINTJ_OTEL=1` env + `agent.otel`；`close()` 调 dispose
  - `apps/server/__tests__/otel-wiring.spec.ts`：4 个 wiring 验证（代码 / env / 默认关 / 显式关）
  - `pnpm-workspace.yaml`：加 `packages/telemetry/*`；根 `tsconfig.json` refs 加新包
  - `docs/architecture/phase3-8-otel.md`：阶段记录 + 选型 + 7 类陷阱
- **Phase 3.7 产出**（上一轮）：见 [`phase3-7-desktop-e2e.md`](./phase3-7-desktop-e2e.md)
- **Phase 3.5 产出**（上一轮）：见 [`phase3-5-dormant-approval-ui.md`](./phase3-5-dormant-approval-ui.md)
- **Phase 3.4 产出**：见 [`phase3-4-dormant-persistence.md`](./phase3-4-dormant-persistence.md)
- **未提交变更**：仓库根 `.dockerignore` / `.env.example` / `deploy.sh` / `docker-compose.yml` / `nginx.conf` 仍未跟踪（Python v2 部署相关，**不属于本阶段范围**）

## 二、下次开机第一步：自检

```powershell
cd F:\openINTJ\ts
pnpm install                                             # 确认依赖未漂移
pnpm lint                                                # exit 0
pnpm exec turbo run typecheck --concurrency=1            # 33/33 successful
pnpm exec turbo run test --concurrency=1                 # 430 PASS / 11 skipped（CI 模式）

# 想跑真盘 e2e（需要 @lancedb/lancedb + better-sqlite3 已装）：
$env:OPENINTJ_E2E="1"
pnpm exec turbo run test --concurrency=1                 # 441 PASS / 0 skipped
Remove-Item env:OPENINTJ_E2E

# 想跑 Desktop E2E（Playwright + 真 Electron + 真 BrowserWindow，约 35s）：
$env:OPENINTJ_PLAYWRIGHT="1"
pnpm --filter @openintj/desktop run e2e                  # 7/7 passed（含 build）
Remove-Item env:OPENINTJ_PLAYWRIGHT

# 验证 RFC-003 装配 opt-in：
$env:OPENINTJ_DORMANT="1"
$env:OPENINTJ_RETRIEVAL_MODE="hybrid"
$env:OPENINTJ_RATE_LIMIT_QPS="5"
pnpm --filter @openintj/server exec vitest run __tests__/dormant.spec.ts __tests__/hybrid-retrieve.spec.ts __tests__/rate-limited-llm.spec.ts
Remove-Item env:OPENINTJ_DORMANT, env:OPENINTJ_RETRIEVAL_MODE, env:OPENINTJ_RATE_LIMIT_QPS

# 重新生成 Python v2 ↔ TS 行为对齐 fixture（极少需要；Python v2 已冻结）：
cd F:\openINTJ
py scripts/python-parity/generate_fixtures.py            # 重写 4 份 fixture JSON
```

> Windows 下 turbo `--concurrency=1` 是统一策略：避免并行 tsc / esbuild 抢内存导致的 V8 OOM 和 esbuild "service was stopped"。
> e2e 测试需要 30s 超时（`describe(..., { timeout: 30_000 }, ...)`），LanceDB 首次建表 + 重新打开比较慢。
> 远端 CI：见 `.github/workflows/ci.yml`，触发条件是 push / PR 改 `ts/**`。

## 三、Phase 3 候选路线（按推荐顺序）

来自 [`phase2-complete.md` §九](./phase2-complete.md#九未完成--后续路线)，下表加上"开工成本 / 收益"维度。

| # | 任务 | 开工成本 | 收益 | 推荐度 | 状态 |
|---|---|---|---|:-:|:-:|
| 1 | ~~Python v2 ↔ TS 行为对齐测试~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-20 完成（Phase 3.6） |
| 2 | ~~真实持久化 e2e~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-09 完成 |
| 3 | **嵌入基准**：simple vs xenova vs ollama 在固定语料上的 nDCG | 低 | 中 | ⭐⭐ | 待办 |
| 4 | ~~Desktop E2E（Playwright + Electron）~~ | ~~中~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-20 完成（Phase 3.7） |
| 5 | ~~RFC-003 装配进主 Agent~~ | ~~中~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-11 完成 |
| 6 | **打包发布**：electron-builder Win/macOS + electron-updater | 高 | 中 | ⭐ | 待办 |
| 7 | ~~可观测性：Hooks → OpenTelemetry~~ | ~~低~~ | ~~低-中~~ | ⭐ | ✅ 2026-05-20 完成（Phase 3.8） |
| 8 | ~~GitHub Actions CI 工作流~~ | ~~低~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-09 完成 |
| 9.A | ~~Dormant 持久化（SqliteDormantStore + hydrate）~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-19 完成（Phase 3.4） |
| 9.B | ~~Dormant 审批 UI（preload + DormantPanel + tab 布局）~~ | ~~中~~ | ~~中-高~~ | ⭐⭐⭐ | ✅ 2026-05-19 完成（Phase 3.5） |
| 10 | **HybridRetriever LanceDB FTS 路径**：大规模 fragments 时换 LanceDB 原生 FTS，避免每次重建索引 | 中 | 中 | ⭐⭐ | 待办（3.3 衍生） |
| 11 | **Dormant 事件清理**：`pruneEvents(olderThanTs)` / LRU 防 `dormant_events` 无限增长 | 低 | 中 | ⭐⭐ | 待办（3.4 衍生） |
| 12 | **Parity 扩展**：governance plane / Hooks / ContextEngine 接进 parity 网 | 中 | 中 | ⭐⭐ | 待办（3.6 衍生） |

**默认推荐下一站**：

- **#3 嵌入基准**（低成本，给默认选型一个数据支持；下一阶段顺手填）
- 或 **#11 dormant 事件清理**（Phase 3.4 留下的小尾巴；30 行 SQL + 几个 spec）
- 或 **#12 parity 扩展**（顺势补 governance / Hooks / ContextEngine 进对齐网）
- 或 **#6 打包发布**（electron-builder Win/macOS + electron-updater；体感工程量最大但收益最直观）
- 或 **#10 HybridRetriever LanceDB FTS**（3.3 衍生；N>10k fragment 时性能升级）
- 或 **Phase 3.8.1 OTel 扩展**：Hono route + Electron IPC 自动 span 接到 agent span 树

## 四、上下文复盘清单

按这个顺序读，10 分钟即可回到工作状态：

1. [`docs/architecture/phase2-complete.md`](./phase2-complete.md) —— **唯一最重要**，10 节覆盖一切
2. [`CHANGELOG.md`](../../CHANGELOG.md) —— `3.0.0-alpha.0..2`（Phase 2 / 持久化 / CI）
3. [`docs/architecture/python-reference.md`](./python-reference.md) —— Python v2 冻结说明 + 已知遗留清单
4. [`docs/rfcs/`](../rfcs/) —— RFC-001..004
5. [`docs/agent-architecture-research_20260422.md`](../agent-architecture-research_20260422.md) —— 最初的架构论证底稿

## 五、当前已知陷阱 / 注意事项

- **CI / 本地统一用 turbo `--concurrency=1`**：跨 OS 一致策略，规避 tsc V8 OOM 与 esbuild 抢占
- **不要**直接编辑 `packages/*/dist/*` —— 改 `src` 后必须 `pnpm exec turbo run build` 才能让消费方测试看到
- **不要**碰 Python 仓库根目录代码（已冻结为 `v2.0-python-reference`，仅作语义参考）
- **LanceDB 关键陷阱**（这一轮踩的坑）：
  - 必须用 `apache-arrow` 显式声明 `FixedSizeList<Float32, N>` schema，**不能**靠 seed-row 推断（旧版 fallback 已保留）
  - SQL `where` 子句中 camelCase 字段必须 **双引号包裹**（`"fragmentId" = '...'`）
  - LanceDB 返回的 `embedding` 是 TypedArray / Arrow Vector，`taskTags` 是 Arrow Vector —— `lancedb.ts` 的 `normalizeEmbedding` / `normalizeStringArray` 会兜底转 `number[]` / `string[]`
  - `apache-arrow` 是 `@openintj/storage-lance` 的 **直接依赖**（不是 peer），因为 `init()` 必用
- **e2e 测试**：默认 skip。需要 `OPENINTJ_E2E=1`，且 `apps/server` / `apps/desktop` / `plane-memory` 工作区都已装 `@lancedb/lancedb` + `better-sqlite3`
- **turbo 缓存**：`OPENINTJ_E2E` / `OPENINTJ_DATA_DIR` / `OPENINTJ_DESKTOP_NO_PERSIST` / `OPENINTJ_LANCE_DEBUG` 已纳入 cache key（`turbo.json`），切换 env 会强制 invalidate test 任务
- **biome 已放宽**：`useLiteralKeys` / `noNonNullAssertion` / `noUnusedTemplateLiteral` 等 13 条与历史代码冲突的规则已关；不要再因为 lint 报错就批量改业务代码
- `@xenova/transformers` 仍是 peer dependency，按需 `pnpm add`

## 六、累计产出文件清单（Phase 3.1 → 3.6）

### Phase 3.1（持久化 e2e）

新增：

- `ts/packages/planes/memory/src/persistence-factory.ts`
- `ts/packages/planes/memory/__tests__/persistence-factory.spec.ts`
- `ts/apps/server/__tests__/persistence-e2e.spec.ts`
- `ts/apps/desktop/__tests__/agent-persistence.spec.ts`
- `docs/architecture/phase3-1-persistence.md`

改动：

- `ts/packages/storage/lance/src/lancedb.ts`：显式 schema + 双引号 SQL + 类型规范化
- `ts/packages/storage/lance/package.json`：`apache-arrow` 移到 `dependencies`
- `ts/packages/planes/memory/src/index.ts`：导出工厂
- `ts/apps/server/src/agent.ts`：用工厂 + dataDir 选项 + close
- `ts/apps/desktop/src/main/agent.ts`：用工厂 + dataDir 选项 + close
- `ts/apps/desktop/src/main/index.ts`：`app.getPath('userData')` + before-quit
- 各 `package.json`：装 peer deps（`@lancedb/lancedb` / `better-sqlite3`）

### Phase 3.2（GitHub Actions CI）

新增：

- `.github/workflows/ci.yml`（仓库根；旧的 `ts/.github/workflows/ci.yml` 已删除）

改动：

- `ts/turbo.json`：`test` 任务 cache key 加入 `OPENINTJ_*` env
- `ts/biome.json`：放宽与历史代码冲突的 13 条规则
- 全仓 107 个 `tsconfig.json` / `package.json` 被 biome formatter 自动收紧
- `docs/architecture/phase2-complete.md`：§九 #1 划掉
- `CHANGELOG.md`：新增 `3.0.0-alpha.1`、`3.0.0-alpha.2` 条目

### Phase 3.3（RFC-003 装配进主 Agent）

新增：

- `ts/packages/dormant/src/dormant-runtime.ts` + `__tests__/dormant-runtime.spec.ts`
- `ts/packages/concurrency/src/rate-limited-llm.ts`
- `ts/apps/server/src/hybrid-retrieve.ts`
- `ts/apps/server/__tests__/dormant.spec.ts` / `hybrid-retrieve.spec.ts` / `rate-limited-llm.spec.ts`
- `docs/architecture/phase3-3-rfc3-wiring.md`

改动：

- `ts/packages/dormant/src/index.ts`：导出 `DormantRuntime`
- `ts/packages/concurrency/src/index.ts`：导出 `RateLimitedLlmClient`
- `ts/apps/server/{src/agent.ts, src/routes.ts, src/rate-limited-llm.ts, package.json, tsconfig.json}`：装配三方向 opt-in + 路由 + 兼容 re-export
- `ts/apps/desktop/{src/main/agent.ts, src/main/ipc-handlers.ts, src/shared/ipc-protocol.ts, package.json, tsconfig.json}`：镜像 server 端装配 + 5 个 Dormant IPC channel + MEMORY_QUERY mode
- `ts/apps/desktop/__tests__/ipc-handlers.spec.ts`：扩展 5 个新测试
- `CHANGELOG.md`：新增 `3.0.0-alpha.3` 条目

### Phase 3.4（Dormant 持久化 / #9.A）

新增：

- `ts/packages/dormant/src/persistence.ts`（`DormantPersistenceAdapter` + `InMemoryDormantStore` + `DormantSnapshot`）
- `ts/packages/storage/sqlite/src/dormant.ts`（`SqliteDormantStore` + `createSqliteDormantStore`）
- `ts/packages/dormant/__tests__/persistence.spec.ts`（9 个）
- `ts/packages/storage/sqlite/__tests__/dormant.spec.ts`（11 个）
- `ts/apps/server/__tests__/dormant-persistence-e2e.spec.ts`（6 个，CI 2 PASS + 4 skip / E2E 6 PASS）
- `docs/architecture/phase3-4-dormant-persistence.md`

改动：

- `ts/packages/dormant/src/{dormant-runtime.ts, passive-store.ts, internalization-manager.ts, index.ts}`：adapter 槽 + hydrate + restoreState + recordBulk
- `ts/packages/storage/sqlite/src/index.ts`：修复重复 export bug
- `ts/apps/server/src/agent.ts` + `ts/apps/desktop/src/main/agent.ts`：新增 `dormantPersistence` / `dormantDbPath` opts + auto-wire + hydrate + `dormantPersistenceInfo`
- `CHANGELOG.md`：新增 `3.0.0-alpha.4` 条目

### Phase 3.5（Dormant 审批 UI / #9.B）

新增：

- `ts/apps/desktop/src/renderer/components/DormantPanel.tsx`（mine + filter + 卡片 + persona 折叠）
- `docs/architecture/phase3-5-dormant-approval-ui.md`

改动：

- `ts/apps/desktop/src/shared/ipc-protocol.ts`：StatusResponseSchema 补三字段 + 6 个 Dormant DTO/Response/Error schema
- `ts/apps/desktop/src/preload/index.ts`：暴露 5 个 dormant API（联合类型 success | error）
- `ts/apps/desktop/src/renderer/App.tsx`：右侧栏改 tab 布局 + Dormant pending 角标
- `ts/apps/desktop/src/renderer/components/StatusBar.tsx`：补 retrievalMode/persistence/dormant 三段；StatusSnapshot 切到 protocol re-export
- `ts/apps/desktop/src/renderer/components/TrajectoryPanel.tsx`：去外层 chrome（tab 容器统一提供）
- `ts/apps/desktop/__tests__/ipc-handlers.spec.ts`：扩展 6 个新契约测试
- `CHANGELOG.md`：新增 `3.0.0-alpha.5` 条目

### Phase 3.6（Python v2 ↔ TS 行为对齐测试 / #1）

新增：

- `scripts/python-parity/generate_fixtures.py`（Python 端只读取证脚本，覆盖 4 个 slice）
- `scripts/python-parity/README.md`（工具说明 + 已知偏差速查）
- `ts/packages/core/__tests__/parity/python-v2.spec.ts`（23 tests：SimpleEmbedder / cosine / decay）
- `ts/packages/core/__tests__/parity/fixtures/python-v2.json`
- `ts/packages/planes/control/__tests__/parity/python-v2.spec.ts`（21 tests：GoalParser / Planner）
- `ts/packages/planes/control/__tests__/parity/fixtures/python-v2.json`
- `ts/packages/planes/execution/__tests__/parity/python-v2.spec.ts`（17 tests：StateMachine / Executor）
- `ts/packages/planes/execution/__tests__/parity/fixtures/python-v2.json`
- `ts/packages/planes/memory/__tests__/parity/python-v2.spec.ts`（3 tests：Store overflow / Retriever scoring）
- `ts/packages/planes/memory/__tests__/parity/fixtures/python-v2.json`
- `docs/architecture/phase3-6-parity-tests.md`（阶段记录 + 已知偏差矩阵 + 容差策略）

改动：

- `ts/biome.json`：`files.ignore` 加 `**/__tests__/parity/fixtures/**`
- `CHANGELOG.md`：新增 `3.0.0-alpha.6` 条目

### Phase 3.8（Hooks → OpenTelemetry / #7）

新增：

- `ts/packages/telemetry/otel/`（新包 `@openintj/telemetry-otel`）：
  - `package.json`：deps=`@openintj/core` + `@opentelemetry/api`；6 个 SDK 包全标 `peerDependenciesMeta.optional`
  - `src/attach.ts`（~290 行）：`attachOtelToHooks(bus, opts)`，per-traceId span 帧栈 + 6 counter + dispose
  - `src/bootstrap.ts`（~100 行）：`bootstrapNodeOtel(opts)`，懒 import SDK + ProxyTracerProvider 探针 idempotent
  - `src/index.ts`：barrel
  - `__tests__/noop.spec.ts`（2 tests）：未注册 provider 不抛、不产 span
  - `__tests__/spans.spec.ts`（2 tests）：InMemorySpanExporter 断言 parent/child + ERROR 状态 + recordException
  - `__tests__/metrics.spec.ts`（3 tests）：InMemoryMetricExporter 断言 6 counter 累计
  - `__tests__/dispose.spec.ts`（3 tests）：dispose 兜底 end + unregister + 新 iteration 把旧 iter 标 unfinished
- `ts/apps/server/__tests__/otel-wiring.spec.ts`（4 tests）：enableOtel 三通道
- `docs/architecture/phase3-8-otel.md`：阶段记录 + 选型 + 7 类陷阱

改动：

- `ts/pnpm-workspace.yaml`：加 `packages/telemetry/*`
- `ts/tsconfig.json`：refs 加 `packages/telemetry/otel`
- `ts/apps/server/{src/agent.ts, package.json, tsconfig.json}`：
  - `enableOtel` opt + `resolveOtel(opts)` + `agent.otel` + close 调 dispose
  - devDep 加 `@opentelemetry/{api,sdk-trace-base}`（仅 wiring 测试用，运行时不需要）
- `ts/apps/desktop/{src/main/agent.ts, package.json, tsconfig.json}`：镜像 server
- `CHANGELOG.md`：新增 `3.0.0-alpha.8` 条目

### Phase 3.7（Desktop E2E / Playwright + Electron / #4）

新增：

- `ts/apps/desktop/e2e/playwright.config.ts`（workers=1，`OPENINTJ_PLAYWRIGHT=1` 才执行）
- `ts/apps/desktop/e2e/fixtures.ts`（`electronApp` + `page` fixture，默认 mock + no-persist）
- `ts/apps/desktop/e2e/tsconfig.json`（独立 e2e 项目，不污染 src）
- `ts/apps/desktop/e2e/tests/smoke.spec.ts`（5 tests：boot / status / chat / trajectory / dormant tab）
- `ts/apps/desktop/e2e/tests/dormant.spec.ts`（2 tests：mine 按钮 + 扫描摘要，需 `OPENINTJ_DORMANT=1`）
- `docs/architecture/phase3-7-desktop-e2e.md`（阶段记录 + 两个坑 + CI 集成）

改动：

- `ts/apps/desktop/src/main/index.ts`：preload 路径 `../preload/index.js` → `../preload/index.mjs`（修历史 silent fail）
- `ts/apps/desktop/package.json`：加 devDep `@playwright/test ^1.60`；
  `typecheck` 串第二段 `tsc --noEmit -p e2e/tsconfig.json`；
  新 script `e2e`（build + run）/ `e2e:run`（只 run）
- `ts/biome.json`：`files.ignore` 加 `**/test-results/**` 与 `**/playwright-report/**`
- `.github/workflows/ci.yml`：新增 `e2e-desktop` job（Ubuntu 24.04 + xvfb，需要 libnss3/libgtk-3-0 等运行时）
- `CHANGELOG.md`：新增 `3.0.0-alpha.7` 条目

---

## 七、Phase 3.3 / 3.4 / 3.5 / 3.6 / 3.7 / 3.8 关键陷阱（接续时回看）

1. **`DormantRuntime` 默认 `category: "other"` 时 proposals 为空**
   - PatternMiner 不配 `llmExtract` 会把每个 ngram 打成 "other"
   - `InternalizationManager.defaultMapToField` 忽略 "other"
   - **生产部署务必配 `dormantOpts.minerOpts.llmExtract`** 或自定义 `internalizationOpts.mapToField`
2. **HybridRetriever 每次查询重建索引** —— 中等规模够用，N>10k 切 LanceDB FTS
3. **rate-limit 装饰只覆盖 `chat / visionChat`** —— 未来加 stream/embeddings 接口需要同步扩展
4. **PassiveStore / PersonaConfig 不持久化** —— ✅ Phase 3.4 已解决（auto 模式：`dataDir + enableDormant=true` 自动挂 `SqliteDormantStore`）
5. **IPC 协议向后兼容**：新增字段都是 optional，旧 renderer 仍然能用；新 renderer 调旧 main 会拿到 `dormant_not_enabled` 而不是崩
6. **Phase 3.4 装配点**：装配顺序很重要 —— `await createSqliteDormantStore` → 传入 `DormantRuntime` 的 `adapter` 槽 → `await runtime.hydrate()`。close 时**先 dormant.close 再 persistentStore.close**
7. **Phase 3.4 `SqliteDormantConfigInput`**：`wal` 用 `z.boolean().default(true)`，input/output 类型不一致 —— 用 `z.input<>` 给装配点，`z.infer<>` 给内部
8. **Phase 3.4 `dormant_events` 表无限增长**：当前 PassiveStore 内存有 `maxPassiveEvents` 环形上限，但磁盘表会一直累积。下个 phase 加 `pruneEvents(olderThanTs)`（#11）
9. ~~**桌面端审批 UI 仍未接**~~：✅ Phase 3.5 完成
10. **Phase 3.5 协议联合类型**：preload 5 个 dormant API 全部返回 `Success | Error` 联合类型 —— renderer 必须用 `'error' in r` narrow 才能拿数据。这是为了把"dormant 未启用"这类正常态从 try/catch 里剥离出来
11. **Phase 3.5 类型对齐**：`StatusBar.tsx` 不再本地定义 `StatusSnapshot`，而是 `type StatusSnapshot = StatusResponse`（来自 protocol）。新加字段时只改 ipc-protocol.ts 即可全栈传播
12. **Phase 3.5 renderer 0 测试**：desktop 工作区只有 main-process vitest，没有 jsdom/@testing-library/react。`DormantPanel` 的逻辑分支已被 IPC 契约测试覆盖；UI 渲染留给手动 / Playwright e2e（#4）
13. **Phase 3.6 parity fixture 是 commit-in 资产**：`packages/*/__tests__/parity/fixtures/python-v2.json` 由 `scripts/python-parity/generate_fixtures.py` 一次性生成。CI 不跑 Python；只有 Python v2 端被"延寿活动"修补、或 `generate_fixtures.py` 自身改动时才需要重跑。biome 已 ignore 该目录，**不要**对 fixture 跑格式化。
14. **Phase 3.6 已知偏差矩阵**：详见 `phase3-6-parity-tests.md` §三。不要随手"修齐" Planner delete/execute 模板回 general、或把 TS `Math.LN2` 改回 `0.693` —— 这些偏差**有意保留**（TS 修复或扩展 Python）。
15. **Phase 3.6 决意保留 Python 0.693 近似**：`decayImportance` parity 容差用 `1e-4` 而非 `1e-12`。换 embedder（如 xenova / nomic-embed）也别动这条；它只影响 `decay` 一项的纯数学精度。
16. **Phase 3.6 fixture `schemaVersion=1`**：TS spec 加载时会断言版本；将来如果想加新字段（如 `governance` slice），把 `schemaVersion` 升 2 强制旧 fixture 失效，避免误判。
17. **Phase 3.7 preload `.mjs` 才是正确路径**：electron-vite 默认产物是 `out/preload/index.mjs`。**不要**回退到 `.js`，否则 `window.openintj` 永远 undefined，但 vitest 走 mock electron 路径不会暴露 —— 只有真 Electron 启动才崩。Electron 28+ 原生支持 ESM preload。
18. **Phase 3.7 Windows + Playwright `_electron.launch` 别加 `--no-sandbox`**：该 flag 在 Windows + Electron 33 + Playwright 1.60 这个具体组合下会让 launch 卡死 30s。Linux + xvfb 不需要这个 flag。fixture 已经只传 `[MAIN_ENTRY]`；扩 e2e 用例时也别图省事重新加 flag。
19. **Phase 3.7 e2e 默认 opt-in**：`OPENINTJ_PLAYWRIGHT=1` 才会跑；不设 env 时 playwright.config.ts 顶部直接 `testIgnore: ["**/*"]`。pnpm test / turbo test 不会触发它。CI 走专用 `e2e-desktop` job（已加 `OPENINTJ_PLAYWRIGHT: "1"` env）。
20. **Phase 3.7 strict-mode locator**：`getByText(/mock 模式/)` 会撞 chat 气泡 + trajectory JSON dump。新加 e2e 断言前先用 tailwind 颜色 token 圈父定位（`div.bg-\\[\\#1e1e2e\\]` = 主聊天区，`div.bg-\\[\\#313244\\]` = assistant 气泡）。
21. **Phase 3.8 HookBus traceId ≠ OTel traceId**：前者是 UUID 字符串、后者是 128-bit hex 由 SDK 生成。本适配器把 HookBus traceId 写到 `trace_id` span 属性方便反查。不要让 caller 拿 `agent.otel` 当作 trace context 源。
22. **Phase 3.8 tool 事件必须带 traceId 才能挂对 parent**：`tool.beforeCall` / `tool.afterCall` / `tool.onError` emit 时必须传 `{ traceId }`（ToolHub 真实代码已这么做）。漏传的话 tool span 会挂在 'anon' trace 上，与 iteration / action 失联。写 hook 单测时尤其要注意。
23. **Phase 3.8 OTel SDK 是 optional peer**：`attachOtelToHooks` 只需 `@opentelemetry/api`（已是硬依赖）；`bootstrapNodeOtel` 懒 import 6 个 SDK 包，缺包就 throw。生产部署用 OTLP 时 consumer 自己 `pnpm add @opentelemetry/{sdk-trace-node,exporter-trace-otlp-http,resources,semantic-conventions}`。
24. **Phase 3.8 metric 默认 DELTA**：`InMemoryMetricExporter` 的 `AggregationTemporality.DELTA = 0`；构造时显式传 0，否则跨多次 emit 会丢中间增量。生产 exporter 一般是 CUMULATIVE，行为不同。
25. **Phase 3.8 tool.onError 不 end span**：故意设计，让 `tool.afterCall` 统一收尾（happy-path 一致）。如果业务异常分支不发 afterCall，`dispose()` 会兜底 end 并打 `disposed=true` 标记。

---

**回来工作时**：直接对我说 "继续 Phase 3 的 #X" 或 "先自检一遍" 都可以，我会顺着这份备忘接下去。
