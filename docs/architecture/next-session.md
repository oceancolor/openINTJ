# 下一次工作交接备忘

> 本文件用于工作中断 / 多日离开后快速恢复上下文。
> 上次更新：2026-05-19（Phase 3.5 Dormant 审批 UI 收官当日）

---

## 一、当前停在哪里

- **Phase 3.1（真实持久化 e2e）已收官**，仓库标签：`v3.0.0-alpha.1`
- **Phase 3.2（GitHub Actions CI）已收官**，仓库标签：`v3.0.0-alpha.2`
- **Phase 3.3（RFC-003 装配进主 Agent）已收官**，仓库标签：`v3.0.0-alpha.3`
- **Phase 3.4（Dormant 持久化 #9.A）已收官**，仓库标签：`v3.0.0-alpha.4`
- **Phase 3.5（Dormant 审批 UI #9.B）已收官**，仓库标签：`v3.0.0-alpha.5` ⭐ 本轮新增
- **CI 状态**（本地与 GitHub Actions 同口径）：
  - `pnpm lint` exit 0（2 条 React useExhaustiveDependencies 警告，不阻塞）
  - `pnpm exec turbo run typecheck --concurrency=1` → 33/33 successful
  - `pnpm exec turbo run test --concurrency=1`（CI 模式）→ 33/33 successful，**366 passed + 11 skipped**
  - `OPENINTJ_E2E=1 pnpm exec turbo run test --concurrency=1`（真盘模式）→ 33/33 successful，**377 passed，0 skipped**
  - turbo cache 已经把 `OPENINTJ_E2E` 等 env 算进 cache key，env 切换会强制 invalidate 测试任务
- **本轮主要产出（Phase 3.5 / #9.B Dormant 审批 UI）**：
  - `apps/desktop/src/shared/ipc-protocol.ts`：StatusResponseSchema 补 `persistence` / `retrievalMode` / `dormant` + 6 个 Dormant DTO/Response/Error schema
  - `apps/desktop/src/preload/index.ts`：暴露 5 个 dormant API（mine/list/approve/reject/persona），返回 success | error 联合类型
  - `apps/desktop/src/renderer/components/DormantPanel.tsx`：新组件 —— mine 按钮 / status filter / proposal 卡片（含 approve/reject）/ persona 折叠区
  - `apps/desktop/src/renderer/App.tsx`：右侧栏改 tab 布局（推理轨迹 / Dormant + pending 角标）
  - `apps/desktop/src/renderer/components/StatusBar.tsx`：补 retrievalMode / persistence.mode / dormant 状态；类型从本地 interface 切到 protocol re-export
  - `apps/desktop/__tests__/ipc-handlers.spec.ts`：12 → 18 tests（+6 个新契约测试）
- **Phase 3.4 产出**（上一轮）：见 [`phase3-4-dormant-persistence.md`](./phase3-4-dormant-persistence.md)
- **未提交变更**：仓库根 `.dockerignore` / `.env.example` / `deploy.sh` / `docker-compose.yml` / `nginx.conf` 仍未跟踪（Python v2 部署相关，**不属于本阶段范围**）

## 二、下次开机第一步：自检

```powershell
cd F:\openINTJ\ts
pnpm install                                             # 确认依赖未漂移
pnpm lint                                                # exit 0
pnpm exec turbo run typecheck --concurrency=1            # 33/33 successful
pnpm exec turbo run test --concurrency=1                 # 312 PASS / 7 skipped（CI 模式）

# 想跑真盘 e2e（需要 @lancedb/lancedb + better-sqlite3 已装）：
$env:OPENINTJ_E2E="1"
pnpm exec turbo run test --concurrency=1                 # 318 PASS / 0 skipped
Remove-Item env:OPENINTJ_E2E

# 验证 RFC-003 装配 opt-in：
$env:OPENINTJ_DORMANT="1"
$env:OPENINTJ_RETRIEVAL_MODE="hybrid"
$env:OPENINTJ_RATE_LIMIT_QPS="5"
pnpm --filter @openintj/server exec vitest run __tests__/dormant.spec.ts __tests__/hybrid-retrieve.spec.ts __tests__/rate-limited-llm.spec.ts
Remove-Item env:OPENINTJ_DORMANT, env:OPENINTJ_RETRIEVAL_MODE, env:OPENINTJ_RATE_LIMIT_QPS
```

> Windows 下 turbo `--concurrency=1` 是统一策略：避免并行 tsc / esbuild 抢内存导致的 V8 OOM 和 esbuild "service was stopped"。
> e2e 测试需要 30s 超时（`describe(..., { timeout: 30_000 }, ...)`），LanceDB 首次建表 + 重新打开比较慢。
> 远端 CI：见 `.github/workflows/ci.yml`，触发条件是 push / PR 改 `ts/**`。

## 三、Phase 3 候选路线（按推荐顺序）

来自 [`phase2-complete.md` §九](./phase2-complete.md#九未完成--后续路线)，下表加上"开工成本 / 收益"维度。

| # | 任务 | 开工成本 | 收益 | 推荐度 | 状态 |
|---|---|---|---|:-:|:-:|
| 1 | **Python v2 ↔ TS 行为对齐测试**：固定输入跑两边，断言事件序列等价 | 中 | 高 | ⭐⭐⭐ | 待办 |
| 2 | ~~真实持久化 e2e~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-09 完成 |
| 3 | **嵌入基准**：simple vs xenova vs ollama 在固定语料上的 nDCG | 低 | 中 | ⭐⭐ | 待办 |
| 4 | **Desktop E2E（Playwright + Electron）**：真渲染器跑 mock chat | 中 | 中 | ⭐⭐ | 待办 |
| 5 | ~~RFC-003 装配进主 Agent~~ | ~~中~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-11 完成 |
| 6 | **打包发布**：electron-builder Win/macOS + electron-updater | 高 | 中 | ⭐ | 待办 |
| 7 | **可观测性**：Hooks → OpenTelemetry | 低 | 低-中 | ⭐ | 待办 |
| 8 | ~~GitHub Actions CI 工作流~~ | ~~低~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-09 完成 |
| 9.A | ~~Dormant 持久化（SqliteDormantStore + hydrate）~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-19 完成（Phase 3.4） |
| 9.B | ~~Dormant 审批 UI（preload + DormantPanel + tab 布局）~~ | ~~中~~ | ~~中-高~~ | ⭐⭐⭐ | ✅ 2026-05-19 完成（Phase 3.5） |
| 10 | **HybridRetriever LanceDB FTS 路径**：大规模 fragments 时换 LanceDB 原生 FTS，避免每次重建索引 | 中 | 中 | ⭐⭐ | 待办（3.3 衍生） |
| 11 | **Dormant 事件清理**：`pruneEvents(olderThanTs)` / LRU 防 `dormant_events` 无限增长 | 低 | 中 | ⭐⭐ | 待办（3.4 衍生） |

**默认推荐下一站**：

- **#1 行为对齐测试**（覆盖核心组件，给重构兜底；Python v2 ↔ TS 固定输入断言事件序列等价）
- 或 **#4 Playwright Desktop E2E**（顺手能覆盖 Phase 3.5 的 UI；当前 renderer 0 测试，唯一缺口）
- 或 **#3 嵌入基准**（低成本，给默认选型一个数据支持）
- 或 **#11 dormant 事件清理**（Phase 3.4 留下的小尾巴）

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

## 六、本次会话产出文件清单（Phase 3.1 + 3.2）

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

---

## 七、Phase 3.3 / 3.4 / 3.5 关键陷阱（接续时回看）

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

---

**回来工作时**：直接对我说 "继续 Phase 3 的 #X" 或 "先自检一遍" 都可以，我会顺着这份备忘接下去。
