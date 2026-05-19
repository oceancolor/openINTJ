# Phase 3.4 — Dormant 持久化（#9）收尾报告

> 更新时间：2026-05-19
> 仓库标签：`v3.0.0-alpha.4`
> 前序：[Phase 3.3 RFC-003 装配](./phase3-3-rfc3-wiring.md)

## 一、目标与范围

Phase 3.3 把 Dormant Memory Learning 装配进了 `apps/server` / `apps/desktop`，但
`PassiveStore` / `PersonaConfig` 全部进程内存，**重启即丢学习**。Phase 3.4 收尾这一坑：
给 Dormant 子系统补 SQLite 真盘适配器，让"用户审批过的偏好 / 习惯"在进程关停后继续存在。

非目标：

- 不引入新的桌面 UI 入口（审批 UI 与 #9.B 一并放到 Phase 3.5）
- 不做云端同步（local-first 优先）
- 不做加密（macOS Keychain / Win DPAPI 留给后续 phase）

## 二、设计要点

### 2.1 包间分层

```
@openintj/dormant
  ├── persistence.ts        # DormantPersistenceAdapter 接口 + DormantSnapshot + InMemoryDormantStore
  ├── dormant-runtime.ts    # adapter 槽 + hydrate() + 五条热路径写穿
  ├── passive-store.ts      # +recordBulk()（hydration 批量回填）
  ├── internalization-manager.ts  # +restoreState()（不触发 lastUpdated / version 自增）
  └── index.ts              # 导出 persistence.*

@openintj/storage-sqlite
  ├── dormant.ts            # SqliteDormantStore 实现 + createSqliteDormantStore 工厂
  └── index.ts              # 导出 dormant.*

apps/server / apps/desktop
  └── agent.ts              # opts.dormantPersistence ∈ {auto, memory, real} + opts.dormantDbPath
```

### 2.2 接口契约（`DormantPersistenceAdapter`）

```ts
interface DormantPersistenceAdapter {
  readonly name: string;                              // 便于 audit 与 status() 暴露
  loadAll(): Promise<DormantSnapshot>;                // 仅 hydrate() 调一次
  recordEvent(event: PassiveEvent): void;             // 热路径同步
  upsertProposal(p: InternalizationProposal): void;   // 热路径同步
  savePersona(p: PersonaConfig): void;                // 热路径同步
  clearAll(): void;                                   // reset() 时调
  close(): Promise<void>;                             // 释放底层句柄
}
```

**关键决策**：

1. **写路径同步、不抛错** —— `recordEvent / upsertProposal / savePersona / clearAll` 全同步，
   底层 better-sqlite3 同步即可。写入失败仅 `console.error`，不污染 agent 主循环。
2. **`loadAll` 是仅一次的"恢复点"** —— DormantRuntime 在装配后调用 `hydrate()`，
   把整库内容拉回 PassiveStore + InternalizationManager。多次 hydrate 安全（每次全量覆写）。
3. **`restoreState` 不动 meta** —— InternalizationManager 收到历史 proposals 时只回填，
   不重新触发 `lastUpdated += now()` / `version += 1`（否则会污染审计）。

### 2.3 SQLite schema（migration v1）

```sql
CREATE TABLE dormant_schema_version (version INTEGER PRIMARY KEY);

CREATE TABLE dormant_events (
  eventId      TEXT PRIMARY KEY,
  ts           INTEGER NOT NULL,
  source       TEXT NOT NULL,
  text         TEXT NOT NULL,
  metadataJson TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_devt_ts ON dormant_events(ts);
CREATE INDEX idx_devt_source ON dormant_events(source);

CREATE TABLE dormant_proposals (
  proposalId  TEXT PRIMARY KEY,
  patternJson TEXT NOT NULL,
  targetField TEXT NOT NULL,
  valueJson   TEXT NOT NULL,
  status      TEXT NOT NULL,
  ts          INTEGER NOT NULL,
  decidedAt   INTEGER
);
CREATE INDEX idx_dprop_status ON dormant_proposals(status);
CREATE INDEX idx_dprop_ts ON dormant_proposals(ts);

CREATE TABLE dormant_persona (
  id   INTEGER PRIMARY KEY CHECK (id = 1),
  json TEXT NOT NULL
);
```

**与 `SqliteMetadataStore` 故意解耦**：

- 单独的 `dormant.sqlite` 文件（默认 `${dataDir}/dormant.sqlite`），避免和 fragments_meta /
  audit / sessions 表共表
- 用户可以独立备份 / 删除 dormant 库而不影响主存储
- 后续要换 dormant 后端（e.g. JSON file、远端服务）也容易

### 2.4 装配点 opt-in 矩阵

| `enableDormant` | `dataDir` | `dormantPersistence` | 结果 |
|---|---|---|---|
| `false`（默认） | 任意 | 任意 | `agent.dormant` 是 `undefined`，零开销 |
| `true` | 缺 | `'auto'` / 缺省 | dormant in-memory，不挂 adapter |
| `true` | 有 | `'auto'` / 缺省 | **自动**挂 `SqliteDormantStore @ ${dataDir}/dormant.sqlite` |
| `true` | 任意 | `'memory'` | 强制不挂 adapter（即使 dataDir 存在） |
| `true` | 任意 | `'real'` | 强制挂 adapter；缺 `dbPath` 抛错 |
| `true` | 任意 | 任意 | `opts.dormantOpts.adapter` 显式覆盖以上规则 |

环境变量：

- `OPENINTJ_DORMANT=1` → 等价 `enableDormant: true`
- `OPENINTJ_DORMANT_DB_PATH=/some/path.sqlite` → 覆盖默认路径

### 2.5 status() 扩展

```jsonc
{
  "dormant": {
    "enabled": true,
    "passiveSize": 42,
    "pendingProposals": 3,
    "persistence": {                                  // 仅挂 adapter 时存在
      "adapter": "sqlite-dormant:/data/dormant.sqlite",
      "dbPath": "/data/dormant.sqlite"
    }
  }
}
```

## 三、测试矩阵

| 测试文件 | 数量 | CI 模式 | E2E 模式 | 说明 |
|---|---:|:-:|:-:|---|
| `packages/dormant/__tests__/persistence.spec.ts` | 9 | ✅ | ✅ | InMemoryDormantStore CRUD + hydrate + write-through |
| `packages/storage/sqlite/__tests__/dormant.spec.ts` | 11 | ✅ | ✅ | SqliteDormantStore（`:memory:` 走真 better-sqlite3） |
| `apps/server/__tests__/dormant-persistence-e2e.spec.ts` | 6 | 2 PASS + 4 skip | 6 PASS | server 装配 + 真盘往返 |

**关键 e2e 用例**：`record → mine → approve → close → 重装配 → hydrate → 验证状态恢复`。

## 四、已知限制 / 留尾

1. **桌面端 UI 没有审批面板**：dormant proposals 已经能持久化和恢复，但 desktop renderer
   还没接 `/api/dormant/proposals` IPC channel 的 UI 入口（#9.B，下一个 phase）。
2. **历史 events 是无限增长的**：`dormant_events` 没有自动清理。
   当前 PassiveStore 内存层有 `maxPassiveEvents` 环形上限，但磁盘表会一直累积。
   建议下一个 phase 加 `pruneEvents(olderThanTs)` 接口或 LRU 策略。
3. **跨进程并发**：better-sqlite3 走单进程同步访问；同时跑两个 server / desktop 进程对
   同一个 `dormant.sqlite` 是未定义行为（与 `SqliteMetadataStore` 一致）。
4. **加密**：dormant.sqlite 是明文。**用户敏感偏好不应该走 dormant**（已经在 `propose()`
   层面通过 `defaultMapToField` 限制 category；自定义 mapper 时需自己把关）。
5. **rate-limit 装饰只覆盖 LLM**：dormant.mine() 调用 LLM extract 时也走 rate limit；这一条
   已经在 Phase 3.3 通过 `RateLimitedLlmClient` 自动覆盖，但只是 chat / visionChat 两个接口
   ——未来 stream 接口要单独扩。

## 五、对外可观察变更

- 新增 opts：`dormantPersistence`、`dormantDbPath`
- 新增字段：`agent.dormantPersistenceInfo`、`status().dormant.persistence`
- 新增 export：`@openintj/dormant` → `DormantPersistenceAdapter` / `InMemoryDormantStore` /
  `DormantSnapshot`
- 新增 export：`@openintj/storage-sqlite` → `SqliteDormantStore` / `createSqliteDormantStore` /
  `SqliteDormantConfig`
- 新增 env：`OPENINTJ_DORMANT_DB_PATH`

## 六、下一站候选

按 `next-session.md` §三 顺序：

1. ⭐⭐⭐ **#9.B Dormant 审批 UI**：桌面端真正能看见 proposals 并 approve/reject
2. ⭐⭐⭐ **#1 行为对齐测试**：Python v2 ↔ TS 固定输入断言事件序列等价
3. ⭐⭐ **#3 嵌入基准** / **#4 Desktop Playwright E2E** / **#10 LanceDB FTS 路径**

## 七、自检脚本

```powershell
cd F:\openINTJ\ts
pnpm lint                                            # exit 0（2 条已知 React 警告）
pnpm exec turbo run typecheck --concurrency=1        # 33/33
pnpm exec turbo run test --concurrency=1             # 默认 CI 模式

$env:OPENINTJ_E2E="1"
pnpm exec turbo run test --concurrency=1             # 含真盘 e2e
Remove-Item env:OPENINTJ_E2E
```
