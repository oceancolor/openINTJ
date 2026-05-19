# Phase 3.1 — 真实持久化 e2e（完成报告）

> 完成日期：2026-05-09
> 范围：把 `apps/server` / `apps/desktop` 默认从 in-memory 兜底切到真盘 LanceDB + SQLite，并端到端验证 "写入 → 关闭 → 重启 → 读回"。
> 上一阶段：[`phase2-complete.md`](./phase2-complete.md)

---

## 一、目标对照

| 项 | Phase 2 状态 | Phase 3.1 目标 | 实际交付 |
|---|---|---|---|
| `apps/server` 默认存储 | `InMemory*Store` 兜底 | LanceDB + SQLite，env 可启 | ✅ `OPENINTJ_DATA_DIR` 启用 |
| `apps/desktop` 默认存储 | `InMemory*Store` 兜底 | `userData` 目录写盘 | ✅ Electron `app.getPath('userData')` |
| 持久化装配点 | server / desktop 各自 new | 统一工厂 | ✅ `createPersistentMemoryStore` |
| 进程关闭释放句柄 | 无 | `agent.close()` | ✅ 全部接入 + before-quit 钩子 |
| e2e 写盘往返 | 无 | 单元测试覆盖 | ✅ 7 个新测试（gated by `OPENINTJ_E2E=1`） |
| LanceDB schema | 自动推断 | 显式声明 | ✅ `apache-arrow` 显式 `FixedSizeList<Float32, N>` |

## 二、关键设计

### 2.1 持久化工厂

集中决策点在 `packages/planes/memory/src/persistence-factory.ts`：

```ts
createPersistentMemoryStore({
  dataDir?: string,         // 提供则默认 'real'
  mode?: 'real' | 'memory', // 显式覆盖
  embeddingDim?: number,
  embedder?: EmbeddingProvider,
  storeConfig?: Partial<MemoryStoreConfig>,
  hydrateOnInit?: boolean,
})
```

- `mode === 'real' && !dataDir` → 抛错（fail-fast）
- `mode === 'memory'` 或 默认无 `dataDir` → 走 `InMemoryVectorStore` + `InMemoryMetadataStore`
- 真盘模式自动 `mkdir -p` `dataDir/lancedb/` 与 `dataDir/metadata.db`

### 2.2 LanceDB schema 修正

旧路径："插一行 seed → LanceDB 自动推断 schema → 立刻删除"。
该路径在 `taskTags: []`（空数组）下会抛 "Cannot infer list vector from empty array"，
且 embedding 被推断为 `List<Float64>` 而非 `FixedSizeList<Float32, N>`，向量搜索直接不可用。

新路径：用 `apache-arrow` 显式声明 schema 后调 `db.createEmptyTable(name, schema)`：

```ts
new arrow.Schema([
  new arrow.Field("fragmentId", new arrow.Utf8(), false),
  new arrow.Field("embedding",
    new arrow.FixedSizeList(dim, new arrow.Field("item", new arrow.Float32(), true)),
    false),
  new arrow.Field("taskTags",
    new arrow.List(new arrow.Field("item", new arrow.Utf8(), true)),
    false),
  // ...
]);
```

旧版 LanceDB 无 `createEmptyTable` 时回落到 seed-row + delete，但 seed 的 `taskTags` 用 `["__seed_tag__"]` 占位避免空数组推断失败。

### 2.3 LanceDB SQL 大小写

LanceDB 的 SQL 引擎对未加引号的标识符**自动转小写**，会导致 `"fragmentId" = '...'` 找不到字段。
解决：所有 camelCase 字段在 `where` / `delete` / `IN (...)` 子句中**双引号包裹**：

```sql
"fragmentId" IN ('a', 'b')
"memoryType" IN ('long_term', 'working')
```

### 2.4 LanceDB 返回类型规范化

LanceDB 通过 `arrow-ipc` 反序列化时返回的不是普通 JS 数组：

| 字段 | 实际类型 |
|---|---|
| `embedding` | `Float32Array` 或 Arrow `Vector`（含 `.toArray()`） |
| `taskTags` | Arrow `Vector<Utf8>`（可迭代） |

`VectorRowSchema.parse`（zod）不会自动适配这些类型，会**静默抛错**（被外层 try/catch 吞掉）。
解决：`search()` 内引入两个本地辅助：

```ts
const normalizeEmbedding = (raw: unknown): number[] => {
  if (raw instanceof Float32Array || raw instanceof Float64Array) return Array.from(raw);
  if (Array.isArray(raw)) return raw.map(Number);
  if (typeof (raw as any)?.toArray === "function") return Array.from((raw as any).toArray()).map(Number);
  if (raw && (raw as any)[Symbol.iterator]) return Array.from(raw as Iterable<number>).map(Number);
  return [];
};

const normalizeStringArray = (raw: unknown): string[] => { /* 类似套路 */ };
```

调试模式：`OPENINTJ_LANCE_DEBUG=1` 会把 zod 解析失败的行打到 stderr。

### 2.5 进程关闭路径

```
Electron app.before-quit  ──►  agent.close()  ──►  PersistentMemoryStore.close()
                                                    ├── LanceDBVectorStore.close()
                                                    └── SqliteMetadataStore.close()
```

server 端没有"退出"事件，由调用方在 `Ctrl+C` / `SIGTERM` 处理器里调 `agent.close()`（已在 README 示例里点出）。

## 三、新增 / 修改文件

新增：

- `ts/packages/planes/memory/src/persistence-factory.ts`
- `ts/packages/planes/memory/__tests__/persistence-factory.spec.ts`
- `ts/apps/server/__tests__/persistence-e2e.spec.ts`
- `ts/apps/desktop/__tests__/agent-persistence.spec.ts`
- `docs/architecture/phase3-1-persistence.md`（本文）

修改：

- `ts/packages/storage/lance/src/lancedb.ts`
- `ts/packages/storage/lance/package.json`
- `ts/packages/planes/memory/src/index.ts`
- `ts/apps/server/src/agent.ts`
- `ts/apps/server/package.json`
- `ts/apps/desktop/src/main/agent.ts`
- `ts/apps/desktop/src/main/index.ts`
- `ts/apps/desktop/package.json`
- `docs/architecture/phase2-complete.md`（§九 #1 划掉）
- `CHANGELOG.md`（新增 `3.0.0-alpha.1`）

## 四、CI 验证

| 模式 | 命令 | 结果 |
|---|---|---|
| typecheck | `pnpm -r typecheck` | **19/19 绿** |
| 默认（无 e2e） | `pnpm -r --workspace-concurrency=1 test` | **292/292 绿，7 skipped** |
| 全量真盘 | `$env:OPENINTJ_E2E="1"; pnpm -r --workspace-concurrency=1 test` | **299/299 绿，0 skipped** |

测试数变化：

- `plane-memory`：56 → 61（+5：3 in-mem + 2 e2e）
- `apps/server`：7 → 11（+4 e2e）
- `apps/desktop`：7 → 11（+4：1 in-mem 默认 + 2 in-mem NO_PERSIST + 1 e2e）

合计：**286 → 292（CI）/ 299（E2E）**

## 五、已知限制 / 后续

- **未做并发回归**：多 server 实例同时打开同一 `dataDir` 的并发互锁未测试；LanceDB 0.x 行为与 SQLite WAL 默认是单写者多读者
- **未做大数据量基准**：当前 e2e 只到 ~10 行，未覆盖 LanceDB ANN 索引在 10K+ 行下的延迟
- **退出路径不完备**：server 端缺 `SIGTERM` 处理器示例
- **embedding 维度不可变**：表一旦建立，`embeddingDim` 不能再变；切换 embedder 维度需要新 `dataDir` 或迁移脚本（后者未实现）

这些都不影响 alpha → beta 的硬门槛（"持久化能不能用"），留给后续阶段。

## 六、回滚指引

如果想临时退回 in-memory（例如某台机器上 LanceDB / better-sqlite3 装不上）：

```powershell
# server
Remove-Item env:OPENINTJ_DATA_DIR

# desktop
$env:OPENINTJ_DESKTOP_NO_PERSIST="1"
```

代码里也可以传 `assembleServerAgent({ persistenceMode: 'memory' })` 显式覆盖。

---

**Phase 3.1 完成于 2026-05-09。下一站候选：行为对齐测试 / RFC-003 装配 / GitHub Actions CI。**
