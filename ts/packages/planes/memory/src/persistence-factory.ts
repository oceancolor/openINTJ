import type { EmbeddingProvider, HookBus } from "@openintj/core";
import { InMemoryVectorStore, LanceDBVectorStore, type VectorStore } from "@openintj/storage-lance";
import {
  InMemoryMetadataStore,
  type MetadataStore,
  SqliteCanonicalMemoryStore,
  SqliteMetadataStore,
} from "@openintj/storage-sqlite";
import type { CanonicalMemoryRepository } from "./canonical.js";
import { PersistentMemoryStore } from "./persistent-store.js";
import type { MemoryStoreConfig } from "./store.js";

export type PersistenceMode = "real" | "memory";

export interface CreatePersistentMemoryStoreOpts {
  /**
   * 持久化数据目录。设置后默认走真实磁盘（LanceDB + SQLite）。
   * 不设置则强制走 in-memory（CI / 单元测试用）。
   */
  dataDir?: string;
  /**
   * 显式覆盖模式：
   *  - "real"：要求 dataDir 必须存在，加载真实 LanceDB + SQLite
   *  - "memory"：忽略 dataDir，使用 InMemory*Store
   * 默认按 dataDir 推断（有 dataDir → real，无 → memory）。
   */
  mode?: PersistenceMode;
  /** 向量维度（默认 64，对齐 SimpleEmbedder）。 */
  embeddingDim?: number;
  /** LanceDB 表名（默认 memory_fragments）。 */
  tableName?: string;
  /** SQLite 文件名（默认 metadata.db），相对 dataDir。 */
  sqliteFileName?: string;
  /** 透传给 PersistentMemoryStore.storeConfig（capacity 等）。 */
  storeConfig?: Partial<MemoryStoreConfig>;
  /** 嵌入器（默认 SimpleEmbedder via MemoryStore 构造函数）。 */
  embedder?: EmbeddingProvider;
  /** 透传给 MemoryStore 的 HookBus（emit event.MEMORY_WRITTEN change-feed）。 */
  hooks?: HookBus;
  /** 启动时 hydrate（默认 true）。 */
  hydrateOnInit?: boolean;
  /**
   * G2：启用 canonical 权威源（real 模式下在 dataDir 建 memory-canonical.db）。
   * 默认关——真实数据目录需先跑 legacy dry-run 对账并确认后才开启（迁移门）。
   */
  useCanonical?: boolean;
  /** embedding 空间真实 provider/model（指纹用；缺省取 embedder.name）。 */
  embeddingSpaceInfo?: { provider: string; model: string; transformVersion?: number };
}

export interface PersistenceBackends {
  vectorStore: VectorStore;
  metadataStore: MetadataStore;
  canonical?: CanonicalMemoryRepository;
  mode: PersistenceMode;
  /** 真实模式下：解析后的数据目录。 */
  dataDir?: string;
}

const ensureDir = async (dir: string): Promise<void> => {
  const fs = await import("node:fs/promises");
  await fs.mkdir(dir, { recursive: true });
};

export const buildPersistenceBackends = async (
  opts: CreatePersistentMemoryStoreOpts = {},
): Promise<PersistenceBackends> => {
  const mode: PersistenceMode = opts.mode ?? (opts.dataDir ? "real" : "memory");
  const dim = opts.embeddingDim ?? 64;

  if (mode === "memory") {
    return {
      vectorStore: new InMemoryVectorStore(),
      metadataStore: new InMemoryMetadataStore(),
      mode: "memory",
    };
  }

  if (!opts.dataDir) {
    throw new Error("createPersistentMemoryStore: mode='real' requires opts.dataDir");
  }
  const dataDir = opts.dataDir;
  await ensureDir(dataDir);

  const path = await import("node:path");
  const lanceDir = path.join(dataDir, "lancedb");
  const sqlitePath = path.join(dataDir, opts.sqliteFileName ?? "metadata.db");
  await ensureDir(lanceDir);

  const vectorStore = new LanceDBVectorStore({
    dataDir: lanceDir,
    tableName: opts.tableName ?? "memory_fragments",
    dimension: dim,
  });
  const metadataStore = new SqliteMetadataStore({
    dbPath: sqlitePath,
    wal: true,
  });
  const canonical = opts.useCanonical
    ? (new SqliteCanonicalMemoryStore(
        path.join(dataDir, "memory-canonical.db"),
      ) as unknown as CanonicalMemoryRepository)
    : undefined;

  return {
    vectorStore,
    metadataStore,
    ...(canonical ? { canonical } : {}),
    mode: "real",
    dataDir,
  };
};

/**
 * createPersistentMemoryStore —— 一站式工厂。
 *
 * 用法：
 *   - 真实持久化：`createPersistentMemoryStore({ dataDir: "/path/to/data", embeddingDim: 64 })`
 *   - 内存兜底：  `createPersistentMemoryStore({})` 或 `createPersistentMemoryStore({ mode: "memory" })`
 *
 * 已自动调用 `init()`（含 vectorStore + metadataStore.migrate + hydrate）。
 */
export const createPersistentMemoryStore = async (
  opts: CreatePersistentMemoryStoreOpts = {},
): Promise<PersistentMemoryStore> => {
  const backends = await buildPersistenceBackends(opts);
  const dim = opts.embeddingDim ?? 64;
  const store = new PersistentMemoryStore({
    vectorStore: backends.vectorStore,
    metadataStore: backends.metadataStore,
    ...(backends.canonical ? { canonical: backends.canonical } : {}),
    ...(opts.embeddingSpaceInfo ? { embeddingSpaceInfo: opts.embeddingSpaceInfo } : {}),
    storeConfig: { embeddingDim: dim, ...(opts.storeConfig ?? {}) },
    ...(opts.embedder ? { embedder: opts.embedder } : {}),
    ...(opts.hooks ? { hooks: opts.hooks } : {}),
    ...(opts.hydrateOnInit !== undefined ? { hydrateOnInit: opts.hydrateOnInit } : {}),
  });
  await store.init();
  return store;
};
