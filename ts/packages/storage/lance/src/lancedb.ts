import { z } from "zod";
import {
  type VectorRow,
  VectorRowSchema,
  type VectorSearchOpts,
  type VectorSearchResult,
  type VectorStore,
} from "./types.js";

export const LanceDBStoreConfigSchema = z.object({
  /** 数据目录。LanceDB 在该目录下建表。 */
  dataDir: z.string(),
  /** 表名。 */
  tableName: z.string().default("memory_fragments"),
  /** 向量维度（必填，建表用）。 */
  dimension: z.number().int().positive(),
});
export type LanceDBStoreConfig = z.infer<typeof LanceDBStoreConfigSchema>;

interface LanceQueryBuilder {
  limit(n: number): {
    where(filter: string): { toArray(): Promise<unknown[]> };
    toArray(): Promise<unknown[]>;
  };
}

interface LanceTable {
  add(rows: VectorRow[]): Promise<void>;
  delete(predicate: string): Promise<void>;
  countRows(): Promise<number>;
  /** 向量检索：传 number[]。FTS 检索：传 string + queryType="fts"（LanceDB 支持）。 */
  search(query: readonly number[] | string, queryType?: string): LanceQueryBuilder;
  /** 建索引（含 FTS）；旧版可能没有。 */
  createIndex?(column: string, opts?: unknown): Promise<void>;
  toArrow?: () => Promise<{ toArray(): unknown[] }>;
  query?: () => { toArray(): Promise<unknown[]> };
}

interface LanceDB {
  openTable(name: string): Promise<LanceTable>;
  createTable(name: string, data: VectorRow[]): Promise<LanceTable>;
  createEmptyTable?(name: string, schema: unknown): Promise<LanceTable>;
  tableNames(): Promise<string[]>;
}

interface LanceIndexFactory {
  fts(opts?: unknown): unknown;
}

interface LanceModule {
  connect(uri: string): Promise<LanceDB>;
  /** 索引工厂（含 `Index.fts()`）；旧版可能没有。 */
  Index?: LanceIndexFactory;
}

interface ArrowField {
  new (name: string, type: unknown, nullable?: boolean): unknown;
}

/** 把 Arrow Vector / TypedArray / 普通数组都规范化为 number[]。 */
const normalizeEmbedding = (raw: unknown): number[] => {
  if (raw instanceof Float32Array || raw instanceof Float64Array) {
    return Array.from(raw);
  }
  if (Array.isArray(raw)) {
    return raw.map((x) => Number(x));
  }
  // Arrow Vector：有 toArray()
  const v = raw as { toArray?: () => ArrayLike<number> } | null;
  if (v && typeof v.toArray === "function") {
    return Array.from(v.toArray()).map((x) => Number(x));
  }
  // FixedSizeList Vector：可迭代
  if (raw && typeof (raw as Iterable<number>)[Symbol.iterator] === "function") {
    return Array.from(raw as Iterable<number>).map((x) => Number(x));
  }
  return [];
};

const normalizeStringArray = (raw: unknown): string[] => {
  if (Array.isArray(raw)) return raw.map((x) => String(x));
  const v = raw as { toArray?: () => ArrayLike<unknown> } | null;
  if (v && typeof v.toArray === "function") {
    return Array.from(v.toArray()).map((x) => String(x));
  }
  if (raw && typeof (raw as Iterable<unknown>)[Symbol.iterator] === "function") {
    return Array.from(raw as Iterable<unknown>).map((x) => String(x));
  }
  return [];
};
interface ArrowModule {
  Schema: new (fields: unknown[]) => unknown;
  Field: ArrowField;
  FixedSizeList: new (listSize: number, child: unknown) => unknown;
  Float32: new () => unknown;
  Utf8: new () => unknown;
  Float64: new () => unknown;
  Int64: new () => unknown;
  List: new (child: unknown) => unknown;
}

/**
 * LanceDBVectorStore —— 基于 @lancedb/lancedb 的向量持久化。
 *
 * 注意：
 * - @lancedb/lancedb 是 peerDependency，需用户安装。
 * - 首次 init() 会建表（用一行 placeholder 记录初始化 schema）。
 *
 * 不支持的功能 fallback：
 * - 复杂条件过滤当前用 SQL where 子句简化处理（仅 memoryType 等基础字段）
 */
export class LanceDBVectorStore implements VectorStore {
  readonly name: string;
  readonly config: LanceDBStoreConfig;
  private db?: LanceDB;
  private table?: LanceTable;
  private lanceModule?: LanceModule;
  private _dimension: number;
  /** FTS 能力探测结果：undefined=未探测，true/false=已知。 */
  private _supportsFts?: boolean;

  constructor(config: LanceDBStoreConfig) {
    this.config = LanceDBStoreConfigSchema.parse(config);
    this._dimension = this.config.dimension;
    this.name = `lancedb:${this.config.tableName}`;
  }

  get dimension(): number {
    return this._dimension;
  }

  /** FTS 是否可用（未探测时按「可能支持」返回 true，交由 ensureFtsIndex 落定）。 */
  get supportsFts(): boolean {
    return this._supportsFts !== false;
  }

  async init(): Promise<void> {
    // 通过动态字符串规避 TS 静态解析（peer dep 可能未安装）
    const moduleName = "@lancedb/lancedb";
    const mod = (await import(moduleName).catch((e) => {
      throw new Error(
        `LanceDBVectorStore: failed to load @lancedb/lancedb (peer dep). Install it: pnpm add @lancedb/lancedb. Cause: ${(e as Error).message}`,
      );
    })) as unknown as LanceModule;
    this.lanceModule = mod;
    this.db = await mod.connect(this.config.dataDir);
    const names = await this.db.tableNames();
    if (names.includes(this.config.tableName)) {
      this.table = await this.db.openTable(this.config.tableName);
    } else {
      // LanceDB 从 JS 原生 number[] 默认推断 List<Float64>，但向量搜索需要
      // FixedSizeList<Float32, N>。所以这里用 apache-arrow 显式声明 schema。
      const arrowName = "apache-arrow";
      const arrow = (await import(arrowName)) as unknown as ArrowModule;
      const dim = this._dimension;
      const schema = new arrow.Schema([
        new arrow.Field("fragmentId", new arrow.Utf8(), false),
        new arrow.Field("content", new arrow.Utf8(), false),
        new arrow.Field(
          "embedding",
          new arrow.FixedSizeList(dim, new arrow.Field("item", new arrow.Float32(), true)),
          false,
        ),
        new arrow.Field("memoryType", new arrow.Utf8(), false),
        new arrow.Field("importance", new arrow.Float64(), false),
        new arrow.Field(
          "taskTags",
          new arrow.List(new arrow.Field("item", new arrow.Utf8(), true)),
          false,
        ),
        new arrow.Field("contentHash", new arrow.Utf8(), false),
        new arrow.Field("timestamp", new arrow.Float64(), false),
        new arrow.Field("accessCount", new arrow.Int64(), false),
        new arrow.Field("lastAccessed", new arrow.Float64(), false),
        new arrow.Field("metadataJson", new arrow.Utf8(), false),
        new arrow.Field("summariesJson", new arrow.Utf8(), false),
      ]);
      if (typeof this.db.createEmptyTable === "function") {
        this.table = await this.db.createEmptyTable(this.config.tableName, schema);
      } else {
        // 兜底：旧版 LanceDB 没有 createEmptyTable，仍走 seed-row 路径
        const seed: VectorRow = VectorRowSchema.parse({
          fragmentId: "__seed__",
          content: "",
          embedding: new Array(dim).fill(0),
          memoryType: "long_term",
          importance: 0,
          taskTags: ["__seed_tag__"],
          contentHash: "",
          timestamp: 0,
        });
        this.table = await this.db.createTable(this.config.tableName, [seed]);
        await this.table.delete(`"fragmentId" = '__seed__'`);
      }
    }
  }

  async upsert(rows: readonly VectorRow[]): Promise<void> {
    if (!this.table) throw new Error("LanceDBVectorStore not initialized");
    if (rows.length === 0) return;
    const validated = rows.map((r) => VectorRowSchema.parse(r));
    // upsert = delete + insert（LanceDB v0.5+ 支持 mergeInsert，但用 d+i 兼容性更好）
    const ids = validated.map((r) => `'${r.fragmentId.replace(/'/g, "''")}'`).join(",");
    await this.table.delete(`"fragmentId" IN (${ids})`);
    await this.table.add([...validated]);
  }

  async delete(fragmentIds: readonly string[]): Promise<number> {
    if (!this.table) throw new Error("LanceDBVectorStore not initialized");
    if (fragmentIds.length === 0) return 0;
    const ids = fragmentIds.map((id) => `'${id.replace(/'/g, "''")}'`).join(",");
    await this.table.delete(`"fragmentId" IN (${ids})`);
    return fragmentIds.length; // LanceDB delete 不返回 count，乐观估计
  }

  async search(
    queryEmbedding: readonly number[],
    opts: VectorSearchOpts,
  ): Promise<VectorSearchResult[]> {
    if (!this.table) throw new Error("LanceDBVectorStore not initialized");
    const filters: string[] = [];
    if (opts.memoryTypes && opts.memoryTypes.length > 0) {
      const list = opts.memoryTypes.map((m) => `'${m}'`).join(",");
      filters.push(`"memoryType" IN (${list})`);
    }
    if (opts.minImportance !== undefined) {
      filters.push(`importance >= ${opts.minImportance}`);
    }
    const q = this.table.search(queryEmbedding).limit(opts.topK);
    let raw: unknown[];
    if (filters.length > 0) {
      raw = await q.where(filters.join(" AND ")).toArray();
    } else {
      raw = await q.toArray();
    }
    return this.toResults(raw, "vector", opts.taskTags);
  }

  /**
   * 把 LanceDB 返回的原始行解析为 VectorSearchResult。
   * - vector 模式：score = 1 - `_distance`（cosine 距离越小越相关）。
   * - fts 模式：score = `_score`（BM25 相关分越大越相关）。
   */
  private toResults(
    raw: unknown[],
    mode: "vector" | "fts",
    taskTags?: readonly string[],
  ): VectorSearchResult[] {
    const out: VectorSearchResult[] = [];
    const tagSet = taskTags ? new Set(taskTags) : null;
    for (const item of raw) {
      const obj = item as Record<string, unknown> & { _distance?: number; _score?: number };
      try {
        // LanceDB 返回的 embedding 可能是 TypedArray / Vector / List 子结构。
        // taskTags 可能是 Arrow Vector，需要转回普通数组。
        const embedding = normalizeEmbedding(obj["embedding"]);
        const taskTagsArr = normalizeStringArray(obj["taskTags"]);
        const row = VectorRowSchema.parse({
          fragmentId: String(obj["fragmentId"] ?? ""),
          content: String(obj["content"] ?? ""),
          embedding,
          memoryType: obj["memoryType"] as VectorRow["memoryType"],
          importance: Number(obj["importance"] ?? 0),
          taskTags: taskTagsArr,
          contentHash: String(obj["contentHash"] ?? ""),
          timestamp: Number(obj["timestamp"] ?? 0),
          accessCount: Number(obj["accessCount"] ?? 0),
          lastAccessed: Number(obj["lastAccessed"] ?? 0),
          metadataJson: typeof obj["metadataJson"] === "string" ? obj["metadataJson"] : "{}",
          summariesJson: typeof obj["summariesJson"] === "string" ? obj["summariesJson"] : "{}",
        });
        if (tagSet && !row.taskTags.some((t) => tagSet.has(t))) continue;
        if (mode === "fts") {
          const score = typeof obj._score === "number" ? obj._score : 0;
          out.push({ row, distance: 0, score });
        } else {
          const distance = typeof obj._distance === "number" ? obj._distance : 0;
          out.push({ row, distance, score: 1 - distance });
        }
      } catch (e) {
        if (process.env["OPENINTJ_LANCE_DEBUG"] === "1") {
          console.warn(
            "[LanceDBVectorStore] skipped row due to parse error:",
            (e as Error).message,
            "raw keys:",
            Object.keys(obj),
          );
        }
      }
    }
    return out;
  }

  /**
   * 在 `content` 列上确保建立原生 FTS 索引（幂等）。
   * 旧版 LanceDB 无 `createIndex` / `Index.fts` 时静默降级（`supportsFts` 置 false）。
   */
  async ensureFtsIndex(): Promise<void> {
    if (this._supportsFts !== undefined) return;
    if (!this.table) throw new Error("LanceDBVectorStore not initialized");
    const idxFactory = this.lanceModule?.Index;
    if (
      !idxFactory ||
      typeof idxFactory.fts !== "function" ||
      typeof this.table.createIndex !== "function"
    ) {
      this._supportsFts = false;
      return;
    }
    try {
      await this.table.createIndex("content", { config: idxFactory.fts() });
      this._supportsFts = true;
    } catch (e) {
      const msg = (e as Error).message ?? "";
      // 索引已存在视为可用；其它错误（空表 / 版本不支持）降级为纯向量检索。
      if (/exist/i.test(msg)) {
        this._supportsFts = true;
      } else {
        this._supportsFts = false;
        if (process.env["OPENINTJ_LANCE_DEBUG"] === "1") {
          console.warn("[LanceDBVectorStore] FTS index unavailable, degrade to vector-only:", msg);
        }
      }
    }
  }

  async searchText(query: string, opts: VectorSearchOpts): Promise<VectorSearchResult[]> {
    if (!this.table) throw new Error("LanceDBVectorStore not initialized");
    if (this._supportsFts === undefined) await this.ensureFtsIndex();
    if (this._supportsFts === false) return [];
    const filters: string[] = [];
    if (opts.memoryTypes && opts.memoryTypes.length > 0) {
      const list = opts.memoryTypes.map((m) => `'${m}'`).join(",");
      filters.push(`"memoryType" IN (${list})`);
    }
    if (opts.minImportance !== undefined) {
      filters.push(`importance >= ${opts.minImportance}`);
    }
    try {
      const q = this.table.search(query, "fts").limit(opts.topK);
      const raw =
        filters.length > 0 ? await q.where(filters.join(" AND ")).toArray() : await q.toArray();
      return this.toResults(raw, "fts", opts.taskTags);
    } catch (e) {
      // 查询期失败（如索引未就绪）→ 记一次并降级，避免反复抛错。
      this._supportsFts = false;
      if (process.env["OPENINTJ_LANCE_DEBUG"] === "1") {
        console.warn("[LanceDBVectorStore] FTS query failed, degrade:", (e as Error).message);
      }
      return [];
    }
  }

  async scanAll(): Promise<VectorRow[]> {
    if (!this.table) throw new Error("LanceDBVectorStore not initialized");
    // 简化：用 search([0...], topK=count) 来全扫
    const total = await this.count();
    if (total === 0) return [];
    const dummy = new Array(this._dimension).fill(0);
    const res = await this.search(dummy, { topK: total });
    return res.map((r) => r.row);
  }

  async count(): Promise<number> {
    if (!this.table) throw new Error("LanceDBVectorStore not initialized");
    return this.table.countRows();
  }

  async close(): Promise<void> {
    delete this.table;
    delete this.db;
  }
}
