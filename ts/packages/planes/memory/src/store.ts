import {
  type EmbeddingProvider,
  type MemoryFragment,
  MemoryFragmentSchema,
  type MemoryType,
  SimpleEmbedder,
} from "@openintj/core";

export interface MemoryStoreConfig {
  maxShortTerm: number;
  maxWorking: number;
  /**
   * 兼容字段：当未注入 embedder 时使用 SimpleEmbedder(embeddingDim)。
   * 注入了 embedder 时此字段被 embedder.dimension 覆盖。
   */
  embeddingDim: number;
}

export const DEFAULT_MEMORY_STORE_CONFIG: MemoryStoreConfig = {
  maxShortTerm: 50,
  maxWorking: 20,
  embeddingDim: 64,
};

export interface AddOptions {
  importance?: number;
  taskTags?: readonly string[];
  metadata?: Record<string, unknown>;
  /** 用户自带 embedding；不传则用 store.embedder 生成。 */
  embedding?: readonly number[];
  summaries?: Record<number, string>;
}

export interface MemoryStoreOpts {
  /** 嵌入提供方；不提供则用 SimpleEmbedder(config.embeddingDim)。 */
  embedder?: EmbeddingProvider;
}

/**
 * 记忆存储 v1：三层（短期 / 工作 / 长期），溢出时自动迁移。
 * - 短期溢出 → 长期（不丢）
 * - 工作记忆溢出 → 直接丢弃最旧（属于会话级临时数据）
 *
 * 嵌入策略：
 * - 同步 embedder：addShortTerm/addWorking/addLongTerm 直接同步返回 fragment
 * - 异步 embedder：使用 addShortTermAsync/...Async 系列，否则需提前提供 opts.embedding
 *
 * 默认 embedder = SimpleEmbedder（同步），保证 CI 与现有 API 兼容。
 */
export class MemoryStore {
  readonly config: MemoryStoreConfig;
  readonly embedder: EmbeddingProvider;
  readonly shortTerm: MemoryFragment[] = [];
  readonly working: MemoryFragment[] = [];
  readonly longTerm: MemoryFragment[] = [];

  constructor(cfg: Partial<MemoryStoreConfig> = {}, opts: MemoryStoreOpts = {}) {
    this.config = { ...DEFAULT_MEMORY_STORE_CONFIG, ...cfg };
    this.embedder = opts.embedder ?? new SimpleEmbedder(this.config.embeddingDim);
    // dim 对齐
    (this.config as { embeddingDim: number }).embeddingDim = this.embedder.dimension;
  }

  // ---------- 同步 API（要求 embedder 同步或 opts.embedding 提供） ----------

  addShortTerm(content: string, opts: AddOptions = {}): MemoryFragment {
    const fragment = this.makeFragmentSync(content, opts, 0.5, "short_term");
    this.pushShortTerm(fragment);
    return fragment;
  }

  addWorking(content: string, opts: AddOptions = {}): MemoryFragment {
    const fragment = this.makeFragmentSync(content, opts, 0.7, "working");
    this.pushWorking(fragment);
    return fragment;
  }

  addLongTerm(content: string, opts: AddOptions = {}): MemoryFragment {
    const fragment = this.makeFragmentSync(content, opts, 0.5, "long_term");
    this.longTerm.push(fragment);
    return fragment;
  }

  // ---------- 异步 API（任何 embedder 都可用） ----------

  async addShortTermAsync(content: string, opts: AddOptions = {}): Promise<MemoryFragment> {
    const fragment = await this.makeFragmentAsync(content, opts, 0.5, "short_term");
    this.pushShortTerm(fragment);
    return fragment;
  }

  async addWorkingAsync(content: string, opts: AddOptions = {}): Promise<MemoryFragment> {
    const fragment = await this.makeFragmentAsync(content, opts, 0.7, "working");
    this.pushWorking(fragment);
    return fragment;
  }

  async addLongTermAsync(content: string, opts: AddOptions = {}): Promise<MemoryFragment> {
    const fragment = await this.makeFragmentAsync(content, opts, 0.5, "long_term");
    this.longTerm.push(fragment);
    return fragment;
  }

  clearWorking(): void {
    this.working.length = 0;
  }

  remove(fragmentId: string): boolean {
    for (const list of [this.shortTerm, this.working, this.longTerm]) {
      const idx = list.findIndex((f) => f.fragmentId === fragmentId);
      if (idx >= 0) {
        list.splice(idx, 1);
        return true;
      }
    }
    return false;
  }

  get all(): readonly MemoryFragment[] {
    return [...this.shortTerm, ...this.working, ...this.longTerm];
  }

  get totalCount(): number {
    return this.shortTerm.length + this.working.length + this.longTerm.length;
  }

  countsByTier(): Record<MemoryType, number> {
    return {
      short_term: this.shortTerm.length,
      working: this.working.length,
      long_term: this.longTerm.length,
    };
  }

  // ---------- private ----------

  private pushShortTerm(fragment: MemoryFragment): void {
    this.shortTerm.push(fragment);
    while (this.shortTerm.length > this.config.maxShortTerm) {
      const oldest = this.shortTerm.shift();
      if (oldest) {
        oldest.memoryType = "long_term";
        this.longTerm.push(oldest);
      }
    }
  }

  private pushWorking(fragment: MemoryFragment): void {
    this.working.push(fragment);
    while (this.working.length > this.config.maxWorking) {
      this.working.shift();
    }
  }

  private makeFragmentSync(
    content: string,
    opts: AddOptions,
    defaultImportance: number,
    memoryType: MemoryType,
  ): MemoryFragment {
    let embedding: number[];
    if (opts.embedding !== undefined) {
      embedding = [...opts.embedding];
    } else {
      const r = this.embedder.embed(content);
      if (r instanceof Promise) {
        throw new Error(
          `MemoryStore: embedder '${this.embedder.name}' is async; use addShortTermAsync/... instead`,
        );
      }
      embedding = r;
    }
    return MemoryFragmentSchema.parse({
      content,
      embedding,
      importance: opts.importance ?? defaultImportance,
      taskTags: opts.taskTags ? [...opts.taskTags] : [],
      metadata: opts.metadata ?? {},
      summaries: opts.summaries ?? {},
      memoryType,
    });
  }

  private async makeFragmentAsync(
    content: string,
    opts: AddOptions,
    defaultImportance: number,
    memoryType: MemoryType,
  ): Promise<MemoryFragment> {
    const embedding =
      opts.embedding !== undefined
        ? [...opts.embedding]
        : await Promise.resolve(this.embedder.embed(content));
    return MemoryFragmentSchema.parse({
      content,
      embedding,
      importance: opts.importance ?? defaultImportance,
      taskTags: opts.taskTags ? [...opts.taskTags] : [],
      metadata: opts.metadata ?? {},
      summaries: opts.summaries ?? {},
      memoryType,
    });
  }
}
