import type { EmbeddingProvider, HookBus, MemoryFragment, ShaderConfig } from "@openintj/core";
import { MemoryRetriever, type RankedMemory, type RetrieveOptions } from "./retriever.js";
import { MemoryStore, type MemoryStoreConfig } from "./store.js";

export interface MemoryPlaneOpts {
  storeConfig?: Partial<MemoryStoreConfig>;
  shaderConfig?: Partial<ShaderConfig>;
  hooks?: HookBus;
  clock?: () => number;
  embedder?: EmbeddingProvider;
}

export class MemoryPlane {
  readonly name = "memory-plane";
  readonly store: MemoryStore;
  readonly retriever: MemoryRetriever;
  private readonly hooks?: HookBus;

  constructor(opts: MemoryPlaneOpts = {}) {
    const storeOpts = {
      ...(opts.embedder ? { embedder: opts.embedder } : {}),
      ...(opts.hooks ? { hooks: opts.hooks } : {}),
    };
    this.store = new MemoryStore(opts.storeConfig, storeOpts);
    const retrieverOpts: { clock?: () => number; embedder?: EmbeddingProvider } = {};
    if (opts.clock !== undefined) retrieverOpts.clock = opts.clock;
    if (opts.embedder !== undefined) retrieverOpts.embedder = opts.embedder;
    this.retriever = new MemoryRetriever(this.store, opts.shaderConfig, retrieverOpts);
    if (opts.hooks !== undefined) this.hooks = opts.hooks;
  }

  async retrieve(
    query: string,
    opts: RetrieveOptions = {},
    emitOpts?: { traceId?: string; budgetUsage?: number },
  ): Promise<RankedMemory[]> {
    const ranked = await this.retriever.retrieveAsync(query, opts);
    if (this.hooks) {
      const hookOpts = emitOpts?.traceId ? { traceId: emitOpts.traceId } : undefined;
      await this.hooks.emit(
        "event.MEMORY_LOADED",
        {
          count: ranked.length,
          budgetUsage: emitOpts?.budgetUsage ?? 0,
        },
        hookOpts,
      );
    }
    return ranked;
  }

  /**
   * 便捷：直接添加用户输入到短期记忆。
   * extraTags 用于带上分类器 label（与 retriever 的 taskType ×1.3 加成叠加，随使用复利）。
   */
  recordUserInput(content: string, extraTags: readonly string[] = []): MemoryFragment {
    return this.store.addShortTerm(content, {
      taskTags: ["user_input", ...extraTags],
      importance: 0.6,
    });
  }

  recordAssistantOutput(content: string, extraTags: readonly string[] = []): MemoryFragment {
    return this.store.addShortTerm(content, {
      taskTags: ["assistant_output", ...extraTags],
      importance: 0.5,
    });
  }

  getStats(): {
    counts: ReturnType<MemoryStore["countsByTier"]>;
    total: number;
  } {
    return {
      counts: this.store.countsByTier(),
      total: this.store.totalCount,
    };
  }
}
