import {
  type ChatMessage,
  type ContextBudget,
  type ContextWindowSnapshot,
  type EmbeddingProvider,
  type HookBus,
  type ShadedFragment,
  type ShaderConfig,
  type Summarizer,
  type TaskTypeType,
  estimateTokens,
} from "@openintj/core";
import { MemoryRetriever, type RankedMemory } from "./retriever.js";
import { ShaderPipeline } from "./shader/pipeline.js";
import type { MemoryStore } from "./store.js";

export interface ContextEngineOpts {
  store: MemoryStore;
  shaderConfig?: Partial<ShaderConfig>;
  budget?: Partial<ContextBudget>;
  summarize?: Summarizer;
  hooks?: HookBus;
  clock?: () => number;
  /** 注入 retriever（默认基于 store + shaderConfig 自建）。 */
  retriever?: MemoryRetriever;
  /** 注入 embedder（默认用 store.embedder）。 */
  embedder?: EmbeddingProvider;
}

export interface BuildContextInput {
  query: string;
  history: ChatMessage[];
  taskType: TaskTypeType;
  systemPrompt: string;
  /** 由调用方提供的检索 topK；不提供则用 shaderConfig.maxFragmentsPerQuery。 */
  topK?: number;
  traceId?: string;
}

/**
 * ContextEngine —— 上下文窗口构建器
 *
 * 流程：
 *  1) 用 MemoryRetriever 拉相关记忆 (RankedMemory[])
 *  2) ShaderPipeline.process(...) → ShadedFragment[]
 *  3) 注入到 system prompt 后 + history 拼接成最终 messages
 *  4) 检查总 token 是否超 budget；若超则触发 emit('event.CONTEXT_COMPACTED')
 */
export class ContextEngine {
  readonly store: MemoryStore;
  readonly retriever: MemoryRetriever;
  readonly pipeline: ShaderPipeline;
  private readonly hooks?: HookBus;

  constructor(opts: ContextEngineOpts) {
    this.store = opts.store;
    const retrieverOpts: { clock?: () => number; embedder?: EmbeddingProvider } = {};
    if (opts.clock !== undefined) retrieverOpts.clock = opts.clock;
    if (opts.embedder !== undefined) retrieverOpts.embedder = opts.embedder;
    this.retriever =
      opts.retriever ?? new MemoryRetriever(opts.store, opts.shaderConfig, retrieverOpts);
    const pipeOpts: ConstructorParameters<typeof ShaderPipeline>[0] = {};
    if (opts.shaderConfig !== undefined) pipeOpts.config = opts.shaderConfig;
    if (opts.budget !== undefined) pipeOpts.budget = opts.budget;
    if (opts.summarize !== undefined) pipeOpts.summarize = opts.summarize;
    if (opts.hooks !== undefined) pipeOpts.hooks = opts.hooks;
    if (opts.clock !== undefined) pipeOpts.clock = opts.clock;
    this.pipeline = new ShaderPipeline(pipeOpts);
    if (opts.hooks !== undefined) this.hooks = opts.hooks;
  }

  async build(input: BuildContextInput): Promise<ContextWindowSnapshot> {
    // 1) 检索（自动适配 sync / async embedder）
    const retrieveOpts: { topK?: number; taskType?: TaskTypeType } = {
      taskType: input.taskType,
    };
    if (input.topK !== undefined) retrieveOpts.topK = input.topK;
    const ranked: RankedMemory[] = await this.retriever.retrieveAsync(input.query, retrieveOpts);

    // 2) 着色
    const shaderOpts = input.traceId ? { traceId: input.traceId } : undefined;
    const shadeResult = await this.pipeline.process(ranked, input.taskType, shaderOpts);

    // 3) 拼最终 messages：systemPrompt 内嵌 memory，history + 当前 user query
    const memorySection = this.formatMemorySection(shadeResult.shaded);
    const finalSystem = memorySection
      ? `${input.systemPrompt.trim()}\n\n[记忆参考]\n${memorySection}`
      : input.systemPrompt;

    const messages: ChatMessage[] = [...input.history, { role: "user", content: input.query }];

    // 4) 计算 token 总量
    const memoryTokens = shadeResult.totalTokens;
    const inferredConvTokens =
      estimateTokens(input.systemPrompt) +
      messages.reduce(
        (s, m) => s + estimateTokens(typeof m.content === "string" ? m.content : ""),
        0,
      );
    // 只 patch memoryTokens；conversationTokens 由 caller 通过 patchBudget 管理
    // （我们再加上本次推断的对话量到累积值，便于自动估算）
    const prevSnapshot = this.pipeline.budgetTracker.snapshot;
    this.pipeline.patchBudget({
      memoryTokens,
      conversationTokens: prevSnapshot.conversationTokens + inferredConvTokens,
    });

    const budget = this.pipeline.budgetTracker.snapshot;
    const totalTokens = memoryTokens + budget.conversationTokens;

    // 5) 自动压缩通知（不主动重新跑管线，由 caller 决定下一步；只发事件）
    if (this.hooks && this.pipeline.budgetTracker.needsCompaction(0.8)) {
      const hookOpts = input.traceId ? { traceId: input.traceId } : undefined;
      await this.hooks.emit(
        "event.CONTEXT_COMPACTED",
        {
          compactedMessages: shadeResult.outputCount,
          newBudgetUsage: this.pipeline.budgetTracker.usageRatio,
        },
        hookOpts,
      );
    }

    return {
      systemPrompt: finalSystem,
      messages,
      memoryFragments: shadeResult.shaded,
      totalTokens,
      budget: {
        maxTokens: budget.maxTokens,
        used: totalTokens,
        available: this.pipeline.budgetTracker.availableTokens,
      },
    };
  }

  private formatMemorySection(shaded: ShadedFragment[]): string {
    if (shaded.length === 0) return "";
    return shaded
      .map(
        (s, i) =>
          `[#${i + 1} score=${s.score} lod=${s.lod} importance=${s.importance}]\n${s.content}`,
      )
      .join("\n\n");
  }

  /** 让 caller 通知系统提示 token 占用变更（影响 memoryBudget 计算）。 */
  patchBudget(patch: Partial<ContextBudget>): void {
    this.pipeline.patchBudget(patch);
  }

  /** 获取当前 budget 状态。 */
  get budget(): ContextBudget {
    return this.pipeline.budgetTracker.snapshot;
  }
}

void (null as unknown as ContextWindowSnapshot);
