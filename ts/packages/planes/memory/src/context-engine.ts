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
import { type MemoryScope, isFragmentVisibleToScope } from "./scope.js";
import { isCurrentlyValid } from "./validity.js";
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
  /**
   * 可选候选召回器（A1.3 opt-in）：提供后 build() 用它替代默认 MemoryRetriever 做召回，
   * 返回 RankedMemory[] 后仍走原 ShaderPipeline。用于把 session 级 HybridRetriever 接进主循环。
   * 默认 undefined → 行为零变化。
   */
  candidateRetrieve?: (
    query: string,
    opts: { topK?: number; taskType?: TaskTypeType; scope?: MemoryScope },
  ) => Promise<RankedMemory[]>;
}

export interface BuildContextInput {
  query: string;
  history: ChatMessage[];
  taskType: TaskTypeType;
  systemPrompt: string;
  /** 由调用方提供的检索 topK；不提供则用 shaderConfig.maxFragmentsPerQuery。 */
  topK?: number;
  traceId?: string;
  /** RFC-021 F04: verified read scope; invisible memories never enter the prompt. */
  scope?: MemoryScope;
  /**
   * F10：本次注入的使用事件 id（run/turn 级）。同一 id 重复 build 不重复记使用；
   * 不传则不记使用（retrieval 本身永远不算使用）。
   */
  usageEventId?: string;
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
  private readonly candidateRetrieve?: ContextEngineOpts["candidateRetrieve"];
  /** F10：已记过使用的 usageEventId 集合（同事件重复 build 幂等）。 */
  private readonly recordedUsageEvents = new Set<string>();

  constructor(opts: ContextEngineOpts) {
    this.store = opts.store;
    const retrieverOpts: { clock?: () => number; embedder?: EmbeddingProvider } = {};
    if (opts.clock !== undefined) retrieverOpts.clock = opts.clock;
    if (opts.embedder !== undefined) retrieverOpts.embedder = opts.embedder;
    this.retriever =
      opts.retriever ?? new MemoryRetriever(opts.store, opts.shaderConfig, retrieverOpts);
    if (opts.candidateRetrieve !== undefined) this.candidateRetrieve = opts.candidateRetrieve;
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
    // 1) 检索（自动适配 sync / async embedder）。
    //    注入了 candidateRetrieve（A1.3 opt-in）则走它，否则用默认 MemoryRetriever。
    const retrieveOpts: { topK?: number; taskType?: TaskTypeType; scope?: MemoryScope } = {
      taskType: input.taskType,
    };
    if (input.topK !== undefined) retrieveOpts.topK = input.topK;
    if (input.scope !== undefined) retrieveOpts.scope = input.scope;
    const unfiltered: RankedMemory[] = this.candidateRetrieve
      ? await this.candidateRetrieve(input.query, retrieveOpts)
      : await this.retriever.retrieveAsync(input.query, retrieveOpts);
    // The default retriever filters before topK; an injected candidate
    // retriever only gets post-filtering here (defense in depth, and the
    // reason hosts should prefer pre-filtered candidate sources).
    // F08: validity (expired/superseded) is filtered at every entry, same as scope.
    const ranked: RankedMemory[] = unfiltered.filter(
      (r) =>
        isCurrentlyValid(r.fragment) &&
        (input.scope === undefined || isFragmentVisibleToScope(r.fragment, input.scope)),
    );

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
    // 只 patch memoryTokens；conversationTokens 按 F09 改为「当前完整消息集」
    // 的绝对量，不再逐轮累加同一历史（重复累加会虚报预算压力）。取与既有
    // 基线（宿主预置/上一轮完整集）的较大值，避免覆盖宿主给的占用基线。
    const prevSnapshot = this.pipeline.budgetTracker.snapshot;
    this.pipeline.patchBudget({
      memoryTokens,
      conversationTokens: Math.max(prevSnapshot.conversationTokens, inferredConvTokens),
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

    // F10：真正进入最终 prompt 的片段才记使用；同 usageEventId 幂等。
    if (input.usageEventId && !this.recordedUsageEvents.has(input.usageEventId)) {
      this.recordedUsageEvents.add(input.usageEventId);
      const injectedIds = shadeResult.shaded.map((s) => s.fragmentId);
      if (injectedIds.length > 0) {
        this.store.recordUsage(injectedIds);
        if (this.hooks) {
          await this.hooks.emit("event.MEMORY_INJECTED", {
            fragmentIds: injectedIds,
            usageEventId: input.usageEventId,
          });
        }
      }
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

  /** 上下文窗口占用：对话 + 记忆 + 系统提示，不含 reserved。 */
  usage(): { used: number; max: number; ratio: number } {
    const snap = this.budget;
    const used =
      snap.systemPromptTokens + snap.conversationTokens + snap.memoryTokens + snap.toolTokens;
    return {
      used,
      max: snap.maxTokens,
      ratio: this.pipeline.budgetTracker.usageRatio,
    };
  }
}

void (null as unknown as ContextWindowSnapshot);
