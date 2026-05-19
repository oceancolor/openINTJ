import {
  type ContextBudget,
  ContextBudgetTracker,
  type HookBus,
  type ShadedFragment,
  type ShaderConfig,
  ShaderConfigSchema,
  ShaderMode,
  type ShaderModeType,
  type Summarizer,
  TASK_SHADER_MAP,
  type TaskTypeType,
} from "@openintj/core";
import type { RankedMemory } from "../retriever.js";
import { fragmentShader } from "./fragment.js";
import { geometryShader } from "./geometry.js";
import { vertexShader } from "./vertex.js";

export interface ShaderPipelineOpts {
  config?: Partial<ShaderConfig>;
  budget?: Partial<ContextBudget>;
  summarize?: Summarizer;
  hooks?: HookBus;
  /** 时间注入（用于测试 + 决定时间衰减）。 */
  clock?: () => number;
}

export interface ShaderRunResult {
  shaded: ShadedFragment[];
  appliedMode: ShaderModeType;
  /** 经过整个管线后保留的 fragment 数（vs 输入数）。 */
  inputCount: number;
  outputCount: number;
  /** 累积 token 消耗（按 estimateTokens）。 */
  totalTokens: number;
}

/**
 * Memory Shader Pipeline —— 检索结果 → V→G→F → 着色后片段
 * 对齐 Python memory_plane.ShaderPipeline.process。
 */
export class ShaderPipeline {
  readonly config: ShaderConfig;
  readonly budgetTracker: ContextBudgetTracker;
  private readonly summarize?: Summarizer;
  private readonly hooks?: HookBus;
  private readonly clock: () => number;

  constructor(opts: ShaderPipelineOpts = {}) {
    this.config = ShaderConfigSchema.parse(opts.config ?? {});
    this.budgetTracker = new ContextBudgetTracker(opts.budget);
    if (opts.summarize !== undefined) this.summarize = opts.summarize;
    if (opts.hooks !== undefined) this.hooks = opts.hooks;
    this.clock = opts.clock ?? (() => Date.now() / 1000);
  }

  /** 决定最终 shader 模式（自适应 → 根据预算占用率挑选具体模式）。 */
  resolveMode(taskType: TaskTypeType): { mode: ShaderModeType; budgetRatio: number } {
    let mode = TASK_SHADER_MAP[taskType] ?? this.config.mode;
    const ratio = this.budgetTracker.usageRatio;
    if (mode === ShaderMode.ADAPTIVE) {
      if (this.budgetTracker.needsCompaction(this.config.compactionThreshold)) {
        mode = ShaderMode.LOW_FIDELITY;
      } else if (ratio < 0.5) {
        mode = ShaderMode.HIGH_FIDELITY;
      } else {
        mode = ShaderMode.HYBRID;
      }
    }
    return { mode, budgetRatio: ratio };
  }

  async process(
    ranked: RankedMemory[],
    taskType: TaskTypeType,
    opts?: { traceId?: string },
  ): Promise<ShaderRunResult> {
    if (ranked.length === 0) {
      return {
        shaded: [],
        appliedMode: this.config.mode,
        inputCount: 0,
        outputCount: 0,
        totalTokens: 0,
      };
    }

    const { mode, budgetRatio } = this.resolveMode(taskType);

    // V — 顶点
    const lodAssigned = vertexShader(ranked, mode, budgetRatio);

    // G — 几何
    const filtered = geometryShader(lodAssigned, {
      config: this.config,
      nowSec: this.clock(),
    });

    // F — 片元
    const fragOpts: Parameters<typeof fragmentShader>[1] = {
      config: this.config,
      shaderMode: mode,
      memoryBudgetTokens: this.budgetTracker.memoryBudget,
      nowSec: this.clock(),
    };
    if (this.summarize !== undefined) fragOpts.summarize = this.summarize;
    const shaded = await fragmentShader(filtered, fragOpts);

    const totalTokens = shaded.reduce((s, f) => s + f.tokens, 0);

    if (this.hooks) {
      const hookOpts = opts?.traceId ? { traceId: opts.traceId } : undefined;
      // 用 lod 的均值作为代表（否则要循环每段发；选第一段简化）
      const repLod = shaded[0]?.lod ?? 0;
      await this.hooks.emit("event.SHADER_APPLIED", { mode, lod: repLod }, hookOpts);
    }

    return {
      shaded,
      appliedMode: mode,
      inputCount: ranked.length,
      outputCount: shaded.length,
      totalTokens,
    };
  }

  /** 通知预算变化（外部调用方法）。 */
  patchBudget(patch: Partial<ContextBudget>): void {
    this.budgetTracker.patch(patch);
  }
}
