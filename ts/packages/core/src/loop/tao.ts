import { randomUUID } from "node:crypto";
import type { HookBus } from "../hooks/bus.js";
import {
  type ShaderModeType,
  TASK_SHADER_MAP,
  TaskType,
  type TaskTypeType,
} from "../types/shader.js";
import type { ToolDescriptor } from "../types/tool.js";
import type { ReactStateMachine } from "./react.js";
import type {
  ChatMessage,
  ImagePayload,
  TaoConfig,
  TaoContext,
  TaoResult,
  TaoStatus,
  TrajectoryEntry,
} from "./types.js";

export interface TaoMessageBuilder {
  /** 给定本轮 query 和已发生的 trajectory，构造下一轮 ReAct 输入消息。 */
  build(input: {
    query: string;
    trajectory: TrajectoryEntry[];
    iteration: number;
    history: ChatMessage[];
  }): { messages: ChatMessage[]; systemPrompt: string };
}

/** 每轮动态构造「基础 system prompt」的入参（可据此检索并注入记忆）。 */
export interface TaoContextInput {
  query: string;
  iteration: number;
  history: ChatMessage[];
  trajectory: TrajectoryEntry[];
  taskType: TaskTypeType;
  traceId: string;
}

/**
 * 可选：每轮调用，返回注入了记忆/上下文的「基础 system prompt」。
 * 提供后**取代**静态 `systemPrompt`（仍会再叠加 builder 的轮次提示与工具说明）。
 * 支持异步，便于接 ContextEngine 这类需要向量检索的实现。
 */
export type TaoContextProvider = (input: TaoContextInput) => Promise<string> | string;

export interface TaoDeps {
  config: TaoConfig;
  hooks: HookBus;
  react: ReactStateMachine;
  /** 默认按 TaskType → ShaderMode 选择；可注入定制策略。 */
  shaderSelector?: (task: TaskTypeType) => ShaderModeType;
  /** 决定下一轮是否继续（多轮 Tao 时使用）。 */
  needsContinue?: (ctx: TaoContext) => boolean;
  /** 提供本次任务可用工具集（来自 ExecutionPlane.toolHub.list）。 */
  availableTools: () => ToolDescriptor[];
  /** 把对话历史 + 用户 query 组装成 ReAct 输入。默认实现见 defaultBuilder。 */
  messageBuilder?: TaoMessageBuilder;
  /** 可选：从 query 推断 TaskType。默认使用 detectTaskType 启发式。 */
  taskClassifier?: (query: string) => TaskTypeType;
  /** 用户系统提示。 */
  systemPrompt?: string;
  /**
   * 可选：每轮动态构造基础 system prompt（用于注入检索到的记忆）。
   * 提供后取代静态 `systemPrompt`；不提供则回退到 `systemPrompt`。
   */
  contextProvider?: TaoContextProvider;
}

const defaultBuilder: TaoMessageBuilder = {
  build({ query, trajectory, iteration, history }) {
    const systemPrompt = [
      "你是 OpenINTJ Agent。",
      iteration > 1
        ? `这是 TAO 宏循环的第 ${iteration} 轮，请基于上一轮的轨迹进一步细化。`
        : "请理解用户意图并按需调用工具。",
    ].join(" ");
    const trajSummary = trajectory.length === 0 ? "" : `\n[上一轮关键步骤数: ${trajectory.length}]`;
    return {
      systemPrompt,
      messages: [
        ...history,
        {
          role: "user",
          content: `${query}${trajSummary}`,
        },
      ],
    };
  },
};

/**
 * 关键词启发式任务分类（零 token、本地）。TaoLoop 默认分类器，
 * 也作为 @openintj/classifier 的冷启动/低置信兜底。
 */
export const detectTaskType = (query: string): TaskTypeType => {
  const t = query.toLowerCase();
  if (
    t.includes("代码") ||
    t.includes("function") ||
    t.includes("class") ||
    t.includes("写一个") ||
    t.includes("实现") ||
    t.includes("修复 bug") ||
    t.includes("bug")
  ) {
    return TaskType.CODE_GENERATION;
  }
  if (
    t.includes("文档") ||
    t.includes("readme") ||
    t.includes("教程") ||
    t.includes("写一份") ||
    t.includes("写作")
  ) {
    return TaskType.TECHNICAL_WRITING;
  }
  if (t.includes("分析") || t.includes("评估") || t.includes("对比")) {
    return TaskType.ANALYSIS;
  }
  if (t.includes("规划") || t.includes("方案") || t.includes("计划")) {
    return TaskType.PLANNING;
  }
  if (query.length < 30) return TaskType.QUICK_RESPONSE;
  return TaskType.GENERAL_CHAT;
};

const defaultNeedsContinue = (ctx: TaoContext, maxIter: number): boolean => {
  if (ctx.iteration >= maxIter) return false;
  const lastTraj = ctx.trajectory.at(-1);
  // 当 ReAct 最后一步是 final，且 finalAnswer 已设置 → 终止
  if (lastTraj?.state.type === "final") return false;
  if (ctx.finalAnswer && ctx.finalAnswer.trim().length > 0) return false;
  return true;
};

export class TaoLoop {
  protected readonly _config: TaoConfig;
  protected readonly _hooks: HookBus;
  protected readonly _react: ReactStateMachine;
  private readonly shaderSelector: (task: TaskTypeType) => ShaderModeType;
  private readonly needsContinueFn: (ctx: TaoContext) => boolean;
  private readonly availableTools: () => ToolDescriptor[];
  private readonly builder: TaoMessageBuilder;
  private readonly classifier: (query: string) => TaskTypeType;
  private readonly systemPrompt: string;
  private readonly contextProvider: TaoContextProvider | undefined;

  constructor(deps: TaoDeps) {
    this._config = deps.config;
    this._hooks = deps.hooks;
    this._react = deps.react;
    this.shaderSelector = deps.shaderSelector ?? ((t: TaskTypeType) => TASK_SHADER_MAP[t]);
    this.needsContinueFn =
      deps.needsContinue ??
      ((ctx: TaoContext) => defaultNeedsContinue(ctx, this._config.maxTaoIterations));
    this.availableTools = deps.availableTools;
    this.builder = deps.messageBuilder ?? defaultBuilder;
    this.classifier = deps.taskClassifier ?? detectTaskType;
    this.systemPrompt = deps.systemPrompt ?? "";
    this.contextProvider = deps.contextProvider;
  }

  async run(
    query: string,
    opts: {
      imageData?: ImagePayload;
      traceId?: string;
      /** 外部预分类结果：提供后跳过内部 classifier（前端分类器接入点）。 */
      taskType?: TaskTypeType;
      /** 按本次 run 覆盖 enableReact：false 走单次 LLM 调用（降 token），不改全局配置。 */
      enableReact?: boolean;
    } = {},
  ): Promise<TaoResult> {
    const traceId = opts.traceId ?? randomUUID();
    const startTime = Date.now();
    const taskType = opts.taskType ?? this.classifier(query);
    const shaderMode = this.shaderSelector(taskType);
    const enableReact = opts.enableReact ?? this._config.enableReact;

    const ctx: TaoContext = {
      traceId,
      query,
      ...(opts.imageData ? { imageData: opts.imageData } : {}),
      iteration: 0,
      trajectory: [],
      metrics: {},
    };

    const history: ChatMessage[] = [];
    let totalReactSteps = 0;
    let totalTokensSpent = 0;
    let status: TaoStatus = "completed";
    let failureReason: string | undefined;

    // ============================================================
    // 主循环：每一轮一次 Think → Act → Observe
    // ============================================================
    const runStart = Date.now();
    while (true) {
      ctx.iteration++;
      if (ctx.iteration > this._config.maxTaoIterations) {
        ctx.iteration = this._config.maxTaoIterations;
        status = "max_iter_reached";
        break;
      }

      const taoOpts = traceId ? { traceId } : undefined;

      // ----- Think -----
      await this._hooks.emit(
        "tao.beforeThink",
        { query: ctx.query, iteration: ctx.iteration },
        taoOpts,
      );

      const built = this.builder.build({
        query: ctx.query,
        trajectory: ctx.trajectory,
        iteration: ctx.iteration,
        history,
      });

      // 模拟"plan"占位（v1 由 ReAct 负责真正规划）
      const planLike = {
        planId: randomUUID(),
        totalSteps: 0,
      };
      await this._hooks.emit(
        "tao.afterThink",
        { plan: planLike, iteration: ctx.iteration },
        taoOpts,
      );

      // ----- Act -----
      const tools = this.availableTools();
      await this._hooks.emit(
        "tao.beforeAct",
        {
          plan: planLike,
          availableTools: tools,
          iteration: ctx.iteration,
        },
        taoOpts,
      );

      // 基础 system prompt：优先用 contextProvider（可注入检索到的记忆），否则用静态 systemPrompt。
      let baseSystem = this.systemPrompt;
      if (this.contextProvider) {
        try {
          baseSystem = await this.contextProvider({
            query: ctx.query,
            iteration: ctx.iteration,
            history,
            trajectory: ctx.trajectory,
            taskType,
            traceId,
          });
        } catch {
          // 记忆注入失败绝不阻断主循环，回退静态 systemPrompt。
          baseSystem = this.systemPrompt;
        }
      }
      const finalSystemPrompt =
        baseSystem.length > 0 ? `${baseSystem}\n\n${built.systemPrompt}` : built.systemPrompt;

      const reactInput = {
        messages: built.messages,
        availableTools: tools,
        taoIteration: ctx.iteration,
        systemPrompt: finalSystemPrompt,
      };

      const reactOpts = traceId ? { traceId } : undefined;
      // enableReact=false 退化为单次 LLM 调用（不跑微循环、不下发工具）。
      const react = enableReact
        ? await this._react.run(reactInput, reactOpts)
        : await this._react.runSingle(reactInput, reactOpts);

      ctx.trajectory.push(...react.trajectory);
      totalReactSteps += react.iterations;
      totalTokensSpent += react.totalTokensSpent;

      await this._hooks.emit(
        "tao.afterAct",
        {
          reactOutput: {
            finalAnswer: react.finalAnswer,
            status: react.status,
            iterations: react.iterations,
          },
          iteration: ctx.iteration,
        },
        taoOpts,
      );

      // ----- Observe -----
      await this._hooks.emit(
        "tao.beforeObserve",
        {
          trajectory: ctx.trajectory.map((t) => ({ timestamp: t.timestamp })),
          iteration: ctx.iteration,
        },
        taoOpts,
      );

      // ReAct 失败模式直接传播；正常模式以 reactOutput.finalAnswer 作为本轮答案
      if (react.status === "fail_fast") {
        status = "failed";
        failureReason = react.failedTool
          ? `tool ${react.failedTool.tool} failed: ${react.failedTool.error}`
          : "react fail_fast";
        ctx.finalAnswer = react.finalAnswer;
      } else {
        ctx.finalAnswer = react.finalAnswer;
      }

      // 把本轮 user 与 assistant 入会话历史，供下一轮使用
      history.push({ role: "user", content: query });
      history.push({ role: "assistant", content: react.finalAnswer });

      const continues = this.needsContinueFn(ctx);
      await this._hooks.emit(
        "tao.afterObserve",
        { needsContinue: continues, iteration: ctx.iteration },
        taoOpts,
      );

      if (status === "failed") break;
      if (!continues) break;

      // 超时检查
      if (Date.now() - runStart > this._config.taoTimeoutMs) {
        status = "timeout";
        failureReason = `tao timeout after ${this._config.taoTimeoutMs}ms`;
        break;
      }
    }

    ctx.metrics["totalDurationMs"] = Date.now() - startTime;
    ctx.metrics["totalReactSteps"] = totalReactSteps;
    ctx.metrics["totalTokensSpent"] = totalTokensSpent;

    await this._hooks.emit(
      "event.LOOP_ITERATION",
      { taoIter: ctx.iteration, metrics: ctx.metrics },
      traceId ? { traceId } : undefined,
    );

    const result: TaoResult = {
      traceId,
      status,
      finalAnswer: ctx.finalAnswer ?? "",
      iterations: ctx.iteration,
      reactTotalSteps: totalReactSteps,
      totalTokensSpent,
      durationMs: Date.now() - startTime,
      trajectory: ctx.trajectory,
      taskType,
      shaderMode,
      metrics: ctx.metrics,
    };
    if (failureReason) result.failureReason = failureReason;
    return result;
  }
}

export const __taoTest__ = { defaultBuilder, detectTaskType, defaultNeedsContinue };
