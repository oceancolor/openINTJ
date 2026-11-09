import type { HookBus } from "../hooks/bus.js";
import {
  journalThought,
  journalToolCall,
  journalToolResult,
  journalUser,
} from "../journal/sink.js";
import type { JournalTurnContext } from "../journal/types.js";
import type { RunContext } from "../run-context.js";
import { AgentError, ErrorCode, isContextOverflowError } from "../types/errors.js";
import { estimateTokens } from "../types/memory-fragment.js";
import type { ToolCallResult, ToolDescriptor } from "../types/tool.js";
import type {
  ChatMessage,
  LlmClient,
  ReactConfig,
  ReactInput,
  ReactOutput,
  ReactState,
  ReactStatus,
  ReactStopCondition,
  TrajectoryEntry,
} from "./types.js";

/**
 * 工具调用执行器接口（外部注入；通常是 ExecutionPlane.toolHub.call）。
 * 不在 core 引用 plane-execution，避免反向依赖。
 */
export type ReactToolRunner = (
  toolName: string,
  params: Record<string, unknown>,
  opts?: { traceId?: string; timeoutMs?: number; signal?: AbortSignal; runId?: string },
) => Promise<ToolCallResult>;

export interface ReactDeps {
  config: ReactConfig;
  hooks: HookBus;
  llm: LlmClient;
  toolRunner: ReactToolRunner;
}

export interface ReactRunOptions {
  traceId?: string;
  signal?: AbortSignal;
  /** Per-run client override used by conversation-scoped model selection. */
  llm?: LlmClient;
  /** RFC-009 optional journal; missing or throwing sink never fails the loop. */
  journal?: JournalTurnContext;
  /** RFC-021 F02: host-verified run identity, forwarded to hooks/tools. */
  run?: RunContext;
  /**
   * RFC-012 overflow recovery (at most once per run): invoked when the
   * provider rejects the conversation as too long. Receives the history
   * segment (`input.messages` without the trailing current query) and
   * returns a compacted replacement, or undefined to rethrow the original
   * error. Messages appended during this turn are always preserved.
   */
  onContextOverflow?: (
    history: readonly ChatMessage[],
  ) => Promise<readonly ChatMessage[] | undefined>;
}

/**
 * ReAct 输出格式（LLM 必须遵守）：
 *
 *   Thought: <推理过程>
 *   Action: <tool_name>           # 可选；不存在则视为终止
 *   Action-Input: <json string>    # Action 存在时必需
 *
 * 或终止形式：
 *
 *   Thought: <推理过程>
 *   FINAL: <最终答案>
 */
interface ParsedThought {
  thought: string;
  isFinal: boolean;
  finalAnswer?: string;
  action?: { tool: string; params: Record<string, unknown> };
  parseError?: string;
}

const parseLlmThought = (text: string): ParsedThought => {
  const trimmed = text.trim();
  if (!trimmed) {
    return { thought: "", isFinal: false, parseError: "模型返回空内容" };
  }
  // 1) 先看是否有 FINAL
  const finalMatch = trimmed.match(/FINAL\s*:\s*([\s\S]+)$/im);
  const thoughtMatch = trimmed.match(/Thought\s*:\s*([\s\S]+?)(?=\n\s*(?:Action|FINAL)\s*:|$)/i);
  const thought = thoughtMatch?.[1]?.trim() ?? trimmed;

  if (finalMatch) {
    const finalAnswer = finalMatch[1]!.trim();
    if (!finalAnswer) {
      return { thought, isFinal: false, parseError: "模型返回空内容" };
    }
    if (/(?:^|\n)\s*(?:Thought|Action|Action-Input|FINAL)\s*:/i.test(finalAnswer)) {
      return {
        thought,
        isFinal: false,
        parseError: "FINAL 内容混入了内部 ReAct 协议标记",
      };
    }
    return {
      thought,
      isFinal: true,
      finalAnswer,
    };
  }

  const actionMatch = trimmed.match(/Action\s*:\s*([^\n]+)/i);
  const inputMatch = trimmed.match(
    /Action-Input\s*:\s*([\s\S]+?)(?=\n\s*(?:Thought|Action|FINAL)\s*:|$)/i,
  );

  if (actionMatch) {
    const tool = actionMatch[1]!.trim();
    let params: Record<string, unknown> = {};
    if (inputMatch) {
      const raw = inputMatch[1]!.trim();
      try {
        const v = JSON.parse(raw);
        if (typeof v === "object" && v !== null && !Array.isArray(v)) {
          params = v as Record<string, unknown>;
        } else {
          params = { value: v };
        }
      } catch (err) {
        return {
          thought,
          isFinal: false,
          parseError: `Action-Input JSON 解析失败: ${(err as Error).message}`,
        };
      }
    }
    return {
      thought,
      isFinal: false,
      action: { tool, params },
    };
  }

  // 既无 FINAL 也无 Action：当作隐式终止
  return {
    thought,
    isFinal: true,
    finalAnswer: thoughtMatch ? thought : trimmed,
  };
};

const buildSystemPrompt = (baseSystemPrompt: string, tools: ToolDescriptor[]): string => {
  if (tools.length === 0) {
    return baseSystemPrompt;
  }
  const toolDoc = tools
    .map((t) => `- ${t.name}: ${t.description}\n    输入: ${JSON.stringify(t.inputSchema)}`)
    .join("\n");
  return [
    baseSystemPrompt.trim(),
    "",
    "你是一个使用 ReAct 模式的 Agent。每一轮请按以下格式输出：",
    "",
    "Thought: <你的推理>",
    "Action: <工具名>",
    "Action-Input: <严格的 JSON 对象>",
    "",
    "当你已得到答案，请按以下格式输出（不要再调用工具）：",
    "",
    "Thought: <你的最终推理>",
    "FINAL: <最终答案>",
    "",
    "可用工具：",
    toolDoc,
  ].join("\n");
};

const stopConditionKey = (cond: ReactStopCondition): string => cond.kind;

/** 策略拒绝、工具不存在：继续 ReAct，让模型换工具。真正的执行崩溃仍 failFast。 */
export const isRecoverableToolError = (error: string | undefined): boolean =>
  Boolean(error && /未启用|白名单|工具未注册|unknown tool|not in (the )?allowlist/i.test(error));

const checkDuplicate = (trajectory: TrajectoryEntry[], threshold: number): boolean => {
  const actions: Array<{ tool: string; key: string }> = [];
  for (const t of trajectory) {
    if (t.state.type === "action") {
      const key = `${t.state.tool}::${JSON.stringify(t.state.params)}`;
      actions.push({ tool: t.state.tool, key });
    }
  }
  if (actions.length < threshold) return false;
  const last = actions.at(-1)!;
  let count = 0;
  for (const a of actions) if (a.key === last.key) count++;
  return count >= threshold;
};

export class ReactStateMachine {
  protected readonly _config: ReactConfig;
  protected readonly _hooks: HookBus;
  protected readonly _llm: LlmClient;
  protected readonly _toolRunner: ReactToolRunner;

  constructor(deps: ReactDeps) {
    this._config = deps.config;
    this._hooks = deps.hooks;
    this._llm = deps.llm;
    this._toolRunner = deps.toolRunner;
  }

  /**
   * One LLM call with single-shot RFC-012 overflow recovery: on a classified
   * context-overflow error, ask the host callback for a compacted history,
   * splice it over the history segment of `conversation` (system prompt,
   * current query, and this turn's appended steps stay), and retry once.
   * A second overflow — or no replacement — rethrows the original error.
   */
  private async chatWithOverflowRecovery(args: {
    llm: LlmClient;
    conversation: ChatMessage[];
    input: ReactInput;
    opts: ReactRunOptions;
    state: { recovered: boolean };
  }): Promise<string> {
    const { llm, conversation, input, opts, state } = args;
    const chatOpts = {
      temperature: 0.4,
      maxTokens: 1024,
      ...(opts.signal ? { signal: opts.signal } : {}),
    };
    try {
      return await llm.chat(conversation, chatOpts);
    } catch (err) {
      if (state.recovered || !opts.onContextOverflow || !isContextOverflowError(err)) {
        throw err;
      }
      state.recovered = true;
      let replacement: readonly ChatMessage[] | undefined;
      try {
        replacement = await opts.onContextOverflow(input.messages.slice(0, -1));
      } catch {
        replacement = undefined;
      }
      if (!replacement) throw err;
      conversation.splice(1, Math.max(0, input.messages.length - 1), ...replacement);
      return llm.chat(conversation, chatOpts);
    }
  }

  async run(input: ReactInput, opts: ReactRunOptions = {}): Promise<ReactOutput> {
    opts.signal?.throwIfAborted();
    const llm = opts.llm ?? this._llm;
    const trajectory: TrajectoryEntry[] = [];
    const traceId = opts.traceId;
    const conversation: ChatMessage[] = [
      { role: "system", content: buildSystemPrompt(input.systemPrompt, input.availableTools) },
      ...input.messages,
    ];

    let totalTokens = 0;
    let finalAnswer = "";
    let status: ReactStatus = "ok";
    let failedTool: { tool: string; error: string } | undefined;
    let iter = 0;
    const overflowState = { recovered: false };

    while (iter < this._config.maxIterations) {
      opts.signal?.throwIfAborted();
      iter++;

      // ============================================================
      // Thought 阶段
      // ============================================================
      if (this._hooks) {
        const beforeOpts = {
          ...(traceId ? { traceId } : {}),
          ...(opts.run ? { runId: opts.run.runId } : {}),
        };
        const thoughtHook = await this._hooks.emitOutcome(
          "react.beforeThought",
          {
            context: { systemPrompt: input.systemPrompt },
            reactIter: iter,
            taoIter: input.taoIteration,
          },
          beforeOpts,
        );
        if (thoughtHook.cancelled) {
          continue;
        }
        for (const message of thoughtHook.payload.steerMessages ?? []) {
          conversation.push({ role: "user", content: message });
          journalUser(opts.journal, message, { source: "steer" });
        }
      }

      const t0 = Date.now();
      const llmText = await this.chatWithOverflowRecovery({
        llm,
        conversation,
        input,
        opts,
        state: overflowState,
      });
      const parsed = parseLlmThought(llmText);
      totalTokens += estimateTokens(llmText);
      journalThought(opts.journal, iter, parsed.thought);
      const afterOpts = traceId ? { traceId } : undefined;
      await this._hooks.emit(
        "event.ASSISTANT_DELTA",
        {
          traceId: traceId ?? "",
          turnId: opts.journal?.turnId ?? "ephemeral",
          step: iter,
          text: parsed.isFinal ? (parsed.finalAnswer ?? parsed.thought) : parsed.thought,
          channel: parsed.isFinal ? "answer" : "thought",
        },
        afterOpts,
      );

      const thoughtState: ReactState = {
        type: "thought",
        content: parsed.thought,
        iteration: iter,
      };
      trajectory.push({
        timestamp: Date.now() / 1000,
        state: thoughtState,
        durationMs: Date.now() - t0,
      });
      await this._hooks.emit(
        "react.afterThought",
        {
          thought: parsed.thought,
          reactIter: iter,
          taoIter: input.taoIteration,
        },
        afterOpts,
      );

      // 把 LLM 输出放回会话上下文（供下一轮 reflection）
      conversation.push({ role: "assistant", content: llmText });

      // ============================================================
      // 终止判定（explicitFinal）
      // ============================================================
      if (parsed.isFinal) {
        finalAnswer = parsed.finalAnswer ?? parsed.thought;
        trajectory.push({
          timestamp: Date.now() / 1000,
          state: { type: "final", answer: finalAnswer },
          durationMs: 0,
        });
        if (this.hasStop("explicitFinal")) {
          await this._hooks.emit(
            "react.onStopCondition",
            { kind: "explicitFinal", reactIter: iter },
            afterOpts,
          );
        }
        break;
      }

      if (parsed.parseError) {
        // 解析失败：把错误回灌为观察，让 LLM 自我修正
        conversation.push({
          role: "tool",
          name: "parse_error",
          content: `[ParseError] ${parsed.parseError}\n请重新输出：需要工具时使用合法 Action-Input JSON；已能回答时仅输出一次 FINAL，且 FINAL 后只放面向用户的答案。`,
        });
        continue;
      }

      // ============================================================
      // tokenBudgetExceeded
      // ============================================================
      const tokenStop = this._config.stopConditions.find((s) => s.kind === "tokenBudgetExceeded") as
        | Extract<ReactStopCondition, { kind: "tokenBudgetExceeded" }>
        | undefined;
      if (tokenStop && totalTokens > tokenStop.maxTokens) {
        status = "token_overflow";
        finalAnswer = parsed.thought || "[超出 token 预算，已中断]";
        await this._hooks.emit(
          "react.onStopCondition",
          { kind: "tokenBudgetExceeded", reactIter: iter },
          afterOpts,
        );
        break;
      }

      // ============================================================
      // Action 阶段
      // ============================================================
      if (!parsed.action) {
        // 既无 final 又无 action（理论上不会到这里，安全网）
        finalAnswer = parsed.thought;
        break;
      }

      const actionState: ReactState = {
        type: "action",
        tool: parsed.action.tool,
        params: parsed.action.params,
        iteration: iter,
      };
      trajectory.push({
        timestamp: Date.now() / 1000,
        state: actionState,
        durationMs: 0,
      });

      // duplicateToolCall 检查（在执行之前）
      const dupStop = this._config.stopConditions.find((s) => s.kind === "duplicateToolCall") as
        | Extract<ReactStopCondition, { kind: "duplicateToolCall" }>
        | undefined;
      if (dupStop && checkDuplicate(trajectory, dupStop.threshold)) {
        status = "duplicate_loop";
        finalAnswer = `[ReAct 检测到重复工具调用 ${parsed.action.tool}，已中断] ${parsed.thought}`;
        await this._hooks.emit(
          "react.onStopCondition",
          { kind: "duplicateToolCall", reactIter: iter },
          afterOpts,
        );
        break;
      }

      const beforeActionOpts = traceId ? { traceId } : undefined;
      const actionHook = await this._hooks.emitOutcome(
        "react.beforeAction",
        {
          tool: parsed.action.tool,
          params: parsed.action.params,
          reactIter: iter,
          taoIter: input.taoIteration,
        },
        beforeActionOpts,
      );
      if (actionHook.cancelled) {
        conversation.push({
          role: "tool",
          name: parsed.action.tool,
          content: "[cancelled by hook]",
        });
        continue;
      }
      const actionPayload = actionHook.payload;

      const callOpts: {
        traceId?: string;
        timeoutMs: number;
        signal?: AbortSignal;
        runId?: string;
      } = {
        timeoutMs: this._config.stepTimeoutMs,
      };
      if (traceId) callOpts.traceId = traceId;
      if (opts.signal) callOpts.signal = opts.signal;
      if (opts.run) callOpts.runId = opts.run.runId;
      journalToolCall(
        opts.journal,
        iter,
        actionPayload.tool,
        (actionPayload.params as Record<string, unknown>) ?? parsed.action.params,
      );
      const toolResult = await this._toolRunner(
        actionPayload.tool,
        (actionPayload.params as Record<string, unknown>) ?? parsed.action.params,
        callOpts,
      );
      const observation = this.truncateObservation(this.formatObservation(toolResult));
      journalToolResult(
        opts.journal,
        iter,
        toolResult.toolName,
        toolResult.success,
        observation,
        !toolResult.success && isRecoverableToolError(toolResult.error),
      );

      const obsState: ReactState = {
        type: "observation",
        toolResult,
        iteration: iter,
      };
      trajectory.push({
        timestamp: Date.now() / 1000,
        state: obsState,
        durationMs: toolResult.durationMs,
      });

      await this._hooks.emit(
        "react.afterAction",
        {
          toolResult,
          reactIter: iter,
          taoIter: input.taoIteration,
        },
        afterOpts,
      );

      // ============================================================
      // failFast 检查
      // 白名单/未启用/未注册是可恢复观察：模型应改用其它工具（例如 write_file 写 SVG）。
      // ============================================================
      if (
        !toolResult.success &&
        this.hasStop("failFast") &&
        !isRecoverableToolError(toolResult.error)
      ) {
        status = "fail_fast";
        failedTool = {
          tool: toolResult.toolName,
          error: toolResult.error ?? "unknown",
        };
        finalAnswer = `[工具 ${toolResult.toolName} 失败: ${
          toolResult.error ?? "unknown"
        }] ${parsed.thought}`;
        await this._hooks.emit(
          "react.onStopCondition",
          { kind: "failFast", reactIter: iter },
          afterOpts,
        );
        break;
      }

      // 把观察结果回灌给 LLM
      totalTokens += estimateTokens(observation);
      conversation.push({
        role: "tool",
        name: toolResult.toolName,
        toolCallId: toolResult.callId,
        content: observation,
      });
    }

    if (iter >= this._config.maxIterations && !finalAnswer) {
      status = "max_iter";
      const bestThought = trajectory
        .filter(
          (entry): entry is TrajectoryEntry & { state: Extract<ReactState, { type: "thought" }> } =>
            entry.state.type === "thought",
        )
        .map((entry) => entry.state.content.trim())
        .filter(Boolean)
        .sort((a, b) => b.length - a.length)[0];
      finalAnswer = bestThought ?? "[达到最大 ReAct 迭代次数，未能收敛]";
    }

    const output: ReactOutput = {
      finalAnswer,
      trajectory,
      iterations: iter,
      status,
      totalTokensSpent: totalTokens,
    };
    if (failedTool) {
      output.failedTool = failedTool;
    }
    return output;
  }

  /**
   * 退化路径（`TaoConfig.enableReact=false`）：跳过 ReAct 微循环，做**单次 LLM 调用**直接作答。
   * 不下发工具说明、不解析 Thought/Action 协议——用于"纯对话 / 快速响应"等无需工具的场景，
   * 省 token、省时延。仍发出 react.beforeThought / afterThought / onStopCondition 以保持可观测一致。
   */
  async runSingle(input: ReactInput, opts: ReactRunOptions = {}): Promise<ReactOutput> {
    opts.signal?.throwIfAborted();
    const llm = opts.llm ?? this._llm;
    const traceId = opts.traceId;
    const hookOpts = {
      ...(traceId ? { traceId } : {}),
      ...(opts.run ? { runId: opts.run.runId } : {}),
    };

    const thoughtHook = await this._hooks.emitOutcome(
      "react.beforeThought",
      { context: { systemPrompt: input.systemPrompt }, reactIter: 1, taoIter: input.taoIteration },
      hookOpts,
    );

    const steerMessages = thoughtHook.payload.steerMessages ?? [];
    for (const content of steerMessages) {
      journalUser(opts.journal, content, { source: "steer" });
    }
    const conversation: ChatMessage[] = [
      { role: "system", content: input.systemPrompt },
      ...input.messages,
      ...steerMessages.map((content) => ({
        role: "user" as const,
        content,
      })),
    ];
    const t0 = Date.now();
    const llmText = await this.chatWithOverflowRecovery({
      llm,
      conversation,
      input,
      opts,
      state: { recovered: false },
    });
    opts.signal?.throwIfAborted();
    const parsed = parseLlmThought(llmText);
    journalThought(opts.journal, 1, parsed.thought);
    const answer = (parsed.finalAnswer ?? parsed.thought).trim();
    await this._hooks.emit(
      "event.ASSISTANT_DELTA",
      {
        traceId: traceId ?? "",
        turnId: opts.journal?.turnId ?? "ephemeral",
        step: 1,
        text: answer,
        channel: "answer",
      },
      hookOpts,
    );
    if (!answer) {
      throw new AgentError({
        code: ErrorCode.INTERNAL_ERROR,
        message: "模型返回空内容",
        retriable: true,
      });
    }

    await this._hooks.emit(
      "react.afterThought",
      { thought: answer, reactIter: 1, taoIter: input.taoIteration },
      hookOpts,
    );

    const trajectory: TrajectoryEntry[] = [
      {
        timestamp: Date.now() / 1000,
        state: { type: "final", answer },
        durationMs: Date.now() - t0,
      },
    ];

    await this._hooks.emit(
      "react.onStopCondition",
      { kind: "explicitFinal", reactIter: 1 },
      hookOpts,
    );

    return {
      finalAnswer: answer,
      trajectory,
      iterations: 1,
      status: "ok",
      totalTokensSpent: estimateTokens(llmText),
    };
  }

  private hasStop(kind: ReactStopCondition["kind"]): boolean {
    return this._config.stopConditions.some((s) => stopConditionKey(s) === kind);
  }

  private formatObservation(r: ToolCallResult): string {
    if (r.success) {
      const out = typeof r.output === "string" ? r.output : JSON.stringify(r.output, null, 2);
      return `[Observation] ${r.toolName} 成功:\n${out}`;
    }
    return `[Observation] ${r.toolName} 失败: ${r.error ?? "unknown"}`;
  }

  private truncateObservation(text: string): string {
    if (text.length <= this._config.observationMaxChars) return text;
    return `${text.slice(0, this._config.observationMaxChars)}\n…[已截断 ${
      text.length - this._config.observationMaxChars
    } 字符]`;
  }
}

// 导出 helper 供测试使用（不在 public API 中保证稳定性）
export const __test__ = {
  parseLlmThought,
  buildSystemPrompt,
  checkDuplicate,
  isRecoverableToolError,
};

// 占位错误码引用（避免未使用警告）
void AgentError;
void ErrorCode;
