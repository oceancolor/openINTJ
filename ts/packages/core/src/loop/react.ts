import type { HookBus } from "../hooks/bus.js";
import { AgentError, ErrorCode } from "../types/errors.js";
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
  opts?: { traceId?: string; timeoutMs?: number },
) => Promise<ToolCallResult>;

export interface ReactDeps {
  config: ReactConfig;
  hooks: HookBus;
  llm: LlmClient;
  toolRunner: ReactToolRunner;
}

export interface ReactRunOptions {
  traceId?: string;
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
  // 1) 先看是否有 FINAL
  const finalMatch = trimmed.match(/FINAL\s*:\s*([\s\S]+)$/m);
  const thoughtMatch = trimmed.match(/Thought\s*:\s*([\s\S]+?)(?=\n\s*(?:Action|FINAL)\s*:|$)/);
  const thought = thoughtMatch?.[1]?.trim() ?? trimmed;

  if (finalMatch) {
    return {
      thought,
      isFinal: true,
      finalAnswer: finalMatch[1]!.trim(),
    };
  }

  const actionMatch = trimmed.match(/Action\s*:\s*([^\n]+)/);
  const inputMatch = trimmed.match(
    /Action-Input\s*:\s*([\s\S]+?)(?=\n\s*(?:Thought|Action|FINAL)\s*:|$)/,
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
    finalAnswer: trimmed,
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

  async run(input: ReactInput, opts: ReactRunOptions = {}): Promise<ReactOutput> {
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

    while (iter < this._config.maxIterations) {
      iter++;

      // ============================================================
      // Thought 阶段
      // ============================================================
      if (this._hooks) {
        const beforeOpts = traceId ? { traceId } : undefined;
        await this._hooks.emit(
          "react.beforeThought",
          {
            context: { systemPrompt: input.systemPrompt },
            reactIter: iter,
            taoIter: input.taoIteration,
          },
          beforeOpts,
        );
      }

      const t0 = Date.now();
      const llmText = await this._llm.chat(conversation, {
        temperature: 0.4,
        maxTokens: 1024,
      });
      const parsed = parseLlmThought(llmText);
      totalTokens += estimateTokens(llmText);

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

      const afterOpts = traceId ? { traceId } : undefined;
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
          content: `[ParseError] ${parsed.parseError}\n请严格按 ReAct 格式输出 Action-Input 为合法 JSON。`,
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
      const actionPayload = await this._hooks.emit(
        "react.beforeAction",
        {
          tool: parsed.action.tool,
          params: parsed.action.params,
          reactIter: iter,
          taoIter: input.taoIteration,
        },
        beforeActionOpts,
      );

      const callOpts: { traceId?: string; timeoutMs: number } = {
        timeoutMs: this._config.stepTimeoutMs,
      };
      if (traceId) callOpts.traceId = traceId;
      const toolResult = await this._toolRunner(
        actionPayload.tool,
        (actionPayload.params as Record<string, unknown>) ?? parsed.action.params,
        callOpts,
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
      // ============================================================
      if (!toolResult.success && this.hasStop("failFast")) {
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
      const obsText = this.formatObservation(toolResult);
      totalTokens += estimateTokens(obsText);
      conversation.push({
        role: "tool",
        name: toolResult.toolName,
        toolCallId: toolResult.callId,
        content: this.truncateObservation(obsText),
      });
    }

    if (iter >= this._config.maxIterations && !finalAnswer) {
      status = "max_iter";
      finalAnswer = "[达到最大 ReAct 迭代次数，未能收敛]";
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
export const __test__ = { parseLlmThought, buildSystemPrompt, checkDuplicate };

// 占位错误码引用（避免未使用警告）
void AgentError;
void ErrorCode;
