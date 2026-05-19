import { AgentError, ErrorCode, type HookBus } from "@openintj/core";
import { StepStateMachine } from "./state-machine.js";
import { ToolHub } from "./tool-hub.js";
import type { ExecutionMode, ExecutionResult, Step } from "./types.js";

export interface ExecutorOpts {
  toolHub?: ToolHub;
  stateMachine?: StepStateMachine;
  hooks?: HookBus;
  /** 注册内置工具？默认 true。 */
  registerBuiltins?: boolean;
}

/**
 * 执行器 —— 顺序 / 并行执行步骤。
 * 修复 Python v2 死重试 bug：失败时实际重新进入 ready→running 循环，
 * 而不是只迁移状态后立即标记 failed。
 */
export class Executor {
  readonly toolHub: ToolHub;
  private readonly sm: StepStateMachine;
  private readonly hooks?: HookBus;

  constructor(opts: ExecutorOpts = {}) {
    this.toolHub =
      opts.toolHub ?? (opts.hooks ? new ToolHub({ hooks: opts.hooks }) : new ToolHub());
    this.sm = opts.stateMachine ?? new StepStateMachine();
    if (opts.hooks !== undefined) this.hooks = opts.hooks;
    if ((opts.registerBuiltins ?? true) && this.toolHub.list().length === 0) {
      this.toolHub.registerBuiltinTools();
    }
  }

  async execute(
    steps: Step[],
    mode: ExecutionMode = "sequential",
    opts?: { traceId?: string },
  ): Promise<ExecutionResult> {
    const start = Date.now();
    const finished: string[] = [];
    const failed: string[] = [];
    const skipped: string[] = [];
    const errors: ExecutionResult["errors"] = [];

    if (mode === "sequential") {
      for (const step of steps) {
        await this.runOne(step, opts?.traceId, finished, failed, errors);
      }
    } else if (mode === "parallel") {
      await Promise.all(
        steps.map((step) => this.runOne(step, opts?.traceId, finished, failed, errors)),
      );
    } else if (mode === "human_approval") {
      // 占位：进入 waiting_approval，由调用方驱动
      for (const step of steps) {
        this.sm.transition(step, "ready");
        // skip running — caller will do it
      }
    }

    return {
      success: failed.length === 0,
      mode,
      finishedSteps: finished,
      failedSteps: failed,
      skippedSteps: skipped,
      errors,
      totalDurationMs: Date.now() - start,
    };
  }

  /** 执行单个步骤，含真正的重试循环（已修复 Python v2 bug）。 */
  private async runOne(
    step: Step,
    traceId: string | undefined,
    finished: string[],
    failed: string[],
    errors: ExecutionResult["errors"],
  ): Promise<void> {
    while (true) {
      this.sm.transition(step, "ready");
      this.sm.transition(step, "running");

      try {
        if (this.toolHub.has(step.action)) {
          const callOpts: { traceId?: string; timeoutMs: number } = {
            timeoutMs: step.timeoutMs,
          };
          if (traceId) callOpts.traceId = traceId;
          const result = await this.toolHub.call(step.action, step.params, callOpts);
          step.result = result.output;
          if (!result.success) {
            throw new AgentError({
              code: ErrorCode.TOOL_FAILED,
              message: result.error ?? "工具调用失败",
              retriable: true,
              details: { tool: step.action },
            });
          }
        } else {
          step.result = { action: step.action, status: "executed-no-tool" };
        }
        this.sm.transition(step, "completed");
        finished.push(step.stepId);
        return;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        const code = err instanceof AgentError ? err.code : ErrorCode.EXECUTION_FAILED;
        const retriable = err instanceof AgentError ? err.retriable : false;
        step.error = message;
        this.sm.transition(step, "failed");

        if (retriable && this.sm.canRetry(step)) {
          step.retryCount++;
          // 真正重试：转回 ready，循环再次执行
          continue;
        }

        failed.push(step.stepId);
        errors.push({
          stepId: step.stepId,
          error: message,
          retryCount: step.retryCount,
          errorCode: code,
        });
        return;
      }
    }
  }

  // 暴露 hooks 用于外部注入（如需要）
  get bus(): HookBus | undefined {
    return this.hooks;
  }
}
