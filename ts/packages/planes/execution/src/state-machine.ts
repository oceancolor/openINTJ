import { AgentError, ErrorCode } from "@openintj/core";
import type { Step, StepState } from "./types.js";

const TRANSITIONS: Readonly<Record<StepState, ReadonlySet<StepState>>> = Object.freeze({
  pending: new Set<StepState>(["ready", "skipped"]),
  ready: new Set<StepState>(["running", "skipped"]),
  running: new Set<StepState>(["completed", "failed", "waiting_approval"]),
  completed: new Set<StepState>(),
  failed: new Set<StepState>(["ready"]),
  skipped: new Set<StepState>(),
  waiting_approval: new Set<StepState>(["running", "skipped", "failed"]),
});

export interface StepTransitionEvent {
  stepId: string;
  from: StepState;
  to: StepState;
  timestampSec: number;
}

export class StepStateMachine {
  private readonly clock: () => number;

  constructor(opts?: { clock?: () => number }) {
    this.clock = opts?.clock ?? (() => Date.now() / 1000);
  }

  transition(step: Step, target: StepState): StepTransitionEvent {
    const allowed = TRANSITIONS[step.state];
    if (!allowed.has(target)) {
      throw new AgentError({
        code: ErrorCode.STATE_TRANSITION_INVALID,
        message: `非法状态转换: ${step.state} → ${target}`,
        details: { stepId: step.stepId, from: step.state, to: target },
      });
    }

    const from = step.state;
    step.state = target;

    const now = this.clock();
    if (target === "running" && step.startedAt === 0) {
      step.startedAt = now;
    } else if (target === "completed" || target === "failed") {
      step.finishedAt = now;
    } else if (target === "ready" && from === "failed") {
      // 重试：清空 finishedAt 让下一轮重新计时；保留 startedAt 作为首次开始
      step.finishedAt = 0;
    }

    return { stepId: step.stepId, from, to: target, timestampSec: now };
  }

  /** 判断是否可重试：失败 + retryCount < maxRetries。 */
  canRetry(step: Step): boolean {
    return step.state === "failed" && step.retryCount < step.maxRetries;
  }
}
