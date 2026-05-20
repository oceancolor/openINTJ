/**
 * Python v2 ↔ TS 行为对齐测试 —— execution slice
 * ===============================================
 *
 * 覆盖执行平面两个核心组件：
 *
 *   1. StepStateMachine.transition — 状态机合法转换表 + 错误抛出
 *   2. Executor.execute            — sequential / parallel 模式下的事件轨迹
 *
 * 协议层差异（与 Python v2 行为等价但落地形态不同）：
 *   - Python `StepStateMachine.transition` 返回 framework_core.Event(event_type=...),
 *     其中 target=RUNNING/READY/SKIPPED/WAITING_APPROVAL → STEP_STARTED,
 *           target=COMPLETED → STEP_FINISHED,
 *           target=FAILED → STEP_FAILED;
 *   - TS `StepStateMachine.transition` 直接返回 `{ stepId, from, to, timestampSec }`，
 *     不再带符号化 event_type。本测试用相同映射表在 TS 侧重建 eventType 字符串，
 *     以便和 Python fixture 中的 `eventType` / `eventTrace.type` 严格对齐。
 *
 * 已知偏差：
 *   - Python Executor 的失败重试有 FAILED→FAILED 死循环 bug（见 python-reference.md §三.1）；
 *     TS 已修复（真正重试）。本 fixture 只跑全成功路径以避开该差异。
 *
 * 重新生成 fixture：
 *   py scripts/python-parity/generate_fixtures.py
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AgentError } from "@openintj/core";
import { describe, expect, it } from "vitest";
import {
  type ExecutionMode,
  Executor,
  type Step,
  StepSchema,
  type StepState,
  StepStateMachine,
} from "../../src/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

/** Python v2 的 target → event_type 映射；TS 端复用以对齐 fixture。 */
const EVENT_TYPE_FOR_TARGET: Record<StepState, string> = {
  ready: "STEP_STARTED",
  running: "STEP_STARTED",
  completed: "STEP_FINISHED",
  failed: "STEP_FAILED",
  skipped: "STEP_STARTED",
  waiting_approval: "STEP_STARTED",
  pending: "STEP_STARTED", // Python 不会主动 transition 到 pending；保留兜底
};

interface TransitionCase {
  from: StepState;
  to: StepState;
  allowed: boolean;
  eventType?: string;
  eventSource?: string;
  eventPayload?: { step_id: string; from: string; to: string };
  errorCode?: string;
}

interface ExecutionCase {
  mode: ExecutionMode;
  steps: Array<{ stepId: string; action: string }>;
  expected: {
    success: boolean;
    finishedSteps: string[];
    failedSteps: string[];
    eventTrace: Array<{
      type: string;
      stepId: string;
      from: string;
      to: string;
    }>;
  };
}

interface Fixture {
  schemaVersion: number;
  generatedFrom: string;
  notes: Record<string, unknown>;
  transitions: TransitionCase[];
  executions: ExecutionCase[];
}

const FIXTURE_PATH = resolve(__dirname, "./fixtures/python-v2.json");

const loadFixture = (): Fixture => {
  const raw = readFileSync(FIXTURE_PATH, "utf8");
  const parsed = JSON.parse(raw) as Fixture;
  if (parsed.schemaVersion !== 1) {
    throw new Error(
      `[parity:execution] unsupported fixture schemaVersion=${parsed.schemaVersion}; regenerate via scripts/python-parity/generate_fixtures.py`,
    );
  }
  return parsed;
};

/** 让 fixture 里 stepId 走过 StepSchema 的最小 Step 构造器。 */
const makeStep = (stepId: string, action: string, state: StepState): Step =>
  StepSchema.parse({ stepId, action, state });

/**
 * 录制版 StepStateMachine：每次 transition 都记一条 {type, stepId, from, to}。
 * 用相同的 EVENT_TYPE_FOR_TARGET 映射给出 Python 兼容的 type 字符串。
 */
class RecordingStateMachine extends StepStateMachine {
  readonly trace: Array<{
    type: string;
    stepId: string;
    from: string;
    to: string;
  }> = [];

  override transition(step: Step, target: StepState) {
    const from = step.state;
    const event = super.transition(step, target);
    this.trace.push({
      type: EVENT_TYPE_FOR_TARGET[target] ?? "STEP_STARTED",
      stepId: step.stepId,
      from,
      to: target,
    });
    return event;
  }
}

describe("parity:execution ← Python v2", () => {
  const fixture = loadFixture();

  describe("StepStateMachine.transition vs Python state machine", () => {
    for (const c of fixture.transitions) {
      it(`${c.from} → ${c.to} : ${c.allowed ? "allowed" : "rejected"}`, () => {
        const sm = new StepStateMachine();
        const step = makeStep(`t-${c.from}-${c.to}`, "noop", c.from);

        if (c.allowed) {
          const got = sm.transition(step, c.to);
          expect(got.from).toBe(c.from);
          expect(got.to).toBe(c.to);
          expect(got.stepId).toBe(step.stepId);
          const expectedType = c.eventType ?? EVENT_TYPE_FOR_TARGET[c.to] ?? "STEP_STARTED";
          expect(EVENT_TYPE_FOR_TARGET[c.to]).toBe(expectedType);
          expect(c.eventPayload?.step_id).toBe(step.stepId);
          expect(c.eventPayload?.from).toBe(c.from);
          expect(c.eventPayload?.to).toBe(c.to);
        } else {
          let thrown: unknown;
          try {
            sm.transition(step, c.to);
          } catch (err) {
            thrown = err;
          }
          expect(thrown).toBeInstanceOf(AgentError);
          if (thrown instanceof AgentError) {
            // Python 端用 EXECUTION_FAILED；TS 用更细化的 STATE_TRANSITION_INVALID。
            // 两者都属于"非法转换被拒绝"语义，仅错误码命名不同。
            expect(["STATE_TRANSITION_INVALID", "EXECUTION_FAILED"]).toContain(thrown.code);
          }
        }
      });
    }
  });

  describe("Executor.execute vs Python Executor.execute (event trace)", () => {
    for (const c of fixture.executions) {
      it(`mode='${c.mode}' → finishedSteps + eventTrace`, async () => {
        const recorder = new RecordingStateMachine();
        const executor = new Executor({
          stateMachine: recorder,
          registerBuiltins: false, // 与 Python 等价：测试 action 都未注册，走 default 分支
        });
        const steps = c.steps.map((s) => makeStep(s.stepId, s.action, "pending" as StepState));

        const result = await executor.execute(steps, c.mode);
        expect(result.success).toBe(c.expected.success);
        expect(result.finishedSteps).toEqual(c.expected.finishedSteps);
        expect(result.failedSteps).toEqual(c.expected.failedSteps);
        expect(recorder.trace).toEqual(c.expected.eventTrace);
      });
    }
  });
});
