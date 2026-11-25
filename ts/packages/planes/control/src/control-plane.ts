import {
  type Command,
  CommandSchema,
  CommandType,
  TaskType,
  type TaskTypeType,
} from "@openintj/core";
import { GoalParser } from "./goal-parser.js";
import type { PlanGraph } from "./plan-graph.js";
import { Planner } from "./planner.js";
import type { ParsedGoal, PlanStep } from "./types.js";

export interface ControlPlaneOpts {
  goalParser?: GoalParser;
  planner?: Planner;
}

export class ControlPlane {
  readonly name = "control-plane";
  readonly goalParser: GoalParser;
  readonly planner: Planner;

  constructor(opts: ControlPlaneOpts = {}) {
    this.goalParser = opts.goalParser ?? new GoalParser();
    this.planner = opts.planner ?? new Planner();
  }

  processInput(
    rawInput: string,
    taskType: TaskTypeType = TaskType.GENERAL_CHAT,
  ): { goal: ParsedGoal; plan: PlanGraph } {
    const goal = this.goalParser.parse(rawInput, taskType);
    const plan = this.planner.createPlan(goal);
    return { goal, plan };
  }

  makePlanCommand(payload: Record<string, unknown>): Command {
    return CommandSchema.parse({
      commandType: CommandType.PLAN,
      target: "planner",
      payload,
    });
  }

  makeExecuteCommand(step: PlanStep): Command {
    return CommandSchema.parse({
      commandType: CommandType.EXECUTE,
      target: "executor",
      payload: {
        stepId: step.stepId,
        action: step.action,
        params: step.params,
      },
    });
  }

  makeToolCommand(toolName: string, params: Record<string, unknown>): Command {
    return CommandSchema.parse({
      commandType: CommandType.TOOL_CALL,
      target: toolName,
      payload: params,
    });
  }

  /**
   * P1-4：把 PlanStep 接到 ToolHub（经 makeExecuteCommand / makeToolCommand）。
   * 抽象 action（无 hub 工具）返回 null，由调用方回退到 prompt 路径。
   */
  async dispatchPlanStep(
    step: PlanStep,
    callTool: (name: string, params: Record<string, unknown>) => Promise<unknown>,
  ): Promise<{ command: Command; result: unknown } | null> {
    const executeCmd = this.makeExecuteCommand(step);
    void executeCmd; // 保留 EXECUTE 命令构造，供审计/编排层消费
    if (!step.action || step.action === "respond" || step.action === "think" || step.action === "act") {
      return null;
    }
    // 仅当 action 看起来像工具名（含下划线或常见内置）时派发
    const toolish = /^(?:[a-z][a-z0-9_]*)$/i.test(step.action);
    if (!toolish) return null;
    const toolCmd = this.makeToolCommand(step.action, step.params ?? {});
    const result = await callTool(step.action, step.params ?? {});
    return { command: toolCmd, result };
  }
}
