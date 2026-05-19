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
}
