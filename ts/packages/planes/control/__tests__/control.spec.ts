import { CommandType, TaskType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { ControlPlane, GoalParser, PlanGraph, Planner, buildPlanStep } from "../src/index.js";

describe("GoalParser", () => {
  it("recognizes Chinese intents", () => {
    const p = new GoalParser();
    expect(p.parse("帮我创建一个文件").intent).toBe("create");
    expect(p.parse("修改这段代码").intent).toBe("modify");
    expect(p.parse("查询数据库").intent).toBe("query");
    expect(p.parse("规划一个方案").intent).toBe("plan");
    expect(p.parse("天气怎么样？").intent).toBe("general");
  });

  it("extracts entities from quoted strings (CJK quotes too)", () => {
    const p = new GoalParser();
    expect(p.parse('请创建 "user.config.json" 和 \u201Cmain.ts\u201D').entities).toEqual([
      "user.config.json",
      "main.ts",
    ]);
  });

  it("escalates priority for urgent words", () => {
    const p = new GoalParser();
    expect(p.parse("普通任务").priority).toBe(5);
    expect(p.parse("紧急修复 bug").priority).toBe(9);
    expect(p.parse("urgent: deploy").priority).toBe(9);
  });

  it("escalates priority by task type", () => {
    const p = new GoalParser();
    expect(p.parse("写代码", TaskType.CODE_GENERATION).priority).toBe(7);
    expect(p.parse("快速回答", TaskType.QUICK_RESPONSE).priority).toBe(8);
  });
});

describe("Planner", () => {
  it("emits create plan with 4 steps and chain dependencies", () => {
    const planner = new Planner();
    const parser = new GoalParser();
    const goal = parser.parse("创建一个 README");
    const plan = planner.createPlan(goal);
    expect(plan.steps).toHaveLength(4);
    expect(plan.steps.map((s) => s.action)).toEqual(["analyze", "design", "implement", "verify"]);
    expect(plan.steps[3]!.dependencies).toEqual(["s3"]);
  });

  it("falls back to general template for unknown intent", () => {
    const planner = new Planner();
    const parser = new GoalParser();
    const goal = parser.parse("how are you?");
    const plan = planner.createPlan(goal);
    expect(plan.steps).toHaveLength(3);
    expect(plan.steps.map((s) => s.action)).toEqual(["think", "act", "respond"]);
  });

  it("supports custom templates override", () => {
    const planner = new Planner({
      templates: {
        general: () => [buildPlanStep({ stepId: "x1", action: "custom" })],
      },
    });
    const parser = new GoalParser();
    const plan = planner.createPlan(parser.parse("hi"));
    expect(plan.steps).toHaveLength(1);
    expect(plan.steps[0]!.action).toBe("custom");
  });
});

describe("PlanGraph", () => {
  it("computes progress correctly", () => {
    const planner = new Planner();
    const parser = new GoalParser();
    const plan = planner.createPlan(parser.parse("查询数据库"));
    expect(plan.progress).toBe(0);
    plan.markStep("s1", "completed");
    expect(plan.progress).toBeCloseTo(1 / 3, 5);
  });

  it("getReadySteps respects dependencies", () => {
    const planner = new Planner();
    const parser = new GoalParser();
    const plan = planner.createPlan(parser.parse("创建 feature"));
    expect(plan.getReadySteps().map((s) => s.stepId)).toEqual(["s1"]);
    plan.markStep("s1", "completed");
    expect(plan.getReadySteps().map((s) => s.stepId)).toEqual(["s2"]);
  });

  it("topoOrder respects DAG and detects cycles", () => {
    const goal = new GoalParser().parse("plan x");
    const steps = [
      buildPlanStep({ stepId: "a", action: "x" }),
      buildPlanStep({ stepId: "b", action: "y", dependencies: ["a"] }),
      buildPlanStep({ stepId: "c", action: "z", dependencies: ["b"] }),
    ];
    const plan = new PlanGraph({ goal, steps });
    expect(plan.topoOrder().map((s) => s.stepId)).toEqual(["a", "b", "c"]);

    const cyclical = [
      buildPlanStep({ stepId: "a", action: "x", dependencies: ["b"] }),
      buildPlanStep({ stepId: "b", action: "y", dependencies: ["a"] }),
    ];
    expect(() => new PlanGraph({ goal, steps: cyclical })).toThrowError(/循环/);
  });
});

describe("ControlPlane", () => {
  it("processInput returns goal+plan", () => {
    const cp = new ControlPlane();
    const { goal, plan } = cp.processInput("创建文件", TaskType.CODE_GENERATION);
    expect(goal.intent).toBe("create");
    expect(plan.totalSteps).toBe(4);
  });

  it("makeXCommand validates with zod", () => {
    const cp = new ControlPlane();
    const cmd = cp.makeToolCommand("read_file", { path: "x" });
    expect(cmd.commandType).toBe(CommandType.TOOL_CALL);
    expect(cmd.target).toBe("read_file");
    expect(cmd.payload).toEqual({ path: "x" });
  });
});
