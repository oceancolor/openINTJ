import { TaskType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import {
  ControlPlane,
  GoalParser,
  PlanGraph,
  Planner,
  advisePlan,
  annotatePlanSteps,
  collectKnowledgeFacts,
  dijkstraSchedule,
  evaluatePlanPath,
  formatNextGoalQuery,
  resolvePlanAdviceForRun,
  shouldAdvisePlan,
} from "../src/index.js";
import { buildPlanStep } from "../src/plan-graph.js";

describe("knowledge facts", () => {
  it("splits known goal/target from missing environment", () => {
    const facts = collectKnowledgeFacts({
      query: "把 openINTJ CLI 迁移方案做到可发布",
      entities: ["openINTJ CLI"],
      structure: { goal: "迁移 CLI", deliverables: ["安装包"] },
    });
    expect(facts.find((f) => f.slot === "goal")?.known).toBe(true);
    expect(facts.find((f) => f.slot === "target")?.known).toBe(true);
    expect(facts.find((f) => f.slot === "deliverable")?.known).toBe(true);
    expect(facts.find((f) => f.slot === "environment")?.known).toBe(false);
  });
});

describe("path search", () => {
  it("Dijkstra prefers a known ready branch before clarifying an unknown sibling", () => {
    const goal = new GoalParser().parse("分析后执行未知目标");
    const steps = [
      buildPlanStep({ stepId: "s1", action: "retrieve", description: "盘点已知" }),
      buildPlanStep({
        stepId: "s2",
        action: "execute",
        description: "部署到生产",
        dependencies: [],
      }),
      buildPlanStep({
        stepId: "s3",
        action: "synthesize",
        description: "汇总",
        dependencies: ["s1", "s2"],
      }),
    ];
    const plan = new PlanGraph({ goal, steps });
    const annotated = annotatePlanSteps(plan.steps, { query: "部署到生产并汇总已知资料" });
    expect(annotated.find((a) => a.step.stepId === "s1")?.known).toBe(true);
    expect(annotated.find((a) => a.step.stepId === "s2")?.known).toBe(false);
    const path = dijkstraSchedule(annotated);
    expect(path.nextStepId).toBe("s1");
    expect(path.dijkstraOrder[0]).toBe("s1");
    expect(evaluatePlanPath(annotated).annealingOrder[0]).toBe("s1");
  });

  it("annealing and Dijkstra agree on next hop for a linear template", () => {
    const { plan } = new ControlPlane().processInput("规划迁移方案", TaskType.PLANNING);
    const annotated = annotatePlanSteps(plan.steps, {
      query: "规划 TypeScript CLI 三阶段迁移方案，并列出每阶段交付物",
      structure: { deliverables: ["阶段交付物"] },
    });
    const path = evaluatePlanPath(annotated);
    expect(path.nextStepId).toBe(plan.steps[0]!.stepId);
    expect(path.annealingOrder[0]).toBe(path.dijkstraOrder[0]);
  });
});

describe("advisePlan", () => {
  it("asks to fill unknowns when the next hop itself is blocked", () => {
    const goal = new GoalParser().parse("执行删除");
    const plan = new PlanGraph({
      goal,
      steps: [buildPlanStep({ stepId: "s1", action: "execute", description: "部署到生产" })],
    });
    const advice = advisePlan(plan, { query: "部署到生产" });
    expect(advice.action).toBe("clarify");
    expect(advice.questions.length).toBeGreaterThan(0);
    expect(advice.questions.join("")).toMatch(/环境|目标/);
    expect(advice.unknown.some((f) => !f.known)).toBe(true);
  });

  it("recommends the first known step and defers later unknowns", () => {
    const { plan } = new ControlPlane().processInput(
      "规划 TypeScript CLI 三阶段迁移方案，并列出每阶段交付物",
      TaskType.PLANNING,
    );
    const advice = advisePlan(plan, {
      query: "规划 TypeScript CLI 三阶段迁移方案，并列出每阶段交付物",
      structure: { deliverables: ["迁移文档", "安装包"] },
    });
    expect(advice.action).toBe("proceed");
    expect(advice.nextGoal).toBe("目标分解");
    expect(formatNextGoalQuery("原始任务", advice)).toContain("【规划】下一轮目标：目标分解");
  });
});

describe("resolvePlanAdviceForRun", () => {
  it("skips chat/quick tasks", () => {
    const decision = resolvePlanAdviceForRun(new ControlPlane(), {
      query: "你好",
      executionQuery: "你好",
      taskType: TaskType.QUICK_RESPONSE,
    });
    expect(decision.kind).toBe("skip");
    expect(shouldAdvisePlan(TaskType.PLANNING)).toBe(true);
  });

  it("prefixes the recommended next goal for a well-specified plan", () => {
    const decision = resolvePlanAdviceForRun(new ControlPlane(), {
      query: "规划 TypeScript CLI 三阶段迁移方案，并列出每阶段交付物",
      executionQuery: "规划 TypeScript CLI 三阶段迁移方案，并列出每阶段交付物",
      taskType: TaskType.PLANNING,
      structure: { deliverables: ["迁移文档"] },
    });
    expect(decision.kind).toBe("proceed");
    if (decision.kind !== "proceed") return;
    expect(decision.executionQuery).toContain("下一轮目标：目标分解");
    expect(decision.advice.path.method).toBe("dijkstra");
  });

  it("asks to fill slots when the cheapest next hop itself is blocked", () => {
    const decision = resolvePlanAdviceForRun(
      new ControlPlane({
        planner: new Planner({
          templates: {
            plan: () => [
              buildPlanStep({ stepId: "s1", action: "execute", description: "发布上线" }),
            ],
          },
        }),
      }),
      {
        query: "规划发布上线",
        executionQuery: "规划发布上线",
        taskType: TaskType.PLANNING,
      },
    );
    expect(decision.kind).toBe("clarify");
    if (decision.kind !== "clarify") return;
    expect(decision.questions.join("")).toMatch(/环境|目标/);
  });
});
