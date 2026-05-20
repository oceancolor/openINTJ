/**
 * Python v2 ↔ TS 行为对齐测试 —— control slice
 * =============================================
 *
 * 覆盖控制平面两个核心组件：
 *
 *   1. GoalParser.parse       — 中英文意图关键字提取 + 引号实体抽取 + 优先级估算
 *   2. Planner.createPlan     — 意图 → 固定 PlanStep DAG 模板
 *
 * 已知偏差（详见 fixture.notes.plannerDivergence）：
 *   - Python Planner 把 `delete`/`execute` intent 都落到 general 分支（3 步 think/act/respond）；
 *   - TS Planner 给二者提供了专门的 3 步专用模板（验证存在/审批/删除 与 校验/执行/汇报）。
 *   - 因此本测试只对齐公共 5 个 intent：create / modify / query / plan / general。
 *
 * 重新生成 fixture：
 *   py scripts/python-parity/generate_fixtures.py
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { TaskType, type TaskTypeType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { GoalParser, Planner } from "../../src/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

interface ParsedGoalCase {
  input: string;
  taskType: TaskTypeType;
  expected: {
    intent: string;
    entities: string[];
    priority: number;
  };
}

interface PlanCase {
  intent: string;
  expected: Array<{
    stepId: string;
    action: string;
    description: string;
    dependencies: string[];
    status: string;
  }>;
}

interface Fixture {
  schemaVersion: number;
  generatedFrom: string;
  notes: Record<string, unknown>;
  parsedGoals: ParsedGoalCase[];
  plans: PlanCase[];
}

const FIXTURE_PATH = resolve(__dirname, "./fixtures/python-v2.json");

const loadFixture = (): Fixture => {
  const raw = readFileSync(FIXTURE_PATH, "utf8");
  const parsed = JSON.parse(raw) as Fixture;
  if (parsed.schemaVersion !== 1) {
    throw new Error(
      `[parity:control] unsupported fixture schemaVersion=${parsed.schemaVersion}; regenerate via scripts/python-parity/generate_fixtures.py`,
    );
  }
  return parsed;
};

describe("parity:control ← Python v2", () => {
  const fixture = loadFixture();

  describe("GoalParser.parse vs control_plane.GoalParser.parse", () => {
    const parser = new GoalParser();

    for (const c of fixture.parsedGoals) {
      const safeKey = JSON.stringify(c.input).slice(0, 40);
      it(`intent/entities/priority: ${safeKey} (taskType=${c.taskType})`, () => {
        // 确认 fixture 的 taskType 字符串是合法 TaskType enum 值
        const tt = Object.values(TaskType).includes(c.taskType as TaskTypeType)
          ? (c.taskType as TaskTypeType)
          : TaskType.GENERAL_CHAT;

        const got = parser.parse(c.input, tt);
        expect(got.intent).toBe(c.expected.intent);
        expect(got.entities).toEqual(c.expected.entities);
        expect(got.priority).toBe(c.expected.priority);
      });
    }
  });

  describe("Planner.createPlan vs control_plane.Planner.create_plan", () => {
    const planner = new Planner();
    const parser = new GoalParser();

    for (const c of fixture.plans) {
      it(`intent='${c.intent}' → PlanGraph.steps shape`, () => {
        // 通过解析一个最小输入引出对应 intent 的 ParsedGoal
        // （Planner.createPlan 只依赖 goal.intent）
        const stubGoal = parser.parse("_parity_");
        const goal = { ...stubGoal, intent: c.intent as typeof stubGoal.intent };
        const plan = planner.createPlan(goal);

        expect(plan.steps).toHaveLength(c.expected.length);
        for (let i = 0; i < c.expected.length; i++) {
          const want = c.expected[i];
          if (!want) continue;
          const got = plan.steps[i];
          expect(got, `step[${i}] for intent='${c.intent}'`).toBeDefined();
          if (!got) continue;
          expect(got.stepId).toBe(want.stepId);
          expect(got.action).toBe(want.action);
          expect(got.description).toBe(want.description);
          expect(got.dependencies).toEqual(want.dependencies);
          expect(got.status).toBe(want.status);
        }
      });
    }
  });
});
