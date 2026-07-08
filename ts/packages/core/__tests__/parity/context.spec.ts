/**
 * Python v2 ↔ TS 行为对齐测试 —— context slice（#12 扩展）
 * ========================================================
 *
 * 覆盖 ContextEngine 的**确定性内核**（两端 build_context 整体架构不同，
 * 但这两块语义严格等价，是上层上下文装配的地基）：
 *
 *   1. ContextBudget 算术   — availableTokens / usageRatio / memoryBudget / needsCompaction
 *                             (framework_core.ContextBudget ↔ TS ContextBudgetTracker)
 *   2. task → shader 映射    — ShaderConfig.get_shader_for_task ↔ getShaderForTask
 *
 * fixture 来自 scripts/python-parity/generate_fixtures.py（context slice）。
 * 重新生成：py scripts/python-parity/generate_fixtures.py
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  ContextBudgetTracker,
  type ShaderModeType,
  type TaskTypeType,
  getShaderForTask,
} from "../../src/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

interface BudgetCase {
  input: {
    maxTokens: number;
    reservedTokens: number;
    systemPromptTokens: number;
    conversationTokens: number;
    memoryTokens: number;
    toolTokens: number;
  };
  expected: {
    availableTokens: number;
    usageRatio: number;
    memoryBudget: number;
    needsCompaction: Record<string, boolean>;
  };
}

interface Fixture {
  schemaVersion: number;
  generatedFrom: string;
  budgets: BudgetCase[];
  shaderForTask: Array<{ taskType: string; expected: string }>;
}

const FIXTURE_PATH = resolve(__dirname, "./fixtures/context.json");

const loadFixture = (): Fixture => {
  const parsed = JSON.parse(readFileSync(FIXTURE_PATH, "utf8")) as Fixture;
  if (parsed.schemaVersion !== 1) {
    throw new Error(
      `[parity:context] unsupported fixture schemaVersion=${parsed.schemaVersion}; regenerate via scripts/python-parity/generate_fixtures.py`,
    );
  }
  return parsed;
};

const TOL_RATIO = 1e-12;

describe("parity:context ← Python v2", () => {
  const fixture = loadFixture();

  describe("ContextBudget 算术 vs ContextBudgetTracker", () => {
    for (const [i, c] of fixture.budgets.entries()) {
      it(`budget#${i} (conv=${c.input.conversationTokens}, mem=${c.input.memoryTokens})`, () => {
        const t = new ContextBudgetTracker(c.input);
        expect(t.availableTokens).toBe(c.expected.availableTokens);
        expect(t.usageRatio).toBeCloseTo(c.expected.usageRatio, 12);
        expect(Math.abs(t.usageRatio - c.expected.usageRatio)).toBeLessThan(TOL_RATIO);
        expect(t.memoryBudget).toBe(c.expected.memoryBudget);
        for (const [threshold, expected] of Object.entries(c.expected.needsCompaction)) {
          expect(t.needsCompaction(Number(threshold))).toBe(expected);
        }
      });
    }
  });

  describe("get_shader_for_task vs getShaderForTask", () => {
    for (const c of fixture.shaderForTask) {
      it(`${c.taskType} → ${c.expected}`, () => {
        expect(getShaderForTask(c.taskType as TaskTypeType)).toBe(c.expected as ShaderModeType);
      });
    }
  });
});
