/**
 * Python v2 ↔ TS 行为对齐测试 —— governance slice
 * ================================================
 *
 * 对齐 governance_plane.PolicyEngine.check 的判定结果（result / riskLevel / 阻断行为）。
 * fixture 由 scripts/python-parity/generate_fixtures.py 的 gen_governance() 生成。
 *
 * 重新生成 fixture：
 *   py scripts/python-parity/generate_fixtures.py
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AgentError, CommandSchema, ErrorCode } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { PolicyEngine } from "../../src/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

interface PolicyCheck {
  commandType: string;
  target: string;
  strictMode: boolean;
  expected: {
    allowed: boolean;
    result: string;
    riskLevel: string;
    errorCode?: string;
  };
}

interface Fixture {
  schemaVersion: number;
  generatedFrom: string;
  policyChecks: PolicyCheck[];
}

const FIXTURE_PATH = resolve(__dirname, "./fixtures/python-v2.json");

const loadFixture = (): Fixture => {
  const parsed = JSON.parse(readFileSync(FIXTURE_PATH, "utf8")) as Fixture;
  if (parsed.schemaVersion !== 1) {
    throw new Error(
      `[parity:governance] unsupported fixture schemaVersion=${parsed.schemaVersion}; regenerate via scripts/python-parity/generate_fixtures.py`,
    );
  }
  return parsed;
};

describe("parity:governance ← Python v2", () => {
  const fixture = loadFixture();

  describe("PolicyEngine.check vs PolicyEngine.check", () => {
    for (const c of fixture.policyChecks) {
      const label = `${c.target} (strict=${c.strictMode})`;
      it(label, () => {
        const engine = new PolicyEngine({ strictMode: c.strictMode });
        const command = CommandSchema.parse({
          commandType: c.commandType,
          target: c.target,
        });

        if (c.expected.allowed) {
          const event = engine.check(command);
          expect(event.result).toBe(c.expected.result);
          expect(event.riskLevel).toBe(c.expected.riskLevel);
        } else {
          let thrown: unknown;
          try {
            engine.check(command);
          } catch (e) {
            thrown = e;
          }
          expect(thrown).toBeInstanceOf(AgentError);
          expect((thrown as AgentError).code).toBe(ErrorCode.POLICY_BLOCKED);
          expect(c.expected.errorCode).toBe("POLICY_BLOCKED");
        }
      });
    }
  });
});
