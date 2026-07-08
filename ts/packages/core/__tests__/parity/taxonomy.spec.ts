/**
 * Python v2 ↔ TS 行为对齐测试 —— taxonomy slice（#12 扩展）
 * ========================================================
 *
 * 跨实现共享的**枚举契约**。这是 Python v2「Hooks/事件」最接近的对齐面：
 * TS HookBus 发出的框架事件用同一套 `EventType`；`CommandType` 是控制平面指令类型；
 * `ErrorCode` 是对客户端/日志暴露的错误契约。
 *
 * 断言：
 *   - EventType / CommandType：TS 与 Python 逐条相等（name→value 完全一致）。
 *   - ErrorCode：Python 的每一项都必须在 TS 中存在且值相等；TS 可额外扩展
 *     （HOOK_ERROR / STATE_TRANSITION_INVALID / LOOP_LIMIT_REACHED / REACT_DUPLICATE_LOOP —— hook/react 专用）。
 *
 * fixture 来自 scripts/python-parity/generate_fixtures.py（taxonomy slice）。
 * 重新生成：py scripts/python-parity/generate_fixtures.py
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CommandType, ErrorCode, EventType } from "../../src/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

interface Fixture {
  schemaVersion: number;
  generatedFrom: string;
  eventType: Record<string, string>;
  commandType: Record<string, string>;
  errorCode: Record<string, string>;
}

const FIXTURE_PATH = resolve(__dirname, "./fixtures/taxonomy.json");

const loadFixture = (): Fixture => {
  const parsed = JSON.parse(readFileSync(FIXTURE_PATH, "utf8")) as Fixture;
  if (parsed.schemaVersion !== 1) {
    throw new Error(
      `[parity:taxonomy] unsupported fixture schemaVersion=${parsed.schemaVersion}; regenerate via scripts/python-parity/generate_fixtures.py`,
    );
  }
  return parsed;
};

describe("parity:taxonomy ← Python v2", () => {
  const fixture = loadFixture();

  it("EventType 与 Python 逐条相等（含数量一致）", () => {
    expect(EventType).toEqual(fixture.eventType);
  });

  it("CommandType 与 Python 逐条相等（含数量一致）", () => {
    expect(CommandType).toEqual(fixture.commandType);
  });

  describe("ErrorCode：Python ⊆ TS（值相等）", () => {
    for (const [name, value] of Object.entries(loadFixture().errorCode)) {
      it(`${name} = ${value}`, () => {
        expect((ErrorCode as Record<string, string>)[name]).toBe(value);
      });
    }
  });

  it("TS ErrorCode 是 Python 的超集（仅新增 hook/react 专用码）", () => {
    const tsOnly = Object.keys(ErrorCode).filter((k) => !(k in fixture.errorCode));
    expect(tsOnly.sort()).toEqual(
      [
        "HOOK_ERROR",
        "LOOP_LIMIT_REACHED",
        "REACT_DUPLICATE_LOOP",
        "STATE_TRANSITION_INVALID",
      ].sort(),
    );
  });
});
