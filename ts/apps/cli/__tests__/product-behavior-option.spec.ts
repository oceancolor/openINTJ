import { describe, expect, it } from "vitest";
import { parseProductBehaviorCohort } from "../src/product-behavior-option.js";

describe("CLI Product Behavior cohort option", () => {
  it.each([
    ["treatment", true],
    ["on", true],
    ["1", true],
    ["control", false],
    ["off", false],
    ["0", false],
  ] as const)("maps %s to %s", (input, expected) => {
    expect(parseProductBehaviorCohort(input)).toBe(expected);
  });

  it("rejects unknown cohorts", () => {
    expect(() => parseProductBehaviorCohort("maybe")).toThrow(/treatment\|control/);
  });
});
