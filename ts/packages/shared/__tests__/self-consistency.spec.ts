import { afterEach, describe, expect, it } from "vitest";
import { resolveSelfConsistency, selectConsistentAnswer } from "../src/self-consistency.js";

describe("selectConsistentAnswer", () => {
  it("returns undefined for empty input", () => {
    expect(selectConsistentAnswer([])).toBeUndefined();
  });

  it("returns the only candidate without voting", () => {
    expect(selectConsistentAnswer([{ finalAnswer: "x" }])?.finalAnswer).toBe("x");
  });

  it("majority picks the most frequent normalized answer", () => {
    const got = selectConsistentAnswer([
      { finalAnswer: "Paris" },
      { finalAnswer: "paris " },
      { finalAnswer: "London" },
    ]);
    expect(got?.finalAnswer).toBe("Paris");
  });

  it("majority tie-breaks by longest answer", () => {
    const got = selectConsistentAnswer([
      { finalAnswer: "short" },
      { finalAnswer: "a much longer answer" },
    ]);
    // 各 1 票 → 平票，取更长的
    expect(got?.finalAnswer).toBe("a much longer answer");
  });

  it("longest strategy ignores frequency", () => {
    const got = selectConsistentAnswer(
      [{ finalAnswer: "aa" }, { finalAnswer: "aa" }, { finalAnswer: "bbbb" }],
      "longest",
    );
    expect(got?.finalAnswer).toBe("bbbb");
  });

  it("first strategy returns first candidate", () => {
    const got = selectConsistentAnswer(
      [{ finalAnswer: "one" }, { finalAnswer: "two" }],
      "first",
    );
    expect(got?.finalAnswer).toBe("one");
  });
});

describe("resolveSelfConsistency", () => {
  const KEYS = ["OPENINTJ_SELF_CONSISTENCY", "OPENINTJ_SELF_CONSISTENCY_STRATEGY"];
  afterEach(() => {
    for (const k of KEYS) delete process.env[k];
  });

  it("returns undefined when disabled (no opts / no env)", () => {
    expect(resolveSelfConsistency()).toBeUndefined();
    expect(resolveSelfConsistency({ samples: 1 })).toBeUndefined();
  });

  it("uses opts.samples and default majority strategy", () => {
    expect(resolveSelfConsistency({ samples: 3 })).toEqual({ samples: 3, strategy: "majority" });
  });

  it("reads samples + strategy from env", () => {
    process.env["OPENINTJ_SELF_CONSISTENCY"] = "4";
    process.env["OPENINTJ_SELF_CONSISTENCY_STRATEGY"] = "longest";
    expect(resolveSelfConsistency()).toEqual({ samples: 4, strategy: "longest" });
  });

  it("clamps samples to a max of 8", () => {
    expect(resolveSelfConsistency({ samples: 99 })?.samples).toBe(8);
  });
});
