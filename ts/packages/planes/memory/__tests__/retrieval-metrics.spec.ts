import { describe, expect, it } from "vitest";
import {
  dcg,
  evaluateRanker,
  ndcgAtK,
  precisionAtK,
  recallAtK,
  reciprocalRank,
} from "../src/eval/retrieval-metrics.js";

describe("retrieval-metrics", () => {
  it("dcg: 完美排序 > 逆序", () => {
    expect(dcg([3, 2, 1])).toBeGreaterThan(dcg([1, 2, 3]));
  });

  it("ndcgAtK: 理想排序 = 1.0", () => {
    const rel = new Map([
      ["a", 3],
      ["b", 2],
      ["c", 1],
    ]);
    expect(ndcgAtK(["a", "b", "c"], rel, 3)).toBeCloseTo(1, 10);
  });

  it("ndcgAtK: 逆序 < 1.0 且 > 0", () => {
    const rel = new Map([
      ["a", 3],
      ["b", 2],
      ["c", 1],
    ]);
    const v = ndcgAtK(["c", "b", "a"], rel, 3);
    expect(v).toBeLessThan(1);
    expect(v).toBeGreaterThan(0);
  });

  it("ndcgAtK: 无相关文档返回 0", () => {
    expect(ndcgAtK(["x"], new Map(), 5)).toBe(0);
  });

  it("recallAtK / precisionAtK", () => {
    const relevant = new Set(["a", "b", "c", "d"]);
    expect(recallAtK(["a", "b", "z"], relevant, 3)).toBeCloseTo(0.5, 10);
    expect(precisionAtK(["a", "b", "z"], relevant, 3)).toBeCloseTo(2 / 3, 10);
  });

  it("reciprocalRank: 第一个相关文档排第 2 → 0.5", () => {
    expect(reciprocalRank(["x", "a"], new Set(["a"]))).toBeCloseTo(0.5, 10);
    expect(reciprocalRank(["x", "y"], new Set(["a"]))).toBe(0);
  });

  it("evaluateRanker: 宏平均", () => {
    const cases = [
      { query: "q1", relevant: new Map([["a", 1]]) },
      { query: "q2", relevant: new Map([["b", 1]]) },
    ];
    const perfect = evaluateRanker(cases, (q) => (q === "q1" ? ["a"] : ["b"]), 5);
    expect(perfect.ndcg).toBeCloseTo(1, 10);
    expect(perfect.recall).toBeCloseTo(1, 10);
    expect(perfect.n).toBe(2);
  });
});
