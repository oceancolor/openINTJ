/**
 * Python v2 ↔ TS 行为对齐测试 —— memory slice
 * ============================================
 *
 * 覆盖记忆平面两个核心组件：
 *
 *   1. MemoryStore add* overflow — 短期 → 长期 自动迁移
 *   2. MemoryRetriever.retrieve  — 评分公式 (relevance/keyword/decay) 与最终排序
 *
 * 半衰期口径对齐（详见 fixture.notes.halfLifeAlignment）：
 *   - Python MemoryRetriever 写死用 `max_summary_length / 10` 当半衰期（v2 已知 bug）；
 *   - TS MemoryRetriever 使用 `ShaderConfig.recencyHalfLifeHours`（修正后的字段）；
 *   - fixture 生成器把 Python 的 `max_summary_length = 240` → Python 半衰期 = 24h，
 *     与 TS 默认值对齐，使两边在同一组评分上严格可比。
 *
 * 容差：
 *   - 评分组件 `relevance` / `keyword` 是纯位运算 + 浮点，应当 bit-identical (≤1e-12)。
 *   - `recency` 与最终 `score` 受 Python `0.693 ≈ ln(2)` 近似影响，使用 `1e-4` 容差。
 *
 * 重新生成 fixture：
 *   py scripts/python-parity/generate_fixtures.py
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { MemoryFragmentSchema, SimpleEmbedder } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { MemoryRetriever, MemoryStore } from "../../src/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

interface StoreOverflowCase {
  maxShortTerm: number;
  adds: string[];
  expected: {
    shortTerm: string[];
    longTerm: string[];
  };
}

interface RetrievalCase {
  query: string;
  testNow: number;
  shader: {
    recencyHalfLifeHours: number;
    relevanceWeight: number;
    recencyWeight: number;
    importanceWeight: number;
  };
  fragments: Array<{
    content: string;
    importance: number;
    ageSeconds: number;
    taskTags: string[];
  }>;
  expected: Array<{
    content: string;
    score: number;
    components: { relevance: number; keyword: number; recency: number };
    importance: number;
    taskTags: string[];
  }>;
}

interface Fixture {
  schemaVersion: number;
  generatedFrom: string;
  notes: Record<string, unknown>;
  storeOverflows: StoreOverflowCase[];
  retrieval: RetrievalCase;
}

const FIXTURE_PATH = resolve(__dirname, "./fixtures/python-v2.json");

const loadFixture = (): Fixture => {
  const raw = readFileSync(FIXTURE_PATH, "utf8");
  const parsed = JSON.parse(raw) as Fixture;
  if (parsed.schemaVersion !== 1) {
    throw new Error(
      `[parity:memory] unsupported fixture schemaVersion=${parsed.schemaVersion}; regenerate via scripts/python-parity/generate_fixtures.py`,
    );
  }
  return parsed;
};

const TOL_COMPONENT_PURE = 1e-12; // relevance / keyword 是纯位运算
const TOL_RECENCY = 1e-4; // Python 0.693 ≈ ln(2) 引入
const TOL_SCORE = 1e-4;

describe("parity:memory ← Python v2", () => {
  const fixture = loadFixture();

  describe("MemoryStore overflow vs Python add_short_term", () => {
    for (const c of fixture.storeOverflows) {
      it(`maxShortTerm=${c.maxShortTerm} + ${c.adds.length} adds → short/long split`, () => {
        const store = new MemoryStore({ maxShortTerm: c.maxShortTerm });
        for (const content of c.adds) {
          store.addShortTerm(content);
        }
        expect(store.shortTerm.map((f) => f.content)).toEqual(c.expected.shortTerm);
        expect(store.longTerm.map((f) => f.content)).toEqual(c.expected.longTerm);
      });
    }
  });

  describe("MemoryRetriever.retrieve vs Python MemoryRetriever.retrieve", () => {
    const c = fixture.retrieval;

    it("score components & final ranking align with Python", () => {
      const embedder = new SimpleEmbedder(64);
      const store = new MemoryStore({}, { embedder });

      // 关键：fragment.timestamp = testNow - ageSeconds，
      // 配合后面 retriever 的 clock 注入返回 testNow，让 ageHours 严格 = ageSeconds/3600。
      for (const f of c.fragments) {
        const fragment = MemoryFragmentSchema.parse({
          content: f.content,
          importance: f.importance,
          embedding: embedder.embed(f.content),
          taskTags: f.taskTags,
          timestamp: c.testNow - f.ageSeconds,
          memoryType: "short_term",
        });
        store.shortTerm.push(fragment);
      }

      const retriever = new MemoryRetriever(
        store,
        {
          recencyHalfLifeHours: c.shader.recencyHalfLifeHours,
          relevanceWeight: c.shader.relevanceWeight,
          recencyWeight: c.shader.recencyWeight,
          importanceWeight: c.shader.importanceWeight,
          importanceThreshold: 0, // 与 Python `min_importance=0.0` 默认一致
          maxFragmentsPerQuery: 10,
        },
        { clock: () => c.testNow },
      );

      const ranked = retriever.retrieve(c.query, { topK: 10, minImportance: 0 });

      expect(ranked).toHaveLength(c.expected.length);

      // 按 content → expected 索引
      for (let i = 0; i < c.expected.length; i++) {
        const want = c.expected[i];
        const got = ranked[i];
        expect(got, `ranked[${i}]`).toBeDefined();
        if (!got || !want) continue;

        // 顺序必须严格一致（Python 按 score 降序，TS 同上）
        expect(got.fragment.content, `rank ${i} content`).toBe(want.content);

        // 纯位运算分量
        expect(Math.abs(got.components.relevance - want.components.relevance)).toBeLessThanOrEqual(
          TOL_COMPONENT_PURE,
        );
        expect(Math.abs(got.components.keyword - want.components.keyword)).toBeLessThanOrEqual(
          TOL_COMPONENT_PURE,
        );
        // recency 含 exp(0.693 vs LN2) 差异
        expect(Math.abs(got.components.recency - want.components.recency)).toBeLessThanOrEqual(
          TOL_RECENCY,
        );
        // 最终 score 同上
        expect(Math.abs(got.score - want.score)).toBeLessThanOrEqual(TOL_SCORE);
      }
    });
  });
});
