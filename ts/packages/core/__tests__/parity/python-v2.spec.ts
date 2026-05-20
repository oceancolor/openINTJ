/**
 * Python v2 ↔ TS 行为对齐测试 —— core slice
 * ==========================================
 *
 * 固定 fixture 来自 scripts/python-parity/generate_fixtures.py，
 * 覆盖三个最底层的纯函数：
 *
 *   1. SimpleEmbedder           — SHA-256 → 64 维伪向量
 *   2. cosineSimilarity         — 余弦相似度
 *   3. decayImportance          — 半衰期指数衰减
 *
 * 这三件是上层 MemoryRetriever / ContextEngine / DormantStore 的"数值地基"，
 * 必须与 Python v2 严格一致，否则后续所有评分都飘。
 *
 * 重新生成 fixture：
 *   py scripts/python-parity/generate_fixtures.py
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { SimpleEmbedder, cosineSimilarity, decayImportance } from "../../src/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

interface Fixture {
  schemaVersion: number;
  generatedFrom: string;
  embeddings: ReadonlyArray<{ input: string; dim: number; vector: number[] }>;
  cosineSimilarities: ReadonlyArray<{
    a: number[];
    b: number[];
    expected: number;
  }>;
  decayImportance: ReadonlyArray<{
    importance: number;
    halfLifeHours: number;
    ageSeconds: number;
    expected: number;
  }>;
}

const FIXTURE_PATH = resolve(__dirname, "./fixtures/python-v2.json");

const loadFixture = (): Fixture => {
  const raw = readFileSync(FIXTURE_PATH, "utf8");
  const parsed = JSON.parse(raw) as Fixture;
  if (parsed.schemaVersion !== 1) {
    throw new Error(
      `[parity:core] unsupported fixture schemaVersion=${parsed.schemaVersion}; regenerate via scripts/python-parity/generate_fixtures.py`,
    );
  }
  return parsed;
};

const TOL_EMBEDDING = 1e-12; // SHA256 → 纯位运算，应当 bit-identical
const TOL_COSINE = 1e-12; // 纯浮点点积/sqrt，应当 bit-identical
/**
 * decay 容差：Python 用 0.693 当 ln(2)，TS 用 Math.LN2 (=0.6931471805599453)。
 * 相对误差 ≈ 2.1e-4，单半衰期 absolute err ≈ 7e-5；多半衰期累积仍 < 1e-4。
 * 这是 TS 端**有意的精度提升**——保留 Python 行为作为参考，但允许 TS 更精确。
 */
const TOL_DECAY = 1e-4;

describe("parity:core ← Python v2", () => {
  const fixture = loadFixture();

  describe("SimpleEmbedder vs simple_embedding", () => {
    for (const { input, dim, vector } of fixture.embeddings) {
      it(`dim=${dim} input=${JSON.stringify(input).slice(0, 32)}`, () => {
        const embedder = new SimpleEmbedder(dim);
        const got = embedder.embed(input);
        expect(got).toHaveLength(vector.length);
        for (let i = 0; i < vector.length; i++) {
          const a = got[i] ?? Number.NaN;
          const b = vector[i] ?? Number.NaN;
          expect(Math.abs(a - b)).toBeLessThanOrEqual(TOL_EMBEDDING);
        }
      });
    }
  });

  describe("cosineSimilarity vs cosine_similarity", () => {
    for (const { a, b, expected } of fixture.cosineSimilarities) {
      it(`cos(a[${a.length}], b[${b.length}]) ≈ ${expected.toFixed(6)}`, () => {
        const got = cosineSimilarity(a, b);
        expect(Math.abs(got - expected)).toBeLessThanOrEqual(TOL_COSINE);
      });
    }
  });

  describe("decayImportance vs MemoryFragment.decay_importance", () => {
    for (const c of fixture.decayImportance) {
      it(`imp=${c.importance} halfLife=${c.halfLifeHours}h age=${c.ageSeconds}s ≈ ${c.expected.toFixed(6)}`, () => {
        // 构造一个"虚拟 fragment"：timestamp = now - age, now 任意
        const now = 1_000_000;
        const fragment = {
          importance: c.importance,
          timestamp: now - c.ageSeconds,
        };
        const got = decayImportance(fragment, c.halfLifeHours, now);
        expect(Math.abs(got - c.expected)).toBeLessThanOrEqual(TOL_DECAY);
      });
    }
  });
});
