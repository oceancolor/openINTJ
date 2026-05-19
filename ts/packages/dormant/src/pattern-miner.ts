import { randomUUID } from "node:crypto";
import { type DormantPattern, DormantPatternSchema, type PassiveEvent } from "./types.js";

export interface PatternMinerOpts {
  /** n-gram 大小，默认 3-grams。 */
  ngramSize: number;
  /** 最小出现次数才算 pattern。 */
  minFrequency: number;
  /** 最小置信度（计算公式见下）。 */
  minConfidence: number;
  /** 可选 LLM 抽取器 —— 用于把高频 n-gram 转成自然语言描述。 */
  llmExtract?: (
    ngram: string,
    samples: PassiveEvent[],
  ) => Promise<{
    description: string;
    category: DormantPattern["category"];
  }>;
}

export const DEFAULT_PATTERN_OPTS: PatternMinerOpts = {
  ngramSize: 3,
  minFrequency: 3,
  minConfidence: 0.4,
};

/**
 * Tokenize：
 *  - 拉丁字符按 whitespace 切词
 *  - 中日韩字符按字符级切（每个 CJK 字符为一 token）
 *  - 这是简单的 v0.1 实现；生产可换 jieba/ICU
 */
const tokenize = (text: string): string[] => {
  const cleaned = text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ");
  const tokens: string[] = [];
  let buf = "";
  const flushBuf = (): void => {
    if (buf.length > 0) {
      tokens.push(buf);
      buf = "";
    }
  };
  // CJK Unified + Hiragana + Katakana + Hangul ranges
  const isCjk = (cp: number): boolean =>
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x3040 && cp <= 0x30ff) ||
    (cp >= 0xac00 && cp <= 0xd7af);
  for (const ch of cleaned) {
    const cp = ch.codePointAt(0) ?? 0;
    if (isCjk(cp)) {
      flushBuf();
      tokens.push(ch);
    } else if (/\s/.test(ch)) {
      flushBuf();
    } else {
      buf += ch;
    }
  }
  flushBuf();
  return tokens.filter((t) => t.length > 0);
};

const buildNgrams = (tokens: string[], n: number): string[] => {
  if (tokens.length < n) return [];
  const out: string[] = [];
  for (let i = 0; i <= tokens.length - n; i++) {
    out.push(tokens.slice(i, i + n).join(" "));
  }
  return out;
};

/**
 * PatternMiner —— 从 PassiveStore 事件中提取 n-gram 模式。
 *
 * 算法：
 *  1) tokenize 用户/agent 文本
 *  2) 滑窗 n-gram 计数
 *  3) 频次 >= minFrequency 进入候选
 *  4) 置信度 = freq / totalEvents（粗糙但足够 v0.1）
 *  5) 可选注入 llmExtract 把 n-gram 翻译为人类可读 description
 */
export class PatternMiner {
  readonly opts: PatternMinerOpts;

  constructor(opts: Partial<PatternMinerOpts> = {}) {
    this.opts = { ...DEFAULT_PATTERN_OPTS, ...opts };
  }

  async mine(events: readonly PassiveEvent[]): Promise<DormantPattern[]> {
    if (events.length === 0) return [];

    const ngramFreq = new Map<string, number>();
    const ngramEvidence = new Map<string, Set<string>>();

    for (const e of events) {
      const tokens = tokenize(e.text);
      const grams = buildNgrams(tokens, this.opts.ngramSize);
      const seenInDoc = new Set<string>();
      for (const g of grams) {
        if (seenInDoc.has(g)) continue; // 单文档不重复计入 freq（DF-like）
        seenInDoc.add(g);
        ngramFreq.set(g, (ngramFreq.get(g) ?? 0) + 1);
        let evidences = ngramEvidence.get(g);
        if (!evidences) {
          evidences = new Set<string>();
          ngramEvidence.set(g, evidences);
        }
        evidences.add(e.eventId);
      }
    }

    const totalEvents = events.length;
    const patterns: DormantPattern[] = [];
    for (const [ngram, freq] of ngramFreq.entries()) {
      if (freq < this.opts.minFrequency) continue;
      const confidence = Math.min(1, freq / totalEvents);
      if (confidence < this.opts.minConfidence) continue;

      const evidenceIds = [...(ngramEvidence.get(ngram) ?? [])];
      const samples = events.filter((e) => evidenceIds.includes(e.eventId));

      let description = ngram;
      let category: DormantPattern["category"] = "other";
      if (this.opts.llmExtract) {
        try {
          const r = await this.opts.llmExtract(ngram, samples.slice(0, 5));
          description = r.description;
          category = r.category;
        } catch {
          description = `频繁出现的短语: "${ngram}"`;
        }
      } else {
        description = `频繁出现的短语: "${ngram}"`;
      }

      patterns.push(
        DormantPatternSchema.parse({
          patternId: randomUUID(),
          description,
          evidenceIds,
          frequency: freq,
          confidence,
          category,
          ts: Date.now(),
        }),
      );
    }

    // 按置信度降序
    patterns.sort((a, b) => b.confidence - a.confidence);
    return patterns;
  }
}
