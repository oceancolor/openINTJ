/**
 * 长跑评测（long-run evaluation）——把「越用越好」做成可测量的指标。
 *
 * 动机（roadmap A2）：单轮评测看不出记忆/分类器的复利效应。这里按**有先后依赖的
 * query 序列**驱动 Agent：前几轮写入信息，后几轮应当「回忆」起来并受益。逐轮记录：
 *  - 检索命中（expectRecall 金片段是否出现在注入上下文 / 答案里）
 *  - token 花费（result.totalTokensSpent）
 *  - judge 通过
 * 再聚合成「改进曲线」（后半 vs 前半 recall 差）与会话级汇总，并支持 memory-on/off、
 * classifier-on/off 等 A/B 对比。
 *
 * 纯编排、零依赖：Agent 由调用方包成 `LongRunAgent`，judge / 金片段由场景脚本提供。
 */

export interface LongRunTurn {
  query: string;
  /** 本轮答案判定（true = 完成）。不提供则不计入通过率。 */
  judge?: (answer: string) => boolean | Promise<boolean>;
  /** 期望被「回忆」的金片段：出现在注入上下文或答案中即算命中（不区分大小写）。 */
  expectRecall?: string;
}

export interface LongRunScript {
  id: string;
  description?: string;
  turns: readonly LongRunTurn[];
}

export interface LongRunAgentOutput {
  finalAnswer: string;
  /** 本轮 token 花费（如 TaoResult.totalTokensSpent）。 */
  tokensSpent: number;
  /** 注入到本轮上下文的记忆文本（用于命中判定）；不提供则回退到对答案做命中判定。 */
  injectedContext?: string;
}

export type LongRunAgent = (query: string) => Promise<LongRunAgentOutput> | LongRunAgentOutput;

export interface LongRunTurnResult {
  index: number;
  query: string;
  answer: string;
  tokensSpent: number;
  /** 是否提供了 judge。 */
  judged: boolean;
  passed: boolean;
  /** 是否提供了 expectRecall。 */
  recallExpected: boolean;
  recallHit: boolean;
  durationMs: number;
  error?: string;
}

export interface LongRunImprovement {
  firstHalfRecall: number;
  secondHalfRecall: number;
  /** secondHalf - firstHalf；>0 表示「越用越好」。 */
  delta: number;
}

export interface LongRunSessionResult {
  scriptId: string;
  turns: LongRunTurnResult[];
  totalTokens: number;
  /** judged 轮里的通过率（无 judged 轮则为 0）。 */
  passRate: number;
  /** expectRecall 轮里的命中率（无该类轮则为 0）。 */
  recallRate: number;
  improvement: LongRunImprovement;
}

const containsCI = (haystack: string, needle: string): boolean =>
  haystack.toLowerCase().includes(needle.trim().toLowerCase());

/** 顺序跑完一个脚本，逐轮记录命中 / token / judge，并算改进曲线。 */
export const runLongRunSession = async (
  agent: LongRunAgent,
  script: LongRunScript,
): Promise<LongRunSessionResult> => {
  const turns: LongRunTurnResult[] = [];
  for (let i = 0; i < script.turns.length; i++) {
    const turn = script.turns[i]!;
    const t0 = Date.now();
    try {
      const out = await agent(turn.query);
      const passed = turn.judge ? await turn.judge(out.finalAnswer) : false;
      const recallExpected = turn.expectRecall !== undefined;
      let recallHit = false;
      if (recallExpected) {
        const haystack = out.injectedContext ?? out.finalAnswer;
        recallHit = containsCI(haystack, turn.expectRecall as string);
      }
      turns.push({
        index: i,
        query: turn.query,
        answer: out.finalAnswer,
        tokensSpent: out.tokensSpent,
        judged: turn.judge !== undefined,
        passed,
        recallExpected,
        recallHit,
        durationMs: Date.now() - t0,
      });
    } catch (e) {
      turns.push({
        index: i,
        query: turn.query,
        answer: "",
        tokensSpent: 0,
        judged: turn.judge !== undefined,
        passed: false,
        recallExpected: turn.expectRecall !== undefined,
        recallHit: false,
        durationMs: Date.now() - t0,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  const judgedTurns = turns.filter((t) => t.judged);
  const recallTurns = turns.filter((t) => t.recallExpected);
  const passRate =
    judgedTurns.length > 0 ? judgedTurns.filter((t) => t.passed).length / judgedTurns.length : 0;
  const recallRate =
    recallTurns.length > 0 ? recallTurns.filter((t) => t.recallHit).length / recallTurns.length : 0;

  return {
    scriptId: script.id,
    turns,
    totalTokens: turns.reduce((s, t) => s + t.tokensSpent, 0),
    passRate,
    recallRate,
    improvement: computeImprovement(recallTurns),
  };
};

/** 把 expectRecall 轮按时序对半切，比较后半 vs 前半命中率（含中点向上取整给后半）。 */
const computeImprovement = (recallTurns: readonly LongRunTurnResult[]): LongRunImprovement => {
  if (recallTurns.length < 2) {
    return { firstHalfRecall: 0, secondHalfRecall: 0, delta: 0 };
  }
  const mid = Math.floor(recallTurns.length / 2);
  const first = recallTurns.slice(0, mid);
  const second = recallTurns.slice(mid);
  const rate = (xs: readonly LongRunTurnResult[]): number =>
    xs.length > 0 ? xs.filter((t) => t.recallHit).length / xs.length : 0;
  const firstHalfRecall = rate(first);
  const secondHalfRecall = rate(second);
  return {
    firstHalfRecall,
    secondHalfRecall,
    delta: secondHalfRecall - firstHalfRecall,
  };
};

export interface LongRunVariantResult {
  variant: string;
  session: LongRunSessionResult;
  score: number;
}

export interface LongRunAbReport {
  scriptId: string;
  variants: LongRunVariantResult[];
  /** 得分最高的变体名。 */
  winner: string;
}

/**
 * 默认打分：召回率为主，通过率次之，再对 token 做轻惩罚。
 * 越「越用越好 + 完成度高 + 省 token」得分越高。
 */
export const defaultLongRunScore = (s: LongRunSessionResult): number => {
  const tokenPenalty = s.totalTokens / 100000; // 10万 token ≈ 扣 1 分
  return s.recallRate * 2 + s.passRate - tokenPenalty;
};

/** 把会话结果格式化成单行汇总（仿 retrieval-benchmark 风格）。 */
export const formatLongRunRow = (s: LongRunSessionResult): string => {
  const pct = (x: number): string => `${(x * 100).toFixed(1)}%`;
  return (
    `[longrun] ${s.scriptId.padEnd(18)} ` +
    `recall=${pct(s.recallRate)}  pass=${pct(s.passRate)}  ` +
    `tokens=${s.totalTokens}  ` +
    `improve(Δrecall)=${s.improvement.delta >= 0 ? "+" : ""}${pct(s.improvement.delta)} ` +
    `(${pct(s.improvement.firstHalfRecall)}→${pct(s.improvement.secondHalfRecall)})`
  );
};

/** 逐轮明细表（控制台用）。 */
export const formatLongRunTurns = (s: LongRunSessionResult): string => {
  const header = "idx | recall | pass | tokens | query";
  const rows = s.turns.map((t) => {
    const recall = t.recallExpected ? (t.recallHit ? "HIT " : "miss") : "—   ";
    const pass = t.judged ? (t.passed ? "ok " : "no ") : "—  ";
    const q = t.query.length > 40 ? `${t.query.slice(0, 39)}…` : t.query;
    return `${String(t.index).padStart(3)} | ${recall}   | ${pass}  | ${String(t.tokensSpent).padStart(6)} | ${q}`;
  });
  return [header, ...rows].join("\n");
};

/** A/B 报告的可读汇总（每变体一行 + winner）。 */
export const formatLongRunAb = (report: LongRunAbReport): string => {
  const lines = report.variants.map(
    (v) => `  ${v.variant.padEnd(16)} score=${v.score.toFixed(3)}  ${formatLongRunRow(v.session)}`,
  );
  return [`[longrun-ab] script=${report.scriptId}  winner=${report.winner}`, ...lines].join("\n");
};

/**
 * 对多个变体（如 memory-on / memory-off、classifier-on / off）跑同一脚本并打分对比。
 * 每个变体是一个独立的 LongRunAgent 工厂（保证状态隔离）。
 */
export const runLongRunAb = async (
  variants: Record<string, () => LongRunAgent>,
  script: LongRunScript,
  score: (s: LongRunSessionResult) => number = defaultLongRunScore,
): Promise<LongRunAbReport> => {
  const results: LongRunVariantResult[] = [];
  for (const [variant, makeAgent] of Object.entries(variants)) {
    const session = await runLongRunSession(makeAgent(), script);
    results.push({ variant, session, score: score(session) });
  }
  results.sort((a, b) => b.score - a.score);
  return {
    scriptId: script.id,
    variants: results,
    winner: results[0]?.variant ?? "",
  };
};
