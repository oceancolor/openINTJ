/**
 * A/B 测试脚手架 —— RFC-003 方向三"越用越好"的长跑可观测验证骨架。
 *
 * 目标：在同一批 query 上对比两个（或多个）变体（如「persona 注入 ON」vs「OFF」），
 * 用一个评分函数量化每次回答，聚合出每变体的平均分 / 胜场，给出可解释的对比结论。
 *
 * 这是**纯编排**：不绑定具体 Agent，调用方注入 `run`（跑一次得到产物）与 `score`（打分）。
 * 便于在测试里用确定性桩验证骨架，也便于线上接真实 Agent + 人工/自动评分跑长跑。
 */

export interface AbVariant<Ctx, Out> {
  /** 变体名（如 "persona-on" / "persona-off"）。 */
  name: string;
  /** 跑一次：给定 query 与共享上下文，产出可评分的结果。 */
  run: (query: string, ctx: Ctx) => Promise<Out> | Out;
}

export interface AbTestOpts<Ctx, Out> {
  variants: ReadonlyArray<AbVariant<Ctx, Out>>;
  /** 评测用 query 集合。 */
  queries: readonly string[];
  /** 对单次产物打分（越高越好）。 */
  score: (out: Out, query: string, variantName: string) => number;
  /** 每个变体的共享上下文工厂（如各自独立的 Agent 实例）。默认 `undefined`。 */
  makeContext?: (variantName: string) => Ctx;
}

export interface AbVariantStat {
  name: string;
  trials: number;
  totalScore: number;
  meanScore: number;
  /** 在逐 query 对比中，本变体取得（并列）最高分的次数。 */
  wins: number;
}

export interface AbTestReport {
  perVariant: AbVariantStat[];
  /** 平均分最高的变体名；并列时取第一个。 */
  winner: string | undefined;
  /** 评测的 query 数。 */
  queryCount: number;
}

/**
 * 运行一次 A/B 评测。对每个 query，所有变体各跑一次并打分；
 * 逐 query 记 win（最高分者，允许并列），最后聚合。
 */
export const runAbTest = async <Ctx, Out>(
  opts: AbTestOpts<Ctx, Out>,
): Promise<AbTestReport> => {
  const stats = new Map<string, AbVariantStat>();
  const contexts = new Map<string, Ctx>();
  for (const v of opts.variants) {
    stats.set(v.name, { name: v.name, trials: 0, totalScore: 0, meanScore: 0, wins: 0 });
    contexts.set(v.name, opts.makeContext ? opts.makeContext(v.name) : (undefined as Ctx));
  }

  for (const query of opts.queries) {
    const roundScores: Array<{ name: string; score: number }> = [];
    for (const v of opts.variants) {
      const out = await v.run(query, contexts.get(v.name) as Ctx);
      const s = opts.score(out, query, v.name);
      const st = stats.get(v.name)!;
      st.trials += 1;
      st.totalScore += s;
      roundScores.push({ name: v.name, score: s });
    }
    // 本 query 的最高分（并列都记 win）。
    const top = Math.max(...roundScores.map((r) => r.score));
    for (const r of roundScores) {
      if (r.score === top) stats.get(r.name)!.wins += 1;
    }
  }

  const perVariant: AbVariantStat[] = [];
  for (const st of stats.values()) {
    st.meanScore = st.trials > 0 ? st.totalScore / st.trials : 0;
    perVariant.push(st);
  }

  let winner: AbVariantStat | undefined;
  for (const st of perVariant) {
    if (!winner || st.meanScore > winner.meanScore) winner = st;
  }

  return {
    perVariant,
    winner: winner?.name,
    queryCount: opts.queries.length,
  };
};
