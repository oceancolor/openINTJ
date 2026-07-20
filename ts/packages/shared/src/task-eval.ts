/**
 * 端到端"任务完成度"评测脚手架。
 *
 * 给一组任务（query + 判定函数），驱动任意 `run(query) → { finalAnswer }` 的 Agent，
 * 逐任务判定通过与否，聚合出完成率。是把"Agent 到底有没有把活干完"量化的最小骨架：
 *  - 纯编排、零依赖；judge 由调用方提供（关键词命中 / 正则 / 调用真实评审 LLM 均可）。
 *  - 确定性可测（注入桩 runner）；接真实 Agent 时通常 gated（需 LLM key）。
 */

export interface TaskCase {
  id: string;
  query: string;
  /** 判定 Agent 输出是否完成任务；第二参数可读取可选结构化运行证据。 */
  judge: (answer: string, output?: RunnerOutput) => boolean | Promise<boolean>;
  /** 可选：人类可读的期望描述（仅用于报告）。 */
  expectation?: string;
}

export interface TaskEvalResult {
  id: string;
  query: string;
  answer: string;
  passed: boolean;
  /** 单任务耗时（ms）。 */
  durationMs: number;
  /** run 抛错时记录（视为不通过）。 */
  error?: string;
  /** runner 提供的结构化证据，便于报告/调试；不提供时保持旧行为。 */
  evidence?: RunEvidence;
}

export interface TaskEvalReport {
  results: TaskEvalResult[];
  passed: number;
  total: number;
  /** 完成率 0-1。 */
  completionRate: number;
}

export interface RunEvidence {
  /** 规范化工具调用顺序；T3 等 judge 可据此验证真实工具使用。 */
  toolsUsed?: readonly string[];
  /** 搜索结果是否包含可核验来源；区分真实 provider 与 mock/失败兜底。 */
  searchEvidence?: "none" | "unavailable" | "reliable";
  /** 原始或精简 trajectory，供调用方自定义 judge。 */
  trajectory?: readonly unknown[];
}

export interface RunnerOutput {
  finalAnswer: string;
  evidence?: RunEvidence;
}

/**
 * 顺序跑完所有任务（默认）。如需并行，调用方可自行用 forkJoin 包 run。
 * run 抛错不会中断整个套件——该任务记为失败并继续，保证"评测套件总能跑完出报告"。
 */
export const evaluateTasks = async (
  tasks: readonly TaskCase[],
  run: (query: string) => Promise<RunnerOutput> | RunnerOutput,
): Promise<TaskEvalReport> => {
  const results: TaskEvalResult[] = [];
  for (const t of tasks) {
    const t0 = Date.now();
    try {
      const out = await run(t.query);
      const passed = await t.judge(out.finalAnswer, out);
      results.push({
        id: t.id,
        query: t.query,
        answer: out.finalAnswer,
        passed,
        durationMs: Date.now() - t0,
        ...(out.evidence ? { evidence: out.evidence } : {}),
      });
    } catch (e) {
      results.push({
        id: t.id,
        query: t.query,
        answer: "",
        passed: false,
        durationMs: Date.now() - t0,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }
  const passed = results.filter((r) => r.passed).length;
  const total = results.length;
  return { results, passed, total, completionRate: total > 0 ? passed / total : 0 };
};

/** 便捷 judge：答案（不区分大小写）包含所有给定关键词。 */
export const judgeContainsAll =
  (...keywords: string[]) =>
  (answer: string): boolean => {
    const a = answer.toLowerCase();
    return keywords.every((k) => a.includes(k.toLowerCase()));
  };

/** 便捷 judge：答案非空（去空白后长度 > 0）。 */
export const judgeNonEmpty = (answer: string): boolean => answer.trim().length > 0;
