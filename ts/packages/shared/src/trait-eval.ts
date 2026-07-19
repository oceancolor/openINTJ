/**
 * RFC-006 trait A/B 评测：在 task-eval 之上聚合 trait 通过率与基线对比。
 */

import { type RunnerOutput, type TaskEvalReport, evaluateTasks } from "./task-eval.js";
import { TRAIT_SCENARIOS, type TraitScenario } from "./trait-scenarios.js";

export interface TraitEvalResult {
  trait: string;
  passed: number;
  total: number;
  rate: number;
}

export interface TraitEvalReport extends TaskEvalReport {
  byTrait: TraitEvalResult[];
  /** RFC-006 首期基线：各 trait 最低通过率（可调）。 */
  baselineMet: boolean;
}

/** 默认 trait 通过率基线（首期 RFC 设定）。 */
export const TRAIT_PASS_BASELINE = 0.6;

export const evaluateTraits = async (
  run: (query: string) => Promise<RunnerOutput> | RunnerOutput,
  scenarios: readonly TraitScenario[] = TRAIT_SCENARIOS,
  baseline = TRAIT_PASS_BASELINE,
): Promise<TraitEvalReport> => {
  const expanded = scenarios.flatMap((scenario) => [
    scenario,
    ...(scenario.counterExample
      ? [
          {
            id: `${scenario.id}:contrast`,
            query: scenario.counterExample.query,
            judge: scenario.counterExample.judge,
            expectation: `对照：${scenario.expectation ?? scenario.trait}`,
          },
        ]
      : []),
  ]);
  const report = await evaluateTasks(expanded, run);
  const byTraitMap = new Map<string, { passed: number; total: number }>();
  for (const r of report.results) {
    const sourceId = r.id.endsWith(":contrast") ? r.id.slice(0, -":contrast".length) : r.id;
    const sc = scenarios.find((s) => s.id === sourceId);
    const trait = sc?.trait ?? "unknown";
    const cur = byTraitMap.get(trait) ?? { passed: 0, total: 0 };
    cur.total++;
    if (r.passed) cur.passed++;
    byTraitMap.set(trait, cur);
  }
  const byTrait: TraitEvalResult[] = [...byTraitMap.entries()].map(([trait, v]) => ({
    trait,
    passed: v.passed,
    total: v.total,
    rate: v.total > 0 ? v.passed / v.total : 0,
  }));
  const baselineMet = byTrait.every((t) => t.rate >= baseline || t.total === 0);
  return { ...report, byTrait, baselineMet };
};

export interface TraitAbReport {
  treatment: TraitEvalReport;
  control: TraitEvalReport;
  completionRateDelta: number;
}

/** 用同一场景比较 Product Behavior treatment 与 control。 */
export const evaluateTraitAb = async (
  treatmentRun: (query: string) => Promise<RunnerOutput> | RunnerOutput,
  controlRun: (query: string) => Promise<RunnerOutput> | RunnerOutput,
  scenarios: readonly TraitScenario[] = TRAIT_SCENARIOS,
): Promise<TraitAbReport> => {
  const [treatment, control] = await Promise.all([
    evaluateTraits(treatmentRun, scenarios),
    evaluateTraits(controlRun, scenarios),
  ]);
  return {
    treatment,
    control,
    completionRateDelta: treatment.completionRate - control.completionRate,
  };
};
