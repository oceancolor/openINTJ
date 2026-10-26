import type { AnnotatedPlanStep } from "./knowledge.js";

export interface PathSearchResult {
  /** Ordered step ids from Dijkstra (exact on small DAGs). */
  dijkstraOrder: string[];
  /** Ordered step ids from simulated annealing (tie-break / larger graphs). */
  annealingOrder: string[];
  /** First action to take: a plan step id. */
  nextStepId: string;
  pathCost: number;
  method: "dijkstra" | "annealing";
}

const UNKNOWN_WHILE_KNOWN_READY = 500;
const CLARIFY_COST = 1;

export interface PathSearchRng {
  next(): number;
}

export const mulberry32 = (seed: number): PathSearchRng => {
  let a = seed >>> 0;
  return {
    next: () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
};

const readyIds = (annotated: readonly AnnotatedPlanStep[], done: number): string[] => {
  const byId = new Map(annotated.map((item, i) => [item.step.stepId, i]));
  return annotated
    .filter((item, i) => {
      if ((done & (1 << i)) !== 0) return false;
      return item.step.dependencies.every((dep) => {
        const idx = byId.get(dep);
        return idx === undefined || (done & (1 << idx)) !== 0;
      });
    })
    .map((item) => item.step.stepId);
};

const transitionCost = (
  item: AnnotatedPlanStep,
  annotated: readonly AnnotatedPlanStep[],
  done: number,
): number => {
  const ready = readyIds(annotated, done);
  const knownReady = ready.some((id) => {
    const other = annotated.find((a) => a.step.stepId === id);
    return other?.known === true;
  });
  let cost = item.cost;
  if (!item.known) {
    cost += CLARIFY_COST;
    if (knownReady) cost += UNKNOWN_WHILE_KNOWN_READY;
  }
  return cost;
};

/**
 * Shortest schedule over the subset lattice (n ≤ 16). Prefers completing known
 * ready work before paying to clarify an unknown step.
 */
export const dijkstraSchedule = (annotated: readonly AnnotatedPlanStep[]): PathSearchResult => {
  const n = annotated.length;
  if (n === 0) {
    return {
      dijkstraOrder: [],
      annealingOrder: [],
      nextStepId: "",
      pathCost: 0,
      method: "dijkstra",
    };
  }
  const goal = (1 << n) - 1;
  const dist = new Array<number>(1 << n).fill(Number.POSITIVE_INFINITY);
  const prev = new Array<{ mask: number; stepIndex: number } | undefined>(1 << n);
  dist[0] = 0;
  const heap: number[] = [0];
  const less = (a: number, b: number): boolean => dist[a]! < dist[b]!;

  while (heap.length > 0) {
    heap.sort((a, b) => (less(a, b) ? -1 : 1));
    const mask = heap.shift()!;
    if (mask === goal) break;
    const ready = readyIds(annotated, mask);
    for (const id of ready) {
      const idx = annotated.findIndex((item) => item.step.stepId === id);
      if (idx < 0) continue;
      const next = mask | (1 << idx);
      if (next === mask) continue;
      const cost = dist[mask]! + transitionCost(annotated[idx]!, annotated, mask);
      if (cost < dist[next]!) {
        dist[next] = cost;
        prev[next] = { mask, stepIndex: idx };
        heap.push(next);
      }
    }
  }

  const order: string[] = [];
  let cursor = goal;
  while (cursor !== 0 && prev[cursor]) {
    const step = prev[cursor]!;
    order.push(annotated[step.stepIndex]!.step.stepId);
    cursor = step.mask;
  }
  order.reverse();
  return {
    dijkstraOrder: order,
    annealingOrder: order,
    nextStepId: order[0] ?? annotated[0]!.step.stepId,
    pathCost: dist[goal] ?? 0,
    method: "dijkstra",
  };
};

const topologicalPermutationsSeed = (
  annotated: readonly AnnotatedPlanStep[],
  rng: PathSearchRng,
): string[] => {
  const done = new Set<string>();
  const order: string[] = [];
  const remaining = [...annotated];
  while (order.length < annotated.length) {
    const ready = remaining.filter(
      (item) => !done.has(item.step.stepId) && item.step.dependencies.every((dep) => done.has(dep)),
    );
    if (ready.length === 0) break;
    const pick = ready[Math.floor(rng.next() * ready.length)]!;
    order.push(pick.step.stepId);
    done.add(pick.step.stepId);
  }
  return order;
};

const scheduleEnergy = (annotated: readonly AnnotatedPlanStep[], order: string[]): number => {
  const byId = new Map(annotated.map((item) => [item.step.stepId, item]));
  let energy = 0;
  const completed = new Set<string>();
  for (const id of order) {
    const item = byId.get(id);
    if (!item) continue;
    let mask = 0;
    annotated.forEach((step, i) => {
      if (completed.has(step.step.stepId)) mask |= 1 << i;
    });
    energy += transitionCost(item, annotated, mask);
    completed.add(id);
  }
  return energy;
};

const neighborSwap = (order: string[], rng: PathSearchRng): string[] => {
  if (order.length < 2) return [...order];
  const next = [...order];
  const i = Math.floor(rng.next() * next.length);
  let j = Math.floor(rng.next() * next.length);
  if (j === i) j = (j + 1) % next.length;
  const a = next[i]!;
  next[i] = next[j]!;
  next[j] = a;
  return next;
};

const isValidTopo = (annotated: readonly AnnotatedPlanStep[], order: string[]): boolean => {
  const index = new Map(order.map((id, i) => [id, i]));
  for (const item of annotated) {
    const at = index.get(item.step.stepId);
    if (at === undefined) return false;
    for (const dep of item.step.dependencies) {
      const depAt = index.get(dep);
      if (depAt !== undefined && depAt >= at) return false;
    }
  }
  return true;
};

/**
 * Simulated annealing over topological orders. Used as a cross-check / fallback
 * when the DAG is larger than the bitmask Dijkstra limit.
 */
export const annealSchedule = (
  annotated: readonly AnnotatedPlanStep[],
  opts: { seed?: number; steps?: number } = {},
): PathSearchResult => {
  if (annotated.length === 0) {
    return {
      dijkstraOrder: [],
      annealingOrder: [],
      nextStepId: "",
      pathCost: 0,
      method: "annealing",
    };
  }
  const rng = mulberry32(opts.seed ?? 20261025);
  const steps = opts.steps ?? 200;
  let current = topologicalPermutationsSeed(annotated, rng);
  if (!isValidTopo(annotated, current)) {
    current = dijkstraSchedule(annotated).dijkstraOrder;
  }
  let best = current;
  let currentE = scheduleEnergy(annotated, current);
  let bestE = currentE;
  for (let t = 0; t < steps; t++) {
    const temp = 1 - t / steps;
    let cand = neighborSwap(current, rng);
    if (!isValidTopo(annotated, cand)) cand = current;
    const e = scheduleEnergy(annotated, cand);
    const accept = e <= currentE || rng.next() < Math.exp((currentE - e) / Math.max(temp, 1e-6));
    if (accept) {
      current = cand;
      currentE = e;
      if (e < bestE) {
        best = cand;
        bestE = e;
      }
    }
  }
  return {
    dijkstraOrder: best,
    annealingOrder: best,
    nextStepId: best[0] ?? annotated[0]!.step.stepId,
    pathCost: bestE,
    method: "annealing",
  };
};

export const evaluatePlanPath = (annotated: readonly AnnotatedPlanStep[]): PathSearchResult => {
  if (annotated.length === 0 || annotated.length > 16) {
    const annealed = annealSchedule(annotated);
    return { ...annealed, method: "annealing" };
  }
  const exact = dijkstraSchedule(annotated);
  const annealed = annealSchedule(annotated);
  return {
    ...exact,
    annealingOrder: annealed.annealingOrder,
    method: "dijkstra",
  };
};
