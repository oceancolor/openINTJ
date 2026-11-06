import type { TaskGraph } from "./plan-graph-adapter.js";
import type { TaskRunState } from "./task-pool.js";

export interface StoredTaskNode {
  taskId: string;
  state: TaskRunState;
  attempt: number;
  result?: unknown;
  error?: string;
  updatedAt: number;
}

export interface StoredTaskRun {
  runId: string;
  planId: string;
  status: "running" | "completed" | "failed" | "cancelled";
  graph: TaskGraph;
  nodes: readonly StoredTaskNode[];
  createdAt: number;
  updatedAt: number;
}

/**
 * Durable boundary for orchestration snapshots. Implementations must be
 * idempotent: TaskPool persists the complete run snapshot after transitions.
 */
export interface TaskStore {
  saveRun(run: StoredTaskRun): Promise<void>;
  loadRun(runId: string): Promise<StoredTaskRun | undefined>;
  listIncompleteRuns(): Promise<readonly StoredTaskRun[]>;
  /** Most recently updated runs first; `limit` is clamped to 1..200. */
  listRuns(limit?: number): Promise<readonly StoredTaskRun[]>;
}

export const normalizeTaskRunListLimit = (limit = 50): number =>
  Math.min(200, Math.max(1, Math.floor(Number.isFinite(limit) ? limit : 50)));

export class MemoryTaskStore implements TaskStore {
  private readonly runs = new Map<string, StoredTaskRun>();

  async saveRun(run: StoredTaskRun): Promise<void> {
    this.runs.set(run.runId, structuredClone(run));
  }

  async loadRun(runId: string): Promise<StoredTaskRun | undefined> {
    const run = this.runs.get(runId);
    return run ? structuredClone(run) : undefined;
  }

  async listIncompleteRuns(): Promise<readonly StoredTaskRun[]> {
    return [...this.runs.values()]
      .filter((run) => run.status === "running")
      .sort((a, b) => a.createdAt - b.createdAt || a.runId.localeCompare(b.runId))
      .map((run) => structuredClone(run));
  }

  async listRuns(limit = 50): Promise<readonly StoredTaskRun[]> {
    return [...this.runs.values()]
      .sort(
        (a, b) =>
          b.updatedAt - a.updatedAt || b.createdAt - a.createdAt || a.runId.localeCompare(b.runId),
      )
      .slice(0, normalizeTaskRunListLimit(limit))
      .map((run) => structuredClone(run));
  }
}
