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
}

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
}
