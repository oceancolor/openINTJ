import { randomUUID } from "node:crypto";
import { computeNextRunAt, isInQuietHours, parseTimeOfDay, quietHoursEndAt } from "./engine.js";
import { InMemoryScheduleStore, type ScheduleStore } from "./store.js";
import type {
  QuietHours,
  ScheduleRunner,
  ScheduledJob,
  ScheduledJobInput,
  ScheduledJobPatch,
  SchedulerEvent,
} from "./types.js";

export interface SchedulerServiceOpts {
  now?: () => number;
  tickMs?: number;
  quietHours?: QuietHours;
  /** Retries after the first failure (default 2). */
  maxRetries?: number;
  /** First retry delay; doubles each attempt (default 60s). */
  retryDelayMs?: number;
  /** When false, start() is a no-op (CRUD still works). Default true. */
  enabled?: boolean;
}

const validateTrigger = (job: ScheduledJobInput | ScheduledJobPatch): void => {
  const trigger = job.trigger;
  if (!trigger) return;
  if (trigger.kind === "interval") {
    if (!Number.isFinite(trigger.everyMs) || trigger.everyMs < 1_000) {
      throw new Error("interval.everyMs must be >= 1000");
    }
  }
  if (trigger.kind === "daily") parseTimeOfDay(trigger.time);
  if (trigger.kind === "once" && !Number.isFinite(trigger.at)) {
    throw new Error("once.at must be a finite timestamp");
  }
};

/**
 * RFC-015 scheduler. Persistent jobs are fired through an attached runner
 * (hosts wire `handle.run`); missed triggers coalesce into one catch-up run.
 */
export class SchedulerService {
  readonly store: ScheduleStore;
  private readonly now: () => number;
  private readonly tickMs: number;
  private readonly quietHours?: QuietHours;
  private readonly maxRetries: number;
  private readonly retryDelayMs: number;
  private readonly enabledOpt: boolean;
  private runner?: ScheduleRunner;
  private timer: ReturnType<typeof setInterval> | undefined;
  private readonly listeners = new Set<(event: SchedulerEvent) => void>();
  private readonly running = new Set<string>();
  private readonly attempts = new Map<string, number>();
  private ticking = false;

  constructor(store: ScheduleStore = new InMemoryScheduleStore(), opts: SchedulerServiceOpts = {}) {
    this.store = store;
    this.now = opts.now ?? (() => Date.now());
    this.tickMs = opts.tickMs ?? 10_000;
    if (opts.quietHours) this.quietHours = opts.quietHours;
    this.maxRetries = opts.maxRetries ?? 2;
    this.retryDelayMs = opts.retryDelayMs ?? 60_000;
    this.enabledOpt = opts.enabled ?? true;
  }

  on(listener: (event: SchedulerEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  attachRunner(runner: ScheduleRunner): void {
    this.runner = runner;
  }

  private emit(event: SchedulerEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch {
        // Listener failures must never break the engine.
      }
    }
  }

  list(): readonly ScheduledJob[] {
    return [...this.store.list()].sort(
      (a, b) => (a.nextRunAt ?? Number.MAX_SAFE_INTEGER) - (b.nextRunAt ?? Number.MAX_SAFE_INTEGER),
    );
  }

  get(id: string): ScheduledJob | undefined {
    return this.store.list().find((job) => job.id === id);
  }

  add(input: ScheduledJobInput): ScheduledJob {
    if (!input.title.trim()) throw new Error("title must not be empty");
    if (!input.prompt.trim()) throw new Error("prompt must not be empty");
    validateTrigger(input);
    if (input.maxRuns !== undefined && (!Number.isInteger(input.maxRuns) || input.maxRuns < 1)) {
      throw new Error("maxRuns must be a positive integer");
    }
    const now = this.now();
    const job: ScheduledJob = {
      id: randomUUID(),
      title: input.title,
      prompt: input.prompt,
      ...(input.conversationId ? { conversationId: input.conversationId } : {}),
      trigger: input.trigger,
      enabled: input.enabled ?? true,
      ...(input.maxRuns ? { maxRuns: input.maxRuns } : {}),
      createdAt: now,
      runCount: 0,
    };
    this.recomputeNextRunAt(job);
    this.store.upsert(job);
    return job;
  }

  private recomputeNextRunAt(job: ScheduledJob): void {
    const next =
      job.trigger.kind === "once"
        ? job.trigger.at
        : computeNextRunAt(job.trigger, this.now(), job.createdAt);
    if (next === undefined) delete job.nextRunAt;
    else job.nextRunAt = next;
  }

  update(id: string, patch: ScheduledJobPatch): ScheduledJob | undefined {
    const job = this.get(id);
    if (!job) return undefined;
    validateTrigger(patch);
    if (patch.title !== undefined) job.title = patch.title;
    if (patch.prompt !== undefined) job.prompt = patch.prompt;
    if (patch.conversationId !== undefined) job.conversationId = patch.conversationId;
    if (patch.maxRuns !== undefined) job.maxRuns = patch.maxRuns;
    if (patch.trigger !== undefined) job.trigger = patch.trigger;
    if (patch.enabled !== undefined) job.enabled = patch.enabled;
    if (patch.trigger !== undefined || patch.enabled === true) {
      this.recomputeNextRunAt(job);
    }
    this.store.upsert(job);
    return job;
  }

  remove(id: string): boolean {
    this.attempts.delete(id);
    return this.store.remove(id);
  }

  setEnabled(id: string, enabled: boolean): boolean {
    return this.update(id, { enabled }) !== undefined;
  }

  start(): void {
    if (!this.enabledOpt || this.timer) return;
    void this.tick(); // catch-up pass for anything missed while offline
    this.timer = setInterval(() => void this.tick(), this.tickMs);
    (this.timer as { unref?: () => void }).unref?.();
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = undefined;
  }

  get started(): boolean {
    return this.timer !== undefined;
  }

  /** One engine pass: fire every due job once, serially. Safe to call anytime. */
  async tick(): Promise<void> {
    if (this.ticking || !this.runner) return;
    this.ticking = true;
    try {
      const now = this.now();
      const due = this.list().filter(
        (job) =>
          job.enabled &&
          job.nextRunAt !== undefined &&
          job.nextRunAt <= now &&
          !this.running.has(job.id),
      );
      for (const snapshot of due) {
        const job = this.get(snapshot.id);
        if (!job || !job.enabled || job.nextRunAt === undefined || job.nextRunAt > this.now()) {
          continue;
        }
        if (this.quietHours && isInQuietHours(now, this.quietHours)) {
          const end = quietHoursEndAt(now, this.quietHours);
          if (end !== undefined && end > job.nextRunAt) {
            job.nextRunAt = end;
            this.store.upsert(job);
            this.emit({ kind: "deferred", job, at: now });
            continue;
          }
        }
        await this.fire(job);
      }
    } finally {
      this.ticking = false;
    }
  }

  private advanceAfterRun(job: ScheduledJob, succeeded: boolean): void {
    if (job.trigger.kind === "once") {
      job.enabled = false;
      delete job.nextRunAt;
      return;
    }
    if (succeeded && job.maxRuns !== undefined && job.runCount >= job.maxRuns) {
      job.enabled = false;
      delete job.nextRunAt;
      return;
    }
    const next = computeNextRunAt(job.trigger, this.now(), job.createdAt);
    if (next === undefined) {
      job.enabled = false;
      delete job.nextRunAt;
    } else {
      job.nextRunAt = next;
    }
  }

  private async fire(job: ScheduledJob): Promise<void> {
    const runner = this.runner;
    if (!runner || job.nextRunAt === undefined) return;
    this.running.add(job.id);
    const scheduledAt = job.nextRunAt;
    const firedAt = this.now();
    this.emit({ kind: "fired", job, at: firedAt, lateMs: Math.max(0, firedAt - scheduledAt) });
    try {
      await runner(job);
      job.lastRunAt = firedAt;
      job.runCount += 1;
      job.lastStatus = "completed";
      delete job.lastError;
      this.attempts.delete(job.id);
      this.advanceAfterRun(job, true);
      this.store.upsert(job);
      this.emit({ kind: "completed", job, at: this.now() });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      job.lastRunAt = firedAt;
      job.lastStatus = "failed";
      job.lastError = message;
      const attempts = (this.attempts.get(job.id) ?? 0) + 1;
      if (attempts <= this.maxRetries) {
        this.attempts.set(job.id, attempts);
        job.nextRunAt = this.now() + this.retryDelayMs * 2 ** (attempts - 1);
        this.store.upsert(job);
        this.emit({ kind: "failed", job, at: this.now(), error: message, willRetry: true });
      } else {
        this.attempts.delete(job.id);
        this.advanceAfterRun(job, false);
        this.store.upsert(job);
        this.emit({ kind: "failed", job, at: this.now(), error: message, willRetry: false });
      }
    } finally {
      this.running.delete(job.id);
    }
  }
}
