import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { ScheduledJob } from "./types.js";

export interface ScheduleStore {
  list(): readonly ScheduledJob[];
  upsert(job: ScheduledJob): void;
  remove(id: string): boolean;
  replaceAll(jobs: readonly ScheduledJob[]): void;
}

export class InMemoryScheduleStore implements ScheduleStore {
  private jobs: ScheduledJob[] = [];

  list(): readonly ScheduledJob[] {
    return [...this.jobs];
  }

  upsert(job: ScheduledJob): void {
    const idx = this.jobs.findIndex((item) => item.id === job.id);
    if (idx >= 0) this.jobs[idx] = job;
    else this.jobs.push(job);
  }

  remove(id: string): boolean {
    const before = this.jobs.length;
    this.jobs = this.jobs.filter((item) => item.id !== id);
    return this.jobs.length !== before;
  }

  replaceAll(jobs: readonly ScheduledJob[]): void {
    this.jobs = [...jobs];
  }
}

/** Single-file JSON store (<dataDir>/schedules.json). Whole-file synchronous writes. */
export class JsonScheduleStore implements ScheduleStore {
  private readonly inner = new InMemoryScheduleStore();

  constructor(private readonly filePath: string) {
    if (existsSync(filePath)) {
      try {
        const parsed = JSON.parse(readFileSync(filePath, "utf8")) as { jobs?: ScheduledJob[] };
        if (Array.isArray(parsed.jobs)) this.inner.replaceAll(parsed.jobs);
      } catch {
        // A corrupt schedule file must never block agent startup; start empty.
      }
    }
  }

  private flush(): void {
    mkdirSync(dirname(this.filePath), { recursive: true });
    writeFileSync(
      this.filePath,
      JSON.stringify({ version: 1, jobs: this.inner.list() }, null, 2),
      "utf8",
    );
  }

  list(): readonly ScheduledJob[] {
    return this.inner.list();
  }

  upsert(job: ScheduledJob): void {
    this.inner.upsert(job);
    this.flush();
  }

  remove(id: string): boolean {
    const removed = this.inner.remove(id);
    if (removed) this.flush();
    return removed;
  }

  replaceAll(jobs: readonly ScheduledJob[]): void {
    this.inner.replaceAll(jobs);
    this.flush();
  }
}
