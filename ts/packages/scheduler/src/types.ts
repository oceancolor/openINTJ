export type ScheduleTrigger =
  | { kind: "once"; at: number }
  | { kind: "interval"; everyMs: number; startAt?: number }
  | { kind: "daily"; time: string }; // local "HH:mm"

export interface ScheduledJob {
  id: string;
  title: string;
  prompt: string;
  conversationId?: string;
  trigger: ScheduleTrigger;
  enabled: boolean;
  maxRuns?: number;
  createdAt: number;
  lastRunAt?: number;
  nextRunAt?: number;
  runCount: number;
  lastStatus?: "completed" | "failed";
  lastError?: string;
}

export interface ScheduledJobInput {
  title: string;
  prompt: string;
  conversationId?: string;
  trigger: ScheduleTrigger;
  enabled?: boolean;
  maxRuns?: number;
}

export interface ScheduledJobPatch {
  title?: string;
  prompt?: string;
  conversationId?: string;
  trigger?: ScheduleTrigger;
  enabled?: boolean;
  maxRuns?: number;
}

export type SchedulerEventKind = "fired" | "completed" | "failed" | "deferred";

export interface SchedulerEvent {
  kind: SchedulerEventKind;
  job: ScheduledJob;
  at: number;
  /** How late the fire was versus nextRunAt (catch-up indicator). */
  lateMs?: number;
  error?: string;
  willRetry?: boolean;
}

/** Host-supplied executor; typically `handle.run(job.prompt)`. Throws on failure. */
export type ScheduleRunner = (job: ScheduledJob) => Promise<void>;

export interface QuietHours {
  start: string; // "HH:mm" local
  end: string; // "HH:mm" local
}
