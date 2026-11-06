import type { ChannelEvent, ChannelEventKind } from "./types.js";

interface SchedulerEventLike {
  kind: string;
  job: { id: string; title: string };
  at: number;
  error?: string;
  lateMs?: number;
}

const KIND_MAP: Record<string, ChannelEventKind> = {
  fired: "schedule.fired",
  completed: "schedule.completed",
  failed: "schedule.failed",
  deferred: "schedule.deferred",
};

/** Map a scheduler lifecycle event to an outbound channel event (RFC-018). */
export const channelEventFromScheduler = (event: SchedulerEventLike): ChannelEvent => {
  const kind = KIND_MAP[event.kind] ?? "schedule.fired";
  const title = event.job.title;
  const body =
    kind === "schedule.completed"
      ? `定时任务「${title}」已完成。`
      : kind === "schedule.failed"
        ? `定时任务「${title}」执行失败${event.error ? `：${event.error}` : ""}。`
        : kind === "schedule.deferred"
          ? `定时任务「${title}」因安静时段已顺延。`
          : `定时任务「${title}」已触发。`;
  return {
    kind,
    title,
    body,
    at: event.at,
    data: {
      jobId: event.job.id,
      ...(event.lateMs !== undefined ? { lateMs: event.lateMs } : {}),
    },
  };
};
