import type { QuietHours, ScheduleTrigger } from "./types.js";

export const TIME_OF_DAY_RE = /^([01]\d|2[0-3]):[0-5]\d$/;

export const parseTimeOfDay = (value: string): { hours: number; minutes: number } => {
  const match = TIME_OF_DAY_RE.exec(value);
  if (!match) throw new Error(`invalid time of day: ${value}`);
  return { hours: Number(match[1]), minutes: Number(value.slice(3, 5)) };
};

const atLocalTime = (fromMs: number, hours: number, minutes: number, dayOffset = 0): number => {
  const d = new Date(fromMs);
  d.setDate(d.getDate() + dayOffset);
  d.setHours(hours, minutes, 0, 0);
  return d.getTime();
};

/**
 * Next fire time strictly after `fromMs`. `anchorMs` is the job creation time
 * (interval anchor when the trigger has no explicit startAt). Returns
 * undefined when the trigger can never fire again (e.g. a past `once`).
 */
export const computeNextRunAt = (
  trigger: ScheduleTrigger,
  fromMs: number,
  anchorMs: number,
): number | undefined => {
  switch (trigger.kind) {
    case "once":
      return trigger.at > fromMs ? trigger.at : undefined;
    case "interval": {
      const anchor = trigger.startAt ?? anchorMs;
      if (fromMs < anchor) return anchor;
      const elapsed = Math.floor((fromMs - anchor) / trigger.everyMs);
      return anchor + (elapsed + 1) * trigger.everyMs;
    }
    case "daily": {
      const { hours, minutes } = parseTimeOfDay(trigger.time);
      const today = atLocalTime(fromMs, hours, minutes);
      return today > fromMs ? today : atLocalTime(fromMs, hours, minutes, 1);
    }
  }
};

const minutesOfDay = (ms: number): number => {
  const d = new Date(ms);
  return d.getHours() * 60 + d.getMinutes();
};

const timeToMinutes = (value: string): number => {
  const { hours, minutes } = parseTimeOfDay(value);
  return hours * 60 + minutes;
};

export const isInQuietHours = (nowMs: number, quiet: QuietHours): boolean => {
  const start = timeToMinutes(quiet.start);
  const end = timeToMinutes(quiet.end);
  const cur = minutesOfDay(nowMs);
  if (start === end) return false;
  if (start < end) return cur >= start && cur < end;
  return cur >= start || cur < end; // overnight window, e.g. 22:00–07:00
};

/** End of the current quiet window (epoch ms); undefined when not quiet. */
export const quietHoursEndAt = (nowMs: number, quiet: QuietHours): number | undefined => {
  if (!isInQuietHours(nowMs, quiet)) return undefined;
  const { hours, minutes } = parseTimeOfDay(quiet.end);
  const start = timeToMinutes(quiet.start);
  const end = timeToMinutes(quiet.end);
  const cur = minutesOfDay(nowMs);
  // Overnight window and we are in the evening part → ends tomorrow morning.
  const dayOffset = start > end && cur >= start ? 1 : 0;
  return atLocalTime(nowMs, hours, minutes, dayOffset);
};
