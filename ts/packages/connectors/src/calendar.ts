import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { Connector, ConnectorTool } from "./types.js";

export interface CalendarConnectorOpts {
  id?: string;
  /** Path to an .ics file. Exactly one of icsPath / icsText. */
  icsPath?: string;
  /** Inline ICS text (tests / pre-fetched content). */
  icsText?: string;
  /** Injectable clock for upcoming queries (default Date.now). */
  now?: () => number;
  /** RFC-017: expose create/delete write tools (requires icsPath; default off). */
  writable?: boolean;
}

export interface CalendarEvent {
  uid: string;
  summary: string;
  description?: string;
  location?: string;
  start: number;
  end?: number;
  allDay: boolean;
}

interface ParsedEvent extends CalendarEvent {
  rrule?: RRule;
}

interface RRule {
  freq: "DAILY" | "WEEKLY" | "MONTHLY" | "YEARLY";
  interval: number;
  count?: number;
  until?: number;
}

const FREQ_STEP_MS: Record<RRule["freq"], number> = {
  DAILY: 86_400_000,
  WEEKLY: 7 * 86_400_000,
  MONTHLY: 0, // handled via Date arithmetic
  YEARLY: 0,
};

/** Parse an ICS date/date-time value. DATE → local midnight; trailing Z → UTC; else local. */
export const parseIcsDate = (value: string): { ms: number; allDay: boolean } => {
  const dateOnly = /^(\d{4})(\d{2})(\d{2})$/.exec(value);
  if (dateOnly) {
    return {
      ms: new Date(Number(dateOnly[1]), Number(dateOnly[2]) - 1, Number(dateOnly[3])).getTime(),
      allDay: true,
    };
  }
  const dateTime = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(Z)?$/.exec(value);
  if (!dateTime) throw new Error(`unsupported ICS date value: ${value}`);
  const [, y, mo, d, h, mi, s, z] = dateTime as unknown as [
    string,
    string,
    string,
    string,
    string,
    string,
    string,
    string?,
  ];
  const ms = z
    ? Date.UTC(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi), Number(s))
    : new Date(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi), Number(s)).getTime();
  return { ms, allDay: false };
};

/** Unfold ICS content lines (continuation lines start with space/tab). */
const unfold = (text: string): string[] =>
  text
    .replace(/\r\n[ \t]/g, "")
    .replace(/\n[ \t]/g, "")
    .split(/\r\n|\n|\r/)
    .filter((line) => line.length > 0);

const unescapeText = (value: string): string =>
  value.replace(/\\n/g, "\n").replace(/\\,/g, ",").replace(/\\;/g, ";").replace(/\\\\/g, "\\");

export const parseIcs = (text: string): ParsedEvent[] => {
  const events: ParsedEvent[] = [];
  let current: Record<string, string> | undefined;
  for (const line of unfold(text)) {
    if (line === "BEGIN:VEVENT") {
      current = {};
      continue;
    }
    if (line === "END:VEVENT") {
      if (current) {
        const event = toEvent(current);
        if (event) events.push(event);
      }
      current = undefined;
      continue;
    }
    if (!current) continue;
    const colon = line.indexOf(":");
    if (colon < 0) continue;
    const key = line.slice(0, colon).split(";")[0]!.toUpperCase();
    current[key] = line.slice(colon + 1);
  }
  return events;
};

const parseRRule = (value: string, eventStart: number): RRule | undefined => {
  const parts = new Map<string, string>();
  for (const part of value.split(";")) {
    const eq = part.indexOf("=");
    if (eq > 0) parts.set(part.slice(0, eq).toUpperCase(), part.slice(eq + 1));
  }
  const freq = parts.get("FREQ")?.toUpperCase();
  if (freq !== "DAILY" && freq !== "WEEKLY" && freq !== "MONTHLY" && freq !== "YEARLY") {
    return undefined;
  }
  const rule: RRule = {
    freq,
    interval: Math.max(1, Number.parseInt(parts.get("INTERVAL") ?? "1", 10) || 1),
  };
  const count = Number.parseInt(parts.get("COUNT") ?? "", 10);
  if (Number.isFinite(count) && count > 0) rule.count = count;
  const untilRaw = parts.get("UNTIL");
  if (untilRaw) {
    try {
      rule.until = parseIcsDate(untilRaw).ms;
    } catch {
      // Ignore malformed UNTIL; fall back to window-bounded expansion.
    }
  }
  void eventStart;
  return rule;
};

const toEvent = (raw: Record<string, string>): ParsedEvent | undefined => {
  const dtstart = raw["DTSTART"];
  if (!dtstart) return undefined;
  let start: { ms: number; allDay: boolean };
  try {
    start = parseIcsDate(dtstart);
  } catch {
    return undefined;
  }
  const event: ParsedEvent = {
    uid: raw["UID"] ?? "",
    summary: unescapeText(raw["SUMMARY"] ?? "(无标题)"),
    start: start.ms,
    allDay: start.allDay,
  };
  if (raw["DESCRIPTION"]) event.description = unescapeText(raw["DESCRIPTION"]);
  if (raw["LOCATION"]) event.location = unescapeText(raw["LOCATION"]);
  if (raw["DTEND"]) {
    try {
      event.end = parseIcsDate(raw["DTEND"]).ms;
    } catch {
      // keep event without end
    }
  }
  if (raw["RRULE"]) {
    const rule = parseRRule(raw["RRULE"], start.ms);
    if (rule) event.rrule = rule;
  }
  return event;
};

const stepOccurrence = (startMs: number, rule: RRule, steps: number): number => {
  if (rule.freq === "MONTHLY" || rule.freq === "YEARLY") {
    const d = new Date(startMs);
    if (rule.freq === "MONTHLY") d.setMonth(d.getMonth() + rule.interval * steps);
    else d.setFullYear(d.getFullYear() + rule.interval * steps);
    return d.getTime();
  }
  return startMs + FREQ_STEP_MS[rule.freq] * rule.interval * steps;
};

/**
 * Expand an event into occurrences within [fromMs, toMs]. Non-recurring
 * events yield themselves when inside the window. Bounded to 500 iterations.
 */
export const expandEvent = (event: ParsedEvent, fromMs: number, toMs: number): CalendarEvent[] => {
  const { rrule, ...base } = event;
  if (!rrule) {
    return event.start >= fromMs && event.start <= toMs ? [base] : [];
  }
  const duration = event.end !== undefined ? event.end - event.start : 0;
  const out: CalendarEvent[] = [];
  for (let step = 0; step < 500; step += 1) {
    if (rrule.count !== undefined && step >= rrule.count) break;
    const occurrence = stepOccurrence(event.start, rrule, step);
    if (rrule.until !== undefined && occurrence > rrule.until) break;
    if (occurrence > toMs) break;
    if (occurrence >= fromMs) {
      out.push({
        ...base,
        start: occurrence,
        ...(event.end !== undefined ? { end: occurrence + duration } : {}),
      });
    }
  }
  return out;
};

const baseDescriptor = {
  permissions: ["filesystem.read"],
  idempotent: true,
  errorSemantics: "fail_fast" as const,
  timeoutS: 30,
};

const writeDescriptor = {
  permissions: ["filesystem.write"],
  idempotent: false,
  errorSemantics: "fail_fast" as const,
  timeoutS: 30,
};

const pad2 = (n: number): string => String(n).padStart(2, "0");

/** Format ms as an ICS floating local date-time (matches parseIcsDate). */
const formatIcsLocal = (ms: number): string => {
  const d = new Date(ms);
  return `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}T${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
};

const formatIcsDateOnly = (ms: number): string => {
  const d = new Date(ms);
  return `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}`;
};

const formatIcsUtc = (ms: number): string => {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}${pad2(d.getUTCMonth() + 1)}${pad2(d.getUTCDate())}T${pad2(d.getUTCHours())}${pad2(d.getUTCMinutes())}${pad2(d.getUTCSeconds())}Z`;
};

const escapeText = (value: string): string =>
  value.replace(/\\/g, "\\\\").replace(/;/g, "\\;").replace(/,/g, "\\,").replace(/\r?\n/g, "\\n");

const parseWhen = (value: unknown, field: string): number => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const ms = Date.parse(value);
    if (Number.isFinite(ms)) return ms;
  }
  throw new Error(`missing or invalid param: ${field}`);
};

/** Read-only connector over a local iCalendar (.ics) file. */
export const createCalendarConnector = (opts: CalendarConnectorOpts): Connector => {
  const id = opts.id ?? "calendar";
  const now = opts.now ?? (() => Date.now());
  let cache: { stamp: string; events: ParsedEvent[] } | undefined;

  const load = (): ParsedEvent[] => {
    if (opts.icsText !== undefined) {
      if (!cache) cache = { stamp: "inline", events: parseIcs(opts.icsText) };
      return cache.events;
    }
    if (!opts.icsPath) throw new Error("calendar connector requires icsPath or icsText");
    const stat = statSync(opts.icsPath);
    const stamp = `${stat.mtimeMs}:${stat.size}`;
    if (cache?.stamp !== stamp) {
      cache = { stamp, events: parseIcs(readFileSync(opts.icsPath, "utf8")) };
    }
    return cache.events;
  };

  // RFC-017 write side (only exposed when writable && icsPath).
  const readRawIcs = (): string => {
    if (!opts.icsPath) throw new Error("calendar writes require icsPath");
    if (!existsSync(opts.icsPath)) {
      return "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//OpenINTJ//Connectors//CN\r\nEND:VCALENDAR\r\n";
    }
    return readFileSync(opts.icsPath, "utf8");
  };

  /** Atomic rewrite: previous content kept at `<ics>.bak`, tmp file renamed over. */
  const writeRawIcs = (text: string): void => {
    if (!opts.icsPath) throw new Error("calendar writes require icsPath");
    mkdirSync(dirname(opts.icsPath), { recursive: true });
    if (existsSync(opts.icsPath)) {
      writeFileSync(`${opts.icsPath}.bak`, readFileSync(opts.icsPath, "utf8"), "utf8");
    }
    const tmp = `${opts.icsPath}.tmp-${process.pid}`;
    writeFileSync(tmp, text, "utf8");
    renameSync(tmp, opts.icsPath);
    cache = undefined;
  };

  const tools: ConnectorTool[] = [
    {
      descriptor: {
        name: "calendar_upcoming",
        description: "列出未来 N 天（默认 7）的日程（含周期日程展开），按开始时间排序。",
        inputSchema: { days: "number?", limit: "number?" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const days = typeof params["days"] === "number" && params["days"] > 0 ? params["days"] : 7;
        const limit =
          typeof params["limit"] === "number" ? Math.min(Math.max(params["limit"], 1), 100) : 20;
        const from = now();
        const to = from + days * 86_400_000;
        const events = load()
          .flatMap((event) => expandEvent(event, from, to))
          .sort((a, b) => a.start - b.start)
          .slice(0, limit);
        return { from, to, events };
      },
    },
    {
      descriptor: {
        name: "calendar_search",
        description: "按关键词搜索日程标题/描述/地点（全部时间范围，不展开周期）。",
        inputSchema: { query: "string", limit: "number?" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const query = typeof params["query"] === "string" ? params["query"].toLowerCase() : "";
        if (!query) throw new Error("missing or invalid param: query");
        const limit =
          typeof params["limit"] === "number" ? Math.min(Math.max(params["limit"], 1), 100) : 20;
        const events = load()
          .filter((event) =>
            [event.summary, event.description ?? "", event.location ?? ""].some((field) =>
              field.toLowerCase().includes(query),
            ),
          )
          .sort((a, b) => a.start - b.start)
          .slice(0, limit)
          .map(({ rrule: _rrule, ...event }) => event);
        return { query: params["query"], events };
      },
    },
  ];

  if (opts.writable && opts.icsPath) {
    tools.push(
      {
        descriptor: {
          name: "calendar_create_event",
          description:
            "在日历中新建一条日程（summary + start 必填，start 为 ISO 时间或毫秒时间戳；allDay 为全天事件）。写操作，需审批。",
          inputSchema: {
            summary: "string",
            start: "string|number",
            end: "string|number?",
            description: "string?",
            location: "string?",
            allDay: "boolean?",
          },
          outputSchema: {},
          ...writeDescriptor,
        },
        handler: (params) => {
          const summary = params["summary"];
          if (typeof summary !== "string" || !summary.trim()) {
            throw new Error("missing or invalid param: summary");
          }
          const allDay = params["allDay"] === true;
          const start = parseWhen(params["start"], "start");
          const end = params["end"] === undefined ? undefined : parseWhen(params["end"], "end");
          const uid = `openintj-${randomUUID()}@openintj.local`;
          const lines = [
            "BEGIN:VEVENT",
            `UID:${uid}`,
            `DTSTAMP:${formatIcsUtc(now())}`,
            allDay
              ? `DTSTART;VALUE=DATE:${formatIcsDateOnly(start)}`
              : `DTSTART:${formatIcsLocal(start)}`,
          ];
          if (end !== undefined) {
            lines.push(
              allDay
                ? `DTEND;VALUE=DATE:${formatIcsDateOnly(end)}`
                : `DTEND:${formatIcsLocal(end)}`,
            );
          }
          lines.push(`SUMMARY:${escapeText(summary)}`);
          if (typeof params["description"] === "string" && params["description"]) {
            lines.push(`DESCRIPTION:${escapeText(params["description"])}`);
          }
          if (typeof params["location"] === "string" && params["location"]) {
            lines.push(`LOCATION:${escapeText(params["location"])}`);
          }
          lines.push("END:VEVENT");
          const text = readRawIcs();
          const at = text.toUpperCase().lastIndexOf("END:VCALENDAR");
          const block = `${lines.join("\r\n")}\r\n`;
          writeRawIcs(
            at >= 0 ? `${text.slice(0, at)}${block}${text.slice(at)}` : `${text}${block}`,
          );
          return { uid, summary, start, ...(end !== undefined ? { end } : {}), allDay };
        },
      },
      {
        descriptor: {
          name: "calendar_delete_event",
          description:
            "按 UID 删除一条日程（周期日程删除整列）。写操作，需审批。UID 可从 calendar_upcoming / calendar_search 结果取得。",
          inputSchema: { uid: "string" },
          outputSchema: {},
          ...writeDescriptor,
        },
        handler: (params) => {
          const uid = params["uid"];
          if (typeof uid !== "string" || !uid) throw new Error("missing or invalid param: uid");
          const lines = readRawIcs().split(/\r\n|\n|\r/);
          const out: string[] = [];
          let removed = false;
          for (let i = 0; i < lines.length; ) {
            if (lines[i] === "BEGIN:VEVENT") {
              let j = i;
              const block: string[] = [];
              while (j < lines.length && lines[j] !== "END:VEVENT") {
                block.push(lines[j]!);
                j += 1;
              }
              if (j < lines.length) {
                block.push(lines[j]!);
                j += 1;
              }
              if (block.some((line) => line.toUpperCase() === `UID:${uid.toUpperCase()}`)) {
                removed = true;
              } else {
                out.push(...block);
              }
              i = j;
              continue;
            }
            if (lines[i]) out.push(lines[i]!);
            i += 1;
          }
          if (!removed) throw new Error(`event not found: ${uid}`);
          writeRawIcs(`${out.join("\r\n")}\r\n`);
          return { uid, deleted: true };
        },
      },
    );
  }

  return {
    id,
    displayName: "日历（ICS 文件）",
    tools: () => tools,
    status: () => {
      try {
        const events = load();
        return {
          id,
          displayName: "日历（ICS 文件）",
          ok: true,
          detail: `${events.length} events`,
          toolCount: tools.length,
        };
      } catch (e) {
        return {
          id,
          displayName: "日历（ICS 文件）",
          ok: false,
          detail: (e as Error).message,
          toolCount: tools.length,
        };
      }
    },
  };
};
