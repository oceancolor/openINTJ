import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  JsonScheduleStore,
  SchedulerService,
  computeNextRunAt,
  isInQuietHours,
} from "../src/index.js";

const at = (day: number, h: number, m = 0): number => new Date(2026, 0, day, h, m).getTime();

describe("computeNextRunAt", () => {
  it("interval advances by whole periods from the anchor", () => {
    const trigger = { kind: "interval", everyMs: 60_000 } as const;
    expect(computeNextRunAt(trigger, at(5, 8), at(5, 8))).toBe(at(5, 8) + 60_000);
    expect(computeNextRunAt(trigger, at(5, 8) + 90_000, at(5, 8))).toBe(at(5, 8) + 120_000);
  });

  it("daily rolls to tomorrow after the time has passed", () => {
    const trigger = { kind: "daily", time: "09:30" } as const;
    expect(computeNextRunAt(trigger, at(5, 8), at(5, 8))).toBe(at(5, 9, 30));
    expect(computeNextRunAt(trigger, at(5, 10), at(5, 8))).toBe(at(6, 9, 30));
  });

  it("quiet hours handle overnight windows", () => {
    const quiet = { start: "22:00", end: "07:00" };
    expect(isInQuietHours(at(5, 23), quiet)).toBe(true);
    expect(isInQuietHours(at(5, 3), quiet)).toBe(true);
    expect(isInQuietHours(at(5, 12), quiet)).toBe(false);
  });
});

describe("SchedulerService (fake clock)", () => {
  const makeService = (opts?: ConstructorParameters<typeof SchedulerService>[1]) => {
    let now = at(5, 8);
    const fired: string[] = [];
    const service = new SchedulerService(undefined, { now: () => now, ...opts });
    service.attachRunner(async (job) => {
      fired.push(job.id);
    });
    return {
      service,
      fired,
      setNow: (ms: number) => {
        now = ms;
      },
      getNow: () => now,
    };
  };

  it("once jobs fire at their time and then disable", async () => {
    const { service, fired, setNow, getNow } = makeService();
    const job = service.add({
      title: "提醒",
      prompt: "喝水",
      trigger: { kind: "once", at: getNow() + 60_000 },
    });
    await service.tick();
    expect(fired).toEqual([]);
    setNow(getNow() + 60_000);
    await service.tick();
    expect(fired).toEqual([job.id]);
    const saved = service.get(job.id)!;
    expect(saved.enabled).toBe(false);
    expect(saved.runCount).toBe(1);
    expect(saved.lastStatus).toBe("completed");
    await service.tick();
    expect(fired).toEqual([job.id]);
  });

  it("missed interval triggers coalesce into one catch-up run with lateMs", async () => {
    const { service, fired, setNow, getNow } = makeService();
    const events: Array<{ kind: string; lateMs?: number }> = [];
    service.on((e) => events.push(e));
    const job = service.add({
      title: "汇总",
      prompt: "汇总",
      trigger: { kind: "interval", everyMs: 60_000 },
    });
    // Offline for ten periods.
    setNow(getNow() + 10 * 60_000);
    await service.tick();
    expect(fired).toEqual([job.id]);
    expect(events.find((e) => e.kind === "fired")?.lateMs).toBeGreaterThan(0);
    expect(service.get(job.id)!.nextRunAt).toBeGreaterThan(getNow());
  });

  it("quiet hours defer instead of dropping", async () => {
    const { service, fired, setNow } = makeService({
      quietHours: { start: "22:00", end: "07:00" },
    });
    const events: string[] = [];
    service.on((e) => events.push(e.kind));
    service.add({ title: "夜报", prompt: "夜报", trigger: { kind: "once", at: at(5, 23) } });
    setNow(at(5, 23));
    await service.tick();
    expect(fired).toEqual([]);
    expect(events).toContain("deferred");
    setNow(at(6, 7));
    await service.tick();
    expect(fired).toHaveLength(1);
  });

  it("failures retry with backoff, then record failed and reschedule", async () => {
    let now = at(5, 8);
    let calls = 0;
    const service = new SchedulerService(undefined, {
      now: () => now,
      maxRetries: 1,
      retryDelayMs: 30_000,
    });
    service.attachRunner(async () => {
      calls += 1;
      throw new Error("boom");
    });
    const job = service.add({
      title: "易错",
      prompt: "易错",
      trigger: { kind: "interval", everyMs: 600_000 },
    });
    now += 600_000;
    await service.tick();
    expect(calls).toBe(1);
    expect(service.get(job.id)!.lastStatus).toBe("failed");
    // Not yet time for the retry.
    await service.tick();
    expect(calls).toBe(1);
    now += 30_000;
    await service.tick();
    expect(calls).toBe(2);
    const saved = service.get(job.id)!;
    expect(saved.lastError).toBe("boom");
    // Retries exhausted: next regular occurrence is scheduled.
    expect(saved.nextRunAt).toBeGreaterThan(now);
  });

  it("maxRuns disables the job after enough successes", async () => {
    const { service, setNow, getNow } = makeService();
    const job = service.add({
      title: "两次",
      prompt: "两次",
      maxRuns: 2,
      trigger: { kind: "interval", everyMs: 60_000 },
    });
    setNow(getNow() + 60_000);
    await service.tick();
    expect(service.get(job.id)!.enabled).toBe(true);
    setNow(getNow() + 60_000);
    await service.tick();
    expect(service.get(job.id)!.enabled).toBe(false);
    expect(service.get(job.id)!.runCount).toBe(2);
  });

  it("rejects invalid inputs", () => {
    const { service } = makeService();
    expect(() =>
      service.add({ title: "", prompt: "x", trigger: { kind: "once", at: 1 } }),
    ).toThrow();
    expect(() =>
      service.add({ title: "x", prompt: "x", trigger: { kind: "interval", everyMs: 10 } }),
    ).toThrow();
    expect(() =>
      service.add({ title: "x", prompt: "x", trigger: { kind: "daily", time: "25:00" } }),
    ).toThrow();
  });

  it("JsonScheduleStore persists jobs across instances", async () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-sched-"));
    try {
      const file = join(dir, "schedules.json");
      let now = at(5, 8);
      const first = new SchedulerService(new JsonScheduleStore(file), { now: () => now });
      const job = first.add({
        title: "持久",
        prompt: "持久",
        trigger: { kind: "once", at: now + 5_000 },
      });
      const second = new SchedulerService(new JsonScheduleStore(file), { now: () => now });
      expect(second.get(job.id)?.title).toBe("持久");
      const fired: string[] = [];
      second.attachRunner(async (j) => {
        fired.push(j.id);
      });
      now += 5_000;
      await second.tick();
      expect(fired).toEqual([job.id]);
      const third = new SchedulerService(new JsonScheduleStore(file));
      expect(third.get(job.id)?.runCount).toBe(1);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
