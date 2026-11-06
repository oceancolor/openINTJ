import { IdentityService } from "@openintj/identity";
import { ProfileService } from "@openintj/profile";
import { SchedulerService } from "@openintj/scheduler";
import { describe, expect, it } from "vitest";
import { applySyncBundle, buildSyncBundle, validateSyncBundle } from "../src/index.js";

function makeServices() {
  return {
    profile: new ProfileService(),
    scheduler: new SchedulerService(),
    identity: new IdentityService(),
  };
}

describe("sync bundle v1", () => {
  it("round-trips profile + schedules preserving ids", () => {
    const a = makeServices();
    a.profile.upsert({ section: "identity", key: "name", value: "Ben" });
    const job = a.scheduler.add({
      title: "日报",
      prompt: "写日报",
      trigger: { kind: "daily", time: "09:00" },
    });
    const bundle = buildSyncBundle(a);
    const b = makeServices();
    const report = applySyncBundle(bundle, b);
    expect(report.profileUpserted).toBe(1);
    expect(report.schedulesAdded).toBe(1);
    expect(b.profile.list().map((e) => e.value)).toContain("Ben");
    expect(b.scheduler.list().map((j) => j.id)).toEqual([job.id]);
  });

  it("profile LWW: newer wins, local user entry never overwritten by dormant", () => {
    const a = makeServices();
    const b = makeServices();
    b.profile.upsert({ section: "preferences", key: "tone", value: "简洁", source: "user" });
    a.profile.upsert({ section: "preferences", key: "tone", value: "详尽", source: "dormant" });
    const report = applySyncBundle(buildSyncBundle(a), b);
    expect(report.profileSkipped).toBe(1);
    expect(b.profile.list().find((e) => e.key === "tone")?.value).toBe("简洁");
  });

  it("dismissed tombstones union and delete live entries", () => {
    const a = makeServices();
    const b = makeServices();
    b.profile.upsert({ section: "notes", key: "old", value: "x" });
    a.profile.store.dismiss("notes:old");
    const report = applySyncBundle(buildSyncBundle(a), b);
    expect(report.dismissedApplied).toBe(1);
    expect(b.profile.list().find((e) => e.key === "old")).toBeUndefined();
    expect(b.profile.store.listDismissed()).toContain("notes:old");
  });

  it("existing schedule ids are kept verbatim", () => {
    const a = makeServices();
    const b = makeServices();
    const job = a.scheduler.add({
      title: "A",
      prompt: "p",
      trigger: { kind: "interval", everyMs: 60_000 },
    });
    applySyncBundle(buildSyncBundle(a), b);
    const edited = b.scheduler.update(job.id, { title: "本地改名" });
    expect(edited?.title).toBe("本地改名");
    const report = applySyncBundle(buildSyncBundle(a), b);
    expect(report.schedulesKept).toBe(1);
    expect(b.scheduler.get(job.id)?.title).toBe("本地改名");
  });

  it("rejects malformed bundles before applying", () => {
    expect(() => validateSyncBundle({ version: 2 })).toThrow(/version/);
    expect(() => validateSyncBundle(null)).toThrow();
    expect(() =>
      applySyncBundle(
        {
          version: 1,
          exportedAt: "x",
          profile: { entries: [{ bad: 1 }], dismissed: [] },
          schedules: [],
        } as never,
        makeServices(),
      ),
    ).toThrow();
  });

  it("identity persons union by id; accounts union by account id (RFC-020)", () => {
    const a = makeServices();
    const b = makeServices();
    const person = a.identity.assertAccount({
      displayName: "Ben",
      platform: "gravatar",
      kind: "email",
      value: "ben@example.com",
    });
    a.identity.assertAccount({
      personId: person.id,
      platform: "github",
      kind: "url",
      value: "https://github.com/ben",
    });
    const bundle = buildSyncBundle(a);
    expect(bundle.identity?.persons).toHaveLength(1);

    const first = applySyncBundle(bundle, b);
    expect(first.identityPersonsAdded).toBe(1);
    expect(first.identityAccountsAdded).toBe(2);
    expect(b.identity.listPersons()[0]?.accounts).toHaveLength(2);

    // Re-apply is a no-op for identity.
    const second = applySyncBundle(bundle, b);
    expect(second.identityPersonsAdded).toBe(0);
    expect(second.identityAccountsAdded).toBe(0);

    // A new account on a known person merges in; local extras are kept.
    a.identity.assertAccount({
      personId: person.id,
      platform: "x",
      kind: "url",
      value: "https://x.com/ben",
    });
    const third = applySyncBundle(buildSyncBundle(a), b);
    expect(third.identityPersonsAdded).toBe(0);
    expect(third.identityAccountsAdded).toBe(1);
    expect(b.identity.listPersons()[0]?.accounts).toHaveLength(3);

    // Malformed identity sections are rejected up front.
    expect(() => validateSyncBundle({ ...bundle, identity: { persons: [{ id: 1 }] } })).toThrow(
      /identity/,
    );
  });

  it("old bundles without an identity section still apply", () => {
    const a = makeServices();
    const bundle = buildSyncBundle(a);
    delete bundle.identity;
    const b = makeServices();
    const report = applySyncBundle(bundle, b);
    expect(report.identityPersonsAdded).toBe(0);
    expect(report.profileUpserted).toBe(0);
  });
});
