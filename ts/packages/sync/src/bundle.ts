import { PROFILE_SECTIONS, type ProfileEntry } from "@openintj/profile";
import type { ScheduledJob } from "@openintj/scheduler";
import type { SyncApplyReport, SyncBundle, SyncServices } from "./types.js";

const SECTIONS = new Set<string>(PROFILE_SECTIONS);

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function keyOf(section: string, key: string): string {
  return `${section}:${key}`;
}

/** Throws TypeError on any structural problem — callers validate before applying. */
export function validateSyncBundle(raw: unknown): SyncBundle {
  if (!isRecord(raw)) throw new TypeError("sync bundle must be an object");
  if (raw.version !== 1)
    throw new TypeError(`unsupported sync bundle version: ${String(raw.version)}`);
  if (typeof raw.exportedAt !== "string") throw new TypeError("sync bundle missing exportedAt");
  if (
    !isRecord(raw.profile) ||
    !Array.isArray(raw.profile.entries) ||
    !Array.isArray(raw.profile.dismissed)
  ) {
    throw new TypeError("sync bundle profile section malformed");
  }
  for (const e of raw.profile.entries as unknown[]) {
    if (!isRecord(e)) throw new TypeError("profile entry must be an object");
    if (typeof e.section !== "string" || !SECTIONS.has(e.section)) {
      throw new TypeError(`unknown profile section: ${String(e.section)}`);
    }
    if (typeof e.id !== "string" || typeof e.key !== "string" || typeof e.value !== "string") {
      throw new TypeError("profile entry missing id/key/value");
    }
    if (typeof e.updatedAt !== "number" || typeof e.createdAt !== "number") {
      throw new TypeError("profile entry missing timestamps");
    }
    if (e.source !== "user" && e.source !== "dormant" && e.source !== "import") {
      throw new TypeError(`unknown profile source: ${String(e.source)}`);
    }
  }
  for (const d of raw.profile.dismissed as unknown[]) {
    if (typeof d !== "string") throw new TypeError("dismissed keys must be strings");
  }
  if (raw.identity !== undefined) {
    if (!isRecord(raw.identity) || !Array.isArray(raw.identity.persons)) {
      throw new TypeError("sync bundle identity section malformed");
    }
    for (const p of raw.identity.persons as unknown[]) {
      if (!isRecord(p)) throw new TypeError("identity person must be an object");
      if (typeof p.id !== "string" || typeof p.displayName !== "string") {
        throw new TypeError("identity person missing id/displayName");
      }
      if (!Array.isArray(p.accounts)) {
        throw new TypeError("identity person accounts malformed");
      }
      for (const a of p.accounts as unknown[]) {
        if (!isRecord(a)) throw new TypeError("identity account must be an object");
        if (
          typeof a.id !== "string" ||
          typeof a.platform !== "string" ||
          typeof a.value !== "string"
        ) {
          throw new TypeError("identity account missing id/platform/value");
        }
      }
    }
  }
  if (!Array.isArray(raw.schedules)) throw new TypeError("sync bundle schedules malformed");
  for (const j of raw.schedules as unknown[]) {
    if (!isRecord(j)) throw new TypeError("scheduled job must be an object");
    if (typeof j.id !== "string" || typeof j.title !== "string" || typeof j.prompt !== "string") {
      throw new TypeError("scheduled job missing id/title/prompt");
    }
    if (!isRecord(j.trigger) || typeof j.trigger.kind !== "string") {
      throw new TypeError("scheduled job missing trigger");
    }
  }
  return raw as unknown as SyncBundle;
}

export function buildSyncBundle(services: SyncServices, now: number = Date.now()): SyncBundle {
  return {
    version: 1,
    exportedAt: new Date(now).toISOString(),
    profile: {
      entries: [...services.profile.store.list()],
      dismissed: [...services.profile.store.listDismissed()],
    },
    schedules: [...services.scheduler.store.list()],
    ...(services.identity ? { identity: { persons: [...services.identity.listPersons()] } } : {}),
  };
}

/**
 * Merge semantics (v1): profile last-write-wins by updatedAt per (section,key),
 * a local `user` entry is never overwritten by non-user incoming data, dismissed
 * tombstones union and delete the corresponding live entry; schedules union by
 * id with existing jobs kept verbatim (edits/deletes do not propagate in v1).
 */
export function applySyncBundle(bundle: SyncBundle, services: SyncServices): SyncApplyReport {
  const valid = validateSyncBundle(bundle);
  const report: SyncApplyReport = {
    profileUpserted: 0,
    profileSkipped: 0,
    dismissedApplied: 0,
    schedulesAdded: 0,
    schedulesKept: 0,
    identityPersonsAdded: 0,
    identityAccountsAdded: 0,
  };
  const store = services.profile.store;
  const dismissed = new Set<string>(store.listDismissed());

  for (const key of valid.profile.dismissed) {
    if (!dismissed.has(key)) {
      store.dismiss(key);
      dismissed.add(key);
      report.dismissedApplied += 1;
    }
    const live = store.list().find((e) => keyOf(e.section, e.key) === key);
    if (live !== undefined) store.remove(live.id);
  }

  const byKey = new Map<string, ProfileEntry>();
  for (const e of store.list()) byKey.set(keyOf(e.section, e.key), e);

  for (const incoming of valid.profile.entries) {
    const k = keyOf(incoming.section, incoming.key);
    if (dismissed.has(k)) {
      report.profileSkipped += 1;
      continue;
    }
    const local = byKey.get(k);
    if (local === undefined) {
      store.upsert(incoming);
      byKey.set(k, incoming);
      report.profileUpserted += 1;
      continue;
    }
    if (local.source === "user" && incoming.source !== "user") {
      report.profileSkipped += 1;
      continue;
    }
    if (incoming.updatedAt > local.updatedAt) {
      if (local.id !== incoming.id) store.remove(local.id);
      store.upsert(incoming);
      byKey.set(k, incoming);
      report.profileUpserted += 1;
    } else {
      report.profileSkipped += 1;
    }
  }

  const localJobIds = new Set(services.scheduler.store.list().map((j) => j.id));
  for (const job of valid.schedules as ScheduledJob[]) {
    if (localJobIds.has(job.id)) {
      report.schedulesKept += 1;
    } else {
      services.scheduler.store.upsert(job);
      localJobIds.add(job.id);
      report.schedulesAdded += 1;
    }
  }

  // RFC-020: identity persons union by person id; for a known person,
  // accounts union by account id. Existing records are kept verbatim (v1
  // semantics, same as schedules): edits/deletes do not propagate.
  if (valid.identity && services.identity) {
    const local = [...services.identity.listPersons()];
    const byId = new Map(local.map((p) => [p.id, p]));
    let merged = false;
    for (const incoming of valid.identity.persons) {
      const existing = byId.get(incoming.id);
      if (!existing) {
        local.push(incoming);
        byId.set(incoming.id, incoming);
        report.identityPersonsAdded += 1;
        report.identityAccountsAdded += incoming.accounts.length;
        merged = true;
        continue;
      }
      const knownAccountIds = new Set(existing.accounts.map((a) => a.id));
      const fresh = incoming.accounts.filter((a) => !knownAccountIds.has(a.id));
      if (fresh.length > 0) {
        const at = local.findIndex((p) => p.id === existing.id);
        local[at] = {
          ...existing,
          accounts: [...existing.accounts, ...fresh],
          updatedAt: Math.max(existing.updatedAt, incoming.updatedAt),
        };
        report.identityAccountsAdded += fresh.length;
        merged = true;
      }
    }
    if (merged) services.identity.replacePersons(local);
  }
  return report;
}
