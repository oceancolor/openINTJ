import type { IdentityPerson, IdentityService } from "@openintj/identity";
import type { ProfileEntry, ProfileService } from "@openintj/profile";
import type { ScheduledJob, SchedulerService } from "@openintj/scheduler";

export interface SyncBundle {
  version: 1;
  /** ISO timestamp of export. */
  exportedAt: string;
  profile: { entries: ProfileEntry[]; dismissed: string[] };
  schedules: ScheduledJob[];
  /** RFC-020 identity graph (optional: absent in pre-identity bundles). */
  identity?: { persons: IdentityPerson[] };
}

export interface SyncApplyReport {
  profileUpserted: number;
  profileSkipped: number;
  dismissedApplied: number;
  schedulesAdded: number;
  schedulesKept: number;
  identityPersonsAdded: number;
  identityAccountsAdded: number;
}

export interface SyncServices {
  profile: ProfileService;
  scheduler: SchedulerService;
  /** RFC-020: when present, identity persons travel with the bundle. */
  identity?: IdentityService;
}
