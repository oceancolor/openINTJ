import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { type CapabilityGrant, type GrantStore, InMemoryGrantStore } from "./approval.js";

/** Approval-derived grants are session-only by design (RFC-013 two-tier rule). */
const isPersistable = (grant: CapabilityGrant): boolean => grant.source !== "approval";

/**
 * File-backed grant store (`<dataDir>/grants.json`), mirroring the profile /
 * scheduler JSON stores: whole-file synchronous writes, corrupt file starts
 * empty, expired grants are pruned on every write. Only explicit (manual)
 * grants land on disk — a one-time approval must never silently become a
 * permanent authorization across restarts.
 */
export class JsonGrantStore implements GrantStore {
  private readonly inner = new InMemoryGrantStore();

  constructor(private readonly filePath: string) {
    if (existsSync(filePath)) {
      try {
        const parsed = JSON.parse(readFileSync(filePath, "utf8")) as {
          grants?: CapabilityGrant[];
        };
        const now = Date.now();
        for (const grant of parsed.grants ?? []) {
          if (!isPersistable(grant)) continue;
          if (grant.expiresAt && grant.expiresAt <= now) continue;
          this.inner.grant(grant);
        }
      } catch {
        // A corrupt grants file must never block agent startup; start empty.
      }
    }
  }

  private flush(): void {
    mkdirSync(dirname(this.filePath), { recursive: true });
    writeFileSync(
      this.filePath,
      JSON.stringify({ version: 1, grants: this.inner.list().filter(isPersistable) }, null, 2),
      "utf8",
    );
  }

  grant(input: Parameters<GrantStore["grant"]>[0]): CapabilityGrant {
    const grant = this.inner.grant(input);
    this.flush();
    return grant;
  }

  revoke(id: string): boolean {
    const removed = this.inner.revoke(id);
    if (removed) this.flush();
    return removed;
  }

  list(): readonly CapabilityGrant[] {
    return this.inner.list();
  }

  covers(input: Parameters<GrantStore["covers"]>[0]): boolean {
    return this.inner.covers(input);
  }
}
