import { createHash } from "node:crypto";
import { normalizeEmail } from "./normalize.js";
import type { IdentityProposal, IdentityResolver } from "./types.js";

export interface GravatarResolverOpts {
  /** Injectable for tests; defaults to global fetch. */
  fetchFn?: typeof fetch;
  baseUrl?: string;
  now?: () => number;
  /** Per-request timeout (default 8000ms). */
  timeoutMs?: number;
}

interface GravatarProfile {
  display_name?: string;
  profile_url?: string;
  accounts?: { domain?: string; url?: string; verified?: boolean; shortname?: string }[];
  links?: { label?: string; url?: string }[];
}

const platformOf = (url: string): string => {
  try {
    return new URL(url).hostname.replace(/^www\./, "").split(".")[0] ?? "web";
  } catch {
    return "web";
  }
};

/**
 * Email is the only seed with a legitimate public surface: Gravatar's keyless
 * profile API keyed by the SHA-256 of the normalized address. Only the hash
 * leaves the machine. 404 means "no public profile" — an empty result, not
 * an error, and never evidence the person has no accounts anywhere.
 */
export const createGravatarResolver = (opts: GravatarResolverOpts = {}): IdentityResolver => {
  const fetchFn = opts.fetchFn ?? fetch;
  const baseUrl = (opts.baseUrl ?? "https://api.gravatar.com").replace(/\/$/, "");
  const now = opts.now ?? (() => Date.now());
  return {
    id: "gravatar",
    async resolve(seed) {
      if (seed.kind !== "email") return [];
      const email = normalizeEmail(seed.value);
      const hash = createHash("sha256").update(email).digest("hex");
      const res = await fetchFn(`${baseUrl}/v3/profiles/${hash}`, {
        headers: { accept: "application/json" },
        // A hung public endpoint must not stall a resolve forever.
        signal: AbortSignal.timeout(opts.timeoutMs ?? 8_000),
      });
      if (res.status === 404) return [];
      if (!res.ok) throw new Error(`gravatar_http_${res.status}`);
      const profile = (await res.json()) as GravatarProfile;
      const verifiedAt = now();
      const groupKey = `gravatar:${hash}`;
      const displayName = profile.display_name?.trim() || undefined;
      const proposals: IdentityProposal[] = [
        {
          id: `gravatar:self`,
          personId: null,
          groupKey,
          ...(displayName ? { displayName } : {}),
          account: {
            platform: "gravatar",
            kind: "email",
            value: email,
            ...(profile.profile_url ? { url: profile.profile_url } : {}),
            source: "gravatar",
            evidence: "Gravatar public profile keyed by this email hash",
            confidence: 0.95,
            verifiedAt,
          },
        },
      ];
      for (const [i, account] of (profile.accounts ?? []).entries()) {
        if (!account.url) continue;
        proposals.push({
          id: `gravatar:account:${i}`,
          personId: null,
          groupKey,
          account: {
            platform: platformOf(account.url),
            kind: "url",
            value: account.url,
            url: account.url,
            source: "gravatar",
            evidence: account.verified
              ? "Verified account on Gravatar profile"
              : "Account listed on Gravatar profile",
            confidence: account.verified ? 0.9 : 0.6,
            verifiedAt: account.verified ? verifiedAt : null,
          },
        });
      }
      for (const [i, link] of (profile.links ?? []).entries()) {
        if (!link.url) continue;
        proposals.push({
          id: `gravatar:link:${i}`,
          personId: null,
          groupKey,
          account: {
            platform: platformOf(link.url),
            kind: "url",
            value: link.url,
            url: link.url,
            source: "gravatar",
            evidence: `Link on Gravatar profile${link.label ? ` (${link.label})` : ""}`,
            confidence: 0.6,
            verifiedAt: null,
          },
        });
      }
      return proposals;
    },
  };
};
