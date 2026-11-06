/** RFC-020 identity graph types. See docs/rfcs/RFC-020-identity-graph.md. */

export type IdentitySeedKind = "email" | "phone" | "wechat" | "username" | "url";

export type LinkedAccountSource = "gravatar" | "mail" | "calendar" | "notes" | "manual";

export interface IdentitySeed {
  kind: IdentitySeedKind;
  value: string;
}

export interface LinkedAccount {
  id: string;
  /** Where the account lives: "github", "mail", "wechat", "gravatar", ... */
  platform: string;
  kind: IdentitySeedKind;
  /** Normalized value (email lowercased, phone digits, url as-is). */
  value: string;
  url?: string;
  source: LinkedAccountSource;
  /** Human-readable evidence: which document / public profile backs this. */
  evidence: string;
  /** 0..1 — how strongly the evidence ties the account to the person. */
  confidence: number;
  /** Public-surface verification time; null for manual assertions. */
  verifiedAt: number | null;
  createdAt: number;
}

/** A canonical person. Exactly one per real human; accounts hang off it. */
export interface IdentityPerson {
  id: string;
  displayName: string;
  accounts: readonly LinkedAccount[];
  createdAt: number;
  updatedAt: number;
}

/** A candidate produced by a resolver. Never enters the graph unconfirmed. */
export interface IdentityProposal {
  id: string;
  /** Suggested canonical person; null means "create a new person". */
  personId: string | null;
  /** Proposals sharing a groupKey should land on the same new person. */
  groupKey?: string | undefined;
  displayName?: string | undefined;
  account: {
    platform: string;
    kind: IdentitySeedKind;
    value: string;
    url?: string | undefined;
    source: LinkedAccountSource;
    evidence: string;
    confidence: number;
    verifiedAt: number | null;
  };
}

/** A read-only local document a corpus resolver may scan. Never leaves the host. */
export interface IdentityDocument {
  source: "mail" | "calendar" | "notes";
  title: string;
  text: string;
}

export interface IdentityResolver {
  readonly id: string;
  resolve(seed: IdentitySeed): Promise<readonly IdentityProposal[]>;
}

export interface ResolverReport {
  resolverId: string;
  ok: boolean;
  detail?: string;
  count: number;
}

export interface ResolveResult {
  seed: IdentitySeed;
  proposals: readonly IdentityProposal[];
  reports: readonly ResolverReport[];
}

export interface AssertAccountInput {
  /** Existing person to attach to; omit/null to create a new person. */
  personId?: string | null | undefined;
  displayName?: string | undefined;
  platform: string;
  kind: IdentitySeedKind;
  value: string;
  url?: string | undefined;
  evidence?: string | undefined;
}
