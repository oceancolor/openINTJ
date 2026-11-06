import { randomUUID } from "node:crypto";
import { detectSeedKind, normalizeSeed, sameAccountValue } from "./normalize.js";
import { type IdentityStore, InMemoryIdentityStore } from "./store.js";
import type {
  AssertAccountInput,
  IdentityPerson,
  IdentityProposal,
  IdentityResolver,
  LinkedAccount,
  ResolveResult,
} from "./types.js";

export interface IdentityServiceOpts {
  store?: IdentityStore;
  resolvers?: readonly IdentityResolver[];
  now?: () => number;
  /** Fired with the full graph after every mutation (confirm/assert/unlink/replace). */
  onGraphChanged?: (persons: readonly IdentityPerson[]) => void;
}

/**
 * Derive the primary (first person's) profile-facing identity: the
 * highest-confidence email plus a human display name. A displayName that
 * merely echoes an account value (auto-derived) or looks like an email is
 * not promoted to a "name".
 */
export const derivePrimaryIdentity = (
  persons: readonly IdentityPerson[],
): { email?: string; name?: string } => {
  const primary = persons[0];
  if (!primary) return {};
  const email = primary.accounts
    .filter((a) => a.kind === "email")
    .sort((a, b) => b.confidence - a.confidence || a.createdAt - b.createdAt)[0]?.value;
  const candidate = primary.displayName.trim();
  const name =
    candidate && !candidate.includes("@") && !primary.accounts.some((a) => a.value === candidate)
      ? candidate
      : undefined;
  return {
    ...(email ? { email } : {}),
    ...(name ? { name } : {}),
  };
};

export interface ConfirmResult {
  persons: readonly IdentityPerson[];
  linked: readonly LinkedAccount[];
  skipped: readonly string[];
}

const shortId = (): string => randomUUID().slice(0, 8);

/**
 * RFC-020 identity service. The core invariant is proposal-confirm: resolver
 * output never enters the graph without an explicit confirm(), and an
 * account value owned by one person can never be silently moved to another.
 */
export class IdentityService {
  readonly store: IdentityStore;
  private readonly resolvers: readonly IdentityResolver[];
  private readonly now: () => number;
  private readonly onGraphChanged: ((persons: readonly IdentityPerson[]) => void) | undefined;

  constructor(opts: IdentityServiceOpts = {}) {
    this.store = opts.store ?? new InMemoryIdentityStore();
    this.resolvers = opts.resolvers ?? [];
    this.now = opts.now ?? (() => Date.now());
    this.onGraphChanged = opts.onGraphChanged;
  }

  private emitChanged(): void {
    this.onGraphChanged?.(this.store.list());
  }

  listPersons(): readonly IdentityPerson[] {
    return this.store.list();
  }

  async resolve(input: { kind?: string | undefined; value: string }): Promise<ResolveResult> {
    const seed = normalizeSeed(
      (input.kind as ResolveResult["seed"]["kind"]) ?? detectSeedKind(input.value),
      input.value,
    );
    const owner = this.findOwner(seed.kind, seed.value);
    const proposals: IdentityProposal[] = [];
    const reports: ResolveResult["reports"][number][] = [];
    for (const resolver of this.resolvers) {
      try {
        const found = await resolver.resolve(seed);
        for (const proposal of found) {
          proposals.push({
            ...proposal,
            id: `${resolver.id}:${shortId()}:${proposals.length}`,
            personId: proposal.personId ?? owner?.id ?? null,
          });
        }
        reports.push({ resolverId: resolver.id, ok: true, count: found.length });
      } catch (err) {
        reports.push({
          resolverId: resolver.id,
          ok: false,
          detail: err instanceof Error ? err.message : String(err),
          count: 0,
        });
      }
    }
    return { seed, proposals, reports };
  }

  confirm(proposals: readonly IdentityProposal[]): ConfirmResult {
    const persons = [...this.store.list()];
    const linked: LinkedAccount[] = [];
    const skipped: string[] = [];
    const now = this.now();
    const targetByKey = new Map<string, IdentityPerson>();

    const ensurePerson = (proposal: IdentityProposal): IdentityPerson => {
      if (proposal.personId) {
        const existing = persons.find((p) => p.id === proposal.personId);
        if (existing) return existing;
      }
      const key = proposal.personId ?? proposal.groupKey ?? proposal.id;
      const cached = targetByKey.get(key);
      if (cached) return cached;
      const person: IdentityPerson = {
        id: `person_${shortId()}`,
        displayName: proposal.displayName ?? proposal.account.value,
        accounts: [],
        createdAt: now,
        updatedAt: now,
      };
      persons.push(person);
      targetByKey.set(key, person);
      return person;
    };

    for (const proposal of proposals) {
      const owner = this.findOwnerIn(persons, proposal.account.kind, proposal.account.value);
      if (owner) {
        const isTarget =
          proposal.personId === owner.id ||
          (proposal.groupKey !== undefined && targetByKey.get(proposal.groupKey)?.id === owner.id);
        if (isTarget) {
          skipped.push(proposal.account.value);
          continue;
        }
        throw new Error(
          `identity_conflict: ${proposal.account.kind} "${proposal.account.value}" already belongs to "${owner.displayName}"`,
        );
      }
      const person = ensurePerson(proposal);
      const account: LinkedAccount = {
        id: `acct_${shortId()}`,
        platform: proposal.account.platform,
        kind: proposal.account.kind,
        value: proposal.account.value,
        ...(proposal.account.url ? { url: proposal.account.url } : {}),
        source: proposal.account.source,
        evidence: proposal.account.evidence,
        confidence: proposal.account.confidence,
        verifiedAt: proposal.account.verifiedAt,
        createdAt: now,
      };
      const at = persons.findIndex((p) => p.id === person.id);
      persons[at!] = {
        ...person,
        accounts: [...person.accounts, account],
        updatedAt: now,
      };
      targetByKey.set(proposal.personId ?? proposal.groupKey ?? proposal.id, persons[at!]!);
      linked.push(account);
    }
    this.store.replaceAll(persons);
    this.emitChanged();
    return { persons: this.store.list(), linked, skipped };
  }

  /** Manual assertion: "this account is mine". Confidence 1, never "verified". */
  assertAccount(input: AssertAccountInput): IdentityPerson {
    const seed = normalizeSeed(input.kind, input.value);
    const persons = [...this.store.list()];
    const owner = this.findOwnerIn(persons, seed.kind, seed.value);
    if (owner && owner.id !== input.personId) {
      throw new Error(
        `identity_conflict: ${seed.kind} "${seed.value}" already belongs to "${owner.displayName}"`,
      );
    }
    const now = this.now();
    const account: LinkedAccount = {
      id: `acct_${shortId()}`,
      platform: input.platform,
      kind: seed.kind,
      value: seed.value,
      ...(input.url ? { url: input.url } : {}),
      source: "manual",
      evidence: input.evidence ?? "Asserted by the user",
      confidence: 1,
      verifiedAt: null,
      createdAt: now,
    };
    let person = input.personId ? persons.find((p) => p.id === input.personId) : undefined;
    if (owner && !person) person = owner;
    if (!person) {
      person = {
        id: `person_${shortId()}`,
        displayName: input.displayName ?? seed.value,
        accounts: [],
        createdAt: now,
        updatedAt: now,
      };
      persons.push(person);
    }
    if (owner?.id === person.id) return person;
    const at = persons.findIndex((p) => p.id === person!.id);
    persons[at!] = { ...person, accounts: [...person.accounts, account], updatedAt: now };
    this.store.replaceAll(persons);
    this.emitChanged();
    return persons[at!]!;
  }

  /** Remove one linked account; a person left with no accounts is removed too. */
  unlink(accountId: string): { removed: boolean; personRemoved: boolean } {
    const persons = [...this.store.list()];
    let removed = false;
    let personRemoved = false;
    const next: IdentityPerson[] = [];
    for (const person of persons) {
      if (!person.accounts.some((a) => a.id === accountId)) {
        next.push(person);
        continue;
      }
      removed = true;
      const accounts = person.accounts.filter((a) => a.id !== accountId);
      if (accounts.length === 0) {
        personRemoved = true;
        continue;
      }
      next.push({ ...person, accounts, updatedAt: this.now() });
    }
    if (removed) {
      this.store.replaceAll(next);
      this.emitChanged();
    }
    return { removed, personRemoved };
  }

  /**
   * Replace the whole graph (sync import). Malformed persons/accounts are
   * dropped rather than failing the import; the change callback fires so
   * derived surfaces (profile writeback) stay consistent.
   */
  replacePersons(persons: readonly IdentityPerson[]): void {
    const clean: IdentityPerson[] = [];
    for (const person of persons) {
      if (typeof person?.id !== "string" || typeof person?.displayName !== "string") continue;
      if (!Array.isArray(person.accounts)) continue;
      const accounts = person.accounts.filter(
        (a) =>
          typeof a?.id === "string" &&
          typeof a?.platform === "string" &&
          typeof a?.value === "string",
      );
      clean.push({ ...person, accounts });
    }
    this.store.replaceAll(clean);
    this.emitChanged();
  }

  private findOwner(kind: LinkedAccount["kind"], value: string): IdentityPerson | undefined {
    return this.findOwnerIn(this.store.list(), kind, value);
  }

  private findOwnerIn(
    persons: readonly IdentityPerson[],
    kind: LinkedAccount["kind"],
    value: string,
  ): IdentityPerson | undefined {
    return persons.find((person) =>
      person.accounts.some(
        (account) => account.kind === kind && sameAccountValue(kind, account.value, value),
      ),
    );
  }
}
