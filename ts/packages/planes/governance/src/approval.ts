import { randomUUID } from "node:crypto";

/** RFC-013 risk tiers, derived from tool permissions (see classifyToolRisk). */
export type ToolRisk = "read" | "write" | "execute" | "external";

export const TOOL_RISK_ORDER: readonly ToolRisk[] = ["read", "write", "execute", "external"];

export const riskAtLeast = (risk: ToolRisk, floor: ToolRisk): boolean =>
  TOOL_RISK_ORDER.indexOf(risk) >= TOOL_RISK_ORDER.indexOf(floor);

/** Derive risk from descriptor permissions; unknown tools default to write (fail-safe). */
export const classifyToolRisk = (input: {
  tool: string;
  permissions?: readonly string[];
  idempotent?: boolean;
}): ToolRisk => {
  const perms = input.permissions ?? [];
  if (perms.includes("system.execute")) return "execute";
  if (perms.some((p) => p.endsWith(".external") || p === "external.write")) return "external";
  if (perms.includes("filesystem.write") || perms.includes("network.write")) return "write";
  if (perms.includes("filesystem.read") || perms.includes("network.read")) return "read";
  if (input.tool === "execute_command") return "execute";
  if (input.tool === "write_file") return "write";
  if (input.tool === "read_file" || input.tool === "search") return "read";
  return input.idempotent ? "read" : "write";
};

export interface CapabilityGrant {
  id: string;
  /** Tool name or "*". */
  tool: string;
  /** Highest risk this grant covers. */
  risk: ToolRisk;
  /** write: params.path prefix; execute: command first-token/prefix match. */
  scope?: string;
  expiresAt?: number;
  createdAt: number;
  label?: string;
  /**
   * Origin of the grant. `approval` grants are minted implicitly when a user
   * approves a tool call and stay session-only; explicit (`manual`, default)
   * grants are deliberate long-lived authorizations and may persist.
   */
  source?: "approval" | "manual";
}

export interface GrantStore {
  grant(input: Omit<CapabilityGrant, "id" | "createdAt"> & { id?: string }): CapabilityGrant;
  revoke(id: string): boolean;
  list(): readonly CapabilityGrant[];
  /** True when a live grant covers this call. Expired grants never match. */
  covers(input: { tool: string; risk: ToolRisk; params?: Record<string, unknown> }): boolean;
}

const scopeMatches = (grant: CapabilityGrant, params?: Record<string, unknown>): boolean => {
  if (!grant.scope) return true;
  if (!params) return false;
  const path = typeof params["path"] === "string" ? params["path"] : undefined;
  if (path) return path.startsWith(grant.scope);
  const command = typeof params["command"] === "string" ? params["command"] : undefined;
  if (command) return command.startsWith(grant.scope);
  return false;
};

export class InMemoryGrantStore implements GrantStore {
  private readonly grants = new Map<string, CapabilityGrant>();

  grant(input: Omit<CapabilityGrant, "id" | "createdAt"> & { id?: string }): CapabilityGrant {
    const grant: CapabilityGrant = {
      ...input,
      id: input.id ?? randomUUID(),
      createdAt: (input as { createdAt?: number }).createdAt ?? Date.now(),
    };
    this.grants.set(grant.id, grant);
    return grant;
  }

  revoke(id: string): boolean {
    return this.grants.delete(id);
  }

  list(): readonly CapabilityGrant[] {
    const now = Date.now();
    return [...this.grants.values()].filter((g) => !g.expiresAt || g.expiresAt > now);
  }

  covers(input: { tool: string; risk: ToolRisk; params?: Record<string, unknown> }): boolean {
    const now = Date.now();
    for (const grant of this.grants.values()) {
      if (grant.expiresAt && grant.expiresAt <= now) continue;
      if (grant.tool !== "*" && grant.tool !== input.tool) continue;
      if (!riskAtLeast(grant.risk, input.risk)) continue;
      if (!scopeMatches(grant, input.params)) continue;
      return true;
    }
    return false;
  }
}

export interface ApprovalRequest {
  id: string;
  tool: string;
  risk: ToolRisk;
  params: Record<string, unknown>;
  requestedAt: number;
  /** RFC-021 F02: run cancellation signal; aborting denies the parked request. */
  signal?: AbortSignal;
}

export type ApprovalOutcome =
  | { decision: "approved"; grant?: { scope?: string; ttlMs?: number } }
  | { decision: "denied"; reason?: string };

export interface ApprovalBroker {
  request(req: ApprovalRequest): Promise<ApprovalOutcome>;
}

/**
 * Host-driven broker: requests park until the host calls `decide`.
 * Fail-closed — with no host deciding (or after cancel), the outcome is denied.
 */
export class PendingApprovalBroker implements ApprovalBroker {
  private readonly pending = new Map<
    string,
    { req: ApprovalRequest; resolve: (outcome: ApprovalOutcome) => void }
  >();
  private readonly listeners = new Set<(req: ApprovalRequest) => void>();

  onRequest(listener: (req: ApprovalRequest) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  listPending(): readonly ApprovalRequest[] {
    return [...this.pending.values()].map((entry) => entry.req);
  }

  request(req: ApprovalRequest): Promise<ApprovalOutcome> {
    // RFC-021 F02 (T04): an already-cancelled run never parks; aborting a
    // parked request denies it, and a later decide() finds nothing.
    if (req.signal?.aborted) {
      return Promise.resolve({ decision: "denied", reason: "cancelled" });
    }
    return new Promise<ApprovalOutcome>((resolve) => {
      this.pending.set(req.id, { req, resolve });
      req.signal?.addEventListener(
        "abort",
        () => {
          this.cancel(req.id, "cancelled");
        },
        { once: true },
      );
      for (const listener of this.listeners) listener(req);
    });
  }

  decide(id: string, outcome: ApprovalOutcome): boolean {
    const entry = this.pending.get(id);
    if (!entry) return false;
    this.pending.delete(id);
    entry.resolve(outcome);
    return true;
  }

  cancel(id: string, reason = "cancelled"): boolean {
    return this.decide(id, { decision: "denied", reason });
  }

  cancelAll(reason = "cancelled"): void {
    for (const id of [...this.pending.keys()]) this.cancel(id, reason);
  }
}

/** Test/CLI helper: broker that always returns a fixed outcome. */
export const fixedApprovalBroker = (outcome: ApprovalOutcome): ApprovalBroker => ({
  async request() {
    return outcome;
  },
});

export interface ApprovalSetup {
  broker?: ApprovalBroker;
  /** Present when the setup created the broker (hosts decide pending requests on it). */
  pending?: PendingApprovalBroker;
  requiredRisks: ToolRisk[];
}

const parseRisks = (raw: string | undefined): ToolRisk[] =>
  (raw ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter((item): item is ToolRisk => TOOL_RISK_ORDER.includes(item as ToolRisk));

/**
 * Shared host wiring (RFC-013 §4): explicit risks > `OPENINTJ_APPROVAL_RISKS`
 * env > none. A PendingApprovalBroker is created when risks are required and
 * no custom broker is injected.
 */
export const createApprovalSetup = (
  input: {
    requiredRisks?: readonly ToolRisk[];
    broker?: ApprovalBroker;
    env?: NodeJS.ProcessEnv;
  } = {},
): ApprovalSetup => {
  const requiredRisks = input.requiredRisks
    ? [...input.requiredRisks]
    : parseRisks((input.env ?? process.env)["OPENINTJ_APPROVAL_RISKS"]);
  if (requiredRisks.length === 0) return { requiredRisks };
  if (input.broker) return { broker: input.broker, requiredRisks };
  const pending = new PendingApprovalBroker();
  return { broker: pending, pending, requiredRisks };
};
