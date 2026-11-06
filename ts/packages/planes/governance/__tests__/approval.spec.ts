import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CommandSchema, CommandType } from "@openintj/core";
import { afterEach, describe, expect, it } from "vitest";
import {
  GovernancePlane,
  InMemoryGrantStore,
  JsonGrantStore,
  PendingApprovalBroker,
  classifyToolRisk,
  createToolCallGate,
  fixedApprovalBroker,
} from "../src/index.js";

const writeCommand = (path = "notes/a.md") =>
  CommandSchema.parse({
    commandType: CommandType.TOOL_CALL,
    target: "write_file",
    payload: { path, content: "x" },
  });

describe("classifyToolRisk (RFC-013)", () => {
  it("maps permissions and builtin names", () => {
    expect(classifyToolRisk({ tool: "read_file", permissions: ["filesystem.read"] })).toBe("read");
    expect(classifyToolRisk({ tool: "write_file", permissions: ["filesystem.write"] })).toBe(
      "write",
    );
    expect(classifyToolRisk({ tool: "execute_command", permissions: ["system.execute"] })).toBe(
      "execute",
    );
    expect(classifyToolRisk({ tool: "search" })).toBe("read");
  });
});

describe("GovernancePlane approval gate (RFC-013)", () => {
  it("fails closed when approval is required but no broker exists", async () => {
    const gov = new GovernancePlane({ approval: { requiredRisks: ["write"] } });
    await expect(gov.checkToolCall(writeCommand(), { risk: "write" })).rejects.toThrow("需要审批");
    expect(gov.auditTrail.getStats().blockedCount).toBeGreaterThan(0);
  });

  it("approved calls pass and denied calls are blocked", async () => {
    const approving = new GovernancePlane({
      approval: { requiredRisks: ["write"], broker: fixedApprovalBroker({ decision: "approved" }) },
    });
    const event = await approving.checkToolCall(writeCommand(), { risk: "write" });
    expect(event.details["approval"]).toBe("approved");

    const denying = new GovernancePlane({
      approval: {
        requiredRisks: ["write"],
        broker: fixedApprovalBroker({ decision: "denied", reason: "no" }),
      },
    });
    await expect(denying.checkToolCall(writeCommand(), { risk: "write" })).rejects.toThrow(
      "需要审批",
    );
  });

  it("a capability grant skips approval until revoked or expired", async () => {
    let requests = 0;
    const broker = {
      async request() {
        requests += 1;
        return { decision: "approved" as const };
      },
    };
    const grants = new InMemoryGrantStore();
    const grant = grants.grant({ tool: "write_file", risk: "write", scope: "notes/" });
    const gov = new GovernancePlane({ approval: { requiredRisks: ["write"], broker, grants } });
    await gov.checkToolCall(writeCommand("notes/a.md"), { risk: "write" });
    expect(requests).toBe(0);
    await gov.checkToolCall(writeCommand("other/a.md"), { risk: "write" });
    expect(requests).toBe(1);
    grants.revoke(grant.id);
    await gov.checkToolCall(writeCommand("notes/a.md"), { risk: "write" });
    expect(requests).toBe(2);
  });

  it("approval can mint a scoped grant for later calls", async () => {
    const broker = new PendingApprovalBroker();
    const gov = new GovernancePlane({ approval: { requiredRisks: ["write"], broker } });
    const first = gov.checkToolCall(writeCommand("notes/b.md"), { risk: "write" });
    await new Promise((resolve) => setTimeout(resolve, 0));
    const pending = broker.listPending();
    expect(pending).toHaveLength(1);
    broker.decide(pending[0]!.id, {
      decision: "approved",
      grant: { scope: "notes/", ttlMs: 60_000 },
    });
    await first;
    expect(gov.grants.list()).toHaveLength(1);
    // Second call in scope: no new pending request.
    await gov.checkToolCall(writeCommand("notes/c.md"), { risk: "write" });
    expect(broker.listPending()).toHaveLength(0);
  });

  it("read tools never prompt unless configured, and gate forwards descriptor risk", async () => {
    let requests = 0;
    const gov = new GovernancePlane({
      approval: {
        requiredRisks: ["write"],
        broker: {
          async request() {
            requests += 1;
            return { decision: "approved" as const };
          },
        },
      },
    });
    const gate = createToolCallGate(gov);
    await gate({
      tool: "read_file",
      params: { path: "a.md" },
      descriptor: { permissions: ["filesystem.read"], idempotent: true },
    });
    expect(requests).toBe(0);
    await gate({
      tool: "write_file",
      params: { path: "a.md", content: "x" },
      descriptor: { permissions: ["filesystem.write"], idempotent: false },
    });
    expect(requests).toBe(1);
  });
});

describe("JsonGrantStore (RFC-013 persistence)", () => {
  const dirs: string[] = [];
  const makePath = () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-grants-"));
    dirs.push(dir);
    return join(dir, "grants.json");
  };
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  it("explicit grants survive a reload; approval-derived grants stay session-only", () => {
    const path = makePath();
    const first = new JsonGrantStore(path);
    const manual = first.grant({ tool: "write_file", risk: "write", scope: "notes/" });
    first.grant({ tool: "*", risk: "execute", source: "approval", label: "from-approval" });
    expect(first.list()).toHaveLength(2);

    const reloaded = new JsonGrantStore(path);
    expect(reloaded.list().map((g) => g.id)).toEqual([manual.id]);
    expect(
      reloaded.covers({ tool: "write_file", risk: "write", params: { path: "notes/a.md" } }),
    ).toBe(true);
  });

  it("revokes persist and expired grants are pruned on write and skipped on load", () => {
    const path = makePath();
    const first = new JsonGrantStore(path);
    const gone = first.grant({ tool: "a", risk: "read" });
    const expired = first.grant({ tool: "b", risk: "read", expiresAt: Date.now() - 1_000 });
    first.revoke(gone.id);
    const onDisk = JSON.parse(readFileSync(path, "utf8")) as { grants: Array<{ id: string }> };
    expect(onDisk.grants.map((g) => g.id)).not.toContain(gone.id);
    expect(onDisk.grants.map((g) => g.id)).not.toContain(expired.id);

    writeFileSync(
      path,
      JSON.stringify({ version: 1, grants: [{ ...expired, expiresAt: Date.now() - 5 }] }),
      "utf8",
    );
    expect(new JsonGrantStore(path).list()).toHaveLength(0);
  });

  it("a corrupt file starts empty instead of throwing", () => {
    const path = makePath();
    writeFileSync(path, "{not json", "utf8");
    const store = new JsonGrantStore(path);
    expect(store.list()).toEqual([]);
    store.grant({ tool: "x", risk: "read" });
    expect(new JsonGrantStore(path).list()).toHaveLength(1);
  });

  it("grants minted from approvals carry source=approval and never reach disk", async () => {
    const path = makePath();
    const grants = new JsonGrantStore(path);
    const broker = new PendingApprovalBroker();
    const gov = new GovernancePlane({ approval: { requiredRisks: ["write"], broker, grants } });
    const call = gov.checkToolCall(writeCommand("notes/x.md"), { risk: "write" });
    await new Promise((resolve) => setTimeout(resolve, 0));
    const pending = broker.listPending();
    broker.decide(pending[0]!.id, { decision: "approved", grant: { scope: "notes/" } });
    await call;
    expect(gov.grants.list()[0]?.source).toBe("approval");
    const onDisk = JSON.parse(readFileSync(path, "utf8")) as { grants: unknown[] };
    expect(onDisk.grants).toEqual([]);
  });
});
