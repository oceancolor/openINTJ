import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  IdentityService,
  JsonIdentityStore,
  createGravatarResolver,
  createLocalCorpusResolver,
  derivePrimaryIdentity,
  detectSeedKind,
  normalizeSeed,
} from "../src/index.js";

describe("normalize", () => {
  it("detects and normalizes seeds", () => {
    expect(detectSeedKind("Ben@Example.COM")).toBe("email");
    expect(normalizeSeed("email", " Ben@Example.COM ")).toEqual({
      kind: "email",
      value: "ben@example.com",
    });
    expect(normalizeSeed("phone", "+86 138-0013-8000")).toEqual({
      kind: "phone",
      value: "13800138000",
    });
    expect(normalizeSeed("username", "@ben_dev")).toEqual({
      kind: "username",
      value: "ben_dev",
    });
    expect(() => normalizeSeed("email", "not-an-email")).toThrow(/identity_invalid_seed/);
  });
});

describe("gravatar resolver", () => {
  it("maps a public profile into grouped proposals", async () => {
    const email = "ben@example.com";
    const hash = createHash("sha256").update(email).digest("hex");
    const fetchFn = vi.fn(async (url: string) => {
      expect(url).toBe(`https://api.gravatar.com/v3/profiles/${hash}`);
      return {
        ok: true,
        status: 200,
        json: async () => ({
          display_name: "Ben",
          profile_url: "https://gravatar.com/ben",
          accounts: [
            { url: "https://github.com/ben", verified: true },
            { url: "https://x.com/ben", verified: false },
          ],
          links: [{ label: "Blog", url: "https://ben.dev" }],
        }),
      } as Response;
    });
    const resolver = createGravatarResolver({ fetchFn: fetchFn as never, now: () => 42 });
    const proposals = await resolver.resolve({ kind: "email", value: email });
    expect(proposals).toHaveLength(4);
    expect(proposals[0]!.account).toMatchObject({ platform: "gravatar", confidence: 0.95 });
    expect(proposals[0]!.displayName).toBe("Ben");
    expect(proposals[1]!.account).toMatchObject({
      platform: "github",
      confidence: 0.9,
      verifiedAt: 42,
    });
    expect(proposals[2]!.account.confidence).toBe(0.6);
    expect(proposals[3]!.account.platform).toBe("ben");
    expect(new Set(proposals.map((p) => p.groupKey)).size).toBe(1);
  });

  it("treats 404 as empty and ignores non-email seeds", async () => {
    const fetchFn = vi.fn(async () => ({ ok: false, status: 404 }) as Response);
    const resolver = createGravatarResolver({ fetchFn: fetchFn as never });
    expect(await resolver.resolve({ kind: "email", value: "a@b.co" })).toEqual([]);
    expect(await resolver.resolve({ kind: "phone", value: "13800138000" })).toEqual([]);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });
});

const DOCS = [
  {
    source: "mail" as const,
    title: "mail: 简历投递确认",
    text: "联系方式：手机 13800138000，微信 ben_dev_01，备用邮箱 ben@work.example.com，GitHub @ben-gh",
  },
  {
    source: "notes" as const,
    title: "notes: 账号备忘",
    text: "ben@example.com 绑定了微信 ben_dev_01 和知乎 @ben-thinks",
  },
];

describe("local corpus resolver", () => {
  it("proposes co-occurring identifiers for a phone seed", async () => {
    const resolver = createLocalCorpusResolver({ documents: () => DOCS });
    const proposals = await resolver.resolve({ kind: "phone", value: "13800138000" });
    const values = proposals.map((p) => `${p.account.kind}:${p.account.value}`);
    expect(values).toContain("wechat:ben_dev_01");
    expect(values).toContain("email:ben@work.example.com");
    expect(values).toContain("username:ben-gh");
    expect(values).not.toContain("phone:13800138000");
    expect(proposals[0]!.account.source).toBe("mail");
    expect(proposals[0]!.account.evidence).toContain("mail:");
  });

  it("finds nothing without the seed in the corpus", async () => {
    const resolver = createLocalCorpusResolver({ documents: () => DOCS });
    expect(await resolver.resolve({ kind: "email", value: "nobody@x.co" })).toEqual([]);
  });
});

describe("IdentityService", () => {
  const service = () =>
    new IdentityService({
      resolvers: [createLocalCorpusResolver({ documents: () => DOCS })],
      now: () => 1000,
    });

  it("resolve → confirm links accounts onto one new person", async () => {
    const svc = service();
    const result = await svc.resolve({ value: "13800138000" });
    expect(result.reports).toEqual([
      { resolverId: "local-corpus", ok: true, count: result.proposals.length },
    ]);
    expect(svc.listPersons()).toEqual([]); // proposals never auto-enter the graph
    const confirmed = svc.confirm(result.proposals);
    expect(confirmed.linked.length).toBeGreaterThanOrEqual(3);
    expect(svc.listPersons()).toHaveLength(1);
    expect(
      svc
        .listPersons()[0]!
        .accounts.map((a) => a.kind)
        .sort(),
    ).toEqual(["email", "username", "wechat"]);
  });

  it("rejects moving an owned value to another person", async () => {
    const svc = service();
    svc.assertAccount({ platform: "wechat", kind: "wechat", value: "ben_dev_01" });
    const result = await svc.resolve({ value: "13800138000" });
    expect(() => svc.confirm(result.proposals)).toThrow(/identity_conflict/);
  });

  it("assert + unlink round-trip; empty person is removed", () => {
    const svc = service();
    const person = svc.assertAccount({
      platform: "github",
      kind: "url",
      value: "https://github.com/ben",
    });
    expect(person.accounts[0]).toMatchObject({ source: "manual", verifiedAt: null });
    const out = svc.unlink(person.accounts[0]!.id);
    expect(out).toEqual({ removed: true, personRemoved: true });
    expect(svc.listPersons()).toEqual([]);
    expect(svc.unlink("acct_nope")).toEqual({ removed: false, personRemoved: false });
  });

  it("records resolver failures in reports instead of throwing", async () => {
    const svc = new IdentityService({
      resolvers: [
        {
          id: "boom",
          resolve: async () => {
            throw new Error("offline");
          },
        },
      ],
    });
    const result = await svc.resolve({ value: "ben@example.com" });
    expect(result.proposals).toEqual([]);
    expect(result.reports).toEqual([
      { resolverId: "boom", ok: false, detail: "offline", count: 0 },
    ]);
  });
});

describe("derivePrimaryIdentity + graph-change callback", () => {
  it("derives email/name, skips auto-derived names, fires on mutations", () => {
    let notifications = 0;
    const svc = new IdentityService({
      now: () => 7,
      onGraphChanged: () => {
        notifications += 1;
      },
    });
    const manual = svc.assertAccount({
      displayName: "Ben",
      platform: "gravatar",
      kind: "email",
      value: "ben@example.com",
    });
    expect(derivePrimaryIdentity(svc.listPersons())).toEqual({
      email: "ben@example.com",
      name: "Ben",
    });
    svc.assertAccount({
      personId: manual.id,
      displayName: "Ben",
      platform: "github",
      kind: "url",
      value: "https://github.com/ben",
    });
    expect(notifications).toBe(2);
    svc.replacePersons([
      {
        id: "p1",
        displayName: "Benjamin",
        accounts: [
          {
            id: "a1",
            platform: "mail",
            kind: "email",
            value: "b@x.co",
            source: "manual",
            evidence: "t",
            confidence: 0.5,
            verifiedAt: null,
            createdAt: 1,
          },
        ],
        createdAt: 1,
        updatedAt: 1,
      },
      // Malformed person is dropped, not fatal.
      { id: 42 } as never,
    ]);
    expect(svc.listPersons()).toHaveLength(1);
    expect(derivePrimaryIdentity(svc.listPersons())).toEqual({
      email: "b@x.co",
      name: "Benjamin",
    });
    expect(notifications).toBe(3);
  });
});

describe("JsonIdentityStore", () => {
  it("round-trips and survives corrupt content", () => {
    const dir = mkdtempSync(join(tmpdir(), "identity-"));
    const file = join(dir, "identity.json");
    const store = new JsonIdentityStore(file);
    expect(store.list()).toEqual([]);
    const svc = new IdentityService({ store });
    svc.assertAccount({ platform: "mail", kind: "email", value: "ben@example.com" });
    expect(new JsonIdentityStore(file).list()).toHaveLength(1);
    writeFileSync(file, "{not json", "utf8");
    expect(new JsonIdentityStore(file).list()).toEqual([]);
  });
});
