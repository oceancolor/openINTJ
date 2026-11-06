import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  ConnectorRegistry,
  connectorsFromEnv,
  createCalendarConnector,
  createInMemoryTransportPair,
  createMailConnector,
  createMcpConnector,
  createNotesConnector,
  parseIcs,
  parseMbox,
} from "../src/index.js";

describe("notes connector", () => {
  it("lists, reads and searches notes, and rejects path escapes", async () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-notes-"));
    try {
      writeFileSync(join(dir, "plan.md"), "# 迁移计划\n用 TypeScript 重写 CLI。");
      writeFileSync(join(dir, "tea.txt"), "喜欢绿茶");
      writeFileSync(join(dir, "skip.png"), "binary");
      const connector = createNotesConnector({ rootDir: dir });
      const tools = new Map(connector.tools().map((t) => [t.descriptor.name, t.handler]));
      const listed = (await tools.get("notes_list")!({})) as {
        notes: Array<{ path: string }>;
      };
      expect(listed.notes.map((n) => n.path).sort()).toEqual(["plan.md", "tea.txt"]);
      const read = (await tools.get("notes_read")!({ path: "tea.txt" })) as { content: string };
      expect(read.content).toContain("绿茶");
      const hits = (await tools.get("notes_search")!({ query: "typescript" })) as {
        hits: Array<{ path: string }>;
      };
      expect(hits.hits.map((h) => h.path)).toEqual(["plan.md"]);
      expect(() => tools.get("notes_read")!({ path: "../outside.md" })).toThrow(/escapes/);
      expect(connector.tools()[0]!.descriptor).toMatchObject({
        permissions: ["filesystem.read"],
        idempotent: true,
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("calendar connector", () => {
  const ICS = [
    "BEGIN:VCALENDAR",
    "BEGIN:VEVENT",
    "UID:weekly-1",
    "SUMMARY:周会",
    "DTSTART:20260105T090000",
    "DTEND:20260105T100000",
    "RRULE:FREQ=WEEKLY;COUNT=3",
    "END:VEVENT",
    "BEGIN:VEVENT",
    "UID:once-1",
    "SUMMARY:牙医预约",
    "LOCATION:市中心诊所",
    "DTSTART:20260106T143000Z",
    "END:VEVENT",
    "BEGIN:VEVENT",
    "UID:allday-1",
    "SUMMARY:假期",
    "DTSTART;VALUE=DATE:20260107",
    "END:VEVENT",
    "END:VCALENDAR",
  ].join("\r\n");

  it("parses events with timezones and RRULE", () => {
    const events = parseIcs(ICS);
    expect(events).toHaveLength(3);
    expect(events.find((e) => e.uid === "once-1")?.start).toBe(Date.UTC(2026, 0, 6, 14, 30));
    expect(events.find((e) => e.uid === "allday-1")?.allDay).toBe(true);
  });

  it("expands weekly recurrence inside the upcoming window", async () => {
    const now = new Date(2026, 0, 5, 8, 0).getTime(); // Monday 08:00 local
    const connector = createCalendarConnector({ icsText: ICS, now: () => now });
    const upcoming = connector.tools().find((t) => t.descriptor.name === "calendar_upcoming")!;
    const out = (await upcoming.handler({ days: 15 })) as {
      events: Array<{ summary: string; start: number }>;
    };
    const weekly = out.events.filter((e) => e.summary === "周会");
    expect(weekly).toHaveLength(3); // Jan 5, 12, 19 (COUNT=3)
    expect(out.events.some((e) => e.summary === "假期")).toBe(true);
  });
});

describe("mail connector", () => {
  const MBOX = [
    "From alice@example.com Mon Jan  5 09:00:00 2026",
    "From: Alice <alice@example.com>",
    "To: ben@example.com",
    "Subject: =?UTF-8?B?5L2g5aW977yM5pys5ZGo5peg5LqL?=",
    "Date: Mon, 5 Jan 2026 09:00:00 +0800",
    "Content-Type: text/plain; charset=utf-8",
    "",
    "周一上午十点开会，请带上迁移计划。",
    "From bob@example.com Mon Jan  5 10:00:00 2026",
    "From: Bob <bob@example.com>",
    "Subject: Lunch",
    "Date: Mon, 5 Jan 2026 10:00:00 +0800",
    "",
    "Pizza today?",
    "",
  ].join("\n");

  it("parses messages and decodes encoded subjects", () => {
    const messages = parseMbox(MBOX);
    expect(messages).toHaveLength(2);
    expect(messages[0]!.subject).toBe("你好，本周无事");
    expect(messages[0]!.from).toContain("alice@example.com");
  });

  it("lists recent first and searches bodies", async () => {
    const connector = createMailConnector({ mboxText: MBOX });
    const tools = new Map(connector.tools().map((t) => [t.descriptor.name, t.handler]));
    const listed = (await tools.get("mail_list")!({})) as {
      messages: Array<{ subject: string }>;
    };
    expect(listed.messages[0]!.subject).toBe("Lunch");
    const hits = (await tools.get("mail_search")!({ query: "迁移计划" })) as {
      messages: Array<{ id: string }>;
    };
    expect(hits.messages).toHaveLength(1);
    const read = (await tools.get("mail_read")!({ id: hits.messages[0]!.id })) as {
      body: string;
    };
    expect(read.body).toContain("十点开会");
  });
});

describe("writable notes connector (RFC-017)", () => {
  it("creates, appends and trashes notes; refuses overwrite and escapes", async () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-notes-w-"));
    try {
      const connector = createNotesConnector({ rootDir: dir, writable: true });
      const tools = new Map(connector.tools().map((t) => [t.descriptor.name, t]));
      expect(tools.get("notes_create")!.descriptor).toMatchObject({
        permissions: ["filesystem.write"],
        idempotent: false,
      });
      await tools.get("notes_create")!.handler({ path: "ideas/plan.md", content: "# 计划\n" });
      await expect(
        (async () => tools.get("notes_create")!.handler({ path: "ideas/plan.md", content: "x" }))(),
      ).rejects.toThrow();
      await tools.get("notes_append")!.handler({ path: "ideas/plan.md", content: "第二步" });
      const read = (await tools.get("notes_read")!.handler({ path: "ideas/plan.md" })) as {
        content: string;
      };
      expect(read.content).toContain("第二步");
      await expect(
        (async () => tools.get("notes_create")!.handler({ path: "../evil.md", content: "x" }))(),
      ).rejects.toThrow(/escapes/);
      const deleted = (await tools.get("notes_delete")!.handler({
        path: "ideas/plan.md",
      })) as { trashedTo: string };
      expect(deleted.trashedTo).toContain(".trash/");
      expect(existsSync(join(dir, "ideas/plan.md"))).toBe(false);
      expect(existsSync(join(dir, deleted.trashedTo))).toBe(true);
      const listed = (await tools.get("notes_list")!.handler({})) as { notes: unknown[] };
      expect(listed.notes).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("stays read-only by default", () => {
    const connector = createNotesConnector({ rootDir: tmpdir() });
    expect(connector.tools().map((t) => t.descriptor.name)).toEqual([
      "notes_list",
      "notes_read",
      "notes_search",
    ]);
  });
});

describe("writable calendar connector (RFC-017)", () => {
  it("creates and deletes events in the ICS file with a .bak backup", async () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-cal-w-"));
    try {
      const icsPath = join(dir, "cal.ics");
      writeFileSync(icsPath, "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n", "utf8");
      const now = new Date(2026, 0, 5, 8, 0).getTime();
      const connector = createCalendarConnector({ icsPath, writable: true, now: () => now });
      const tools = new Map(connector.tools().map((t) => [t.descriptor.name, t]));
      const created = (await tools.get("calendar_create_event")!.handler({
        summary: "新建日程",
        start: "2026-01-06T10:00:00",
        end: "2026-01-06T11:00:00",
        location: "会议室",
      })) as { uid: string };
      expect(created.uid).toContain("openintj-");
      const upcoming = (await tools.get("calendar_upcoming")!.handler({ days: 3 })) as {
        events: Array<{ summary: string; uid: string }>;
      };
      expect(upcoming.events.map((e) => e.summary)).toEqual(["新建日程"]);
      expect(existsSync(`${icsPath}.bak`)).toBe(true);
      await tools.get("calendar_delete_event")!.handler({ uid: created.uid });
      const after = (await tools.get("calendar_upcoming")!.handler({ days: 3 })) as {
        events: unknown[];
      };
      expect(after.events).toEqual([]);
      await expect(
        (async () => tools.get("calendar_delete_event")!.handler({ uid: "missing" }))(),
      ).rejects.toThrow(/not found/);
      expect(parseIcs(readFileSync(icsPath, "utf8"))).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("does not expose writes for inline ICS text", () => {
    const connector = createCalendarConnector({ icsText: "BEGIN:VCALENDAR\r\nEND:VCALENDAR" });
    expect(connector.tools().map((t) => t.descriptor.name)).not.toContain("calendar_create_event");
  });
});

describe("mail drafts (RFC-017)", () => {
  it("composes a draft that parseMbox reads back, without sending", async () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-mail-w-"));
    try {
      const draftMboxPath = join(dir, "drafts.mbox");
      const connector = createMailConnector({ mboxText: "", draftMboxPath });
      const tools = new Map(connector.tools().map((t) => [t.descriptor.name, t]));
      const compose = tools.get("mail_compose_draft")!;
      expect(compose.descriptor).toMatchObject({
        permissions: ["filesystem.write"],
        idempotent: false,
      });
      await compose.handler({ to: "ben@example.com", subject: "草稿主题", body: "正文内容" });
      const drafts = parseMbox(readFileSync(draftMboxPath, "utf8"));
      expect(drafts).toHaveLength(1);
      expect(drafts[0]!.subject).toBe("草稿主题");
      expect(drafts[0]!.body).toContain("正文内容");
      await expect(
        (async () => compose.handler({ to: "a@b.c\nBcc: evil@x.y", subject: "s", body: "b" }))(),
      ).rejects.toThrow(/invalid header/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("env builders gate writes behind OPENINTJ_CONNECTORS_WRITE", () => {
    const off = connectorsFromEnv({ OPENINTJ_NOTES_DIR: tmpdir() });
    expect(off[0]!.tools().map((t) => t.descriptor.name)).not.toContain("notes_create");
    const on = connectorsFromEnv({
      OPENINTJ_NOTES_DIR: tmpdir(),
      OPENINTJ_CONNECTORS_WRITE: "1",
    });
    expect(on[0]!.tools().map((t) => t.descriptor.name)).toContain("notes_create");
  });
});

describe("MCP connector", () => {
  it("discovers and calls tools over an in-memory transport", async () => {
    const [clientSide, serverSide] = createInMemoryTransportPair();
    serverSide.onMessage((message) => {
      if (message.method === "initialize") {
        serverSide.send({
          jsonrpc: "2.0",
          id: message.id,
          result: { protocolVersion: "2024-11-05", capabilities: {}, serverInfo: { name: "fake" } },
        });
      } else if (message.method === "tools/list") {
        serverSide.send({
          jsonrpc: "2.0",
          id: message.id,
          result: {
            tools: [
              {
                name: "echo",
                description: "回显",
                inputSchema: { type: "object", properties: { text: { type: "string" } } },
              },
            ],
          },
        });
      } else if (message.method === "tools/call") {
        const args = (message.params as { arguments: { text: string } }).arguments;
        serverSide.send({
          jsonrpc: "2.0",
          id: message.id,
          result: { content: [{ type: "text", text: `echo:${args.text}` }] },
        });
      }
    });

    const connector = createMcpConnector({ id: "fake", transport: clientSide });
    const registered: string[] = [];
    const registry = new ConnectorRegistry();
    registry.register(connector);
    registry.attachHub({
      register: (descriptor) => {
        registered.push(descriptor.name);
      },
    });
    expect(registered).toEqual([]); // nothing before init
    await registry.initAll();
    expect(connector.status().ok).toBe(true);
    expect(registered).toEqual(["mcp_fake_echo"]);
    const tool = connector.tools()[0]!;
    // Unknown MCP semantics must classify as write risk under RFC-013.
    expect(tool.descriptor.permissions).toEqual(["mcp.tool"]);
    expect(tool.descriptor.idempotent).toBe(false);
    const out = (await tool.handler({ text: "hi" })) as { text: string };
    expect(out.text).toBe("echo:hi");
  });

  it("a failing MCP server reports ok:false and registers nothing", async () => {
    const [clientSide, serverSide] = createInMemoryTransportPair();
    serverSide.onMessage((message) => {
      serverSide.send({
        jsonrpc: "2.0",
        id: message.id,
        error: { code: -32601, message: "method not found" },
      });
    });
    const connector = createMcpConnector({ id: "bad", transport: clientSide });
    const registry = new ConnectorRegistry();
    registry.register(connector);
    registry.attachHub({ register: () => {} });
    await registry.initAll();
    expect(connector.status().ok).toBe(false);
    expect(connector.tools()).toEqual([]);
  });
});
