import { createCalendarConnector } from "./calendar.js";
import { createMailConnector } from "./mail.js";
import { createMcpConnector } from "./mcp/connector.js";
import { createNotesConnector } from "./notes.js";
import type { Connector } from "./types.js";

interface McpServerConfig {
  id: string;
  command: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
}

/**
 * Build the env-configured connectors (RFC-016 §1, RFC-017 write opt-in). No
 * credentials are read or stored here — only local paths and MCP commands:
 * - OPENINTJ_NOTES_DIR    → notes connector
 * - OPENINTJ_CALENDAR_ICS → calendar connector (.ics path)
 * - OPENINTJ_MAIL_MBOX    → mail connector (mbox path)
 * - OPENINTJ_MCP_SERVERS  → JSON [{id, command, args?, env?, cwd?}]
 * - OPENINTJ_CONNECTORS_WRITE=1 → writable notes/calendar (+ mail drafts)
 * - OPENINTJ_MAIL_DRAFT_MBOX  → drafts mbox path (requires the write flag)
 */
export const connectorsFromEnv = (env: NodeJS.ProcessEnv = process.env): Connector[] => {
  const connectors: Connector[] = [];
  const writable = env["OPENINTJ_CONNECTORS_WRITE"] === "1";
  const notesDir = env["OPENINTJ_NOTES_DIR"];
  if (notesDir) connectors.push(createNotesConnector({ rootDir: notesDir, writable }));
  const icsPath = env["OPENINTJ_CALENDAR_ICS"];
  if (icsPath) connectors.push(createCalendarConnector({ icsPath, writable }));
  const mboxPath = env["OPENINTJ_MAIL_MBOX"];
  if (mboxPath) {
    const draftMboxPath = env["OPENINTJ_MAIL_DRAFT_MBOX"];
    connectors.push(
      createMailConnector({
        mboxPath,
        ...(writable && draftMboxPath ? { draftMboxPath } : {}),
      }),
    );
  }
  const mcpRaw = env["OPENINTJ_MCP_SERVERS"];
  if (mcpRaw) {
    try {
      const servers = JSON.parse(mcpRaw) as McpServerConfig[];
      for (const server of servers) {
        if (!server.id || !server.command) continue;
        connectors.push(
          createMcpConnector({
            id: server.id,
            stdio: {
              command: server.command,
              ...(server.args ? { args: server.args } : {}),
              ...(server.env ? { env: server.env } : {}),
              ...(server.cwd ? { cwd: server.cwd } : {}),
            },
          }),
        );
      }
    } catch {
      // Malformed MCP config must never block agent startup; skip it.
    }
  }
  return connectors;
};
