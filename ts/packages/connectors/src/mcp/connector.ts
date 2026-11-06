import type { Connector, ConnectorTool } from "../types.js";
import { McpClient } from "./client.js";
import { type McpTransport, type StdioTransportOpts, createStdioTransport } from "./transport.js";

export interface McpConnectorOpts {
  /** Server id; becomes the tool namespace `mcp_<id>_*`. */
  id: string;
  displayName?: string;
  /** Stdio launch config. Exactly one of stdio / transport. */
  stdio?: StdioTransportOpts;
  /** Pre-built transport (tests / embedded servers). */
  transport?: McpTransport;
  requestTimeoutMs?: number;
}

const sanitize = (value: string): string => value.replace(/[^a-zA-Z0-9_]/g, "_");

/**
 * MCP connector: discovers a server's tools during init() and registers
 * them namespaced into ToolHub. Unknown tool semantics → default `write`
 * risk under RFC-013 (first call asks for approval on the desktop).
 */
export const createMcpConnector = (opts: McpConnectorOpts): Connector => {
  const displayName = opts.displayName ?? `MCP: ${opts.id}`;
  let client: McpClient | undefined;
  let tools: ConnectorTool[] = [];
  let failure: string | undefined;

  const ensureClient = (): McpClient => {
    if (client) return client;
    const transport =
      opts.transport ??
      (opts.stdio
        ? createStdioTransport(opts.stdio)
        : (() => {
            throw new Error("MCP connector requires stdio config or transport");
          })());
    client = new McpClient(
      transport,
      opts.requestTimeoutMs ? { requestTimeoutMs: opts.requestTimeoutMs } : {},
    );
    return client;
  };

  return {
    id: `mcp-${opts.id}`,
    displayName,
    async init() {
      try {
        const mcp = ensureClient();
        const discovered = await mcp.listTools();
        tools = discovered.map((tool) => ({
          descriptor: {
            name: `mcp_${sanitize(opts.id)}_${sanitize(tool.name)}`,
            description: tool.description ?? `MCP tool ${tool.name} (server ${opts.id})`,
            inputSchema: tool.inputSchema ?? {},
            outputSchema: {},
            // Unknown semantics: not filesystem.read, not idempotent → the
            // RFC-013 classifier defaults this to `write` (approval-gated).
            permissions: ["mcp.tool"],
            idempotent: false,
            errorSemantics: "fail_fast",
            timeoutS: 60,
          },
          handler: async (params) => {
            const result = await mcp.callTool(tool.name, params);
            if (result.isError) throw new Error(result.text || `MCP tool failed: ${tool.name}`);
            return { text: result.text };
          },
        }));
        failure = undefined;
      } catch (e) {
        failure = (e as Error).message;
        tools = [];
        throw e;
      }
    },
    tools: () => tools,
    status: () => ({
      id: `mcp-${opts.id}`,
      displayName,
      ok: failure === undefined,
      ...(failure ? { detail: failure } : {}),
      toolCount: tools.length,
    }),
    close() {
      client?.close();
      client = undefined;
    },
  };
};
