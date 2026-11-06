import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";

export interface JsonRpcMessage {
  jsonrpc: "2.0";
  id?: number | string;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

/** Bidirectional JSON-RPC message channel (newline-delimited JSON for stdio). */
export interface McpTransport {
  send(message: JsonRpcMessage): void;
  onMessage(cb: (message: JsonRpcMessage) => void): void;
  onClose?(cb: () => void): void;
  close(): void;
}

export interface StdioTransportOpts {
  command: string;
  args?: readonly string[];
  env?: Record<string, string>;
  cwd?: string;
}

/** Spawn an MCP server process and speak newline-delimited JSON-RPC over stdio. */
export const createStdioTransport = (opts: StdioTransportOpts): McpTransport => {
  const child: ChildProcessWithoutNullStreams = spawn(opts.command, [...(opts.args ?? [])], {
    env: { ...process.env, ...(opts.env ?? {}) },
    ...(opts.cwd ? { cwd: opts.cwd } : {}),
    stdio: ["pipe", "pipe", "pipe"],
  });
  // Drain stderr so a chatty server can never block on a full pipe.
  child.stderr.on("data", () => {});
  let buffer = "";
  let messageCb: ((message: JsonRpcMessage) => void) | undefined;
  let closeCb: (() => void) | undefined;
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buffer += chunk;
    for (;;) {
      const newline = buffer.indexOf("\n");
      if (newline < 0) break;
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      try {
        messageCb?.(JSON.parse(line) as JsonRpcMessage);
      } catch {
        // Non-JSON noise on stdout must not crash the client.
      }
    }
  });
  child.on("close", () => closeCb?.());
  child.on("error", () => closeCb?.());

  return {
    send(message) {
      child.stdin.write(`${JSON.stringify(message)}\n`);
    },
    onMessage(cb) {
      messageCb = cb;
    },
    onClose(cb) {
      closeCb = cb;
    },
    close() {
      try {
        child.stdin.end();
        child.kill();
      } catch {
        // already gone
      }
    },
  };
};

/** Linked in-memory transport pair (tests, embedded servers). */
export const createInMemoryTransportPair = (): [McpTransport, McpTransport] => {
  let aCb: ((message: JsonRpcMessage) => void) | undefined;
  let bCb: ((message: JsonRpcMessage) => void) | undefined;
  const a: McpTransport = {
    send(message) {
      queueMicrotask(() => bCb?.(message));
    },
    onMessage(cb) {
      aCb = cb;
    },
    close() {},
  };
  const b: McpTransport = {
    send(message) {
      queueMicrotask(() => aCb?.(message));
    },
    onMessage(cb) {
      bCb = cb;
    },
    close() {},
  };
  return [a, b];
};
