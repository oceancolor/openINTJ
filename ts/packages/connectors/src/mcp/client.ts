import type { JsonRpcMessage, McpTransport } from "./transport.js";

export interface McpToolInfo {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

export interface McpCallResult {
  text: string;
  isError: boolean;
  raw: unknown;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const PROTOCOL_VERSION = "2024-11-05";

/** Minimal MCP client (tools only): initialize → tools/list → tools/call. */
export class McpClient {
  private nextId = 1;
  private readonly pending = new Map<number, PendingRequest>();
  private initialized = false;

  constructor(
    private readonly transport: McpTransport,
    private readonly opts: { requestTimeoutMs?: number } = {},
  ) {
    transport.onMessage((message) => this.handleMessage(message));
    transport.onClose?.(() => {
      for (const [id, req] of this.pending) {
        clearTimeout(req.timer);
        req.reject(new Error("MCP transport closed"));
        this.pending.delete(id);
      }
    });
  }

  private handleMessage(message: JsonRpcMessage): void {
    if (message.id === undefined || typeof message.id !== "number") return;
    const req = this.pending.get(message.id);
    if (!req) return;
    this.pending.delete(message.id);
    clearTimeout(req.timer);
    if (message.error) {
      req.reject(new Error(`MCP error ${message.error.code}: ${message.error.message}`));
    } else {
      req.resolve(message.result);
    }
  }

  private request<T>(method: string, params?: unknown): Promise<T> {
    const id = this.nextId++;
    const timeoutMs = this.opts.requestTimeoutMs ?? 30_000;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP request timed out: ${method} (${timeoutMs}ms)`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
        timer,
      });
      this.transport.send({
        jsonrpc: "2.0",
        id,
        method,
        ...(params !== undefined ? { params } : {}),
      });
    });
  }

  private notify(method: string): void {
    this.transport.send({ jsonrpc: "2.0", method });
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    await this.request("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "openintj", version: "0.3.0" },
    });
    this.notify("notifications/initialized");
    this.initialized = true;
  }

  async listTools(): Promise<McpToolInfo[]> {
    await this.initialize();
    const result = await this.request<{ tools?: McpToolInfo[] }>("tools/list");
    return result.tools ?? [];
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<McpCallResult> {
    await this.initialize();
    const result = await this.request<{
      content?: Array<{ type: string; text?: string }>;
      isError?: boolean;
    }>("tools/call", { name, arguments: args });
    const text = (result.content ?? [])
      .filter((part) => part.type === "text" && typeof part.text === "string")
      .map((part) => part.text)
      .join("\n");
    return { text, isError: result.isError === true, raw: result };
  }

  close(): void {
    this.transport.close();
  }
}
