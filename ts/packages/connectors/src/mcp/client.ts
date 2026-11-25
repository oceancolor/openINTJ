import type { JsonRpcMessage, McpTransport } from "./transport.js";

export interface McpToolInfo {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

export interface McpContentPart {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
  /** data: URL for image/data parts (P1-7). */
  dataUrl?: string;
}

export interface McpCallResult {
  text: string;
  isError: boolean;
  raw: unknown;
  /** Full content array including image/data (P1-7). */
  content: McpContentPart[];
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
      content?: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
      isError?: boolean;
    }>("tools/call", { name, arguments: args });
    const content: McpContentPart[] = (result.content ?? []).map((part) => {
      if (
        (part.type === "image" || part.type === "data") &&
        typeof part.data === "string" &&
        part.data.length > 0
      ) {
        const mime = typeof part.mimeType === "string" && part.mimeType ? part.mimeType : "application/octet-stream";
        return {
          type: part.type,
          data: part.data,
          mimeType: mime,
          dataUrl: `data:${mime};base64,${part.data}`,
        };
      }
      if (part.type === "text" && typeof part.text === "string") {
        return { type: "text", text: part.text };
      }
      return { type: part.type, ...(part.text !== undefined ? { text: part.text } : {}) };
    });
    const textParts = content
      .filter((part) => part.type === "text" && typeof part.text === "string")
      .map((part) => part.text as string);
    const dataHints = content
      .filter((part) => part.dataUrl)
      .map((part) => `[${part.type}:${part.mimeType ?? "bin"} dataUrl length=${part.dataUrl!.length}]`);
    const text = [...textParts, ...dataHints].join("\n");
    return { text, isError: result.isError === true, raw: result, content };
  }

  close(): void {
    this.transport.close();
  }
}
