import type { ToolDescriptor, ToolHandler } from "@openintj/core";

export interface ConnectorTool {
  descriptor: ToolDescriptor;
  handler: ToolHandler;
}

export interface ConnectorStatus {
  id: string;
  displayName: string;
  ok: boolean;
  detail?: string;
  toolCount: number;
}

/**
 * A connector turns an external data source into ToolHub tools. All calls
 * keep flowing through the governance gate; connectors get no side channel.
 */
export interface Connector {
  id: string;
  displayName: string;
  /** Async discovery (e.g. MCP handshake). File connectors omit this. */
  init?(): Promise<void>;
  tools(): readonly ConnectorTool[];
  status(): ConnectorStatus;
  close?(): Promise<void> | void;
}

/** Structural subset of ToolHub (avoids a plane-execution dependency). */
export interface ToolRegistrar {
  register(descriptor: ToolDescriptor, handler?: ToolHandler): void;
}
