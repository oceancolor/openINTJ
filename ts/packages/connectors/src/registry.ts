import type { Connector, ConnectorStatus, ToolRegistrar } from "./types.js";

/**
 * Holds the connectors fixed at assemble time. Sync tools register as soon
 * as a hub is attached; async connectors (MCP) register the tools they
 * discover during initAll(). Registration is idempotent per tool name.
 */
export class ConnectorRegistry {
  private readonly connectors: Connector[] = [];
  private readonly registeredTools = new Set<string>();
  private hub: ToolRegistrar | undefined;
  private initialized = false;

  register(connector: Connector): void {
    this.connectors.push(connector);
    this.flushTools(connector);
  }

  attachHub(hub: ToolRegistrar): void {
    this.hub = hub;
    for (const connector of this.connectors) this.flushTools(connector);
  }

  private flushTools(connector: Connector): void {
    if (!this.hub) return;
    for (const tool of connector.tools()) {
      if (this.registeredTools.has(tool.descriptor.name)) continue;
      this.registeredTools.add(tool.descriptor.name);
      this.hub.register(tool.descriptor, tool.handler);
    }
  }

  /** Initialize async connectors and register whatever they discovered. */
  async initAll(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;
    for (const connector of this.connectors) {
      try {
        await connector.init?.();
      } catch {
        // A broken connector reports ok:false in status(); it must not
        // take down the agent or the other connectors.
      }
      this.flushTools(connector);
    }
  }

  list(): readonly Connector[] {
    return [...this.connectors];
  }

  statuses(): ConnectorStatus[] {
    return this.connectors.map((connector) => connector.status());
  }

  async close(): Promise<void> {
    for (const connector of this.connectors) {
      try {
        await connector.close?.();
      } catch {
        // Closing is best-effort.
      }
    }
  }
}
