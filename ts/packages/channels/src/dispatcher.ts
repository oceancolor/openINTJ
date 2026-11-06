import type { Channel, ChannelEvent, ChannelStatus } from "./types.js";

export interface DispatchResult {
  channelId: string;
  ok: boolean;
  error?: string;
  skipped?: boolean;
}

/**
 * Fan-out dispatcher: delivers an event to every accepting channel.
 * A failing channel never blocks the others or the caller; failures are
 * observable both in the returned results and in channel status.
 */
export class ChannelDispatcher {
  private readonly channels = new Map<string, Channel>();

  register(channel: Channel): void {
    this.channels.set(channel.id, channel);
  }

  get(id: string): Channel | undefined {
    return this.channels.get(id);
  }

  list(): ChannelStatus[] {
    return [...this.channels.values()].map((channel) => channel.status());
  }

  async dispatch(event: ChannelEvent): Promise<DispatchResult[]> {
    const results: DispatchResult[] = [];
    for (const channel of this.channels.values()) {
      if (channel.kinds && !channel.kinds.includes(event.kind)) {
        results.push({ channelId: channel.id, ok: true, skipped: true });
        continue;
      }
      try {
        await channel.send(event);
        results.push({ channelId: channel.id, ok: true });
      } catch (error) {
        results.push({
          channelId: channel.id,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    return results;
  }

  /** Send a synthetic test event to one channel. */
  async test(id: string): Promise<DispatchResult> {
    const channel = this.channels.get(id);
    if (!channel) return { channelId: id, ok: false, error: "channel_not_found" };
    const event: ChannelEvent = {
      kind: "test",
      title: "OpenINTJ 通道测试",
      body: "这是一条来自 OpenINTJ 的测试通知。",
      at: Date.now(),
    };
    try {
      await channel.send(event);
      return { channelId: id, ok: true };
    } catch (error) {
      return {
        channelId: id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }
}
