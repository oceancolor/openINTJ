import type { Channel, ChannelEvent, ChannelStatus } from "./types.js";

export interface WebhookChannelOpts {
  id: string;
  url: string;
  /** Sent as `Authorization: Bearer <token>` when set. */
  token?: string;
  headers?: Record<string, string>;
  kinds?: readonly string[];
  timeoutMs?: number;
  /** Injectable fetch (tests). */
  fetchFn?: typeof fetch;
}

/** POST the event as JSON to a webhook URL (ntfy / chat bots / custom). */
export const createWebhookChannel = (opts: WebhookChannelOpts): Channel => {
  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? 10_000;
  let lastSentAt: number | undefined;
  let lastError: string | undefined;
  let sentCount = 0;
  return {
    id: opts.id,
    kind: "webhook",
    ...(opts.kinds ? { kinds: opts.kinds } : {}),
    async send(event: ChannelEvent): Promise<void> {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetchFn(opts.url, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...(opts.token ? { authorization: `Bearer ${opts.token}` } : {}),
            ...(opts.headers ?? {}),
          },
          body: JSON.stringify(event),
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`webhook ${opts.id} responded ${response.status}`);
        lastSentAt = Date.now();
        lastError = undefined;
        sentCount += 1;
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        clearTimeout(timer);
      }
    },
    status(): ChannelStatus {
      return {
        id: opts.id,
        kind: "webhook",
        ok: lastError === undefined,
        detail: opts.url,
        ...(lastSentAt !== undefined ? { lastSentAt } : {}),
        ...(lastError !== undefined ? { lastError } : {}),
        sentCount,
      };
    },
  };
};

/** Records events in memory (tests, local observability). */
export const createMemoryChannel = (opts: { id: string; kinds?: readonly string[] }): Channel => {
  const events: ChannelEvent[] = [];
  return {
    id: opts.id,
    kind: "memory",
    ...(opts.kinds ? { kinds: opts.kinds } : {}),
    async send(event: ChannelEvent): Promise<void> {
      events.push(event);
    },
    status(): ChannelStatus {
      return {
        id: opts.id,
        kind: "memory",
        ok: true,
        ...(events.length > 0 ? { lastSentAt: events.at(-1)!.at } : {}),
        sentCount: events.length,
      };
    },
    // Test/observability helper beyond the Channel contract.
    ...({ events } as object),
  } as Channel & { events: ChannelEvent[] };
};
