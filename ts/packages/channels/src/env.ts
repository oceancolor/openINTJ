import { ChannelDispatcher } from "./dispatcher.js";
import type { Channel } from "./types.js";
import { createWebhookChannel } from "./webhook.js";

interface ChannelConfig {
  id: string;
  kind: string;
  url?: string;
  token?: string;
  headers?: Record<string, string>;
  kinds?: string[];
}

/**
 * Build the env-configured dispatcher (RFC-018). Only explicit env config
 * creates outbound channels; there is no default external destination:
 * - OPENINTJ_CHANNELS → JSON [{id, kind:"webhook", url, token?, kinds?}]
 */
export const channelsFromEnv = (env: NodeJS.ProcessEnv = process.env): ChannelDispatcher => {
  const dispatcher = new ChannelDispatcher();
  const raw = env["OPENINTJ_CHANNELS"];
  if (!raw) return dispatcher;
  try {
    const configs = JSON.parse(raw) as ChannelConfig[];
    for (const config of configs) {
      if (!config.id || config.kind !== "webhook" || !config.url) continue;
      const channel: Channel = createWebhookChannel({
        id: config.id,
        url: config.url,
        ...(config.token ? { token: config.token } : {}),
        ...(config.headers ? { headers: config.headers } : {}),
        ...(config.kinds ? { kinds: config.kinds } : {}),
      });
      dispatcher.register(channel);
    }
  } catch {
    // Malformed channel config must never block agent startup; run with none.
  }
  return dispatcher;
};
