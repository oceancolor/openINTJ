export type ChannelEventKind =
  | "schedule.fired"
  | "schedule.completed"
  | "schedule.failed"
  | "schedule.deferred"
  | "test";

export interface ChannelEvent {
  kind: ChannelEventKind;
  title: string;
  body: string;
  at: number;
  data?: Record<string, unknown>;
}

export interface ChannelStatus {
  id: string;
  kind: string;
  ok: boolean;
  detail?: string;
  lastSentAt?: number;
  lastError?: string;
  sentCount: number;
}

/** An outbound-only notification channel (RFC-018). Channels never receive. */
export interface Channel {
  id: string;
  kind: string;
  /** Event kinds this channel accepts; undefined accepts all. */
  kinds?: readonly string[];
  send(event: ChannelEvent): Promise<void>;
  status(): ChannelStatus;
}
