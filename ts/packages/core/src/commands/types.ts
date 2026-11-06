import type { AgentHandle } from "../handle/types.js";
import type { Journal } from "../journal/types.js";

export interface SlashCommand {
  /** Includes the leading slash, e.g. "/clear". */
  name: string;
  aliases?: string[];
  description: string;
  /** Default: trimmed line equals name/aliases, case-insensitive. */
  match?(text: string): boolean;
  run(ctx: SlashCommandContext): Promise<SlashCommandResult>;
}

export interface SlashCommandContext {
  conversationId?: string;
  journal: Journal;
  handle: AgentHandle;
  raw: string;
  /**
   * RFC-012: host-provided /compact implementation (journal compaction via
   * the runtime Compactor). Absent → /compact reports it is unavailable.
   */
  compact?: () => Promise<{ compacted: boolean; message: string }>;
}

export type SlashCommandResult =
  | { kind: "ok"; message?: string }
  | { kind: "error"; message: string };

export interface SlashCommandRegistry {
  register(command: SlashCommand): void;
  list(): readonly SlashCommand[];
  match(text: string): SlashCommand | undefined;
}
