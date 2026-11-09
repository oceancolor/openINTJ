import type { InputStructuredTaoResult } from "../input-structuring.js";
import type { Journal, JournalTurnContext } from "../journal/types.js";
import type { ChatMessage, LlmClient } from "../loop/types.js";
import type { RunContext } from "../run-context.js";

export type AgentStatus = "idle" | "running" | "cancelling";

export interface AgentHandleRunOpts {
  signal?: AbortSignal;
  history?: readonly ChatMessage[];
  llm?: LlmClient;
  journal?: JournalTurnContext;
  /** RFC-021 F02: run identity created by the handle when absent. */
  run?: RunContext;
  /** RFC-021 F01: "host" = the caller commits the journal final itself. */
  completionOwner?: "loop" | "host";
}

export interface AgentHandle {
  readonly id: string;
  readonly status: AgentStatus;
  readonly journal: Journal;
  run(query: string, opts?: AgentHandleRunOpts): Promise<InputStructuredTaoResult>;
  followup(message: string): void;
  steer(message: string): void;
  inject(message: string): void;
  cancel(cause: "user" | "command" | "dispose", opts?: { keepInbox?: boolean }): void;
  whenIdle(): Promise<void>;
  /** Unregister bus hooks and cancel any active run. Idempotent. */
  dispose(): void;
}
