import type { InputStructuredTaoResult } from "../input-structuring.js";
import type { Journal, JournalTurnContext } from "../journal/types.js";
import type { ChatMessage, LlmClient } from "../loop/types.js";

export type AgentStatus = "idle" | "running" | "cancelling";

export interface AgentHandleRunOpts {
  signal?: AbortSignal;
  history?: readonly ChatMessage[];
  llm?: LlmClient;
  journal?: JournalTurnContext;
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
}
