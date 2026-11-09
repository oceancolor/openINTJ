import { createHash } from "node:crypto";
import type {
  AssistantMessageKind,
  Journal,
  JournalEvent,
  JournalSink,
  JournalTurnContext,
  TurnEndStatus,
} from "./types.js";

export const safeJournalAppend = (
  sink: JournalSink | undefined,
  draft: Parameters<JournalSink["append"]>[0],
): JournalEvent | undefined => {
  if (!sink) return undefined;
  try {
    return sink.append(draft);
  } catch {
    return undefined;
  }
};

const stamp = (
  ctx: JournalTurnContext,
): { conversationId: string; turnId: string; traceId?: string } => {
  const base = { conversationId: ctx.conversationId, turnId: ctx.turnId };
  return ctx.traceId ? { ...base, traceId: ctx.traceId } : base;
};

export const journalTurnStart = (
  ctx: JournalTurnContext | undefined,
  queryPreview?: string,
): void => {
  if (!ctx) return;
  const draft = {
    type: "turn.start" as const,
    ...stamp(ctx),
  };
  safeJournalAppend(ctx.sink, queryPreview ? { ...draft, queryPreview } : draft);
};

export const journalUser = (
  ctx: JournalTurnContext | undefined,
  content: string,
  extra?: { source?: "user" | "steer" | "inject" },
): void => {
  if (!ctx) return;
  const draft = {
    type: "user" as const,
    ...stamp(ctx),
    content,
    ...(extra?.source ? { source: extra.source } : {}),
  };
  const withOriginal =
    ctx.originalInput !== undefined ? { ...draft, originalInput: ctx.originalInput } : draft;
  const withStructure =
    ctx.inputStructure !== undefined
      ? { ...withOriginal, inputStructure: ctx.inputStructure }
      : withOriginal;
  safeJournalAppend(ctx.sink, withStructure);
};

export const journalThought = (
  ctx: JournalTurnContext | undefined,
  step: number,
  content: string,
): void => {
  if (!ctx) return;
  safeJournalAppend(ctx.sink, { type: "react.thought", ...stamp(ctx), step, content });
};

export const journalToolCall = (
  ctx: JournalTurnContext | undefined,
  step: number,
  tool: string,
  params: unknown,
): void => {
  if (!ctx) return;
  safeJournalAppend(ctx.sink, { type: "tool.call", ...stamp(ctx), step, tool, params });
};

export const journalToolResult = (
  ctx: JournalTurnContext | undefined,
  step: number,
  tool: string,
  ok: boolean,
  observation: string,
  recoverable?: boolean,
): void => {
  if (!ctx) return;
  const draft = { type: "tool.result" as const, ...stamp(ctx), step, tool, ok, observation };
  safeJournalAppend(ctx.sink, recoverable !== undefined ? { ...draft, recoverable } : draft);
};

export const journalAssistant = (
  ctx: JournalTurnContext | undefined,
  content: string,
  extra?: { messageKind?: AssistantMessageKind; status?: string },
): void => {
  if (!ctx) return;
  const draft = {
    type: "assistant" as const,
    ...stamp(ctx),
    content,
    messageKind: extra?.messageKind ?? "answer",
  };
  safeJournalAppend(ctx.sink, extra?.status ? { ...draft, status: extra.status } : draft);
};

export const journalTurnEnd = (
  ctx: JournalTurnContext | undefined,
  status: TurnEndStatus,
): void => {
  if (!ctx) return;
  safeJournalAppend(ctx.sink, { type: "turn.end", ...stamp(ctx), status });
};

export const recordClosedTurn = (
  sink: JournalSink,
  input: {
    conversationId: string;
    turnId: string;
    traceId?: string;
    user: { content: string; originalInput?: string; inputStructure?: unknown };
    assistant: { content: string; messageKind?: AssistantMessageKind; status?: string };
    endStatus: TurnEndStatus;
  },
): void => {
  const ctx: JournalTurnContext = {
    conversationId: input.conversationId,
    turnId: input.turnId,
    sink,
  };
  if (input.traceId) ctx.traceId = input.traceId;
  if (input.user.originalInput !== undefined) ctx.originalInput = input.user.originalInput;
  if (input.user.inputStructure !== undefined) ctx.inputStructure = input.user.inputStructure;
  journalTurnStart(ctx, input.user.content.slice(0, 120));
  journalUser(ctx, input.user.content);
  journalAssistant(ctx, input.assistant.content, {
    messageKind: input.assistant.messageKind ?? "answer",
    ...(input.assistant.status ? { status: input.assistant.status } : {}),
  });
  journalTurnEnd(ctx, input.endStatus);
};

/**
 * RFC-021 F01 (T01/T02): host-owned final commit.
 *
 * Unlike `finalizeJournalTurn` (which skips when ANY event exists), this is
 * the single finalization point after host post-processing: partial process
 * events never block it, and the committed assistant text is exactly the
 * answer the user was shown. Re-committing the same final is a replay;
 * committing a different final for an already-ended turn is a conflict and
 * writes nothing. Appends go through the sink directly — a failed commit
 * throws instead of being swallowed, so callers can report COMMIT_FAILED.
 */
export interface HostTurnCommitInput {
  userContent: string;
  originalInput?: string;
  inputStructure?: unknown;
  assistantContent: string;
  messageKind?: AssistantMessageKind;
  status?: string;
  endStatus: TurnEndStatus;
  traceId?: string;
}

export interface HostTurnCommitReceipt {
  outcome: "committed" | "replayed" | "conflict";
  finalHash: string;
  durability: "journal";
}

export const hostTurnFinalHash = (
  input: Pick<HostTurnCommitInput, "assistantContent" | "endStatus">,
): string =>
  createHash("sha256")
    .update(JSON.stringify([input.assistantContent, input.endStatus]))
    .digest("hex");

export const commitHostTurn = (
  journal: Journal,
  ctx: JournalTurnContext,
  input: HostTurnCommitInput,
): HostTurnCommitReceipt => {
  const finalHash = hostTurnFinalHash(input);
  const events = journal.list(ctx.conversationId).filter((e) => e.turnId === ctx.turnId);
  const existingEnd = events.find((e) => e.type === "turn.end");
  if (existingEnd && existingEnd.type === "turn.end") {
    const existingAssistant = [...events]
      .reverse()
      .find((e) => e.type === "assistant" && e.messageKind !== undefined);
    const same =
      existingAssistant !== undefined &&
      existingAssistant.type === "assistant" &&
      existingAssistant.content === input.assistantContent &&
      existingEnd.status === input.endStatus;
    return { outcome: same ? "replayed" : "conflict", finalHash, durability: "journal" };
  }

  const traceId = input.traceId ?? ctx.traceId;
  const base = {
    conversationId: ctx.conversationId,
    turnId: ctx.turnId,
    ...(traceId ? { traceId } : {}),
  };
  if (!events.some((e) => e.type === "turn.start")) {
    journal.append({
      type: "turn.start",
      ...base,
      queryPreview: input.userContent.slice(0, 120),
    });
  }
  const hasUser = events.some((e) => e.type === "user" && (e.source ?? "user") === "user");
  if (!hasUser) {
    journal.append({
      type: "user",
      ...base,
      content: input.userContent,
      ...(input.originalInput !== undefined ? { originalInput: input.originalInput } : {}),
      ...(input.inputStructure !== undefined ? { inputStructure: input.inputStructure } : {}),
    });
  }
  journal.append({
    type: "assistant",
    ...base,
    content: input.assistantContent,
    messageKind: input.messageKind ?? "answer",
    ...(input.status ? { status: input.status } : {}),
  });
  journal.append({ type: "turn.end", ...base, status: input.endStatus });
  return { outcome: "committed", finalHash, durability: "journal" };
};

/** Write a closed turn only when this turnId has no events yet (clarify / preflight / preview). */
export const finalizeJournalTurn = (
  journal: Journal | undefined,
  ctx: JournalTurnContext | undefined,
  input: {
    userContent: string;
    originalInput?: string;
    inputStructure?: unknown;
    assistantContent: string;
    messageKind?: AssistantMessageKind;
    status?: string;
    endStatus: TurnEndStatus;
    traceId?: string;
  },
): void => {
  if (!journal || !ctx) return;
  if (journal.list(ctx.conversationId).some((event) => event.turnId === ctx.turnId)) return;
  recordClosedTurn(journal, {
    conversationId: ctx.conversationId,
    turnId: ctx.turnId,
    ...(input.traceId ? { traceId: input.traceId } : ctx.traceId ? { traceId: ctx.traceId } : {}),
    user: {
      content: input.userContent,
      ...(input.originalInput !== undefined ? { originalInput: input.originalInput } : {}),
      ...(input.inputStructure !== undefined ? { inputStructure: input.inputStructure } : {}),
    },
    assistant: {
      content: input.assistantContent,
      ...(input.messageKind ? { messageKind: input.messageKind } : {}),
      ...(input.status ? { status: input.status } : {}),
    },
    endStatus: input.endStatus,
  });
};
