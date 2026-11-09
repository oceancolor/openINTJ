import type { ChatMessage } from "../loop/types.js";
import type { DerivedChatMessage, DerivedTrajectoryEntry, JournalEvent } from "./types.js";

export const lastClearedSeq = (events: readonly JournalEvent[]): number => {
  let seq = 0;
  for (const event of events) {
    if (event.type === "session.cleared" && event.seq > seq) seq = event.seq;
  }
  return seq;
};

export const activeJournalEvents = (events: readonly JournalEvent[]): JournalEvent[] => {
  const cleared = lastClearedSeq(events);
  return events.filter((event) => event.seq > cleared).sort((a, b) => a.seq - b.seq);
};

const textContent = (value: string): string => value;

/**
 * Model history for the next Tao/ReAct `run()`. Thought / tool.call never leak in.
 * Completed turns → user + assistant. In-progress turns → user + tool observations.
 *
 * RFC-012 compaction: the last `session.compacted` event folds every turn that
 * fully ended before it into a single summary message. Committed events are
 * never rewritten — compaction is purely a projection change, and
 * deriveChatMessages (the UI view) is unaffected.
 */
export const deriveHistory = (events: readonly JournalEvent[]): ChatMessage[] => {
  const active = activeJournalEvents(events);
  const byTurn = new Map<string, JournalEvent[]>();
  const order: string[] = [];
  for (const event of active) {
    if (!event.turnId || event.type === "session.cleared") continue;
    let bucket = byTurn.get(event.turnId);
    if (!bucket) {
      bucket = [];
      byTurn.set(event.turnId, bucket);
      order.push(event.turnId);
    }
    bucket.push(event);
  }

  const compacted = [...active].reverse().find((event) => event.type === "session.compacted");
  const coveredTurns = new Set<string>();
  if (compacted?.type === "session.compacted" && order.includes(compacted.upToTurnId)) {
    if (compacted.includedTurnIds && compacted.includedTurnIds.length > 0) {
      // F09: exact coverage — only turns whose full text entered the summary
      // input are hidden; the prefix shortcut must not hide uncovered turns.
      const completed = new Set(
        order.filter((turnId) =>
          (byTurn.get(turnId) ?? []).some((event) => event.type === "turn.end"),
        ),
      );
      for (const turnId of compacted.includedTurnIds) {
        if (completed.has(turnId)) coveredTurns.add(turnId);
      }
    } else {
      for (const turnId of order) {
        const turn = byTurn.get(turnId) ?? [];
        if (!turn.some((event) => event.type === "turn.end")) break;
        coveredTurns.add(turnId);
        if (turnId === compacted.upToTurnId) break;
      }
    }
  }

  const history: ChatMessage[] = [];
  if (compacted?.type === "session.compacted" && coveredTurns.size > 0) {
    history.push({
      role: "user",
      content: `[历史摘要]（此前 ${coveredTurns.size} 轮对话已压缩，细节省略）\n${compacted.summary}`,
    });
  }
  for (const turnId of order) {
    if (coveredTurns.has(turnId)) continue;
    const turn = byTurn.get(turnId) ?? [];
    const userEvent = turn.find((event) => event.type === "user");
    const assistantEvent = [...turn].reverse().find((event) => event.type === "assistant");
    const ended = turn.some((event) => event.type === "turn.end");
    if (userEvent?.type === "user") {
      history.push({ role: "user", content: textContent(userEvent.content) });
    }
    if (ended) {
      if (assistantEvent?.type === "assistant") {
        history.push({
          role: "assistant",
          content: textContent(assistantEvent.content),
          metadata: { messageKind: assistantEvent.messageKind },
        });
      }
      continue;
    }
    for (const event of turn) {
      if (event.type !== "tool.result") continue;
      history.push({
        role: "tool",
        name: event.tool,
        content: event.observation,
      });
    }
  }
  return history;
};

/** ChatPanel projection: user (prefer originalInput) + assistant. */
export const deriveChatMessages = (events: readonly JournalEvent[]): DerivedChatMessage[] => {
  const active = activeJournalEvents(events);
  const structureByTurn = new Map<string, unknown>();
  for (const event of active) {
    if (event.type === "user" && event.inputStructure !== undefined) {
      structureByTurn.set(event.turnId, event.inputStructure);
    }
  }
  const messages: DerivedChatMessage[] = [];
  for (const event of active) {
    if (event.type === "user") {
      const row: DerivedChatMessage = {
        id: event.id,
        conversationId: event.conversationId,
        role: "user",
        content: event.originalInput ?? event.content,
        messageKind: "message",
        createdAt: event.createdAt,
        turnId: event.turnId,
      };
      if (event.traceId) row.traceId = event.traceId;
      if (event.inputStructure !== undefined) row.inputStructure = event.inputStructure;
      messages.push(row);
    } else if (event.type === "assistant") {
      const row: DerivedChatMessage = {
        id: event.id,
        conversationId: event.conversationId,
        role: "assistant",
        content: event.content,
        messageKind: event.messageKind,
        createdAt: event.createdAt,
        turnId: event.turnId,
      };
      if (event.traceId) row.traceId = event.traceId;
      if (event.status) row.status = event.status;
      const inputStructure = structureByTurn.get(event.turnId);
      if (inputStructure !== undefined) row.inputStructure = inputStructure;
      messages.push(row);
    }
  }
  return messages;
};

export const deriveTrajectory = (events: readonly JournalEvent[]): DerivedTrajectoryEntry[] => {
  const out: DerivedTrajectoryEntry[] = [];
  for (const event of activeJournalEvents(events)) {
    if (
      event.type !== "react.thought" &&
      event.type !== "tool.call" &&
      event.type !== "tool.result" &&
      event.type !== "assistant"
    ) {
      continue;
    }
    const entry: DerivedTrajectoryEntry = {
      seq: event.seq,
      id: event.id,
      conversationId: event.conversationId,
      type: event.type,
      createdAt: event.createdAt,
      content:
        event.type === "react.thought" || event.type === "assistant"
          ? event.content
          : event.type === "tool.call"
            ? event.tool
            : event.observation,
    };
    if (event.turnId) entry.turnId = event.turnId;
    if (event.step !== undefined) entry.step = event.step;
    if (event.type === "assistant") entry.messageKind = event.messageKind;
    if (event.type === "tool.call") {
      entry.tool = event.tool;
      entry.params = event.params;
    }
    if (event.type === "tool.result") {
      entry.tool = event.tool;
      entry.ok = event.ok;
    }
    out.push(entry);
  }
  return out;
};
