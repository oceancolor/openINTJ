import { randomUUID } from "node:crypto";
import { lastClearedSeq } from "./derive.js";
import { type Journal, type JournalEvent, JournalEventSchema } from "./types.js";

export const createInMemoryJournal = (opts?: {
  now?: () => number;
  idFactory?: () => string;
}): Journal => {
  const now = opts?.now ?? Date.now;
  const idFactory = opts?.idFactory ?? randomUUID;
  const events: JournalEvent[] = [];
  const seqByConversation = new Map<string, number>();

  return {
    append(draft) {
      const seq = (seqByConversation.get(draft.conversationId) ?? 0) + 1;
      seqByConversation.set(draft.conversationId, seq);
      const event = JournalEventSchema.parse({
        ...draft,
        seq,
        id: draft.id ?? idFactory(),
        createdAt: draft.createdAt ?? now(),
      });
      events.push(event);
      return event;
    },
    list(conversationId) {
      return events
        .filter((event) => event.conversationId === conversationId)
        .sort((a, b) => a.seq - b.seq);
    },
    listActive(conversationId) {
      const listed = this.list(conversationId);
      const cleared = lastClearedSeq(listed);
      return listed.filter((event) => event.seq > cleared);
    },
    gc(conversationId) {
      const listed = this.list(conversationId);
      const cleared = lastClearedSeq(listed);
      if (cleared <= 0) return 0;
      let removed = 0;
      for (let i = events.length - 1; i >= 0; i--) {
        const event = events[i];
        if (event && event.conversationId === conversationId && event.seq <= cleared) {
          events.splice(i, 1);
          removed++;
        }
      }
      return removed;
    },
  };
};

export const appendSessionCleared = (
  journal: Journal,
  conversationId: string,
  meta?: { by?: "command" | "system"; command?: string },
): JournalEvent => {
  return journal.append({
    type: "session.cleared",
    conversationId,
    by: meta?.by ?? "command",
    ...(meta?.command ? { command: meta.command } : {}),
  });
};

export const appendSessionCompacted = (
  journal: Journal,
  conversationId: string,
  input: {
    summary: string;
    coveredTurns: number;
    upToTurnId: string;
    trigger: "pressure" | "overflow" | "command";
  },
): JournalEvent => {
  return journal.append({
    type: "session.compacted",
    conversationId,
    summary: input.summary,
    coveredTurns: input.coveredTurns,
    upToTurnId: input.upToTurnId,
    trigger: input.trigger,
  });
};
