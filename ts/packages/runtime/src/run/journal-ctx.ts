import { randomUUID } from "node:crypto";
import type { InputStructureResult, Journal, JournalTurnContext } from "@openintj/core";

/**
 * Build the per-turn journal context shared by CLI/server (fallback sink)
 * and desktop (IPC-provided context wins). Returns undefined when the host
 * has no context and no fallback — journaling stays optional (RFC-009).
 */
export const buildJournalTurnContext = (input: {
  provided?: JournalTurnContext;
  fallback?: { conversationId: string; sink: Journal };
  query: string;
  structured?: InputStructureResult;
}): JournalTurnContext | undefined => {
  if (input.provided) {
    return {
      ...input.provided,
      originalInput: input.provided.originalInput ?? input.query,
      ...(input.structured ? { inputStructure: input.structured } : {}),
    };
  }
  if (!input.fallback) return undefined;
  return {
    conversationId: input.fallback.conversationId,
    turnId: randomUUID(),
    sink: input.fallback.sink,
    originalInput: input.query,
    ...(input.structured ? { inputStructure: input.structured } : {}),
  };
};
