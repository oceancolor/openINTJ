import {
  type HostTurnCommitReceipt,
  type Journal,
  type JournalTurnContext,
  type TurnEndStatus,
  commitHostTurn,
} from "@openintj/core";

/**
 * RFC-021 F01: the host-side turn finalization point shared by all hosts.
 *
 * The loop runs with `completionOwner: "host"` and records only process
 * events; after host post-processing (answer revision, artifact fallback,
 * source attachment) the host calls `commitTurn` exactly once. The journal
 * final is therefore byte-identical to the answer the user was shown.
 * Hosts that have not adopted this keep the loop-owned default and must
 * treat their journal final as a draft, not the shown answer.
 */
export interface FinalizedTurn {
  userContent: string;
  originalInput?: string;
  inputStructure?: unknown;
  assistantContent: string;
  messageKind?: "message" | "answer" | "clarification";
  status?: string;
  endStatus: TurnEndStatus;
  traceId?: string;
}

export interface TurnCoordinator {
  commitTurn(
    journal: Journal,
    ctx: JournalTurnContext,
    turn: FinalizedTurn,
  ): HostTurnCommitReceipt;
}

export const createTurnCoordinator = (): TurnCoordinator => ({
  commitTurn: (journal, ctx, turn) =>
    commitHostTurn(journal, ctx, {
      userContent: turn.userContent,
      ...(turn.originalInput !== undefined ? { originalInput: turn.originalInput } : {}),
      ...(turn.inputStructure !== undefined ? { inputStructure: turn.inputStructure } : {}),
      assistantContent: turn.assistantContent,
      ...(turn.messageKind ? { messageKind: turn.messageKind } : {}),
      ...(turn.status ? { status: turn.status } : {}),
      endStatus: turn.endStatus,
      ...(turn.traceId ? { traceId: turn.traceId } : {}),
    }),
});

/** Map a TaoResult status onto the journal terminal vocabulary. */
export const taoStatusToTurnEnd = (status: string, clarify = false): TurnEndStatus => {
  if (clarify) return "clarify";
  switch (status) {
    case "completed":
      return "completed";
    case "cancelled":
      return "cancelled";
    case "timeout":
      return "timeout";
    case "max_iter_reached":
      return "max_iter";
    default:
      return "failed";
  }
};
