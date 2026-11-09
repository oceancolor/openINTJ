import { randomUUID } from "node:crypto";
import type { JournalTurnContext } from "./journal/types.js";

/**
 * RFC-021 F02: immutable run identity. Identity comes from the host, never
 * from model text. Every loop / hook / tool / approval participant of one
 * run carries the same RunContext so control messages (steer / inject /
 * cancel) can be attributed to exactly one run instead of "whoever happens
 * to emit on the shared bus next".
 */
export interface RunContext {
  principalId: string;
  workspaceId: string;
  taskId: string;
  conversationId: string;
  turnId: string;
  runId: string;
  parentRunId?: string;
  branchId?: string;
  traceId: string;
  /** Principal-level control epoch (RFC-021 F12); 0 until the ledger lands. */
  memoryEpoch: number;
  signal?: AbortSignal;
}

export interface CreateRunContextInput {
  conversationId: string;
  principalId?: string;
  workspaceId?: string;
  taskId?: string;
  turnId?: string;
  runId?: string;
  parentRunId?: string;
  branchId?: string;
  traceId?: string;
  memoryEpoch?: number;
  signal?: AbortSignal;
}

export const createRunContext = (input: CreateRunContextInput): RunContext => {
  const turnId = input.turnId ?? randomUUID();
  return {
    principalId: input.principalId ?? "local",
    workspaceId: input.workspaceId ?? "local",
    taskId: input.taskId ?? input.conversationId,
    conversationId: input.conversationId,
    turnId,
    runId: input.runId ?? `run:${input.conversationId}:${turnId}`,
    ...(input.parentRunId ? { parentRunId: input.parentRunId } : {}),
    ...(input.branchId ? { branchId: input.branchId } : {}),
    traceId: input.traceId ?? randomUUID(),
    memoryEpoch: input.memoryEpoch ?? 0,
    ...(input.signal ? { signal: input.signal } : {}),
  };
};

/** Derive the run identity for a journal-scoped turn (stable per turn). */
export const runContextFromJournal = (
  ctx: JournalTurnContext,
  overrides: Partial<CreateRunContextInput> = {},
): RunContext =>
  createRunContext({
    conversationId: ctx.conversationId,
    turnId: ctx.turnId,
    ...(ctx.traceId ? { traceId: ctx.traceId } : {}),
    ...overrides,
  });

/** Child run for followup turns / subagents: new turn + run, same scope. */
export const childRunContext = (
  parent: RunContext,
  overrides: Partial<CreateRunContextInput> = {},
): RunContext =>
  createRunContext({
    principalId: parent.principalId,
    workspaceId: parent.workspaceId,
    taskId: parent.taskId,
    conversationId: parent.conversationId,
    parentRunId: parent.runId,
    traceId: parent.traceId,
    memoryEpoch: parent.memoryEpoch,
    ...(parent.branchId ? { branchId: parent.branchId } : {}),
    ...overrides,
  });

/** Structured scope key (no string-concat ambiguity across id boundaries). */
export const runScopeKey = (ctx: Pick<RunContext, "principalId" | "workspaceId" | "taskId" | "conversationId">): string =>
  JSON.stringify([ctx.principalId, ctx.workspaceId, ctx.taskId, ctx.conversationId]);
