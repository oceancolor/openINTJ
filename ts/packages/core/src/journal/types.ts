import { z } from "zod";

export const JournalEventTypeSchema = z.enum([
  "turn.start",
  "user",
  "react.thought",
  "tool.call",
  "tool.result",
  "assistant",
  "turn.end",
  "session.cleared",
  "session.compacted",
]);
export type JournalEventType = z.infer<typeof JournalEventTypeSchema>;

export const TurnEndStatusSchema = z.enum([
  "completed",
  "failed",
  "cancelled",
  "clarify",
  "max_iter",
  "timeout",
]);
export type TurnEndStatus = z.infer<typeof TurnEndStatusSchema>;

export const AssistantMessageKindSchema = z.enum(["message", "answer", "clarification"]);
export type AssistantMessageKind = z.infer<typeof AssistantMessageKindSchema>;

const JournalEventBaseSchema = z.object({
  seq: z.number().int().positive(),
  id: z.string().min(1),
  conversationId: z.string().min(1),
  turnId: z.string().min(1).optional(),
  step: z.number().int().positive().optional(),
  traceId: z.string().min(1).optional(),
  createdAt: z.number(),
});

export const JournalEventSchema = z.discriminatedUnion("type", [
  JournalEventBaseSchema.extend({
    type: z.literal("turn.start"),
    turnId: z.string().min(1),
    queryPreview: z.string().optional(),
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("user"),
    turnId: z.string().min(1),
    content: z.string(),
    originalInput: z.string().optional(),
    inputStructure: z.unknown().optional(),
    source: z.enum(["user", "steer", "inject"]).optional(),
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("react.thought"),
    turnId: z.string().min(1),
    step: z.number().int().positive(),
    content: z.string(),
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("tool.call"),
    turnId: z.string().min(1),
    step: z.number().int().positive(),
    tool: z.string().min(1),
    params: z.unknown(),
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("tool.result"),
    turnId: z.string().min(1),
    step: z.number().int().positive(),
    tool: z.string().min(1),
    ok: z.boolean(),
    observation: z.string(),
    recoverable: z.boolean().optional(),
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("assistant"),
    turnId: z.string().min(1),
    content: z.string(),
    messageKind: AssistantMessageKindSchema.default("answer"),
    status: z.string().optional(),
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("turn.end"),
    turnId: z.string().min(1),
    status: TurnEndStatusSchema,
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("session.cleared"),
    by: z.enum(["command", "system"]).default("command"),
    command: z.string().optional(),
  }),
  JournalEventBaseSchema.extend({
    type: z.literal("session.compacted"),
    /** LLM-written summary replacing the covered turns in model history. */
    summary: z.string(),
    /** How many completed turns the summary covers. */
    coveredTurns: z.number().int().nonnegative(),
    /** Last covered turn (inclusive); the fold boundary for deriveHistory. */
    upToTurnId: z.string().min(1),
    trigger: z.enum(["pressure", "overflow", "command"]),
  }),
]);
export type JournalEvent = z.infer<typeof JournalEventSchema>;

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export type JournalEventDraft = DistributiveOmit<JournalEvent, "seq" | "id" | "createdAt"> & {
  id?: string;
  createdAt?: number;
};

export interface JournalSink {
  append(draft: JournalEventDraft): JournalEvent;
}

export interface Journal extends JournalSink {
  list(conversationId: string): JournalEvent[];
  listActive(conversationId: string): JournalEvent[];
  gc(conversationId: string): number;
}

export interface JournalTurnContext {
  conversationId: string;
  turnId: string;
  sink: JournalSink;
  traceId?: string;
  /** Display/audit original; journal `user.content` is the execution text. */
  originalInput?: string;
  inputStructure?: unknown;
}

export interface DerivedChatMessage {
  id: string;
  conversationId: string;
  role: "user" | "assistant";
  content: string;
  messageKind: AssistantMessageKind | "message";
  createdAt: number;
  turnId: string;
  traceId?: string;
  status?: string;
  inputStructure?: unknown;
}

export interface DerivedTrajectoryEntry {
  seq: number;
  id: string;
  conversationId: string;
  type: Extract<JournalEventType, "react.thought" | "tool.call" | "tool.result" | "assistant">;
  createdAt: number;
  turnId?: string;
  step?: number;
  content: string;
  tool?: string;
  params?: unknown;
  ok?: boolean;
  messageKind?: AssistantMessageKind;
}
