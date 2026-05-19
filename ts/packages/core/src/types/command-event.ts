import { randomUUID } from "node:crypto";
import { z } from "zod";

export const CommandType = {
  PLAN: "PLAN",
  EXECUTE: "EXECUTE",
  EVALUATE: "EVALUATE",
  REPAIR: "REPAIR",
  SHADER_SELECT: "SHADER_SELECT",
  MEMORY_RETRIEVE: "MEMORY_RETRIEVE",
  TOOL_CALL: "TOOL_CALL",
} as const;

export type CommandTypeType = (typeof CommandType)[keyof typeof CommandType];

export const CommandTypeSchema = z.enum([
  CommandType.PLAN,
  CommandType.EXECUTE,
  CommandType.EVALUATE,
  CommandType.REPAIR,
  CommandType.SHADER_SELECT,
  CommandType.MEMORY_RETRIEVE,
  CommandType.TOOL_CALL,
]);

export const EventType = {
  PLANNED: "PLANNED",
  STEP_STARTED: "STEP_STARTED",
  STEP_FINISHED: "STEP_FINISHED",
  STEP_FAILED: "STEP_FAILED",
  POLICY_BLOCKED: "POLICY_BLOCKED",
  COMPLETED: "COMPLETED",
  SHADER_APPLIED: "SHADER_APPLIED",
  MEMORY_LOADED: "MEMORY_LOADED",
  CONTEXT_COMPACTED: "CONTEXT_COMPACTED",
  TOOL_EXECUTED: "TOOL_EXECUTED",
  CIRCUIT_OPENED: "CIRCUIT_OPENED",
  LOOP_ITERATION: "LOOP_ITERATION",
} as const;

export type EventTypeType = (typeof EventType)[keyof typeof EventType];

export const EventTypeSchema = z.enum([
  EventType.PLANNED,
  EventType.STEP_STARTED,
  EventType.STEP_FINISHED,
  EventType.STEP_FAILED,
  EventType.POLICY_BLOCKED,
  EventType.COMPLETED,
  EventType.SHADER_APPLIED,
  EventType.MEMORY_LOADED,
  EventType.CONTEXT_COMPACTED,
  EventType.TOOL_EXECUTED,
  EventType.CIRCUIT_OPENED,
  EventType.LOOP_ITERATION,
]);

export const CommandSchema = z.object({
  commandType: CommandTypeSchema,
  target: z.string(),
  payload: z.record(z.string(), z.unknown()).default({}),
  commandId: z.string().default(() => randomUUID()),
  createdAt: z.number().default(() => Date.now() / 1000),
});

export type Command = z.infer<typeof CommandSchema>;

export const FrameworkEventSchema = z.object({
  eventType: EventTypeSchema,
  source: z.string(),
  payload: z.record(z.string(), z.unknown()).default({}),
  traceId: z.string().default(""),
  createdAt: z.number().default(() => Date.now() / 1000),
});

export type FrameworkEvent = z.infer<typeof FrameworkEventSchema>;
