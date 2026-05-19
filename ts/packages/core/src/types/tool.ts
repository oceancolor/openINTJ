import { randomUUID } from "node:crypto";
import { z } from "zod";

export const ToolErrorSemanticsSchema = z.enum(["fail_fast", "retry", "ignore"]);
export type ToolErrorSemantics = z.infer<typeof ToolErrorSemanticsSchema>;

export const ToolDescriptorSchema = z.object({
  name: z.string().min(1),
  description: z.string(),
  inputSchema: z.record(z.string(), z.unknown()).default({}),
  outputSchema: z.record(z.string(), z.unknown()).default({}),
  permissions: z.array(z.string()).default([]),
  timeoutS: z.number().int().positive().default(30),
  idempotent: z.boolean().default(false),
  errorSemantics: ToolErrorSemanticsSchema.default("fail_fast"),
});

export type ToolDescriptor = z.infer<typeof ToolDescriptorSchema>;

export const ToolCallResultSchema = z.object({
  toolName: z.string(),
  success: z.boolean(),
  output: z.unknown().optional(),
  error: z.string().optional(),
  durationMs: z.number().nonnegative().default(0),
  traceId: z.string().default(""),
  callId: z.string().default(() => randomUUID()),
});

export type ToolCallResult = z.infer<typeof ToolCallResultSchema>;

export const ToolHandlerSignature = Symbol.for("openintj.toolHandler");
export type ToolHandler = (params: Record<string, unknown>) => Promise<unknown> | unknown;
