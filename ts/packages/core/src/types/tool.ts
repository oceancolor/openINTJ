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

/** Skill frontmatter / LLM 草稿常用 camelCase，与 ToolHub 注册名对齐。 */
const TOOL_NAME_ALIASES: Record<string, string> = {
  readFile: "read_file",
  read_file: "read_file",
  writeFile: "write_file",
  write_file: "write_file",
  executeCommand: "execute_command",
  execute_command: "execute_command",
  search: "search",
};

export const canonicalToolName = (name: string): string => {
  const trimmed = name.trim();
  if (trimmed.length === 0) return trimmed;
  return TOOL_NAME_ALIASES[trimmed] ?? trimmed;
};

export const canonicalToolNames = (names: readonly string[]): string[] => {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const name of names) {
    const canonical = canonicalToolName(name);
    if (canonical.length === 0 || seen.has(canonical)) continue;
    seen.add(canonical);
    out.push(canonical);
  }
  return out;
};
