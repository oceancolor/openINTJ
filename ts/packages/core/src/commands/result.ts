import type { InputStructuredTaoResult } from "../input-structuring.js";
import { TaskType, getShaderForTask } from "../types/shader.js";
import type {
  SlashCommand,
  SlashCommandContext,
  SlashCommandRegistry,
  SlashCommandResult,
} from "./types.js";

export const slashCommandResultToTao = (result: SlashCommandResult): InputStructuredTaoResult => ({
  traceId: "command",
  status: result.kind === "ok" ? "completed" : "failed",
  finalAnswer: result.message ?? (result.kind === "ok" ? "完成。" : result.message),
  iterations: 0,
  reactTotalSteps: 0,
  totalTokensSpent: 0,
  durationMs: 0,
  trajectory: [],
  taskType: TaskType.QUICK_RESPONSE,
  shaderMode: getShaderForTask(TaskType.QUICK_RESPONSE),
  metrics: { slashCommand: 1 },
});

export const invokeSlashCommand = async (
  registry: SlashCommandRegistry,
  text: string,
  ctx: Omit<SlashCommandContext, "raw">,
): Promise<{ command: SlashCommand; result: SlashCommandResult } | undefined> => {
  const command = registry.match(text);
  if (!command) return undefined;
  const result = await command.run({ ...ctx, raw: text });
  return { command, result };
};
