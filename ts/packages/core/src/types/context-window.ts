import { z } from "zod";
import type { ChatMessage } from "../loop/types.js";
import { LODLevelSchema, type LODLevelType } from "./shader.js";

export const ShadedFragmentSchema = z.object({
  fragmentId: z.string(),
  content: z.string(),
  lod: LODLevelSchema,
  score: z.number(),
  shaderMode: z.string(),
  tokens: z.number().int().nonnegative(),
  importance: z.number(),
});
export type ShadedFragment = z.infer<typeof ShadedFragmentSchema>;

export interface ContextWindowSnapshot {
  systemPrompt: string;
  messages: ChatMessage[];
  memoryFragments: ShadedFragment[];
  totalTokens: number;
  budget: {
    maxTokens: number;
    used: number;
    available: number;
  };
}

/**
 * 默认摘要函数：保留开头 2/3 + 结尾 1/3，中间用 " ... " 连接。
 * 与 Python ShaderPipeline._default_summarize 行为对齐。
 */
export const truncateSummarize = (text: string, maxLength: number): string => {
  if (maxLength <= 0) return "";
  if (text.length <= maxLength) return text;
  const head = Math.max(1, Math.floor((maxLength * 2) / 3));
  const tail = maxLength - head - 5;
  if (tail > 0) {
    return `${text.slice(0, head)} ... ${text.slice(-tail)}`;
  }
  return text.slice(0, maxLength);
};

export type Summarizer = (text: string, maxLength: number) => string | Promise<string>;

/** Avoid unused-export tree-shake issue. */
export const _LODLevelTouch = (l: LODLevelType): LODLevelType => l;
