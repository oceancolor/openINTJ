import type { ChatMessage, TaskTypeType } from "@openintj/core";

/**
 * RFC-011 §3 prompt section. Smaller `order` renders earlier.
 * Locked relative order (RFC-006): product-behavior(100) → user-persona(200)
 * → skills(300) → plan-advice(350) → memory(400). New sections must use a
 * free slot or >400; never reorder 100→400.
 */
export interface PromptAssembleCtx {
  query: string;
  conversationId?: string;
  history: readonly ChatMessage[];
  taskType?: TaskTypeType;
  traceId?: string;
  /** RFC-011: plan advice rendered as a system section, never spliced into the user query. */
  planAdviceBlock?: string;
}

export type PromptSectionProvider = (
  ctx: PromptAssembleCtx,
) => string | undefined | Promise<string | undefined>;

export interface PromptSection {
  id: string;
  order: number;
  render: PromptSectionProvider;
}

export interface PromptSectionRegistry {
  register(section: PromptSection): void;
  /** Sections sorted by order (stable by id on ties). Returns a copy. */
  list(): readonly PromptSection[];
  /** Render non-empty sections in order, joined by a blank line. */
  render(ctx: PromptAssembleCtx): Promise<string>;
}

export const PROMPT_SECTION_ORDERS = {
  productBehavior: 100,
  userPersona: 200,
  skills: 300,
  planAdvice: 350,
  memory: 400,
} as const;
