import { z } from "zod";

export const ContextBudgetSchema = z.object({
  maxTokens: z.number().int().positive().default(128_000),
  reservedTokens: z.number().int().nonnegative().default(4096),
  systemPromptTokens: z.number().int().nonnegative().default(0),
  conversationTokens: z.number().int().nonnegative().default(0),
  memoryTokens: z.number().int().nonnegative().default(0),
  toolTokens: z.number().int().nonnegative().default(0),
});

export type ContextBudget = z.infer<typeof ContextBudgetSchema>;

export class ContextBudgetTracker {
  private state: ContextBudget;

  constructor(initial: Partial<ContextBudget> = {}) {
    this.state = ContextBudgetSchema.parse(initial);
  }

  get snapshot(): ContextBudget {
    return { ...this.state };
  }

  patch(patch: Partial<ContextBudget>): void {
    this.state = ContextBudgetSchema.parse({ ...this.state, ...patch });
  }

  get availableTokens(): number {
    const used =
      this.state.systemPromptTokens +
      this.state.conversationTokens +
      this.state.memoryTokens +
      this.state.toolTokens +
      this.state.reservedTokens;
    return Math.max(0, this.state.maxTokens - used);
  }

  get usageRatio(): number {
    const used =
      this.state.systemPromptTokens +
      this.state.conversationTokens +
      this.state.memoryTokens +
      this.state.toolTokens;
    const denominator = Math.max(1, this.state.maxTokens - this.state.reservedTokens);
    return Math.min(1, used / denominator);
  }

  /** 分配给记忆的 token 预算（默认占可用空间的 30%）。 */
  get memoryBudget(): number {
    const totalAvailable =
      this.state.maxTokens - this.state.reservedTokens - this.state.systemPromptTokens;
    return Math.max(0, Math.floor(totalAvailable * 0.3) - this.state.memoryTokens);
  }

  needsCompaction(threshold = 0.8): boolean {
    return this.usageRatio >= threshold;
  }
}
