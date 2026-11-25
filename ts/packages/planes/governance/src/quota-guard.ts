import { type QuotaConfig, QuotaConfigSchema } from "./types.js";

interface QuotaStats {
  apiCallsLastHour: number;
  tokensLastHour: number;
  /** @deprecated 保留字段名兼容；现为当前 run 内工具调用次数。 */
  toolCallsLastMinute: number;
  toolCallsThisRun: number;
  apiQuotaRemaining: number;
  tokenQuotaRemaining: number;
  toolQuotaRemaining: number;
}

/**
 * 滑动窗口配额守卫（API/Token 按小时）+ 工具调用按 run 计费（P1-6）。
 */
export class QuotaGuard {
  readonly config: QuotaConfig;
  private apiCalls: number[] = [];
  private toolCallsThisRun = 0;
  private tokenUsage: Array<[number, number]> = [];
  private readonly clock: () => number;

  constructor(cfg: Partial<QuotaConfig> = {}, opts?: { clock?: () => number }) {
    this.config = QuotaConfigSchema.parse(cfg);
    this.clock = opts?.clock ?? (() => Date.now() / 1000);
  }

  private prune(now: number): void {
    const hourAgo = now - 3600;
    while (this.apiCalls.length > 0 && (this.apiCalls[0] ?? 0) <= hourAgo) {
      this.apiCalls.shift();
    }
    while (this.tokenUsage.length > 0 && (this.tokenUsage[0]?.[0] ?? 0) <= hourAgo) {
      this.tokenUsage.shift();
    }
  }

  /** P1-6：每个 agent run 开始时重置工具调用计数。 */
  beginRun(): void {
    this.toolCallsThisRun = 0;
  }

  checkApiQuota(): boolean {
    const now = this.clock();
    this.prune(now);
    return this.apiCalls.length < this.config.maxApiCallsPerHour;
  }

  checkTokenQuota(): boolean {
    const now = this.clock();
    this.prune(now);
    const total = this.tokenUsage.reduce((sum, [, n]) => sum + n, 0);
    return total < this.config.maxTokensPerHour;
  }

  checkToolQuota(): boolean {
    return this.toolCallsThisRun < this.config.maxToolCallsPerRun;
  }

  recordApiCall(): void {
    this.apiCalls.push(this.clock());
  }

  recordTokenUsage(tokens: number): void {
    if (tokens <= 0) return;
    this.tokenUsage.push([this.clock(), tokens]);
  }

  recordToolCall(): void {
    this.toolCallsThisRun += 1;
  }

  getStats(): QuotaStats {
    const now = this.clock();
    this.prune(now);
    const tokensLastHour = this.tokenUsage.reduce((s, [, n]) => s + n, 0);
    return {
      apiCallsLastHour: this.apiCalls.length,
      tokensLastHour,
      toolCallsLastMinute: this.toolCallsThisRun,
      toolCallsThisRun: this.toolCallsThisRun,
      apiQuotaRemaining: Math.max(0, this.config.maxApiCallsPerHour - this.apiCalls.length),
      tokenQuotaRemaining: Math.max(0, this.config.maxTokensPerHour - tokensLastHour),
      toolQuotaRemaining: Math.max(0, this.config.maxToolCallsPerRun - this.toolCallsThisRun),
    };
  }
}
