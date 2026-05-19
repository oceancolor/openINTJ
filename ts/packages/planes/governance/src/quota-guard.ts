import { type QuotaConfig, QuotaConfigSchema } from "./types.js";

interface QuotaStats {
  apiCallsLastHour: number;
  tokensLastHour: number;
  toolCallsLastMinute: number;
  apiQuotaRemaining: number;
  tokenQuotaRemaining: number;
  toolQuotaRemaining: number;
}

/**
 * 滑动窗口配额守卫。
 * 注：使用 array 在小规模下足够；超大规模可改为环形 deque + 求和缓存。
 */
export class QuotaGuard {
  readonly config: QuotaConfig;
  private apiCalls: number[] = [];
  private toolCalls: number[] = [];
  private tokenUsage: Array<[number, number]> = [];
  private readonly clock: () => number;

  constructor(cfg: Partial<QuotaConfig> = {}, opts?: { clock?: () => number }) {
    this.config = QuotaConfigSchema.parse(cfg);
    this.clock = opts?.clock ?? (() => Date.now() / 1000);
  }

  private prune(now: number): void {
    const hourAgo = now - 3600;
    const minuteAgo = now - 60;
    while (this.apiCalls.length > 0 && (this.apiCalls[0] ?? 0) <= hourAgo) {
      this.apiCalls.shift();
    }
    while (this.toolCalls.length > 0 && (this.toolCalls[0] ?? 0) <= minuteAgo) {
      this.toolCalls.shift();
    }
    while (this.tokenUsage.length > 0 && (this.tokenUsage[0]?.[0] ?? 0) <= hourAgo) {
      this.tokenUsage.shift();
    }
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
    const now = this.clock();
    this.prune(now);
    return this.toolCalls.length < this.config.maxToolCallsPerMinute;
  }

  recordApiCall(): void {
    this.apiCalls.push(this.clock());
  }

  recordTokenUsage(tokens: number): void {
    if (tokens <= 0) return;
    this.tokenUsage.push([this.clock(), tokens]);
  }

  recordToolCall(): void {
    this.toolCalls.push(this.clock());
  }

  getStats(): QuotaStats {
    const now = this.clock();
    this.prune(now);
    const tokensLastHour = this.tokenUsage.reduce((s, [, n]) => s + n, 0);
    return {
      apiCallsLastHour: this.apiCalls.length,
      tokensLastHour,
      toolCallsLastMinute: this.toolCalls.length,
      apiQuotaRemaining: Math.max(0, this.config.maxApiCallsPerHour - this.apiCalls.length),
      tokenQuotaRemaining: Math.max(0, this.config.maxTokensPerHour - tokensLastHour),
      toolQuotaRemaining: Math.max(0, this.config.maxToolCallsPerMinute - this.toolCalls.length),
    };
  }
}
