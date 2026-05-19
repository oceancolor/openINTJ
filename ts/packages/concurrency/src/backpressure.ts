import { Channel } from "./channel.js";

/**
 * Backpressure —— 速率限制 + 背压控制。
 *
 * - tokenBucket: 经典 token bucket 算法，refillRate 每秒补充 tokens
 * - slidingWindow: 用 Channel 实现固定窗口背压（fill rate=1/intervalMs）
 *
 * 用于：保护下游（LLM API、工具执行）免被瞬时洪峰压垮。
 */

export interface TokenBucketOpts {
  capacity: number;
  /** 每秒补充的 tokens 数。 */
  refillRate: number;
}

export class TokenBucket {
  readonly capacity: number;
  readonly refillRate: number;
  private tokens: number;
  private lastRefill: number;
  private waiters: Array<{
    cost: number;
    resolve: () => void;
    reject: (e: Error) => void;
  }> = [];

  constructor(opts: TokenBucketOpts) {
    if (opts.capacity <= 0) throw new Error("TokenBucket capacity must be > 0");
    if (opts.refillRate <= 0) throw new Error("TokenBucket refillRate must be > 0");
    this.capacity = opts.capacity;
    this.refillRate = opts.refillRate;
    this.tokens = opts.capacity;
    this.lastRefill = Date.now();
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    if (elapsed > 0) {
      this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.refillRate);
      this.lastRefill = now;
    }
  }

  tryAcquire(cost = 1): boolean {
    this.refill();
    if (this.tokens >= cost) {
      this.tokens -= cost;
      return true;
    }
    return false;
  }

  /**
   * 异步获取 N 个 tokens。如果 tokens 不足，等待直到 refill 补上。
   * 注意：不保证严格 FIFO，但保证最终一致进度。
   */
  async acquire(cost = 1): Promise<void> {
    if (cost > this.capacity) {
      throw new Error(`TokenBucket: cost ${cost} exceeds capacity ${this.capacity}`);
    }
    if (this.tryAcquire(cost)) return;
    return new Promise<void>((resolve, reject) => {
      this.waiters.push({ cost, resolve, reject });
      this.scheduleProcess();
    });
  }

  private scheduleTimer: ReturnType<typeof setTimeout> | undefined;

  private scheduleProcess(): void {
    if (this.scheduleTimer) return;
    const tick = (): void => {
      this.scheduleTimer = undefined;
      this.refill();
      while (this.waiters.length > 0) {
        const w = this.waiters[0]!;
        if (this.tokens >= w.cost) {
          this.tokens -= w.cost;
          this.waiters.shift();
          w.resolve();
        } else {
          break;
        }
      }
      if (this.waiters.length > 0) {
        const w = this.waiters[0]!;
        const need = w.cost - this.tokens;
        const ms = Math.max(1, Math.ceil((need / this.refillRate) * 1000));
        this.scheduleTimer = setTimeout(tick, ms);
      }
    };
    this.scheduleTimer = setTimeout(tick, 1);
  }

  get availableTokens(): number {
    this.refill();
    return this.tokens;
  }

  get pending(): number {
    return this.waiters.length;
  }
}

/**
 * BackpressureGate —— 把生产者输入限流后送入下游 Channel。
 *
 * 让 producer 自然 await，channel 满时阻塞。
 */
export class BackpressureGate<T> {
  readonly channel: Channel<T>;
  private readonly bucket?: TokenBucket;

  constructor(opts: {
    bufferSize: number;
    rateLimitPerSec?: number;
    burstCapacity?: number;
  }) {
    this.channel = new Channel<T>(opts.bufferSize);
    if (opts.rateLimitPerSec !== undefined) {
      this.bucket = new TokenBucket({
        capacity: opts.burstCapacity ?? opts.rateLimitPerSec,
        refillRate: opts.rateLimitPerSec,
      });
    }
  }

  async send(item: T): Promise<void> {
    if (this.bucket) await this.bucket.acquire(1);
    return this.channel.send(item);
  }

  async recv(): Promise<{ value: T | undefined; done: boolean }> {
    return this.channel.recv();
  }

  close(): void {
    this.channel.close();
  }
}
