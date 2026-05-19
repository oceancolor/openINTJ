import { type CircuitBreakerConfig, DEFAULT_BREAKER } from "./types.js";

export type BreakerState = "closed" | "open" | "half_open";

export class CircuitBreaker {
  readonly config: CircuitBreakerConfig;
  private failureCount = 0;
  private lastFailureMs = 0;
  private _state: BreakerState = "closed";
  private readonly clock: () => number;

  constructor(cfg: Partial<CircuitBreakerConfig> = {}, opts?: { clock?: () => number }) {
    this.config = { ...DEFAULT_BREAKER, ...cfg };
    this.clock = opts?.clock ?? (() => Date.now());
  }

  get state(): BreakerState {
    return this._state;
  }

  recordSuccess(): void {
    this.failureCount = 0;
    this._state = "closed";
  }

  recordFailure(): void {
    this.failureCount++;
    this.lastFailureMs = this.clock();
    if (this.failureCount >= this.config.failureThreshold) {
      this._state = "open";
    }
  }

  canExecute(): boolean {
    if (this._state === "closed") return true;
    if (this._state === "open") {
      if (this.clock() - this.lastFailureMs >= this.config.recoveryTimeoutMs) {
        this._state = "half_open";
        return true;
      }
      return false;
    }
    return true;
  }

  reset(): void {
    this.failureCount = 0;
    this.lastFailureMs = 0;
    this._state = "closed";
  }
}
