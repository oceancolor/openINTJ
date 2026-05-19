import { Mutex } from "@openintj/concurrency";

export type PoolTier = "hot" | "warm" | "cold";

export interface PoolEntry<T> {
  key: string;
  value: T;
  tier: PoolTier;
  lastAccessed: number;
  accessCount: number;
  /** 字节估算（仅信息用途，不影响淘汰）。 */
  sizeHint: number;
}

export interface ObjectPoolOpts<T> {
  /** 各层容量（条目数）。 */
  hotCapacity: number;
  warmCapacity: number;
  coldCapacity: number;
  /** 命中阈值：access >= hot 时进入 hot；< warmDemoteAfter 时降级 cold。 */
  hotPromoteAt: number;
  warmDemoteAfter: number;
  /** 当 cold 满时，cold 中最旧元素被淘汰；可注入 onEvict。 */
  onEvict?: (entry: PoolEntry<T>) => void;
}

const DEFAULTS = {
  hotCapacity: 16,
  warmCapacity: 64,
  coldCapacity: 256,
  hotPromoteAt: 5,
  warmDemoteAfter: 30_000,
};

/**
 * ObjectPool —— hot/warm/cold 三层对象池（LRU + 频率提升）。
 *
 * 用例：缓存 LLM 调用结果、tool 调用结果、检索结果 chunk。
 *
 * 策略：
 *  - get(key)：命中即提升 access count；满阈值则升至 hot
 *  - set(key, val)：默认放入 warm；hot 满时把 LRU 降到 warm
 *  - 周期性 prune：扫描 warm 中超 warmDemoteAfter 未访问的元素降级 cold
 */
export class ObjectPool<T> {
  readonly opts: Required<Omit<ObjectPoolOpts<T>, "onEvict">> & Pick<ObjectPoolOpts<T>, "onEvict">;
  private hot = new Map<string, PoolEntry<T>>();
  private warm = new Map<string, PoolEntry<T>>();
  private cold = new Map<string, PoolEntry<T>>();
  private mutex = new Mutex();

  constructor(opts: Partial<ObjectPoolOpts<T>> = {}) {
    this.opts = {
      hotCapacity: opts.hotCapacity ?? DEFAULTS.hotCapacity,
      warmCapacity: opts.warmCapacity ?? DEFAULTS.warmCapacity,
      coldCapacity: opts.coldCapacity ?? DEFAULTS.coldCapacity,
      hotPromoteAt: opts.hotPromoteAt ?? DEFAULTS.hotPromoteAt,
      warmDemoteAfter: opts.warmDemoteAfter ?? DEFAULTS.warmDemoteAfter,
      ...(opts.onEvict ? { onEvict: opts.onEvict } : {}),
    };
  }

  async get(key: string): Promise<T | undefined> {
    return this.mutex.runExclusive(() => {
      const e = this.hot.get(key) ?? this.warm.get(key) ?? this.cold.get(key);
      if (!e) return undefined;
      e.accessCount++;
      e.lastAccessed = Date.now();
      // promotion
      if (e.tier === "cold" || e.tier === "warm") {
        if (e.accessCount >= this.opts.hotPromoteAt) {
          this.removeFromTier(key);
          e.tier = "hot";
          this.insertHot(e);
        } else if (e.tier === "cold") {
          this.removeFromTier(key);
          e.tier = "warm";
          this.insertWarm(e);
        }
      }
      return e.value;
    });
  }

  async set(key: string, value: T, sizeHint = 0): Promise<void> {
    await this.mutex.runExclusive(() => {
      // 已存在则覆盖在原 tier
      const existing = this.hot.get(key) ?? this.warm.get(key) ?? this.cold.get(key);
      if (existing) {
        existing.value = value;
        existing.sizeHint = sizeHint;
        existing.lastAccessed = Date.now();
        return;
      }
      const e: PoolEntry<T> = {
        key,
        value,
        tier: "warm",
        lastAccessed: Date.now(),
        accessCount: 1,
        sizeHint,
      };
      this.insertWarm(e);
    });
  }

  async delete(key: string): Promise<boolean> {
    return this.mutex.runExclusive(() => {
      if (this.hot.delete(key)) return true;
      if (this.warm.delete(key)) return true;
      if (this.cold.delete(key)) return true;
      return false;
    });
  }

  async prune(): Promise<{ demoted: number; evicted: number }> {
    return this.mutex.runExclusive(() => {
      const now = Date.now();
      let demoted = 0;
      let evicted = 0;
      // warm → cold
      for (const [k, e] of [...this.warm.entries()]) {
        if (now - e.lastAccessed > this.opts.warmDemoteAfter) {
          this.warm.delete(k);
          e.tier = "cold";
          this.insertCold(e);
          demoted++;
        }
      }
      // cold 满 → 逐出 LRU
      while (this.cold.size > this.opts.coldCapacity) {
        const oldest = [...this.cold.entries()].sort(
          (a, b) => a[1].lastAccessed - b[1].lastAccessed,
        )[0];
        if (!oldest) break;
        this.cold.delete(oldest[0]);
        if (this.opts.onEvict) this.opts.onEvict(oldest[1]);
        evicted++;
      }
      return { demoted, evicted };
    });
  }

  stats(): {
    hot: number;
    warm: number;
    cold: number;
    total: number;
  } {
    return {
      hot: this.hot.size,
      warm: this.warm.size,
      cold: this.cold.size,
      total: this.hot.size + this.warm.size + this.cold.size,
    };
  }

  private insertHot(e: PoolEntry<T>): void {
    this.hot.set(e.key, e);
    while (this.hot.size > this.opts.hotCapacity) {
      // 把 LRU 降到 warm
      const lru = [...this.hot.entries()].sort((a, b) => a[1].lastAccessed - b[1].lastAccessed)[0];
      if (!lru) break;
      this.hot.delete(lru[0]);
      lru[1].tier = "warm";
      this.insertWarm(lru[1]);
    }
  }

  private insertWarm(e: PoolEntry<T>): void {
    this.warm.set(e.key, e);
    while (this.warm.size > this.opts.warmCapacity) {
      const lru = [...this.warm.entries()].sort((a, b) => a[1].lastAccessed - b[1].lastAccessed)[0];
      if (!lru) break;
      this.warm.delete(lru[0]);
      lru[1].tier = "cold";
      this.insertCold(lru[1]);
    }
  }

  private insertCold(e: PoolEntry<T>): void {
    this.cold.set(e.key, e);
    while (this.cold.size > this.opts.coldCapacity) {
      const lru = [...this.cold.entries()].sort((a, b) => a[1].lastAccessed - b[1].lastAccessed)[0];
      if (!lru) break;
      this.cold.delete(lru[0]);
      if (this.opts.onEvict) this.opts.onEvict(lru[1]);
    }
  }

  private removeFromTier(key: string): void {
    this.hot.delete(key);
    this.warm.delete(key);
    this.cold.delete(key);
  }
}
