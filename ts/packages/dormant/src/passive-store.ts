import { type PassiveEvent, PassiveEventSchema } from "./types.js";

/**
 * PassiveStore —— 被动层事件缓冲区。
 *
 * 不做任何 LLM 加工，只是把所有交互流水帐记录下来供 PatternMiner 离线分析。
 * 默认环形缓冲，最多保留 N 条；满后丢最旧。
 */
export class PassiveStore {
  private events: PassiveEvent[] = [];
  readonly maxSize: number;

  constructor(maxSize = 10_000) {
    this.maxSize = maxSize;
  }

  record(event: PassiveEvent): void {
    PassiveEventSchema.parse(event);
    this.events.push(event);
    while (this.events.length > this.maxSize) {
      this.events.shift();
    }
  }

  recordBulk(events: readonly PassiveEvent[]): void {
    for (const e of events) this.record(e);
  }

  /** 按 source 过滤 + 时间窗。 */
  query(
    opts: {
      source?: PassiveEvent["source"];
      since?: number;
      limit?: number;
    } = {},
  ): PassiveEvent[] {
    let arr = this.events.slice();
    if (opts.source) arr = arr.filter((e) => e.source === opts.source);
    if (opts.since !== undefined) arr = arr.filter((e) => e.ts >= opts.since!);
    arr.sort((a, b) => b.ts - a.ts);
    if (opts.limit !== undefined) arr = arr.slice(0, opts.limit);
    return arr;
  }

  size(): number {
    return this.events.length;
  }

  /** 删除 ts < olderThanTs 的事件，返回删除条数。 */
  pruneOlderThan(olderThanTs: number): number {
    const before = this.events.length;
    this.events = this.events.filter((e) => e.ts >= olderThanTs);
    return before - this.events.length;
  }

  /** 仅保留最新（按插入序近似时间序）的 maxRows 条事件，删除其余，返回删除条数。 */
  pruneToMax(maxRows: number): number {
    if (maxRows < 0 || this.events.length <= maxRows) return 0;
    const removed = this.events.length - maxRows;
    this.events = this.events.slice(removed);
    return removed;
  }

  clear(): void {
    this.events.length = 0;
  }

  /** 全量导出（用于持久化或离线分析）。 */
  exportAll(): PassiveEvent[] {
    return [...this.events];
  }
}
