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

  clear(): void {
    this.events.length = 0;
  }

  /** 全量导出（用于持久化或离线分析）。 */
  exportAll(): PassiveEvent[] {
    return [...this.events];
  }
}
