import { randomUUID } from "node:crypto";
import { AgentError, ErrorCode } from "../types/errors.js";
import {
  CANCELLABLE_EVENTS,
  type HookContext,
  type HookEventMap,
  type HookHandler,
  type HookInspectResult,
  type HookRegistration,
  type Unregister,
} from "./types.js";

/** 类型擦除的内部 handler 记录（在 on/emit 边界 cast）。 */
interface RegisteredHandler {
  id: string;
  event: string;
  // biome-ignore lint/suspicious/noExplicitAny: 类型擦除，外部 API 仍是强类型
  handler: (ctx: HookContext<any>) => void | Promise<void>;
  priority: number;
  once: boolean;
  tag: string | undefined;
  allowCancel: boolean;
}

const cancellableEvents: ReadonlySet<string> = CANCELLABLE_EVENTS as ReadonlySet<string>;

/**
 * 默认日志器；strictMode=false 下处理 handler 抛错时使用。
 */
export interface HookLogger {
  warn(message: string, details?: Record<string, unknown>): void;
  error(message: string, details?: Record<string, unknown>): void;
}

const defaultLogger: HookLogger = {
  warn: (msg, details) => console.warn("[HookBus]", msg, details ?? ""),
  error: (msg, details) => console.error("[HookBus]", msg, details ?? ""),
};

export class HookBus {
  /** key = event name, value = handlers sorted by priority desc. */
  private readonly handlers = new Map<string, RegisteredHandler[]>();
  /** 严格模式下 handler 抛错向上抛。 */
  strictMode: boolean;
  /** 同 traceId 内的事件栈深度上限（防止递归发事件死锁）。 */
  readonly maxEventStackDepth: number;
  private readonly stackPerTrace = new Map<string, number>();
  private readonly logger: HookLogger;

  constructor(opts?: {
    strictMode?: boolean;
    maxEventStackDepth?: number;
    logger?: HookLogger;
  }) {
    this.strictMode = opts?.strictMode ?? false;
    this.maxEventStackDepth = opts?.maxEventStackDepth ?? 16;
    this.logger = opts?.logger ?? defaultLogger;
  }

  on<E extends keyof HookEventMap>(
    event: E,
    handler: HookHandler<HookEventMap[E]>,
    opts?: HookRegistration,
  ): Unregister {
    const registration: RegisteredHandler = {
      id: randomUUID(),
      event: event as string,
      handler: handler as RegisteredHandler["handler"],
      priority: opts?.priority ?? 0,
      once: opts?.once ?? false,
      tag: opts?.tag,
      allowCancel: opts?.allowCancel ?? true,
    };

    const list = this.handlers.get(event as string) ?? [];
    list.push(registration);
    list.sort((a, b) => b.priority - a.priority);
    this.handlers.set(event as string, list);

    return () => {
      const arr = this.handlers.get(event as string);
      if (!arr) return;
      const idx = arr.findIndex((h) => h.id === registration.id);
      if (idx >= 0) arr.splice(idx, 1);
      if (arr.length === 0) this.handlers.delete(event as string);
    };
  }

  async emit<E extends keyof HookEventMap>(
    event: E,
    payload: HookEventMap[E],
    opts?: { traceId?: string },
  ): Promise<HookEventMap[E]> {
    const list = this.handlers.get(event as string);
    if (!list || list.length === 0) {
      return payload;
    }

    const traceId = opts?.traceId ?? "anon";
    const depth = (this.stackPerTrace.get(traceId) ?? 0) + 1;
    if (depth > this.maxEventStackDepth) {
      throw new AgentError({
        code: ErrorCode.HOOK_ERROR,
        message: `event stack depth exceeded: ${event as string} (traceId=${traceId})`,
        details: { event, traceId, depth, max: this.maxEventStackDepth },
      });
    }
    this.stackPerTrace.set(traceId, depth);

    const isCancellableEvent = cancellableEvents.has(event as string);
    let cancelled = false;
    let executedCount = 0;
    let currentPayload = payload;
    const meta: Record<string, unknown> = {};

    const snapshot = [...list];

    try {
      for (const reg of snapshot) {
        if (cancelled) break;

        const ctx: HookContext<HookEventMap[E]> = {
          eventName: event as string,
          traceId,
          get payload() {
            return currentPayload;
          },
          set payload(v: HookEventMap[E]) {
            currentPayload = v;
          },
          executedCount,
          isCancelled: cancelled,
          cancel: () => {
            if (!isCancellableEvent || !reg.allowCancel) {
              const message = `cancel not allowed for event '${event as string}'`;
              if (this.strictMode) {
                throw new AgentError({
                  code: ErrorCode.HOOK_ERROR,
                  message,
                  details: { event },
                });
              }
              this.logger.warn(message, { event, handlerId: reg.id });
              return;
            }
            cancelled = true;
          },
          replace: (next: HookEventMap[E]) => {
            currentPayload = next;
          },
          meta,
        };

        try {
          const ret = reg.handler(ctx);
          if (ret instanceof Promise) {
            await ret;
          }
        } catch (err) {
          // 框架级错误（如栈深度溢出）总是上抛，不可被 strictMode 吃掉
          if (
            err instanceof AgentError &&
            (err.code === ErrorCode.HOOK_ERROR || err.code === ErrorCode.LOOP_LIMIT_REACHED)
          ) {
            throw err;
          }
          if (this.strictMode) {
            throw err;
          }
          this.logger.error("hook handler threw", {
            event,
            handlerId: reg.id,
            tag: reg.tag,
            error: err instanceof Error ? err.message : String(err),
          });
        }

        executedCount++;
        if (reg.once) {
          const arr = this.handlers.get(event as string);
          if (arr) {
            const idx = arr.findIndex((h) => h.id === reg.id);
            if (idx >= 0) arr.splice(idx, 1);
          }
        }
      }

      return currentPayload;
    } finally {
      const newDepth = (this.stackPerTrace.get(traceId) ?? 1) - 1;
      if (newDepth <= 0) {
        this.stackPerTrace.delete(traceId);
      } else {
        this.stackPerTrace.set(traceId, newDepth);
      }
    }
  }

  offByTag(tag: string): number {
    let removed = 0;
    for (const [event, arr] of this.handlers.entries()) {
      const before = arr.length;
      const filtered = arr.filter((h) => h.tag !== tag);
      if (filtered.length === 0) {
        this.handlers.delete(event);
      } else {
        this.handlers.set(event, filtered);
      }
      removed += before - filtered.length;
    }
    return removed;
  }

  removeAllListeners(event?: keyof HookEventMap): void {
    if (event === undefined) {
      this.handlers.clear();
    } else {
      this.handlers.delete(event as string);
    }
  }

  inspect(): HookInspectResult {
    const byEvent: Record<string, number> = {};
    const byTag: Record<string, number> = {};
    let total = 0;
    for (const [event, arr] of this.handlers.entries()) {
      byEvent[event] = arr.length;
      total += arr.length;
      for (const h of arr) {
        if (h.tag !== undefined) {
          byTag[h.tag] = (byTag[h.tag] ?? 0) + 1;
        }
      }
    }
    return { total, byEvent, byTag, strictMode: this.strictMode };
  }
}
