import { randomUUID } from "node:crypto";
import type { HookBus } from "../hooks/bus.js";
import type { InputStructuredTaoResult } from "../input-structuring.js";
import type { JournalTurnContext } from "../journal/types.js";
import { type RunContext, runContextFromJournal } from "../run-context.js";
import type { AgentHandle, AgentHandleRunOpts, AgentStatus } from "./types.js";

export interface CreateAgentHandleOpts {
  id: string;
  journal: AgentHandle["journal"];
  hooks: HookBus;
  runTurn: (query: string, opts?: AgentHandleRunOpts) => Promise<InputStructuredTaoResult>;
}

const combineSignals = (left?: AbortSignal, right?: AbortSignal): AbortSignal | undefined => {
  if (left && right) {
    const any = (AbortSignal as { any?: (signals: AbortSignal[]) => AbortSignal }).any;
    if (typeof any === "function") return any([left, right]);
    const controller = new AbortController();
    const onAbort = (): void => controller.abort(left.reason ?? right.reason);
    if (left.aborted || right.aborted) onAbort();
    else {
      left.addEventListener("abort", onAbort, { once: true });
      right.addEventListener("abort", onAbort, { once: true });
    }
    return controller.signal;
  }
  return left ?? right;
};

export const createAgentHandle = (opts: CreateAgentHandleOpts): AgentHandle => {
  const inbox = {
    followup: [] as string[],
    steer: [] as string[],
    inject: [] as string[],
  };
  let status: AgentStatus = "idle";
  let abort: AbortController | undefined;
  let currentRunId: string | undefined;
  let cancelRequested = false;
  let disposed = false;
  const idleWaiters: Array<() => void> = [];
  let tail: Promise<void> = Promise.resolve();

  const notifyIdle = (): void => {
    if (status !== "idle") return;
    while (idleWaiters.length > 0) idleWaiters.shift()?.();
  };

  // Read through a call so TS does not narrow the mutable status field.
  const isCancelling = (): boolean => status === "cancelling";

  /**
   * RFC-021 F02 (T03): steer/inject belong to this handle's active run.
   * When the emitter carries a runId it must match exactly; runId-less
   * emitters (legacy hosts) may only be consumed while this handle runs.
   */
  const ownsEmission = (runId: string | undefined): boolean => {
    if (currentRunId === undefined) return false;
    if (runId !== undefined) return runId === currentRunId;
    return status === "running";
  };

  const unregisterSteer = opts.hooks.on("react.beforeThought", (ctx) => {
    if (!ownsEmission(ctx.runId)) return;
    if (inbox.steer.length === 0) return;
    const steerMessages = inbox.steer.splice(0, inbox.steer.length);
    ctx.replace({ ...ctx.payload, steerMessages });
  });
  const unregisterInject = opts.hooks.on("tao.beforeThink", (ctx) => {
    if (!ownsEmission(ctx.runId)) return;
    if (inbox.inject.length === 0) return;
    const injectBlock = inbox.inject.splice(0, inbox.inject.length).join("\n\n");
    ctx.replace({ ...ctx.payload, injectBlock });
  });

  const execute = async (
    query: string,
    runOpts: AgentHandleRunOpts | undefined,
    freshTurn = false,
  ): Promise<InputStructuredTaoResult> => {
    status = "running";
    abort = new AbortController();
    const signal = combineSignals(abort.signal, runOpts?.signal);
    const journalCtx: JournalTurnContext =
      freshTurn || !runOpts?.journal
        ? {
            conversationId: opts.id,
            turnId: randomUUID(),
            sink: opts.journal,
            originalInput: query,
          }
        : runOpts.journal;
    const runCtx: RunContext =
      runOpts?.run && !freshTurn
        ? runOpts.run
        : runContextFromJournal(journalCtx, {
            ...(runOpts?.run
              ? {
                  principalId: runOpts.run.principalId,
                  workspaceId: runOpts.run.workspaceId,
                  taskId: runOpts.run.taskId,
                  parentRunId: runOpts.run.runId,
                }
              : {}),
          });
    currentRunId = runCtx.runId;
    try {
      return await opts.runTurn(query, {
        ...runOpts,
        ...(signal ? { signal } : {}),
        journal: journalCtx,
        run: runCtx,
      });
    } finally {
      if (currentRunId === runCtx.runId) currentRunId = undefined;
      abort = undefined;
    }
  };

  const run: AgentHandle["run"] = (query, runOpts) => {
    if (status === "idle") status = "running";
    cancelRequested = false;
    let resolve!: (value: InputStructuredTaoResult) => void;
    let reject!: (reason: unknown) => void;
    const result = new Promise<InputStructuredTaoResult>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    tail = tail.then(
      async () => {
        try {
          // T04: a run cancelled while still queued must not start.
          if (cancelRequested || status === "cancelling") {
            reject(new Error("cancelled_before_start"));
            return;
          }
          const first = await execute(query, runOpts);
          resolve(first);
          while (!isCancelling() && inbox.followup.length > 0) {
            const next = inbox.followup.shift();
            if (!next) break;
            // Followups are new turns (fresh turnId/runId), never a replay
            // of the parent turn's journal identity.
            await execute(next, runOpts, true);
          }
        } catch (error) {
          reject(error);
        } finally {
          status = "idle";
          notifyIdle();
        }
      },
      () => {
        status = "idle";
        notifyIdle();
      },
    );
    return result;
  };

  const handle: AgentHandle = {
    get id() {
      return opts.id;
    },
    get status() {
      return status;
    },
    get journal() {
      return opts.journal;
    },
    run,
    followup(message) {
      if (status === "idle") void run(message);
      else inbox.followup.push(message);
    },
    steer(message) {
      if (status === "idle") handle.followup(message);
      else inbox.steer.push(message);
    },
    inject(message) {
      inbox.inject.push(message);
    },
    cancel(cause, cancelOpts) {
      if (status === "idle") return;
      cancelRequested = true;
      status = "cancelling";
      if (!cancelOpts?.keepInbox) {
        inbox.followup.length = 0;
        inbox.steer.length = 0;
        inbox.inject.length = 0;
      }
      abort?.abort(cause);
    },
    whenIdle() {
      if (status === "idle") return Promise.resolve();
      return new Promise<void>((resolve) => idleWaiters.push(resolve));
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      unregisterSteer();
      unregisterInject();
      if (status !== "idle") handle.cancel("dispose");
    },
  };
  return handle;
};
