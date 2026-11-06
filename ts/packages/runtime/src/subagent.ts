import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";
import type { ToolDescriptor, ToolHandler } from "@openintj/core";

export interface SubagentChildResult {
  status: string;
  finalAnswer: string;
}

export interface SubagentRecord {
  id: string;
  label: string;
  task: string;
  childConversationId: string;
  status: "running" | "completed" | "failed" | "cancelled";
  startedAt: number;
  endedAt?: number;
  finalAnswer?: string;
  error?: string;
  /** Effective tool allowlist the child ran with (undefined = unrestricted). */
  tools?: readonly string[];
}

export interface SubagentSpawnerOpts {
  /**
   * Host-provided child runner: execute one full agent turn in the child
   * conversation. Must honor the abort signal (agent handles already do).
   * `tools` is the effective allowlist for this child (host cap ∩ per-spawn
   * narrowing); hosts enforce it via ToolHub scope. Undefined = unrestricted.
   */
  runChild: (
    childConversationId: string,
    task: string,
    signal: AbortSignal,
    tools: readonly string[] | undefined,
  ) => Promise<SubagentChildResult>;
  /**
   * Host tool ceiling for every child (tool names), or a resolver evaluated
   * per spawn so late-registered tools (connectors) are included. Per-spawn
   * `tools` can only narrow this further, never widen it.
   */
  tools?: readonly string[] | (() => readonly string[] | undefined);
  /** Delegation depth cap (default 1: children cannot delegate further). */
  maxDepth?: number;
  /** Simultaneous child runs (default 2; excess spawns are rejected). */
  maxConcurrent?: number;
  /** Per-child timeout (default 120_000 ms). */
  timeoutMs?: number;
}

export interface SubagentSpawner {
  spawn(input: {
    task: string;
    label?: string;
    /** Per-spawn narrowing; intersected with the spawner ceiling. */
    tools?: readonly string[];
  }): Promise<SubagentRecord>;
  list(): readonly SubagentRecord[];
}

const intersectTools = (
  cap: readonly string[] | undefined,
  narrow: readonly string[] | undefined,
): readonly string[] | undefined => {
  if (!cap) return narrow;
  if (!narrow) return cap;
  const allowed = new Set(narrow);
  return cap.filter((name) => allowed.has(name));
};

/**
 * RFC-012 §3 in-process subagents. A child is an opaque delegated
 * conversation: same governance as the parent but a capped tool surface
 * (host ceiling ∩ per-spawn narrowing, enforced by the host via ToolHub
 * scope), isolated journal conversation, result returned to the parent as
 * one value.
 * Depth travels through AsyncLocalStorage so a child run that tries to
 * delegate again is rejected instead of recursing.
 */
export const createSubagentSpawner = (opts: SubagentSpawnerOpts): SubagentSpawner => {
  const maxDepth = opts.maxDepth ?? 1;
  const maxConcurrent = opts.maxConcurrent ?? 2;
  const timeoutMs = opts.timeoutMs ?? 120_000;
  const depthContext = new AsyncLocalStorage<{ depth: number }>();
  const records: SubagentRecord[] = [];
  let active = 0;

  const spawn: SubagentSpawner["spawn"] = async ({ task, label, tools: narrow }) => {
    if (typeof task !== "string" || !task.trim()) {
      throw new Error("subagent task must be a non-empty string");
    }
    const depth = depthContext.getStore()?.depth ?? 0;
    if (depth >= maxDepth) {
      throw new Error(`subagent depth limit (${maxDepth}) reached`);
    }
    if (active >= maxConcurrent) {
      throw new Error(`subagent concurrency limit (${maxConcurrent}) reached`);
    }
    const cap = typeof opts.tools === "function" ? opts.tools() : opts.tools;
    const tools = intersectTools(cap, narrow);
    const id = randomUUID().slice(0, 8);
    const record: SubagentRecord = {
      id,
      label: label?.trim() || "subagent",
      task,
      childConversationId: `subagent:${id}`,
      status: "running",
      startedAt: Date.now(),
      ...(tools ? { tools } : {}),
    };
    records.push(record);
    active += 1;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error("subagent timeout")), timeoutMs);
    try {
      const result = await depthContext.run({ depth: depth + 1 }, () =>
        opts.runChild(record.childConversationId, task, controller.signal, tools),
      );
      record.status =
        result.status === "completed"
          ? "completed"
          : result.status === "cancelled"
            ? "cancelled"
            : "failed";
      record.finalAnswer = result.finalAnswer;
      return { ...record };
    } catch (error) {
      record.status = controller.signal.aborted ? "cancelled" : "failed";
      record.error = error instanceof Error ? error.message : String(error);
      throw error;
    } finally {
      clearTimeout(timer);
      record.endedAt = Date.now();
      active -= 1;
    }
  };

  return {
    spawn,
    list: () =>
      [...records].sort((a, b) => b.startedAt - a.startedAt).map((record) => ({ ...record })),
  };
};

/** Model-visible delegation tool (RFC-013 classifies it as write risk). */
export const createDelegateTool = (
  spawner: SubagentSpawner,
): { descriptor: ToolDescriptor; handler: ToolHandler } => ({
  descriptor: {
    name: "delegate_task",
    description:
      "把一个自包含的子任务委托给子 Agent 独立完成（独立上下文），返回它的最终回答。适合调研、局部验证等可隔离的工作；子 Agent 默认只有只读工具（可用 tools 参数在宿主上限内再收窄），不能再继续委托。调用需审批。",
    inputSchema: { task: "string", label: "string?", tools: "string[]?" },
    outputSchema: {},
    permissions: ["agent.delegate"],
    idempotent: false,
    errorSemantics: "fail_fast",
    timeoutS: 180,
  },
  handler: async (params) => {
    const task = params["task"];
    if (typeof task !== "string" || !task.trim()) {
      throw new Error("missing or invalid param: task");
    }
    const label = typeof params["label"] === "string" ? params["label"] : undefined;
    const rawTools = params["tools"];
    const tools = Array.isArray(rawTools)
      ? rawTools.filter((name): name is string => typeof name === "string" && name.length > 0)
      : undefined;
    const record = await spawner.spawn({
      task,
      ...(label ? { label } : {}),
      ...(tools && tools.length > 0 ? { tools } : {}),
    });
    return {
      childConversationId: record.childConversationId,
      status: record.status,
      finalAnswer: record.finalAnswer ?? "",
    };
  },
});
