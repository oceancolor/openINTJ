import { randomUUID } from "node:crypto";
import {
  AgentError,
  ErrorCode,
  type HookBus,
  type LODLevelType,
  type ShaderModeType,
  type ToolCallResult,
  type ToolDescriptor,
  ToolDescriptorSchema,
  type ToolHandler,
} from "@openintj/core";
import { CircuitBreaker } from "./circuit-breaker.js";

/**
 * 工具调用闸门：在每次工具执行前调用，抛错即拒绝该调用。
 *
 * 用于接入治理平面（策略黑名单 / 配额）而**不让 execution 反向依赖 governance** ——
 * 由 agent 负责把 governance.checkToolCall 包成 gate 传进来。gate 收到的是经 `tool.beforeCall`
 * 钩子改写后的最终 params。抛出的错误消息会成为 ToolCallResult.error（治理拒绝不计入熔断）。
 */
export type ToolGate = (ctx: {
  tool: string;
  params: Record<string, unknown>;
  descriptor: ToolDescriptor;
}) => Promise<void> | void;

export interface ToolHubOpts {
  hooks?: HookBus;
  /** 默认熔断器配置。 */
  breakerConfig?: { failureThreshold?: number; recoveryTimeoutMs?: number };
  /** 治理闸门：每次工具调用前执行；抛错即拒绝（不触发熔断）。 */
  gate?: ToolGate;
}

const _unusedTypeCheck: { lod?: LODLevelType; shader?: ShaderModeType } = {};
void _unusedTypeCheck;

export class ToolHub {
  private readonly tools = new Map<string, ToolDescriptor>();
  private readonly handlers = new Map<string, ToolHandler>();
  private readonly breakers = new Map<string, CircuitBreaker>();
  /** 最近调用历史（环形缓冲）。 */
  private readonly history: ToolCallResult[] = [];
  private readonly historyMax = 1000;
  private readonly hooks?: HookBus;
  private readonly breakerConfig: { failureThreshold?: number; recoveryTimeoutMs?: number };
  private readonly gate?: ToolGate;

  constructor(opts: ToolHubOpts = {}) {
    if (opts.hooks !== undefined) this.hooks = opts.hooks;
    this.breakerConfig = opts.breakerConfig ?? {};
    if (opts.gate !== undefined) this.gate = opts.gate;
  }

  register(descriptor: ToolDescriptor, handler?: ToolHandler): void {
    const parsed = ToolDescriptorSchema.parse(descriptor);
    this.tools.set(parsed.name, parsed);
    if (handler) this.handlers.set(parsed.name, handler);
    this.breakers.set(parsed.name, new CircuitBreaker(this.breakerConfig));
  }

  unregister(name: string): void {
    this.tools.delete(name);
    this.handlers.delete(name);
    this.breakers.delete(name);
  }

  has(name: string): boolean {
    return this.tools.has(name);
  }

  get(name: string): ToolDescriptor | undefined {
    return this.tools.get(name);
  }

  list(): ToolDescriptor[] {
    return [...this.tools.values()];
  }

  /** 工具调用：含熔断、超时、钩子事件、错误语义。 */
  async call(
    name: string,
    params: Record<string, unknown>,
    opts?: { traceId?: string; timeoutMs?: number },
  ): Promise<ToolCallResult> {
    const descriptor = this.tools.get(name);
    if (!descriptor) {
      const result: ToolCallResult = {
        toolName: name,
        success: false,
        error: `工具未注册: ${name}`,
        durationMs: 0,
        traceId: opts?.traceId ?? "",
        callId: randomUUID(),
      };
      this.recordHistory(result);
      return result;
    }

    const breaker = this.breakers.get(name);
    if (breaker && !breaker.canExecute()) {
      const result: ToolCallResult = {
        toolName: name,
        success: false,
        error: `熔断器已打开（state=${breaker.state}），暂停调用 ${name}`,
        durationMs: 0,
        traceId: opts?.traceId ?? "",
        callId: randomUUID(),
      };
      if (this.hooks) {
        const emitOpts = opts?.traceId ? { traceId: opts.traceId } : undefined;
        await this.hooks.emit(
          "event.CIRCUIT_OPENED",
          { tool: name, failureCount: descriptor.timeoutS },
          emitOpts,
        );
      }
      this.recordHistory(result);
      return result;
    }

    if (this.hooks) {
      const emitOpts = opts?.traceId ? { traceId: opts.traceId } : undefined;
      const ctx = await this.hooks.emit(
        "tool.beforeCall",
        { tool: name, params, toolDescriptor: descriptor },
        emitOpts,
      );
      // 允许 hook 改写 params（typed payload mutation）
      params = (ctx.params as Record<string, unknown>) ?? params;
    }

    // 治理闸门：策略 / 配额检查（在 params 定型后）。拒绝 = 终态失败结果，不触发熔断
    // （治理拒绝不是工具故障），也不发 tool.onError（避免被当作可重试错误）。
    if (this.gate) {
      try {
        await this.gate({ tool: name, params, descriptor });
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        const blocked: ToolCallResult = {
          toolName: name,
          success: false,
          error: errorMessage,
          durationMs: 0,
          traceId: opts?.traceId ?? "",
          callId: randomUUID(),
        };
        this.recordHistory(blocked);
        if (this.hooks) {
          const emitOpts = opts?.traceId ? { traceId: opts.traceId } : undefined;
          await this.hooks.emit("tool.afterCall", { tool: name, result: blocked }, emitOpts);
        }
        return blocked;
      }
    }

    const handler = this.handlers.get(name);
    const start = Date.now();
    let result: ToolCallResult;

    try {
      const timeoutMs = opts?.timeoutMs ?? descriptor.timeoutS * 1000;
      const output =
        handler !== undefined
          ? await runWithTimeout(handler(params), timeoutMs, name)
          : { status: "no_handler", params };

      result = {
        toolName: name,
        success: true,
        output,
        durationMs: Date.now() - start,
        traceId: opts?.traceId ?? "",
        callId: randomUUID(),
      };
      breaker?.recordSuccess();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      result = {
        toolName: name,
        success: false,
        error: errorMessage,
        durationMs: Date.now() - start,
        traceId: opts?.traceId ?? "",
        callId: randomUUID(),
      };
      breaker?.recordFailure();

      if (this.hooks) {
        const emitOpts = opts?.traceId ? { traceId: opts.traceId } : undefined;
        await this.hooks.emit(
          "tool.onError",
          {
            tool: name,
            error: err instanceof Error ? err : new Error(errorMessage),
            willRetry: descriptor.errorSemantics === "retry",
          },
          emitOpts,
        );
      }
    }

    this.recordHistory(result);

    if (this.hooks) {
      const emitOpts = opts?.traceId ? { traceId: opts.traceId } : undefined;
      await this.hooks.emit("tool.afterCall", { tool: name, result }, emitOpts);
    }

    return result;
  }

  recentHistory(limit = 50): ToolCallResult[] {
    return this.history.slice(-limit);
  }

  private recordHistory(r: ToolCallResult): void {
    this.history.push(r);
    if (this.history.length > this.historyMax) {
      this.history.splice(0, this.history.length - this.historyMax);
    }
  }

  /** 注册 4 个内置工具（参考 pi-mono 四原语）。 */
  registerBuiltinTools(handlers?: {
    readFile?: ToolHandler;
    writeFile?: ToolHandler;
    executeCommand?: ToolHandler;
    search?: ToolHandler;
  }): void {
    this.register(
      ToolDescriptorSchema.parse({
        name: "read_file",
        description: "读取文件内容",
        inputSchema: { path: "string" },
        permissions: ["filesystem.read"],
        idempotent: true,
        errorSemantics: "fail_fast",
      }),
      handlers?.readFile,
    );
    this.register(
      ToolDescriptorSchema.parse({
        name: "write_file",
        description: "写入文件内容",
        inputSchema: { path: "string", content: "string" },
        permissions: ["filesystem.write"],
        idempotent: false,
        errorSemantics: "fail_fast",
      }),
      handlers?.writeFile,
    );
    this.register(
      ToolDescriptorSchema.parse({
        name: "execute_command",
        description: "执行系统命令",
        inputSchema: { command: "string" },
        permissions: ["system.execute"],
        idempotent: false,
        timeoutS: 60,
        errorSemantics: "fail_fast",
      }),
      handlers?.executeCommand,
    );
    this.register(
      ToolDescriptorSchema.parse({
        name: "search",
        description: "搜索信息",
        inputSchema: { query: "string" },
        permissions: ["network.read"],
        idempotent: true,
        errorSemantics: "retry",
      }),
      handlers?.search,
    );
  }
}

const runWithTimeout = async <T>(
  task: T | Promise<T>,
  timeoutMs: number,
  toolName: string,
): Promise<T> => {
  if (!(task instanceof Promise)) return task;
  return await Promise.race([
    task,
    new Promise<T>((_, reject) =>
      setTimeout(
        () =>
          reject(
            new AgentError({
              code: ErrorCode.TIMEOUT,
              message: `工具 ${toolName} 调用超时 (${timeoutMs}ms)`,
              retriable: true,
              details: { tool: toolName, timeoutMs },
            }),
          ),
        timeoutMs,
      ),
    ),
  ]);
};
