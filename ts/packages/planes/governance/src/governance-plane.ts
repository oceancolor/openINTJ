import {
  AgentError,
  type Command,
  CommandSchema,
  CommandType,
  ErrorCode,
  type HookBus,
} from "@openintj/core";
import { AuditTrail } from "./audit-trail.js";
import { PolicyEngine } from "./policy-engine.js";
import { QuotaGuard } from "./quota-guard.js";
import { type AuditEvent, AuditEventSchema } from "./types.js";

export interface GovernancePlaneOpts {
  policyEngine?: PolicyEngine;
  auditTrail?: AuditTrail;
  quotaGuard?: QuotaGuard;
  hooks?: HookBus;
}

export class GovernancePlane {
  readonly name = "governance-plane";
  readonly policyEngine: PolicyEngine;
  readonly auditTrail: AuditTrail;
  readonly quotaGuard: QuotaGuard;
  private readonly hooks: HookBus | undefined;

  constructor(opts: GovernancePlaneOpts = {}) {
    this.policyEngine = opts.policyEngine ?? new PolicyEngine();
    this.auditTrail = opts.auditTrail ?? new AuditTrail();
    this.quotaGuard = opts.quotaGuard ?? new QuotaGuard();
    this.hooks = opts.hooks;
  }

  /**
   * 在命令执行前进行配额 + 策略检查并记录审计。
   * - 配额超限 → blocked + 抛 POLICY_BLOCKED(retriable=true)
   * - 策略黑名单 → 抛 POLICY_BLOCKED(retriable=false)
   * - 通过 hook 暴露 policy.beforeCheck / policy.afterCheck / policy.onBlock
   */
  async checkAndRecord(command: Command): Promise<AuditEvent> {
    if (this.hooks) {
      await this.hooks.emit("policy.beforeCheck", { command });
    }

    if (!this.quotaGuard.checkApiQuota()) {
      const event = AuditEventSchema.parse({
        action: command.commandType,
        target: command.target,
        result: "blocked",
        riskLevel: "high",
        details: { reason: "API 调用配额已用尽" },
      });
      this.auditTrail.record(event);
      if (this.hooks) {
        await this.hooks.emit("policy.onBlock", {
          command,
          auditEvent: event,
          reason: "API 配额已用尽",
        });
      }
      throw new AgentError({
        code: ErrorCode.POLICY_BLOCKED,
        message: "API 调用配额已用尽",
        retriable: true,
        details: { auditEventId: event.eventId },
      });
    }

    let auditEvent: AuditEvent;
    try {
      auditEvent = this.policyEngine.check(command);
    } catch (err) {
      if (err instanceof AgentError && err.code === ErrorCode.POLICY_BLOCKED) {
        const event = AuditEventSchema.parse({
          action: command.commandType,
          target: command.target,
          result: "blocked",
          riskLevel: "critical",
          details: {
            reason: "目标在黑名单中",
            errorCode: err.code,
          },
        });
        this.auditTrail.record(event);
        if (this.hooks) {
          await this.hooks.emit("policy.onBlock", {
            command,
            auditEvent: event,
            reason: err.message,
          });
        }
      }
      throw err;
    }

    this.auditTrail.record(auditEvent);
    this.quotaGuard.recordApiCall();

    if (this.hooks) {
      await this.hooks.emit("policy.afterCheck", { command, auditEvent });
    }

    return auditEvent;
  }

  /**
   * 工具调用前的治理检查（RFC-004 §8：策略边界接进工具执行）。
   * 与 {@link checkAndRecord} 同构，但走**工具调用配额**（每分钟）而非 API 配额：
   * - 工具配额超限 → blocked + 抛 POLICY_BLOCKED(retriable=true)
   * - 策略黑名单 → 抛 POLICY_BLOCKED(retriable=false)
   * - 通过后记审计 + `recordToolCall()`。
   *
   * 供 agent 包成 `ToolHub` 的 gate 使用（execution 不反向依赖 governance）。
   */
  async checkToolCall(command: Command): Promise<AuditEvent> {
    if (this.hooks) {
      await this.hooks.emit("policy.beforeCheck", { command });
    }

    if (!this.quotaGuard.checkToolQuota()) {
      const event = AuditEventSchema.parse({
        action: command.commandType,
        target: command.target,
        result: "blocked",
        riskLevel: "high",
        details: { reason: "工具调用配额已用尽（每分钟）" },
      });
      this.auditTrail.record(event);
      if (this.hooks) {
        await this.hooks.emit("policy.onBlock", {
          command,
          auditEvent: event,
          reason: "工具调用配额已用尽",
        });
      }
      throw new AgentError({
        code: ErrorCode.POLICY_BLOCKED,
        message: `工具调用配额已用尽：${command.target}`,
        retriable: true,
        details: { auditEventId: event.eventId },
      });
    }

    let auditEvent: AuditEvent;
    try {
      auditEvent = this.policyEngine.check(command);
    } catch (err) {
      if (err instanceof AgentError && err.code === ErrorCode.POLICY_BLOCKED) {
        const event = AuditEventSchema.parse({
          action: command.commandType,
          target: command.target,
          result: "blocked",
          riskLevel: "critical",
          details: { reason: "工具在黑名单中", errorCode: err.code },
        });
        this.auditTrail.record(event);
        if (this.hooks) {
          await this.hooks.emit("policy.onBlock", {
            command,
            auditEvent: event,
            reason: err.message,
          });
        }
      }
      throw err;
    }

    this.auditTrail.record(auditEvent);
    this.quotaGuard.recordToolCall();

    if (this.hooks) {
      await this.hooks.emit("policy.afterCheck", { command, auditEvent });
    }

    return auditEvent;
  }

  getStats(): {
    audit: ReturnType<AuditTrail["getStats"]>;
    quota: ReturnType<QuotaGuard["getStats"]>;
    strictMode: boolean;
  } {
    return {
      audit: this.auditTrail.getStats(),
      quota: this.quotaGuard.getStats(),
      strictMode: this.policyEngine.config.strictMode,
    };
  }
}

/**
 * 把 GovernancePlane 包成一个「工具调用闸门」函数，供 `ToolHub({ gate })` 使用。
 *
 * 返回值结构上兼容 execution 的 `ToolGate`（多出的 `descriptor` 参数被忽略）——这样
 * execution 不必反向依赖 governance，agent 只需 `new ToolHub({ gate: createToolCallGate(gov) })`。
 * gate 抛出的 POLICY_BLOCKED 会被 ToolHub 转成 `ToolCallResult.success=false`（不触发熔断）。
 */
export const createToolCallGate =
  (governance: GovernancePlane) =>
  async (ctx: { tool: string; params: Record<string, unknown> }): Promise<void> => {
    await governance.checkToolCall(
      CommandSchema.parse({
        commandType: CommandType.TOOL_CALL,
        target: ctx.tool,
        payload: ctx.params,
      }),
    );
  };
