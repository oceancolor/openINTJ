import { randomUUID } from "node:crypto";
import {
  AgentError,
  type Command,
  CommandSchema,
  CommandType,
  ErrorCode,
  type HookBus,
} from "@openintj/core";
import {
  type ApprovalBroker,
  type GrantStore,
  InMemoryGrantStore,
  type ToolRisk,
  classifyToolRisk,
} from "./approval.js";
import { AuditTrail } from "./audit-trail.js";
import { PolicyEngine } from "./policy-engine.js";
import { QuotaGuard } from "./quota-guard.js";
import { type AuditEvent, AuditEventSchema } from "./types.js";

export interface GovernancePlaneOpts {
  policyEngine?: PolicyEngine;
  auditTrail?: AuditTrail;
  quotaGuard?: QuotaGuard;
  hooks?: HookBus;
  /** RFC-013 interactive approval + capability grants. Default: no risks require approval. */
  approval?: {
    broker?: ApprovalBroker;
    grants?: GrantStore;
    requiredRisks?: readonly ToolRisk[];
  };
}

export class GovernancePlane {
  readonly name = "governance-plane";
  readonly policyEngine: PolicyEngine;
  readonly auditTrail: AuditTrail;
  readonly quotaGuard: QuotaGuard;
  /** RFC-013 capability grants (in-memory unless a store is injected). */
  readonly grants: GrantStore;
  /** RFC-013 approval broker; undefined means approvals fail closed. */
  readonly approvals: ApprovalBroker | undefined;
  private readonly requiredRisks: ReadonlySet<ToolRisk>;
  private readonly hooks: HookBus | undefined;

  constructor(opts: GovernancePlaneOpts = {}) {
    this.policyEngine = opts.policyEngine ?? new PolicyEngine();
    this.auditTrail = opts.auditTrail ?? new AuditTrail();
    this.quotaGuard = opts.quotaGuard ?? new QuotaGuard();
    this.grants = opts.approval?.grants ?? new InMemoryGrantStore();
    this.approvals = opts.approval?.broker;
    this.requiredRisks = new Set(opts.approval?.requiredRisks ?? []);
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
  async checkToolCall(command: Command, opts?: { risk?: ToolRisk }): Promise<AuditEvent> {
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

    // RFC-013 second gate: interactive approval for configured risk tiers.
    // A live capability grant skips the prompt; no broker means fail-closed.
    const risk = opts?.risk ?? classifyToolRisk({ tool: command.target, idempotent: false });
    if (this.requiredRisks.has(risk)) {
      const covered = this.grants.covers({ tool: command.target, risk, params: command.payload });
      if (!covered) {
        const denied = async (reason: string): Promise<never> => {
          const event = AuditEventSchema.parse({
            action: command.commandType,
            target: command.target,
            result: "blocked",
            riskLevel: "high",
            details: { reason, risk, approval: "denied" },
          });
          this.auditTrail.record(event);
          throw new AgentError({
            code: ErrorCode.POLICY_BLOCKED,
            message: `需要审批且未获批准：${command.target}（${reason}）`,
            retriable: false,
            details: { auditEventId: event.eventId, risk },
          });
        };
        if (!this.approvals) {
          await denied("approval_unavailable");
        } else {
          const request = {
            id: randomUUID(),
            tool: command.target,
            risk,
            params: command.payload,
            requestedAt: Date.now(),
          };
          await this.hooks?.emit("event.APPROVAL_REQUESTED", {
            id: request.id,
            tool: request.tool,
            risk,
          });
          const outcome = await this.approvals.request(request);
          await this.hooks?.emit("event.APPROVAL_DECIDED", {
            id: request.id,
            tool: request.tool,
            decision: outcome.decision,
          });
          if (outcome.decision === "denied") {
            await denied(outcome.reason ?? "approval_denied");
          } else if (outcome.grant) {
            this.grants.grant({
              tool: command.target,
              risk,
              ...(outcome.grant.scope ? { scope: outcome.grant.scope } : {}),
              ...(outcome.grant.ttlMs ? { expiresAt: Date.now() + outcome.grant.ttlMs } : {}),
              label: "from-approval",
              source: "approval",
            });
          }
          auditEvent = AuditEventSchema.parse({
            ...auditEvent,
            result: "warning",
            riskLevel: "high",
            details: { ...auditEvent.details, approval: "approved", risk },
          });
        }
      }
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
  async (ctx: {
    tool: string;
    params: Record<string, unknown>;
    descriptor?: { permissions?: readonly string[]; idempotent?: boolean };
  }): Promise<void> => {
    await governance.checkToolCall(
      CommandSchema.parse({
        commandType: CommandType.TOOL_CALL,
        target: ctx.tool,
        payload: ctx.params,
      }),
      {
        risk: classifyToolRisk({
          tool: ctx.tool,
          ...(ctx.descriptor?.permissions ? { permissions: ctx.descriptor.permissions } : {}),
          ...(ctx.descriptor?.idempotent !== undefined
            ? { idempotent: ctx.descriptor.idempotent }
            : {}),
        }),
      },
    );
  };
