import { AgentError, type Command, ErrorCode } from "@openintj/core";
import {
  type AuditEvent,
  AuditEventSchema,
  type PolicyEngineConfig,
  PolicyEngineConfigSchema,
} from "./types.js";

export class PolicyEngine {
  readonly config: PolicyEngineConfig;
  private readonly blocked: Set<string>;
  private readonly approval: Set<string>;
  private readonly white: Set<string>;

  constructor(cfg: Partial<PolicyEngineConfig> = {}) {
    this.config = PolicyEngineConfigSchema.parse(cfg);
    this.blocked = new Set(this.config.blockedTargets);
    this.approval = new Set(this.config.approvalRequired);
    this.white = new Set(this.config.whitelist);
  }

  check(command: Command): AuditEvent {
    if (this.white.has(command.target)) {
      return AuditEventSchema.parse({
        action: command.commandType,
        target: command.target,
        result: "allowed",
        riskLevel: "low",
      });
    }

    if (this.config.strictMode && this.blocked.has(command.target)) {
      const event = AuditEventSchema.parse({
        action: command.commandType,
        target: command.target,
        result: "blocked",
        riskLevel: "critical",
        details: { reason: "目标在黑名单中" },
      });
      throw new AgentError({
        code: ErrorCode.POLICY_BLOCKED,
        message: `策略阻断: 目标 '${command.target}' 被治理策略禁止`,
        retriable: false,
        details: { target: command.target, auditEventId: event.eventId },
      });
    }

    if (this.approval.has(command.target)) {
      return AuditEventSchema.parse({
        action: command.commandType,
        target: command.target,
        result: "warning",
        riskLevel: "high",
        details: { reason: "需要人工审批" },
      });
    }

    return AuditEventSchema.parse({
      action: command.commandType,
      target: command.target,
      result: "allowed",
      riskLevel: "low",
    });
  }

  /** 动态扩展黑名单（运行时）。 */
  block(target: string): void {
    this.blocked.add(target);
  }

  /** 动态扩展白名单（运行时）。 */
  allow(target: string): void {
    this.white.add(target);
  }
}
