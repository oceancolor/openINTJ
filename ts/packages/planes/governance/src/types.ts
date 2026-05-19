import { randomUUID } from "node:crypto";
import { z } from "zod";

export const RiskLevelSchema = z.enum(["low", "medium", "high", "critical"]);
export type RiskLevel = z.infer<typeof RiskLevelSchema>;

export const AuditResultSchema = z.enum(["allowed", "blocked", "warning"]);
export type AuditResult = z.infer<typeof AuditResultSchema>;

export const AuditEventSchema = z.object({
  eventId: z.string().default(() => randomUUID()),
  timestamp: z.number().default(() => Date.now() / 1000),
  action: z.string().default(""),
  actor: z.string().default("agent"),
  target: z.string().default(""),
  result: AuditResultSchema.default("allowed"),
  details: z.record(z.string(), z.unknown()).default({}),
  riskLevel: RiskLevelSchema.default("low"),
});
export type AuditEvent = z.infer<typeof AuditEventSchema>;

export const PolicyEngineConfigSchema = z.object({
  blockedTargets: z
    .array(z.string())
    .default([
      "shell-delete",
      "filesystem-delete-recursive",
      "network-external-unrestricted",
      "credential-access",
    ]),
  approvalRequired: z
    .array(z.string())
    .default([
      "deploy-production",
      "database-migration",
      "config-change-prod",
      "permission-escalation",
    ]),
  whitelist: z.array(z.string()).default(["read_file", "search", "analyze", "think"]),
  allowedPermissions: z
    .array(z.string())
    .default(["filesystem.read", "filesystem.write", "network.read", "system.execute"]),
  strictMode: z.boolean().default(true),
});
export type PolicyEngineConfig = z.infer<typeof PolicyEngineConfigSchema>;

export const QuotaConfigSchema = z.object({
  maxApiCallsPerHour: z.number().int().positive().default(100),
  maxTokensPerHour: z.number().int().positive().default(500_000),
  maxToolCallsPerMinute: z.number().int().positive().default(20),
});
export type QuotaConfig = z.infer<typeof QuotaConfigSchema>;
