import type { AuditEvent, AuditResult, RiskLevel } from "./types.js";

export interface AuditQuery {
  riskLevel?: RiskLevel;
  result?: AuditResult;
  limit?: number;
  /** 起始时间戳（秒）。 */
  since?: number;
}

export interface AuditStats {
  totalEvents: number;
  blockedCount: number;
  warningCount: number;
  allowedCount: number;
  byRisk: Record<RiskLevel, number>;
}

export class AuditTrail {
  /** 用环形缓冲区（双向链表也行；这里 array + tail trim 已够用）。 */
  private events: AuditEvent[] = [];
  readonly maxEvents: number;

  constructor(opts?: { maxEvents?: number }) {
    this.maxEvents = opts?.maxEvents ?? 10_000;
  }

  record(event: AuditEvent): void {
    this.events.push(event);
    if (this.events.length > this.maxEvents) {
      // shift in batches; for steady state max buffer stays bounded
      this.events.splice(0, this.events.length - this.maxEvents);
    }
  }

  query(filter: AuditQuery = {}): AuditEvent[] {
    const { riskLevel, result, since, limit = 100 } = filter;
    let filtered = this.events;
    if (riskLevel !== undefined) {
      filtered = filtered.filter((e) => e.riskLevel === riskLevel);
    }
    if (result !== undefined) {
      filtered = filtered.filter((e) => e.result === result);
    }
    if (since !== undefined) {
      filtered = filtered.filter((e) => e.timestamp >= since);
    }
    return filtered.slice(-limit);
  }

  getStats(): AuditStats {
    const byRisk: Record<RiskLevel, number> = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };
    let blocked = 0;
    let warnings = 0;
    for (const e of this.events) {
      byRisk[e.riskLevel] = (byRisk[e.riskLevel] ?? 0) + 1;
      if (e.result === "blocked") blocked++;
      else if (e.result === "warning") warnings++;
    }
    return {
      totalEvents: this.events.length,
      blockedCount: blocked,
      warningCount: warnings,
      allowedCount: this.events.length - blocked - warnings,
      byRisk,
    };
  }

  /** 清空（仅用于测试或租户切换）。 */
  clear(): void {
    this.events = [];
  }
}
