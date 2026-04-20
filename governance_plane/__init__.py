"""
治理平面 (Governance Plane)
=============================
提供安全治理、策略引擎、审计追踪、配额限制。
OpenINTJ 框架核心组件，参考 OpenClaw 的安全治理机制和企业级合规要求。

核心组件：
- PolicyEngine: 策略引擎，权限隔离与敏感操作保护
- AuditTrail: 审计追踪，完整行为记录
- QuotaGuard: 配额限制，防止资源滥用
- GovernancePlane: 治理平面主控制器
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import time
import uuid

from framework_core import (
    AgentError, ErrorCode, Command, CommandType,
    Event, EventType,
)


# ============================================================
# 1. 审计事件
# ============================================================

@dataclass
class AuditEvent:
    """审计事件记录"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    action: str = ""
    actor: str = "agent"
    target: str = ""
    result: str = "allowed"  # allowed | blocked | warning
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low | medium | high | critical


# ============================================================
# 2. 策略引擎
# ============================================================

@dataclass
class PolicyEngine:
    """
    策略引擎 —— 权限隔离与敏感操作保护
    在命令执行前进行策略审查，阻断高风险操作
    """
    # 高风险目标黑名单
    blocked_targets: Set[str] = field(default_factory=lambda: {
        "shell-delete", "filesystem-delete-recursive",
        "network-external-unrestricted", "credential-access",
    })

    # 需要人工审批的操作
    approval_required: Set[str] = field(default_factory=lambda: {
        "deploy-production", "database-migration",
        "config-change-prod", "permission-escalation",
    })

    # 操作白名单（不受限制）
    whitelist: Set[str] = field(default_factory=lambda: {
        "read_file", "search", "analyze", "think",
    })

    # 权限域
    allowed_permissions: Set[str] = field(default_factory=lambda: {
        "filesystem.read", "filesystem.write",
        "network.read", "system.execute",
    })

    strict_mode: bool = True

    def check(self, command: Command) -> AuditEvent:
        """
        策略检查 —— 在命令执行前审查
        返回审计事件
        """
        # 白名单直接放行
        if command.target in self.whitelist:
            return AuditEvent(
                action=command.command_type.value,
                target=command.target,
                result="allowed",
                risk_level="low",
            )

        # 黑名单直接阻断
        if self.strict_mode and command.target in self.blocked_targets:
            event = AuditEvent(
                action=command.command_type.value,
                target=command.target,
                result="blocked",
                risk_level="critical",
                details={"reason": "目标在黑名单中"},
            )
            raise AgentError(
                code=ErrorCode.POLICY_BLOCKED,
                message=f"策略阻断: 目标 '{command.target}' 被治理策略禁止",
                retriable=False,
                details={"target": command.target, "audit_event_id": event.event_id},
            )

        # 需要审批的操作
        if command.target in self.approval_required:
            return AuditEvent(
                action=command.command_type.value,
                target=command.target,
                result="warning",
                risk_level="high",
                details={"reason": "需要人工审批"},
            )

        # 默认放行
        return AuditEvent(
            action=command.command_type.value,
            target=command.target,
            result="allowed",
            risk_level="low",
        )


# ============================================================
# 3. 审计追踪
# ============================================================

@dataclass
class AuditTrail:
    """
    审计追踪 —— 完整行为记录
    所有操作都会被记录，支持事后审计和故障排查
    """
    events: List[AuditEvent] = field(default_factory=list)
    max_events: int = 10000

    def record(self, event: AuditEvent) -> None:
        """记录审计事件"""
        self.events.append(event)
        # 超出容量时移除最旧的
        while len(self.events) > self.max_events:
            self.events.pop(0)

    def query(self, risk_level: Optional[str] = None,
              result: Optional[str] = None,
              limit: int = 100) -> List[AuditEvent]:
        """查询审计事件"""
        filtered = self.events
        if risk_level:
            filtered = [e for e in filtered if e.risk_level == risk_level]
        if result:
            filtered = [e for e in filtered if e.result == result]
        return filtered[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取审计统计"""
        total = len(self.events)
        blocked = sum(1 for e in self.events if e.result == "blocked")
        warnings = sum(1 for e in self.events if e.result == "warning")
        return {
            "total_events": total,
            "blocked_count": blocked,
            "warning_count": warnings,
            "allowed_count": total - blocked - warnings,
        }


# ============================================================
# 4. 配额限制
# ============================================================

@dataclass
class QuotaGuard:
    """
    配额限制 —— 防止资源滥用
    限制 API 调用次数、token 消耗、工具调用频率等
    """
    # 配额配置
    max_api_calls_per_hour: int = 100
    max_tokens_per_hour: int = 500000
    max_tool_calls_per_minute: int = 20

    # 当前计数
    api_calls: List[float] = field(default_factory=list)
    token_usage: List[tuple] = field(default_factory=list)  # (timestamp, tokens)
    tool_calls: List[float] = field(default_factory=list)

    def check_api_quota(self) -> bool:
        """检查 API 调用配额"""
        now = time.time()
        hour_ago = now - 3600
        self.api_calls = [t for t in self.api_calls if t > hour_ago]
        return len(self.api_calls) < self.max_api_calls_per_hour

    def check_token_quota(self) -> bool:
        """检查 token 配额"""
        now = time.time()
        hour_ago = now - 3600
        self.token_usage = [(t, n) for t, n in self.token_usage if t > hour_ago]
        total = sum(n for _, n in self.token_usage)
        return total < self.max_tokens_per_hour

    def check_tool_quota(self) -> bool:
        """检查工具调用配额"""
        now = time.time()
        minute_ago = now - 60
        self.tool_calls = [t for t in self.tool_calls if t > minute_ago]
        return len(self.tool_calls) < self.max_tool_calls_per_minute

    def record_api_call(self) -> None:
        self.api_calls.append(time.time())

    def record_token_usage(self, tokens: int) -> None:
        self.token_usage.append((time.time(), tokens))

    def record_tool_call(self) -> None:
        self.tool_calls.append(time.time())

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        hour_ago = now - 3600
        minute_ago = now - 60
        return {
            "api_calls_last_hour": len([t for t in self.api_calls if t > hour_ago]),
            "tokens_last_hour": sum(n for t, n in self.token_usage if t > hour_ago),
            "tool_calls_last_minute": len([t for t in self.tool_calls if t > minute_ago]),
            "api_quota_remaining": self.max_api_calls_per_hour - len([t for t in self.api_calls if t > hour_ago]),
        }


# ============================================================
# 5. 治理平面主控制器
# ============================================================

@dataclass
class GovernancePlane:
    """
    治理平面 —— 安全治理的统一入口
    整合 PolicyEngine + AuditTrail + QuotaGuard
    """
    name: str = "governance-plane"
    policy_engine: PolicyEngine = field(default_factory=PolicyEngine)
    audit_trail: AuditTrail = field(default_factory=AuditTrail)
    quota_guard: QuotaGuard = field(default_factory=QuotaGuard)

    def check_and_record(self, command: Command) -> AuditEvent:
        """检查命令并记录审计事件"""
        # 1. 配额检查
        if not self.quota_guard.check_api_quota():
            event = AuditEvent(
                action=command.command_type.value,
                target=command.target,
                result="blocked",
                risk_level="high",
                details={"reason": "API 调用配额已用尽"},
            )
            self.audit_trail.record(event)
            raise AgentError(
                code=ErrorCode.POLICY_BLOCKED,
                message="API 调用配额已用尽",
                retriable=True,
            )

        # 2. 策略检查
        audit_event = self.policy_engine.check(command)
        self.audit_trail.record(audit_event)

        # 3. 记录调用
        self.quota_guard.record_api_call()

        return audit_event

    def get_stats(self) -> Dict[str, Any]:
        return {
            "audit": self.audit_trail.get_stats(),
            "quota": self.quota_guard.get_stats(),
            "strict_mode": self.policy_engine.strict_mode,
        }


__all__ = [
    "GovernancePlane", "PolicyEngine", "AuditTrail", "AuditEvent",
    "QuotaGuard",
]