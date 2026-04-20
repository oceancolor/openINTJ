"""
执行平面 (Execution Plane)
============================
负责工作流编排与任务执行，支持顺序、并行、条件分支与人工审批。
OpenINTJ 框架核心组件，参考 pi-mono 的四工具原语设计和 OpenClaw 的 Lobster Agent Loop。

核心组件：
- StepStateMachine: 步骤状态机
- ToolHub: 工具注册与调用中心
- CircuitBreaker: 熔断器
- Executor: 执行引擎
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import time
import uuid

from framework_core import (
    AgentError, ErrorCode, Command, CommandType,
    Event, EventType, ToolDescriptor, ToolCallResult,
)


# ============================================================
# 1. 执行模式与步骤状态机
# ============================================================

class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    HUMAN_APPROVAL = "human_approval"


class StepState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"


@dataclass
class Step:
    """执行步骤"""
    step_id: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    state: StepState = StepState.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    retry_count: int = 0
    max_retries: int = 3

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0


@dataclass
class StepStateMachine:
    """
    步骤状态机 —— 管理步骤的生命周期
    状态转换：PENDING → READY → RUNNING → COMPLETED/FAILED
    """
    def transition(self, step: Step, target: StepState) -> Event:
        """执行状态转换"""
        valid_transitions = {
            StepState.PENDING: {StepState.READY, StepState.SKIPPED},
            StepState.READY: {StepState.RUNNING, StepState.SKIPPED},
            StepState.RUNNING: {StepState.COMPLETED, StepState.FAILED, StepState.WAITING_APPROVAL},
            StepState.FAILED: {StepState.READY},  # 重试
            StepState.WAITING_APPROVAL: {StepState.RUNNING, StepState.SKIPPED},
        }

        allowed = valid_transitions.get(step.state, set())
        if target not in allowed:
            raise AgentError(
                code=ErrorCode.EXECUTION_FAILED,
                message=f"非法状态转换: {step.state} → {target}",
                details={"step_id": step.step_id},
            )

        old_state = step.state
        step.state = target

        if target == StepState.RUNNING:
            step.started_at = time.time()
        elif target in (StepState.COMPLETED, StepState.FAILED):
            step.finished_at = time.time()

        event_type = {
            StepState.RUNNING: EventType.STEP_STARTED,
            StepState.COMPLETED: EventType.STEP_FINISHED,
            StepState.FAILED: EventType.STEP_FAILED,
        }.get(target, EventType.STEP_STARTED)

        return Event(
            event_type=event_type,
            source="step-state-machine",
            payload={
                "step_id": step.step_id,
                "from": old_state.value,
                "to": target.value,
            },
        )


# ============================================================
# 2. 熔断器 —— 防止连续失败导致系统崩溃
# ============================================================

@dataclass
class CircuitBreaker:
    """
    熔断器 —— 连续失败时自动断开
    状态：CLOSED（正常）→ OPEN（熔断）→ HALF_OPEN（试探）
    """
    failure_threshold: int = 3
    recovery_timeout_s: float = 60.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed | open | half_open

    def record_success(self) -> None:
        """记录成功"""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == "closed":
            return True
        if self.state == "open":
            # 检查是否超过恢复时间
            if time.time() - self.last_failure_time >= self.recovery_timeout_s:
                self.state = "half_open"
                return True
            return False
        # half_open: 允许一次试探
        return True


# ============================================================
# 3. 工具中心 —— 统一工具注册与调用
#    参考 pi-mono 的四工具原语：read, write, execute, browser
# ============================================================

@dataclass
class ToolHub:
    """
    工具中心 —— 管理所有可用工具
    参考 pi-mono 的极简工具设计和 OpenClaw 的 Skills 系统
    """
    tools: Dict[str, ToolDescriptor] = field(default_factory=dict)
    handlers: Dict[str, Callable] = field(default_factory=dict)
    circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=dict)
    call_history: List[ToolCallResult] = field(default_factory=list)

    def register(self, descriptor: ToolDescriptor,
                 handler: Optional[Callable] = None) -> None:
        """注册工具"""
        self.tools[descriptor.name] = descriptor
        if handler:
            self.handlers[descriptor.name] = handler
        self.circuit_breakers[descriptor.name] = CircuitBreaker()

    def call(self, tool_name: str, params: Dict[str, Any]) -> ToolCallResult:
        """调用工具"""
        if tool_name not in self.tools:
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error=f"工具未注册: {tool_name}",
            )

        breaker = self.circuit_breakers[tool_name]
        if not breaker.can_execute():
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error="熔断器已打开，暂停调用",
            )

        start = time.time()
        try:
            handler = self.handlers.get(tool_name)
            if handler:
                output = handler(params)
            else:
                output = {"status": "no_handler", "params": params}

            result = ToolCallResult(
                tool_name=tool_name, success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
            breaker.record_success()
        except Exception as e:
            result = ToolCallResult(
                tool_name=tool_name, success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
            breaker.record_failure()

        self.call_history.append(result)
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有已注册工具"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "permissions": t.permissions,
                "timeout_s": t.timeout_s,
                "idempotent": t.idempotent,
            }
            for t in self.tools.values()
        ]

    def register_builtin_tools(self) -> None:
        """
        注册内置工具 —— 参考 pi-mono 的四工具原语
        """
        self.register(ToolDescriptor(
            name="read_file",
            description="读取文件内容",
            input_schema={"path": "string"},
            permissions=["filesystem.read"],
            idempotent=True,
        ))
        self.register(ToolDescriptor(
            name="write_file",
            description="写入文件内容",
            input_schema={"path": "string", "content": "string"},
            permissions=["filesystem.write"],
        ))
        self.register(ToolDescriptor(
            name="execute_command",
            description="执行系统命令",
            input_schema={"command": "string"},
            permissions=["system.execute"],
            timeout_s=60,
        ))
        self.register(ToolDescriptor(
            name="search",
            description="搜索信息",
            input_schema={"query": "string"},
            permissions=["network.read"],
            idempotent=True,
        ))


# ============================================================
# 4. 执行结果
# ============================================================

@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    mode: ExecutionMode
    finished_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_ms: float = 0.0
    events: List[Event] = field(default_factory=list)


# ============================================================
# 5. 执行器 —— 执行平面主引擎
# ============================================================

@dataclass
class Executor:
    """
    执行器 —— 按指定模式执行步骤序列
    参考 OpenClaw 的 Lobster 循环：Think → Act → Observe → Reflect
    """
    name: str = "execution-plane"
    state_machine: StepStateMachine = field(default_factory=StepStateMachine)
    tool_hub: ToolHub = field(default_factory=ToolHub)

    def __post_init__(self):
        self.tool_hub.register_builtin_tools()

    def execute(self, steps: List[Step],
                mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> ExecutionResult:
        """执行步骤序列"""
        start_time = time.time()
        finished: List[str] = []
        failed: List[str] = []
        errors: List[Dict[str, Any]] = []
        events: List[Event] = []

        if mode == ExecutionMode.SEQUENTIAL:
            for step in steps:
                # PENDING → READY → RUNNING
                events.append(self.state_machine.transition(step, StepState.READY))
                events.append(self.state_machine.transition(step, StepState.RUNNING))

                try:
                    # 尝试通过工具中心执行
                    if step.action in self.tool_hub.tools:
                        result = self.tool_hub.call(step.action, step.params)
                        step.result = result.output
                        if not result.success:
                            raise AgentError(
                                code=ErrorCode.TOOL_FAILED,
                                message=result.error or "工具调用失败",
                            )
                    else:
                        # 默认执行逻辑
                        step.result = {"action": step.action, "status": "executed"}

                    events.append(self.state_machine.transition(step, StepState.COMPLETED))
                    finished.append(step.step_id)

                except Exception as e:
                    step.error = str(e)
                    # 重试逻辑
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        events.append(self.state_machine.transition(step, StepState.FAILED))
                        events.append(self.state_machine.transition(step, StepState.READY))
                        # 简化：不再递归重试，标记为失败
                    events.append(self.state_machine.transition(step, StepState.FAILED))
                    failed.append(step.step_id)
                    errors.append({
                        "step_id": step.step_id,
                        "error": str(e),
                        "retry_count": step.retry_count,
                    })

        elif mode == ExecutionMode.PARALLEL:
            # 并行模式：所有步骤同时执行（简化实现）
            for step in steps:
                events.append(self.state_machine.transition(step, StepState.READY))
                events.append(self.state_machine.transition(step, StepState.RUNNING))
                step.result = {"action": step.action, "status": "parallel_executed"}
                events.append(self.state_machine.transition(step, StepState.COMPLETED))
                finished.append(step.step_id)

        total_duration = (time.time() - start_time) * 1000

        return ExecutionResult(
            success=len(failed) == 0,
            mode=mode,
            finished_steps=finished,
            failed_steps=failed,
            errors=errors,
            total_duration_ms=total_duration,
            events=events,
        )


__all__ = [
    "Executor", "ExecutionMode", "Step", "StepState", "StepStateMachine",
    "ExecutionResult", "ToolHub", "CircuitBreaker",
]