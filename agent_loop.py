"""
Agent Loop —— 核心代理循环
=============================
实现 感知(Perceive) → 决策(Decide) → 行动(Act) → 观察(Observe) → 反馈(Reflect) 闭环。

融合 OpenClaw 的 Lobster 循环模式（Think→Act→Observe→Reflect）、
pi-mono 的极简 Agent Core 设计，以及"记忆着色器"创新机制。

这是 OpenINTJ 框架的运行时核心，类比 3D 引擎的主渲染循环。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import time
import uuid

from framework_core import (
    AgentError, ErrorCode, Command, CommandType,
    Event, EventType, FrameworkConfig, ConfigValidator,
    ContextBudget, ShaderConfig, ShaderMode, TaskType,
    StateSnapshot, ToolDescriptor,
)
from memory_plane import MemoryPlane
from context_engine import ContextEngine, ContextWindow
from control_plane import ControlPlane, PlanGraph, PlanStep
from execution_plane import (
    Executor, ExecutionMode, Step, StepState, ExecutionResult,
)
from governance_plane import GovernancePlane
from llm_client import create_llm_fn, create_vision_llm_fn, get_hunyuan_client


# ============================================================
# 1. Agent 状态
# ============================================================

class AgentState(str, Enum):
    """Agent 生命周期状态"""
    IDLE = "idle"
    PERCEIVING = "perceiving"       # 感知阶段
    DECIDING = "deciding"           # 决策阶段
    ACTING = "acting"               # 行动阶段
    OBSERVING = "observing"         # 观察阶段
    REFLECTING = "reflecting"       # 反馈阶段
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class LoopIteration:
    """单次循环迭代记录"""
    iteration_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    iteration_num: int = 0
    state: AgentState = AgentState.IDLE
    query: str = ""
    task_type: TaskType = TaskType.GENERAL_CHAT
    shader_mode: ShaderMode = ShaderMode.ADAPTIVE
    plan: Optional[PlanGraph] = None
    execution_result: Optional[ExecutionResult] = None
    response: str = ""
    events: List[Event] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    image_data: Optional[Dict[str, Any]] = None  # 图片数据 {"base64": ..., "mime_type": ..., "size_bytes": ...}

    @property
    def duration_ms(self) -> float:
        if self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0


# ============================================================
# 2. Agent Loop 核心
# ============================================================

@dataclass
class AgentLoop:
    """
    Agent Loop —— 自主代理的核心运行循环
    
    循环流程（类比 3D 引擎的帧循环）：
    ┌─────────────────────────────────────────────┐
    │  1. PERCEIVE  感知用户输入 + JIT 加载记忆     │
    │  2. DECIDE    任务分类 + 规划 + 着色器选择     │
    │  3. ACT       执行计划步骤 + 工具调用          │
    │  4. OBSERVE   采集执行结果 + 状态快照          │
    │  5. REFLECT   评估效果 + 策略修正 + 记忆更新   │
    │  → 循环回到 1（如果需要继续）                  │
    └─────────────────────────────────────────────┘
    """
    # 四平面
    control_plane: ControlPlane = field(default_factory=ControlPlane)
    executor: Executor = field(default_factory=Executor)
    memory_plane: MemoryPlane = field(default_factory=MemoryPlane)
    governance: GovernancePlane = field(default_factory=GovernancePlane)

    # 上下文引擎
    context_engine: ContextEngine = field(default_factory=ContextEngine)

    # 配置
    config: Optional[FrameworkConfig] = None
    max_iterations: int = 5
    system_prompt: str = (
        "你是 OpenINTJ，一个基于四平面分层架构的自研 Agent。\n"
        "你具备记忆着色器机制，能根据任务类型动态调整记忆细节级别。\n"
        "你的核心循环是：感知 → 决策 → 行动 → 观察 → 反馈。"
    )

    # 运行状态
    state: AgentState = AgentState.IDLE
    iterations: List[LoopIteration] = field(default_factory=list)
    total_runs: int = 0

    # LLM 调用函数（可替换为真实 LLM）
    llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None
    vision_llm_fn: Optional[Callable] = None  # Vision LLM 调用函数

    def __post_init__(self):
        # 同步记忆平面
        self.context_engine.memory_plane = self.memory_plane
        self.context_engine.set_system_prompt(self.system_prompt)

        # 使用腾讯混元大模型作为 LLM 函数
        if self.llm_fn is None:
            self.llm_fn = create_llm_fn()
        if self.vision_llm_fn is None:
            self.vision_llm_fn = create_vision_llm_fn()

    @staticmethod
    def _mock_llm(messages: List[Dict[str, str]]) -> str:
        """模拟 LLM 响应（生产环境替换为真实 API 调用）"""
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        return f"[OpenINTJ 响应] 已处理您的请求: {last_user[:100]}..."

    def run(self, query: str, image_data: Optional[Dict[str, Any]] = None) -> LoopIteration:
        """
        运行一次完整的 Agent Loop
        这是框架的主入口点
        
        参数:
            query: 用户查询文本
            image_data: 可选的图片数据 {"base64": ..., "mime_type": ..., "size_bytes": ...}
        """
        self.total_runs += 1
        iteration = LoopIteration(
            iteration_num=self.total_runs,
            query=query,
            image_data=image_data,
        )

        try:
            # ===== 1. PERCEIVE 感知 =====
            self._perceive(iteration, query)

            # ===== 2. DECIDE 决策 =====
            self._decide(iteration)

            # ===== 3. ACT 行动 =====
            self._act(iteration)

            # ===== 4. OBSERVE 观察 =====
            self._observe(iteration)

            # ===== 5. REFLECT 反馈 =====
            self._reflect(iteration)

            iteration.state = AgentState.COMPLETED

        except AgentError as e:
            iteration.state = AgentState.ERROR
            iteration.response = f"[错误] {e}"
            iteration.events.append(Event(
                event_type=EventType.STEP_FAILED,
                source="agent-loop",
                payload={"error": str(e), "code": e.code.value},
            ))

        except Exception as e:
            iteration.state = AgentState.ERROR
            iteration.response = f"[内部错误] {e}"

        finally:
            iteration.finished_at = time.time()
            iteration.metrics = self._collect_metrics(iteration)
            self.iterations.append(iteration)
            self.state = AgentState.IDLE

        return iteration

    def _perceive(self, iteration: LoopIteration, query: str) -> None:
        """
        感知阶段 —— 接收输入 + JIT 加载记忆
        类比 3D 引擎的场景加载阶段
        """
        self.state = AgentState.PERCEIVING
        iteration.state = AgentState.PERCEIVING

        # 添加用户消息到上下文
        metadata = {}
        if iteration.image_data:
            metadata["has_image"] = True
            metadata["image_mime_type"] = iteration.image_data.get("mime_type", "image/png")
            metadata["image_size_bytes"] = iteration.image_data.get("size_bytes", 0)
        self.context_engine.add_message("user", query, metadata=metadata,
                                        image_data=iteration.image_data)

        # 任务分类
        task_type = self.context_engine.classifier.classify(query)
        iteration.task_type = task_type

        # JIT 加载记忆（构建上下文窗口）
        context_window = self.context_engine.build_context(query)

        # 确定着色器模式
        shader_config = self.context_engine.shader_config
        iteration.shader_mode = shader_config.get_shader_for_task(task_type)

        iteration.events.append(Event(
            event_type=EventType.MEMORY_LOADED,
            source="perceive",
            payload={
                "task_type": task_type.value,
                "shader_mode": iteration.shader_mode.value,
                "memory_loaded": len(context_window.memory_context),
                "budget_usage": self.context_engine.budget.usage_ratio,
            },
        ))

    def _decide(self, iteration: LoopIteration) -> None:
        """
        决策阶段 —— 目标解析 + 任务规划
        类比 3D 引擎的渲染排序阶段
        """
        self.state = AgentState.DECIDING
        iteration.state = AgentState.DECIDING

        # 通过控制平面处理输入
        plan = self.control_plane.process_input(
            iteration.query, iteration.task_type
        )
        iteration.plan = plan

        # 治理检查：对计划中的每个步骤进行策略审查
        for step in plan.steps:
            cmd = self.control_plane.make_execute_command(
                PlanStep(step_id=step.step_id, action=step.action)
            )
            self.governance.check_and_record(cmd)

        iteration.events.append(Event(
            event_type=EventType.PLANNED,
            source="decide",
            payload={
                "plan_id": plan.plan_id,
                "total_steps": plan.total_steps,
                "intent": plan.goal.intent if plan.goal else "unknown",
            },
        ))

    def _act(self, iteration: LoopIteration) -> None:
        """
        行动阶段 —— 执行计划步骤
        类比 3D 引擎的实际渲染阶段
        """
        self.state = AgentState.ACTING
        iteration.state = AgentState.ACTING

        if not iteration.plan:
            return

        # 将 PlanStep 转换为 execution Step
        exec_steps = [
            Step(step_id=s.step_id, action=s.action, params=s.params)
            for s in iteration.plan.steps
        ]

        # 执行
        result = self.executor.execute(exec_steps, ExecutionMode.SEQUENTIAL)
        iteration.execution_result = result

        # 更新计划状态
        for step_id in result.finished_steps:
            iteration.plan.mark_step(step_id, "completed")
        for step_id in result.failed_steps:
            iteration.plan.mark_step(step_id, "failed")

        # 调用 LLM 生成响应
        context_window = self.context_engine.window
        messages = context_window.to_prompt_messages()
        
        # 检测当前消息是否包含图片，若包含则调用 Vision LLM
        if iteration.image_data and self.vision_llm_fn:
            iteration.response = self.vision_llm_fn(
                messages,
                iteration.image_data["base64"],
                iteration.image_data.get("mime_type", "image/png"),
            )
        else:
            iteration.response = self.llm_fn(messages)

        iteration.events.append(Event(
            event_type=EventType.COMPLETED,
            source="act",
            payload={
                "success": result.success,
                "finished_steps": len(result.finished_steps),
                "failed_steps": len(result.failed_steps),
                "duration_ms": result.total_duration_ms,
            },
        ))

    def _observe(self, iteration: LoopIteration) -> None:
        """
        观察阶段 —— 采集执行结果，保存状态快照
        类比 3D 引擎的后处理阶段
        """
        self.state = AgentState.OBSERVING
        iteration.state = AgentState.OBSERVING

        # 保存状态快照到记忆
        snapshot = StateSnapshot(
            trace_id=iteration.iteration_id,
            status=iteration.state.value,
            metrics=iteration.metrics,
            context={
                "query": iteration.query[:200],
                "task_type": iteration.task_type.value,
                "shader_mode": iteration.shader_mode.value,
                "plan_progress": iteration.plan.progress if iteration.plan else 0,
            },
        )
        self.memory_plane.store.save_snapshot(snapshot)

        # 将 assistant 响应添加到上下文
        if iteration.response:
            self.context_engine.add_message("assistant", iteration.response)

        # 将响应存入工作记忆
        self.memory_plane.ingest(
            f"[任务] {iteration.query[:100]} → [响应] {iteration.response[:200]}",
            "working",
            importance=0.7,
            task_tags=[iteration.task_type.value],
        )

    def _reflect(self, iteration: LoopIteration) -> None:
        """
        反馈阶段 —— 评估效果，策略修正
        类比 3D 引擎的性能分析和自适应调整
        """
        self.state = AgentState.REFLECTING
        iteration.state = AgentState.REFLECTING

        # 评估指标
        metrics = self._collect_metrics(iteration)
        iteration.metrics = metrics

        # 自适应着色器调整
        if metrics.get("budget_usage", 0) > 0.9:
            # 预算紧张，切换到更激进的压缩
            self.context_engine.shader_config.mode = ShaderMode.LOW_FIDELITY
        elif metrics.get("budget_usage", 0) < 0.3:
            # 预算充裕，可以使用高保真
            self.context_engine.shader_config.mode = ShaderMode.HIGH_FIDELITY

        iteration.events.append(Event(
            event_type=EventType.LOOP_ITERATION,
            source="reflect",
            payload={
                "iteration": iteration.iteration_num,
                "metrics": metrics,
                "shader_adjusted": self.context_engine.shader_config.mode.value,
            },
        ))

    def _collect_metrics(self, iteration: LoopIteration) -> Dict[str, float]:
        """收集评估指标"""
        result = iteration.execution_result
        return {
            "duration_ms": iteration.duration_ms,
            "budget_usage": self.context_engine.budget.usage_ratio,
            "memory_count": self.memory_plane.store.total_count,
            "plan_progress": iteration.plan.progress if iteration.plan else 0,
            "success_rate": 1.0 if (result and result.success) else 0.0,
            "compaction_count": self.context_engine.compaction_count,
            "iteration_num": float(iteration.iteration_num),
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取 Agent Loop 统计信息"""
        return {
            "state": self.state.value,
            "total_runs": self.total_runs,
            "total_iterations": len(self.iterations),
            "context": self.context_engine.get_stats(),
            "memory": self.memory_plane.get_stats(),
            "governance": self.governance.get_stats(),
            "tools": self.executor.tool_hub.list_tools(),
        }


# ============================================================
# 3. 框架入口
# ============================================================

@dataclass
class OpenINTJFramework:
    """
    OpenINTJ 框架 —— 顶层编排入口
    组装四平面 + Agent Loop，提供统一的运行接口
    """
    config: Optional[FrameworkConfig] = None
    agent_loop: AgentLoop = field(default_factory=AgentLoop)

    def __post_init__(self):
        if self.config:
            # 应用配置
            self.agent_loop.context_engine.budget.max_tokens = self.config.max_context_tokens
            shader_mode = ShaderMode(self.config.shader_mode)
            self.agent_loop.context_engine.shader_config.mode = shader_mode
            self.agent_loop.governance.policy_engine.strict_mode = self.config.governance_strict

    def run(self, query: str, image_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """运行一次完整的 Agent 循环"""
        iteration = self.agent_loop.run(query, image_data=image_data)
        return {
            "trace_id": iteration.iteration_id,
            "status": iteration.state.value,
            "response": iteration.response,
            "task_type": iteration.task_type.value,
            "shader_mode": iteration.shader_mode.value,
            "plan_progress": iteration.plan.progress if iteration.plan else 0,
            "duration_ms": round(iteration.duration_ms, 2),
            "metrics": iteration.metrics,
            "events_count": len(iteration.events),
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.agent_loop.get_stats()


def bootstrap() -> OpenINTJFramework:
    """引导启动框架"""
    config = FrameworkConfig.load_from_env()
    result = ConfigValidator.validate(config)
    if not result.ok:
        raise AgentError(
            code=ErrorCode.VALIDATION_ERROR,
            message="配置校验失败",
            retriable=False,
            details={"issues": result.issues},
        )
    return OpenINTJFramework(config=config)