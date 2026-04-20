"""
控制平面 (Control Plane)
=========================
OpenINTJ 框架的决策中枢，负责目标解析、任务规划、命令分发。
参考 OpenClaw 的 Gateway 控制平面和 pi-mono 的分层调度设计。

核心组件：
- GoalParser: 目标解析器，将自然语言目标转化为结构化目标
- Planner: 规划器，将目标分解为可执行步骤（Plan Graph）
- CommandDispatcher: 命令分发器，将命令路由到目标平面
- ControlPlane: 控制平面主控制器
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid

from framework_core import (
    AgentError, ErrorCode, Command, CommandType,
    Event, EventType, TaskType,
)


# ============================================================
# 1. 目标解析器
# ============================================================

@dataclass
class ParsedGoal:
    """解析后的结构化目标"""
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_input: str = ""
    task_type: TaskType = TaskType.GENERAL_CHAT
    intent: str = ""                          # 意图描述
    entities: List[str] = field(default_factory=list)  # 提取的实体
    constraints: Dict[str, Any] = field(default_factory=dict)  # 约束条件
    priority: int = 5                         # 优先级 1-10
    created_at: float = field(default_factory=time.time)


@dataclass
class GoalParser:
    """
    目标解析器 —— 将自然语言输入转化为结构化目标
    参考 OpenClaw 的意图识别层
    """
    # 意图关键词映射
    intent_keywords: Dict[str, str] = field(default_factory=lambda: {
        "创建": "create", "生成": "create", "编写": "create", "写": "create",
        "修改": "modify", "更新": "modify", "修复": "modify", "改": "modify",
        "删除": "delete", "移除": "delete",
        "查询": "query", "搜索": "query", "查找": "query", "分析": "query",
        "执行": "execute", "运行": "execute", "部署": "execute",
        "规划": "plan", "设计": "plan", "架构": "plan",
    })

    def parse(self, raw_input: str, task_type: TaskType = TaskType.GENERAL_CHAT) -> ParsedGoal:
        """解析用户输入为结构化目标"""
        intent = self._extract_intent(raw_input)
        entities = self._extract_entities(raw_input)
        priority = self._estimate_priority(raw_input, task_type)

        return ParsedGoal(
            raw_input=raw_input,
            task_type=task_type,
            intent=intent,
            entities=entities,
            priority=priority,
        )

    def _extract_intent(self, text: str) -> str:
        """提取意图"""
        for keyword, intent in self.intent_keywords.items():
            if keyword in text:
                return intent
        return "general"

    def _extract_entities(self, text: str) -> List[str]:
        """提取实体（简化版，生产环境应使用 NER）"""
        # 提取引号内的内容作为实体
        entities = []
        in_quote = False
        current = ""
        for char in text:
            if char in ('"', "'", '"', '"', "'", "'"):
                if in_quote:
                    if current.strip():
                        entities.append(current.strip())
                    current = ""
                in_quote = not in_quote
            elif in_quote:
                current += char
        return entities

    def _estimate_priority(self, text: str, task_type: TaskType) -> int:
        """估算优先级"""
        priority = 5
        urgent_words = {"紧急", "立即", "马上", "urgent", "asap", "critical"}
        if any(w in text.lower() for w in urgent_words):
            priority = 9
        if task_type == TaskType.CODE_GENERATION:
            priority = max(priority, 7)
        elif task_type == TaskType.QUICK_RESPONSE:
            priority = max(priority, 8)
        return priority


# ============================================================
# 2. 规划器 —— Plan Graph
# ============================================================

@dataclass
class PlanStep:
    """计划步骤"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: str = ""
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 依赖的步骤 ID
    estimated_tokens: int = 0
    status: str = "pending"  # pending | running | completed | failed | skipped


@dataclass
class PlanGraph:
    """
    计划图 —— 有向无环图 (DAG) 表示的执行计划
    参考 OpenClaw 的 Todo List 任务分解机制
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: Optional[ParsedGoal] = None
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return self.completed_steps / self.total_steps

    def get_ready_steps(self) -> List[PlanStep]:
        """获取可执行的步骤（所有依赖已完成）"""
        completed_ids = {s.step_id for s in self.steps if s.status == "completed"}
        return [
            s for s in self.steps
            if s.status == "pending" and all(d in completed_ids for d in s.dependencies)
        ]

    def mark_step(self, step_id: str, status: str) -> None:
        """更新步骤状态"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = status
                return


@dataclass
class Planner:
    """
    规划器 —— 将目标分解为可执行步骤
    参考 pi-mono 的极简设计：核心只需要 think → act → observe
    """
    name: str = "planner"

    def create_plan(self, goal: ParsedGoal) -> PlanGraph:
        """根据解析后的目标创建执行计划"""
        plan = PlanGraph(goal=goal)

        if goal.intent == "create":
            plan.steps = [
                PlanStep(step_id="s1", action="analyze", description="分析需求"),
                PlanStep(step_id="s2", action="design", description="设计方案", dependencies=["s1"]),
                PlanStep(step_id="s3", action="implement", description="实现功能", dependencies=["s2"]),
                PlanStep(step_id="s4", action="verify", description="验证结果", dependencies=["s3"]),
            ]
        elif goal.intent == "modify":
            plan.steps = [
                PlanStep(step_id="s1", action="read", description="读取现有内容"),
                PlanStep(step_id="s2", action="analyze", description="分析修改点", dependencies=["s1"]),
                PlanStep(step_id="s3", action="modify", description="执行修改", dependencies=["s2"]),
                PlanStep(step_id="s4", action="verify", description="验证修改", dependencies=["s3"]),
            ]
        elif goal.intent == "query":
            plan.steps = [
                PlanStep(step_id="s1", action="retrieve", description="检索信息"),
                PlanStep(step_id="s2", action="analyze", description="分析结果", dependencies=["s1"]),
                PlanStep(step_id="s3", action="respond", description="生成响应", dependencies=["s2"]),
            ]
        elif goal.intent == "plan":
            plan.steps = [
                PlanStep(step_id="s1", action="decompose", description="目标分解"),
                PlanStep(step_id="s2", action="evaluate", description="方案评估", dependencies=["s1"]),
                PlanStep(step_id="s3", action="synthesize", description="综合输出", dependencies=["s2"]),
            ]
        else:
            # 通用流程：感知 → 思考 → 行动
            plan.steps = [
                PlanStep(step_id="s1", action="think", description="思考分析"),
                PlanStep(step_id="s2", action="act", description="执行操作", dependencies=["s1"]),
                PlanStep(step_id="s3", action="respond", description="生成响应", dependencies=["s2"]),
            ]

        return plan


# ============================================================
# 3. 命令分发器
# ============================================================

class CommandDispatcher(ABC):
    @abstractmethod
    def dispatch(self, command: Command) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class InMemoryDispatcher(CommandDispatcher):
    """内存命令分发器"""
    dispatch_log: List[Dict[str, Any]] = field(default_factory=list)

    def dispatch(self, command: Command) -> Dict[str, Any]:
        result = {
            "accepted": True,
            "command_id": command.command_id,
            "command_type": command.command_type.value,
            "target": command.target,
            "dispatched_at": time.time(),
        }
        self.dispatch_log.append(result)
        return result


# ============================================================
# 4. 控制平面主控制器
# ============================================================

@dataclass
class ControlPlane:
    """
    控制平面 —— 框架的决策中枢
    整合 GoalParser + Planner + CommandDispatcher
    """
    name: str = "control-plane"
    goal_parser: GoalParser = field(default_factory=GoalParser)
    planner: Planner = field(default_factory=Planner)
    dispatcher: CommandDispatcher = field(default_factory=InMemoryDispatcher)

    def process_input(self, raw_input: str,
                      task_type: TaskType = TaskType.GENERAL_CHAT) -> PlanGraph:
        """处理用户输入：解析目标 → 创建计划"""
        goal = self.goal_parser.parse(raw_input, task_type)
        plan = self.planner.create_plan(goal)
        return plan

    def make_plan_command(self, goal: Dict[str, Any]) -> Command:
        """创建计划命令"""
        return Command(command_type=CommandType.PLAN, target="planner", payload=goal)

    def make_execute_command(self, step: PlanStep) -> Command:
        """创建执行命令"""
        return Command(
            command_type=CommandType.EXECUTE,
            target="executor",
            payload={
                "step_id": step.step_id,
                "action": step.action,
                "params": step.params,
            },
        )

    def make_tool_command(self, tool_name: str, params: Dict[str, Any]) -> Command:
        """创建工具调用命令"""
        return Command(
            command_type=CommandType.TOOL_CALL,
            target=tool_name,
            payload=params,
        )


__all__ = [
    "ControlPlane", "CommandDispatcher", "InMemoryDispatcher",
    "GoalParser", "ParsedGoal", "Planner", "PlanGraph", "PlanStep",
]