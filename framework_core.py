"""
OpenINTJ 框架核心模块 (framework_core.py)
============================================
以 OpenClaw 为基础架构，融合 pi-mono 极简分层设计，
以及 3D 引擎 Shader 思想的"记忆着色器"创新机制。

核心类型定义：错误模型、命令/事件体系、配置、记忆着色器类型系统。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import os
import time
import uuid
import hashlib
import math


# ============================================================
# 1. 错误模型 —— 统一错误码与可重试标记
# ============================================================

class ErrorCode(str, Enum):
    CONFIG_MISSING = "CONFIG_MISSING"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    TOOL_FAILED = "TOOL_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SHADER_ERROR = "SHADER_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"


@dataclass
class AgentError(Exception):
    code: ErrorCode
    message: str
    retriable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


# ============================================================
# 2. 命令与事件体系 —— 跨平面通信协议
# ============================================================

class CommandType(str, Enum):
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    EVALUATE = "EVALUATE"
    REPAIR = "REPAIR"
    SHADER_SELECT = "SHADER_SELECT"
    MEMORY_RETRIEVE = "MEMORY_RETRIEVE"
    TOOL_CALL = "TOOL_CALL"


class EventType(str, Enum):
    PLANNED = "PLANNED"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    STEP_FAILED = "STEP_FAILED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    COMPLETED = "COMPLETED"
    SHADER_APPLIED = "SHADER_APPLIED"
    MEMORY_LOADED = "MEMORY_LOADED"
    CONTEXT_COMPACTED = "CONTEXT_COMPACTED"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    CIRCUIT_OPENED = "CIRCUIT_OPENED"
    LOOP_ITERATION = "LOOP_ITERATION"


@dataclass
class Command:
    command_type: CommandType
    target: str
    payload: Dict[str, Any] = field(default_factory=dict)
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


@dataclass
class Event:
    event_type: EventType
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    created_at: float = field(default_factory=time.time)


# ============================================================
# 3. 状态快照 —— 数据平面持久化结构
# ============================================================

@dataclass
class StateSnapshot:
    trace_id: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 4. 记忆着色器类型系统 —— 核心创新
#    借鉴 3D 引擎中 Vertex/Pixel Shader 的管线思想
# ============================================================

class ShaderMode(str, Enum):
    """记忆着色器模式，类比 3D 引擎中不同风格的 Shader"""
    HIGH_FIDELITY = "high_fidelity"      # 高保真：保留原始细节（近景高模）
    LOW_FIDELITY = "low_fidelity"        # 低保真：激进压缩摘要（远景低模）
    HYBRID = "hybrid"                     # 混合模式：部分高保真+部分压缩
    ADAPTIVE = "adaptive"                 # 自适应：根据 token 预算动态调整


class TaskType(str, Enum):
    """任务类型分类，决定着色器策略选择"""
    CODE_GENERATION = "code_generation"       # 高精度代码生成
    TECHNICAL_WRITING = "technical_writing"   # 技术文档撰写
    GENERAL_CHAT = "general_chat"             # 一般对话
    QUICK_RESPONSE = "quick_response"         # 快速响应
    ANALYSIS = "analysis"                     # 分析任务
    PLANNING = "planning"                     # 规划任务


class LODLevel(int, Enum):
    """层次细节级别，类比 3D 引擎 LOD"""
    LOD_0 = 0   # 最高细节 —— 原始内容
    LOD_1 = 1   # 高细节 —— 轻度摘要
    LOD_2 = 2   # 中等细节 —— 中度摘要
    LOD_3 = 3   # 低细节 —— 高度压缩
    LOD_4 = 4   # 最低细节 —— 仅保留关键词/标签


@dataclass
class MemoryFragment:
    """
    记忆片段 —— 记忆存储的基本单元
    类比 3D 引擎中的 Mesh 对象，包含多个 LOD 版本
    """
    fragment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""                                    # 原始内容 (LOD_0)
    summaries: Dict[int, str] = field(default_factory=dict)  # LOD 级别 → 摘要
    embedding: List[float] = field(default_factory=list)     # 向量嵌入
    importance: float = 0.5                              # 重要性评分 [0, 1]
    timestamp: float = field(default_factory=time.time)
    task_tags: List[str] = field(default_factory=list)   # 任务标签
    access_count: int = 0                                # 访问计数
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()

    def get_content_at_lod(self, lod: LODLevel) -> str:
        """获取指定 LOD 级别的内容"""
        if lod == LODLevel.LOD_0:
            return self.content
        return self.summaries.get(lod.value, self.content)

    def estimate_tokens(self, lod: LODLevel) -> int:
        """估算指定 LOD 级别的 token 数量（粗略：1 token ≈ 4 字符）"""
        text = self.get_content_at_lod(lod)
        return max(1, len(text) // 4)

    def decay_importance(self, half_life_hours: float = 24.0) -> float:
        """时间衰减函数，类比 3D 引擎中远处物体的淡出"""
        age_hours = (time.time() - self.timestamp) / 3600
        decay = math.exp(-0.693 * age_hours / half_life_hours)
        return self.importance * decay


@dataclass
class ContextBudget:
    """
    上下文预算管理器 —— 类比 3D 引擎的 GPU 资源预算
    监控 token 使用情况，决定是否需要压缩
    """
    max_tokens: int = 128000          # 最大上下文窗口
    reserved_tokens: int = 4096       # 为输出保留的 token
    system_prompt_tokens: int = 0     # 系统提示词占用
    conversation_tokens: int = 0      # 对话历史占用
    memory_tokens: int = 0            # 记忆内容占用
    tool_tokens: int = 0              # 工具描述占用

    @property
    def available_tokens(self) -> int:
        """可用 token 数量"""
        used = (self.system_prompt_tokens + self.conversation_tokens
                + self.memory_tokens + self.tool_tokens + self.reserved_tokens)
        return max(0, self.max_tokens - used)

    @property
    def usage_ratio(self) -> float:
        """使用率 [0, 1]"""
        used = (self.system_prompt_tokens + self.conversation_tokens
                + self.memory_tokens + self.tool_tokens)
        return min(1.0, used / max(1, self.max_tokens - self.reserved_tokens))

    @property
    def memory_budget(self) -> int:
        """分配给记忆的 token 预算"""
        total_available = self.max_tokens - self.reserved_tokens - self.system_prompt_tokens
        # 记忆最多占可用空间的 30%
        return max(0, int(total_available * 0.3) - self.memory_tokens)

    def needs_compaction(self, threshold: float = 0.8) -> bool:
        """是否需要压缩（类比 3D 引擎的 LOD 切换触发条件）"""
        return self.usage_ratio >= threshold


@dataclass
class ShaderConfig:
    """
    着色器配置 —— 定义着色器管线的参数
    类比 3D 引擎中 Shader 的 uniform 变量
    """
    mode: ShaderMode = ShaderMode.ADAPTIVE
    target_lod: LODLevel = LODLevel.LOD_1
    max_summary_length: int = 200        # 摘要最大长度（字符）
    importance_threshold: float = 0.3    # 重要性过滤阈值
    recency_weight: float = 0.4          # 时间近因权重
    relevance_weight: float = 0.4        # 相关性权重
    importance_weight: float = 0.2       # 重要性权重
    compaction_threshold: float = 0.8    # 触发压缩的使用率阈值
    max_fragments_per_query: int = 10    # 每次查询最大记忆片段数

    # 任务类型 → 着色器模式映射
    task_shader_map: Dict[str, ShaderMode] = field(default_factory=lambda: {
        TaskType.CODE_GENERATION.value: ShaderMode.HIGH_FIDELITY,
        TaskType.TECHNICAL_WRITING.value: ShaderMode.HIGH_FIDELITY,
        TaskType.GENERAL_CHAT.value: ShaderMode.LOW_FIDELITY,
        TaskType.QUICK_RESPONSE.value: ShaderMode.LOW_FIDELITY,
        TaskType.ANALYSIS.value: ShaderMode.HYBRID,
        TaskType.PLANNING.value: ShaderMode.HYBRID,
    })

    def get_shader_for_task(self, task_type: TaskType) -> ShaderMode:
        """根据任务类型获取对应的着色器模式"""
        return self.task_shader_map.get(task_type.value, ShaderMode.ADAPTIVE)

    def get_lod_for_mode(self, mode: ShaderMode, budget_ratio: float) -> LODLevel:
        """根据着色器模式和预算使用率确定 LOD 级别"""
        if mode == ShaderMode.HIGH_FIDELITY:
            return LODLevel.LOD_0 if budget_ratio < 0.6 else LODLevel.LOD_1
        elif mode == ShaderMode.LOW_FIDELITY:
            return LODLevel.LOD_3 if budget_ratio < 0.9 else LODLevel.LOD_4
        elif mode == ShaderMode.HYBRID:
            if budget_ratio < 0.5:
                return LODLevel.LOD_1
            elif budget_ratio < 0.8:
                return LODLevel.LOD_2
            else:
                return LODLevel.LOD_3
        else:  # ADAPTIVE
            if budget_ratio < 0.3:
                return LODLevel.LOD_0
            elif budget_ratio < 0.5:
                return LODLevel.LOD_1
            elif budget_ratio < 0.7:
                return LODLevel.LOD_2
            elif budget_ratio < 0.9:
                return LODLevel.LOD_3
            else:
                return LODLevel.LOD_4


# ============================================================
# 5. 工具描述协议 —— 统一工具接口
#    参考 pi-mono 的四工具原语设计
# ============================================================

@dataclass
class ToolDescriptor:
    """工具描述协议，定义统一的工具接口"""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    timeout_s: int = 30
    idempotent: bool = False
    error_semantics: str = "fail_fast"  # fail_fast | retry | ignore


@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0
    trace_id: str = ""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ============================================================
# 6. 配置与校验
# ============================================================

@dataclass
class FrameworkConfig:
    env: str
    max_retry: int
    default_timeout_s: int
    governance_strict: bool
    max_context_tokens: int = 128000
    shader_mode: str = "adaptive"
    memory_half_life_hours: float = 24.0

    @staticmethod
    def load_from_env() -> "FrameworkConfig":
        required_keys = ["AGENT_ENV", "AGENT_MAX_RETRY", "AGENT_DEFAULT_TIMEOUT_S"]
        missing = [k for k in required_keys if os.getenv(k) is None]
        if missing:
            raise AgentError(
                code=ErrorCode.CONFIG_MISSING,
                message="缺少必需的配置项",
                retriable=False,
                details={"missing_keys": missing},
            )

        try:
            max_retry = int(os.getenv("AGENT_MAX_RETRY", "2"))
            timeout = int(os.getenv("AGENT_DEFAULT_TIMEOUT_S", "30"))
            max_tokens = int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", "128000"))
        except ValueError as exc:
            raise AgentError(
                code=ErrorCode.VALIDATION_ERROR,
                message="数值配置项无效",
                retriable=False,
                details={"error": str(exc)},
            )

        strict = os.getenv("AGENT_GOVERNANCE_STRICT", "true").lower() in {"true", "1", "yes"}
        shader = os.getenv("AGENT_SHADER_MODE", "adaptive")
        half_life = float(os.getenv("AGENT_MEMORY_HALF_LIFE_HOURS", "24.0"))

        return FrameworkConfig(
            env=os.getenv("AGENT_ENV", "dev"),
            max_retry=max_retry,
            default_timeout_s=timeout,
            governance_strict=strict,
            max_context_tokens=max_tokens,
            shader_mode=shader,
            memory_half_life_hours=half_life,
        )


@dataclass
class ValidationResult:
    ok: bool
    issues: List[Dict[str, Any]] = field(default_factory=list)


class ConfigValidator:
    @staticmethod
    def validate(config: FrameworkConfig) -> ValidationResult:
        issues: List[Dict[str, Any]] = []
        if config.max_retry < 0:
            issues.append({"field": "max_retry", "reason": "必须 >= 0"})
        if config.default_timeout_s <= 0:
            issues.append({"field": "default_timeout_s", "reason": "必须 > 0"})
        if config.env not in {"dev", "test", "prod"}:
            issues.append({"field": "env", "reason": "必须为 dev/test/prod 之一"})
        if config.max_context_tokens < 1024:
            issues.append({"field": "max_context_tokens", "reason": "必须 >= 1024"})
        if config.shader_mode not in {m.value for m in ShaderMode}:
            issues.append({"field": "shader_mode", "reason": f"必须为 {[m.value for m in ShaderMode]} 之一"})
        return ValidationResult(ok=len(issues) == 0, issues=issues)