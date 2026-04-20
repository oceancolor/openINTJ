"""
上下文引擎 (Context Engine)
============================
负责 token 预算监控、动态着色器策略选择、JIT（Just-In-Time）加载。
类比 3D 引擎的渲染管线调度器，决定何时加载/卸载资源。

OpenINTJ 核心组件，参考 OpenClaw 的 session compaction 机制和 Anthropic 的 JIT 上下文加载策略。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from framework_core import (
    AgentError, ErrorCode, Event, EventType,
    ContextBudget, ShaderConfig, ShaderMode, LODLevel, TaskType,
    MemoryFragment,
)
from memory_plane import MemoryPlane


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str           # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = 0.0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_data: Optional[Dict[str, Any]] = None  # 图片数据 {"base64": ..., "mime_type": ...}
    has_image: bool = False  # 是否包含图片

    def __post_init__(self):
        if self.token_count == 0:
            # 图片消息按固定 765 tokens 估算
            base_tokens = max(1, len(self.content) // 4)
            if self.has_image or self.image_data:
                self.token_count = base_tokens + 765
                self.has_image = True
            else:
                self.token_count = base_tokens


@dataclass
class ContextWindow:
    """
    上下文窗口 —— 类比 3D 引擎的帧缓冲区
    管理当前 LLM 调用的完整上下文
    """
    system_prompt: str = ""
    messages: List[ConversationMessage] = field(default_factory=list)
    memory_context: List[Dict[str, Any]] = field(default_factory=list)
    tool_descriptions: List[Dict[str, Any]] = field(default_factory=list)

    def get_total_tokens(self) -> int:
        """计算总 token 数"""
        total = max(1, len(self.system_prompt) // 4)
        total += sum(m.token_count for m in self.messages)
        total += sum(item.get("tokens", 0) for item in self.memory_context)
        total += sum(max(1, len(str(t)) // 4) for t in self.tool_descriptions)
        return total

    def to_prompt_messages(self) -> List[Dict[str, Any]]:
        """转换为 LLM API 格式的消息列表（支持多模态）"""
        result = []
        if self.system_prompt:
            # 注入记忆上下文到系统提示
            memory_text = ""
            if self.memory_context:
                memory_text = "\n\n## 相关记忆\n"
                for item in self.memory_context:
                    memory_text += f"- [LOD{item.get('lod', 0)}] {item.get('content', '')}\n"
            result.append({
                "role": "system",
                "content": self.system_prompt + memory_text,
            })
        for msg in self.messages:
            if msg.has_image and msg.image_data:
                # 构建 OpenAI Vision 格式的多模态 content 数组
                content_parts = []
                if msg.content:
                    content_parts.append({
                        "type": "text",
                        "text": msg.content,
                    })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{msg.image_data.get('mime_type', 'image/png')};base64,{msg.image_data.get('base64', '')}",
                    },
                })
                result.append({"role": msg.role, "content": content_parts})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result


@dataclass
class TaskClassifier:
    """
    任务分类器 —— 轻量级任务类型判定
    决定使用哪种着色器模式
    """
    # 关键词 → 任务类型映射
    keyword_map: Dict[str, TaskType] = field(default_factory=lambda: {
        "代码": TaskType.CODE_GENERATION,
        "code": TaskType.CODE_GENERATION,
        "编程": TaskType.CODE_GENERATION,
        "函数": TaskType.CODE_GENERATION,
        "实现": TaskType.CODE_GENERATION,
        "debug": TaskType.CODE_GENERATION,
        "文档": TaskType.TECHNICAL_WRITING,
        "报告": TaskType.TECHNICAL_WRITING,
        "撰写": TaskType.TECHNICAL_WRITING,
        "分析": TaskType.ANALYSIS,
        "比较": TaskType.ANALYSIS,
        "评估": TaskType.ANALYSIS,
        "计划": TaskType.PLANNING,
        "规划": TaskType.PLANNING,
        "设计": TaskType.PLANNING,
        "快速": TaskType.QUICK_RESPONSE,
        "简单": TaskType.QUICK_RESPONSE,
    })

    def classify(self, query: str) -> TaskType:
        """根据查询内容分类任务类型"""
        query_lower = query.lower()
        scores: Dict[TaskType, int] = {}
        for keyword, task_type in self.keyword_map.items():
            if keyword in query_lower:
                scores[task_type] = scores.get(task_type, 0) + 1
        if scores:
            return max(scores, key=scores.get)
        return TaskType.GENERAL_CHAT


@dataclass
class ContextEngine:
    """
    上下文引擎 —— 框架的"渲染调度器"
    
    核心职责：
    1. 监控 token 预算（类比 GPU 资源监控）
    2. 动态选择着色器策略（类比 LOD 切换）
    3. JIT 加载记忆（类比远近景加载）
    4. 会话压缩（类比 OpenClaw 的 session compaction）
    """
    budget: ContextBudget = field(default_factory=ContextBudget)
    shader_config: ShaderConfig = field(default_factory=ShaderConfig)
    memory_plane: MemoryPlane = field(default_factory=MemoryPlane)
    classifier: TaskClassifier = field(default_factory=TaskClassifier)
    window: ContextWindow = field(default_factory=ContextWindow)

    # 压缩历史记录
    compaction_count: int = 0
    events: List[Event] = field(default_factory=list)

    def __post_init__(self):
        # 同步预算到着色器管线
        self.memory_plane.pipeline.budget = self.budget
        self.memory_plane.pipeline.config = self.shader_config

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.window.system_prompt = prompt
        self.budget.system_prompt_tokens = max(1, len(prompt) // 4)

    def add_message(self, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    image_data: Optional[Dict[str, Any]] = None) -> None:
        """添加对话消息（支持多模态）"""
        import time
        has_image = image_data is not None
        msg = ConversationMessage(
            role=role, content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            image_data=image_data,
            has_image=has_image,
        )
        self.window.messages.append(msg)
        self.budget.conversation_tokens += msg.token_count

        # 将用户消息存入短期记忆（仅存文本，不存图片原始数据）
        if role == "user":
            memory_content = content
            if has_image:
                memory_content = f"{content} [用户发送了一张图片]" if content else "[用户发送了一张图片]"
            self.memory_plane.ingest(memory_content, "short_term", importance=0.6)

        # 检查是否需要压缩
        if self.budget.needs_compaction(self.shader_config.compaction_threshold):
            self._compact()

    def build_context(self, query: str) -> ContextWindow:
        """
        构建完整上下文 —— JIT 加载记忆
        
        流程：
        1. 分类任务类型
        2. 检索相关记忆
        3. 通过着色器管线处理
        4. 组装上下文窗口
        """
        # 1. 任务分类
        task_type = self.classifier.classify(query)

        # 2. 更新预算状态
        self._sync_budget()

        # 3. JIT 加载记忆（只在需要时检索）
        memory_results = self.memory_plane.query(
            query, task_type=task_type,
            top_k=self.shader_config.max_fragments_per_query,
        )

        # 4. 更新上下文窗口
        self.window.memory_context = memory_results
        self.budget.memory_tokens = sum(
            item.get("tokens", 0) for item in memory_results
        )

        # 5. 记录事件
        self.events.append(Event(
            event_type=EventType.MEMORY_LOADED,
            source="context-engine",
            payload={
                "task_type": task_type.value,
                "memory_count": len(memory_results),
                "budget_usage": self.budget.usage_ratio,
                "shader_mode": self.shader_config.get_shader_for_task(task_type).value,
            },
        ))

        return self.window

    def _sync_budget(self) -> None:
        """同步预算状态"""
        self.budget.conversation_tokens = sum(
            m.token_count for m in self.window.messages
        )

    def _compact(self) -> None:
        """
        会话压缩 —— 类比 OpenClaw 的 session compaction
        当上下文接近上限时，压缩旧消息
        优先丢弃历史图片数据以节省 token 预算
        """
        # 首先尝试剥离历史消息中的图片数据（保留文本）
        stripped_count = 0
        for msg in self.window.messages[:-2]:  # 保留最近2条的图片
            if msg.has_image and msg.image_data:
                msg.image_data = None  # 丢弃图片原始数据
                msg.content = f"{msg.content} [图片已从上下文中移除]" if msg.content else "[图片已从上下文中移除]"
                msg.token_count = max(1, len(msg.content) // 4)  # 重新计算 token
                stripped_count += 1

        if stripped_count > 0:
            self._sync_budget()
            # 如果剥离图片后预算已足够，不再压缩消息
            if not self.budget.needs_compaction(self.shader_config.compaction_threshold):
                return

        if len(self.window.messages) <= 4:
            return

        # 保留最近 4 条消息，压缩其余
        to_compact = self.window.messages[:-4]
        self.window.messages = self.window.messages[-4:]

        # 生成压缩摘要（图片消息压缩为文本描述）
        summary_parts = []
        for msg in to_compact:
            if msg.has_image:
                summary_parts.append(f"[{msg.role}] {msg.content[:80]} [附带图片]")
            else:
                summary_parts.append(f"[{msg.role}] {msg.content[:100]}")
        summary = "对话历史摘要：\n" + "\n".join(summary_parts)

        # 存入长期记忆
        self.memory_plane.ingest(
            summary, "long_term", importance=0.5,
            task_tags=["compaction"],
            summaries={
                1: summary[:200],
                2: summary[:100],
                3: summary[:50],
            },
        )

        self.compaction_count += 1
        self._sync_budget()

        self.events.append(Event(
            event_type=EventType.CONTEXT_COMPACTED,
            source="context-engine",
            payload={
                "compacted_messages": len(to_compact),
                "compaction_count": self.compaction_count,
                "new_budget_usage": self.budget.usage_ratio,
            },
        ))

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "budget": {
                "max_tokens": self.budget.max_tokens,
                "available_tokens": self.budget.available_tokens,
                "usage_ratio": round(self.budget.usage_ratio, 4),
                "needs_compaction": self.budget.needs_compaction(),
            },
            "conversation": {
                "message_count": len(self.window.messages),
                "conversation_tokens": self.budget.conversation_tokens,
            },
            "memory": self.memory_plane.get_stats(),
            "compaction_count": self.compaction_count,
            "event_count": len(self.events),
        }