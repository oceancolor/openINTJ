"""
记忆平面 (Memory Plane) —— 数据平面核心
==========================================
实现记忆存储、检索、向量相似度、LOD 压缩、着色器处理管线。
类比 3D 引擎的场景图管理 + Shader 管线。

OpenINTJ 核心创新模块，参考 OpenClaw 的 Hybrid RAG 记忆系统（向量+BM25+MMR+时间衰减）
以及 pi-mono 的轻量级本地存储设计。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import time
import hashlib

from framework_core import (
    AgentError, ErrorCode, Event, EventType,
    MemoryFragment, ContextBudget, ShaderConfig,
    ShaderMode, LODLevel, TaskType, StateSnapshot,
)


# ============================================================
# 1. 向量相似度计算 —— 轻量级本地实现
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度计算"""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def simple_embedding(text: str, dim: int = 64) -> List[float]:
    """
    简易文本嵌入（用于演示，生产环境应替换为真实嵌入模型）
    基于字符哈希生成伪向量
    """
    h = hashlib.sha256(text.encode()).hexdigest()
    values = []
    for i in range(dim):
        byte_val = int(h[(i * 2) % len(h):(i * 2 + 2) % len(h) or len(h)], 16)
        values.append((byte_val / 255.0) * 2 - 1)  # 归一化到 [-1, 1]
    norm = math.sqrt(sum(v * v for v in values))
    if norm > 0:
        values = [v / norm for v in values]
    return values


# ============================================================
# 2. 记忆存储 —— 类比 3D 引擎的场景图
# ============================================================

@dataclass
class MemoryStore:
    """
    记忆存储器 —— 管理所有记忆片段
    类比 3D 引擎中的 SceneGraph，层次化组织记忆对象
    """
    # 短期记忆：当前会话上下文（类比近景物体）
    short_term: List[MemoryFragment] = field(default_factory=list)
    # 长期记忆：持久化历史（类比远景物体）
    long_term: List[MemoryFragment] = field(default_factory=list)
    # 工作记忆：当前任务临时状态（类比中景物体）
    working: List[MemoryFragment] = field(default_factory=list)

    # 配置
    max_short_term: int = 50
    max_working: int = 20
    embedding_dim: int = 64

    def add_short_term(self, content: str, importance: float = 0.5,
                       task_tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> MemoryFragment:
        """添加短期记忆"""
        fragment = MemoryFragment(
            content=content,
            embedding=simple_embedding(content, self.embedding_dim),
            importance=importance,
            task_tags=task_tags or [],
            metadata=metadata or {},
        )
        self.short_term.append(fragment)
        # 超出容量时，将最旧的迁移到长期记忆
        while len(self.short_term) > self.max_short_term:
            oldest = self.short_term.pop(0)
            self.long_term.append(oldest)
        return fragment

    def add_long_term(self, content: str, importance: float = 0.5,
                      summaries: Optional[Dict[int, str]] = None,
                      task_tags: Optional[List[str]] = None) -> MemoryFragment:
        """添加长期记忆（含多级 LOD 摘要）"""
        fragment = MemoryFragment(
            content=content,
            summaries=summaries or {},
            embedding=simple_embedding(content, self.embedding_dim),
            importance=importance,
            task_tags=task_tags or [],
        )
        self.long_term.append(fragment)
        return fragment

    def add_working(self, content: str, importance: float = 0.7,
                    task_tags: Optional[List[str]] = None) -> MemoryFragment:
        """添加工作记忆"""
        fragment = MemoryFragment(
            content=content,
            embedding=simple_embedding(content, self.embedding_dim),
            importance=importance,
            task_tags=task_tags or [],
        )
        self.working.append(fragment)
        while len(self.working) > self.max_working:
            self.working.pop(0)
        return fragment

    def clear_working(self) -> None:
        """清空工作记忆"""
        self.working.clear()

    @property
    def all_fragments(self) -> List[MemoryFragment]:
        """获取所有记忆片段"""
        return self.short_term + self.working + self.long_term

    @property
    def total_count(self) -> int:
        return len(self.short_term) + len(self.working) + len(self.long_term)

    def save_snapshot(self, snapshot: StateSnapshot) -> None:
        """保存状态快照到长期记忆"""
        content = f"[快照] trace={snapshot.trace_id} status={snapshot.status}"
        if snapshot.context:
            content += f" context={snapshot.context}"
        self.add_long_term(content, importance=0.6, task_tags=["snapshot"])


# ============================================================
# 3. 记忆检索器 —— 类比 3D 引擎的遮挡剔除
# ============================================================

@dataclass
class MemoryRetriever:
    """
    记忆检索器 —— 混合检索（向量 + 关键词 + 时间衰减）
    类比 3D 引擎的遮挡剔除：只检索与当前查询相关的记忆，剔除无关信息
    """
    store: MemoryStore = field(default_factory=MemoryStore)
    shader_config: ShaderConfig = field(default_factory=ShaderConfig)

    def retrieve(self, query: str, top_k: int = 10,
                 task_type: Optional[TaskType] = None,
                 min_importance: float = 0.0) -> List[Tuple[MemoryFragment, float]]:
        """
        混合检索记忆片段
        返回 (记忆片段, 综合得分) 列表，按得分降序排列
        """
        query_embedding = simple_embedding(query, self.store.embedding_dim)
        query_keywords = set(query.lower().split())

        scored: List[Tuple[MemoryFragment, float]] = []

        for fragment in self.store.all_fragments:
            # 跳过低重要性片段（遮挡剔除）
            if fragment.decay_importance() < min_importance:
                continue

            # 1. 向量相似度得分
            vec_score = cosine_similarity(query_embedding, fragment.embedding)

            # 2. 关键词匹配得分（BM25 简化版）
            content_words = set(fragment.content.lower().split())
            overlap = len(query_keywords & content_words)
            keyword_score = overlap / max(1, len(query_keywords))

            # 3. 时间衰减得分
            recency_score = fragment.decay_importance(
                self.shader_config.max_summary_length / 10  # 近似半衰期
            )

            # 4. 综合得分
            score = (
                self.shader_config.relevance_weight * vec_score
                + self.shader_config.importance_weight * recency_score
                + self.shader_config.recency_weight * keyword_score
            )

            # 任务标签加权
            if task_type and task_type.value in fragment.task_tags:
                score *= 1.3

            scored.append((fragment, score))

        # 按得分降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        # 更新访问计数
        for fragment, _ in scored[:top_k]:
            fragment.access_count += 1
            fragment.last_accessed = time.time()

        return scored[:top_k]


# ============================================================
# 4. 着色器处理管线 —— 核心创新
#    类比 3D 引擎的 Vertex Shader → Fragment Shader 管线
# ============================================================

@dataclass
class ShaderPipeline:
    """
    记忆着色器管线 —— 对检索到的记忆片段进行动态处理
    
    管线流程（类比 3D 渲染管线）：
    1. 顶点着色阶段 (Vertex Shader)：确定每个记忆片段的 LOD 级别
    2. 几何着色阶段 (Geometry Shader)：过滤和重组记忆片段
    3. 片元着色阶段 (Fragment Shader)：生成最终的摘要/压缩输出
    """
    config: ShaderConfig = field(default_factory=ShaderConfig)
    budget: ContextBudget = field(default_factory=ContextBudget)

    # 自定义摘要函数（可替换为 LLM 调用）
    summarize_fn: Optional[Callable[[str, int], str]] = None

    def __post_init__(self):
        if self.summarize_fn is None:
            self.summarize_fn = self._default_summarize

    @staticmethod
    def _default_summarize(text: str, max_length: int) -> str:
        """默认摘要函数（截断式，生产环境应替换为 LLM 摘要）"""
        if len(text) <= max_length:
            return text
        # 保留开头和结尾
        head = max_length * 2 // 3
        tail = max_length - head - 5
        return text[:head] + " ... " + text[-tail:] if tail > 0 else text[:max_length]

    def process(self, fragments: List[Tuple[MemoryFragment, float]],
                task_type: TaskType = TaskType.GENERAL_CHAT) -> List[Dict[str, Any]]:
        """
        着色器管线主处理流程
        输入：检索到的 (记忆片段, 得分) 列表
        输出：处理后的记忆内容列表
        """
        if not fragments:
            return []

        # 1. 确定着色器模式
        shader_mode = self.config.get_shader_for_task(task_type)
        budget_ratio = self.budget.usage_ratio

        # 自适应模式下根据预算动态调整
        if shader_mode == ShaderMode.ADAPTIVE:
            if self.budget.needs_compaction(self.config.compaction_threshold):
                shader_mode = ShaderMode.LOW_FIDELITY
            elif budget_ratio < 0.5:
                shader_mode = ShaderMode.HIGH_FIDELITY
            else:
                shader_mode = ShaderMode.HYBRID

        # 2. 顶点着色阶段：确定每个片段的 LOD
        lod_assignments = self._vertex_shader(fragments, shader_mode, budget_ratio)

        # 3. 几何着色阶段：过滤和重组
        filtered = self._geometry_shader(lod_assignments)

        # 4. 片元着色阶段：生成最终输出
        output = self._fragment_shader(filtered, shader_mode)

        return output

    def _vertex_shader(self, fragments: List[Tuple[MemoryFragment, float]],
                       mode: ShaderMode, budget_ratio: float
                       ) -> List[Tuple[MemoryFragment, float, LODLevel]]:
        """
        顶点着色阶段：为每个记忆片段分配 LOD 级别
        高得分片段获得更高的细节级别（更近的"距离"）
        """
        result = []
        base_lod = self.config.get_lod_for_mode(mode, budget_ratio)

        for i, (fragment, score) in enumerate(fragments):
            # 根据得分排名调整 LOD：排名靠前的获得更高细节
            if mode == ShaderMode.HYBRID:
                # 混合模式：前 30% 高保真，其余低保真
                threshold = max(1, int(len(fragments) * 0.3))
                if i < threshold:
                    lod = LODLevel(max(0, base_lod.value - 1))
                else:
                    lod = LODLevel(min(4, base_lod.value + 1))
            elif mode == ShaderMode.HIGH_FIDELITY:
                lod = LODLevel(max(0, base_lod.value - (1 if score > 0.7 else 0)))
            elif mode == ShaderMode.LOW_FIDELITY:
                lod = LODLevel(min(4, base_lod.value + (1 if score < 0.3 else 0)))
            else:
                lod = base_lod

            result.append((fragment, score, lod))

        return result

    def _geometry_shader(self, assignments: List[Tuple[MemoryFragment, float, LODLevel]]
                         ) -> List[Tuple[MemoryFragment, float, LODLevel]]:
        """
        几何着色阶段：过滤低重要性片段，控制总量
        类比 3D 引擎的视锥体剔除
        """
        # 过滤低于重要性阈值的片段
        filtered = [
            (f, s, l) for f, s, l in assignments
            if f.decay_importance() >= self.config.importance_threshold
        ]

        # 限制最大片段数
        return filtered[:self.config.max_fragments_per_query]

    def _fragment_shader(self, assignments: List[Tuple[MemoryFragment, float, LODLevel]],
                         mode: ShaderMode) -> List[Dict[str, Any]]:
        """
        片元着色阶段：根据 LOD 级别生成最终输出内容
        类比 3D 引擎的像素着色器，决定最终"像素颜色"
        """
        output = []
        remaining_budget = self.budget.memory_budget

        for fragment, score, lod in assignments:
            # 获取对应 LOD 的内容
            content = fragment.get_content_at_lod(lod)

            # 如果没有预生成的摘要，动态生成
            if lod != LODLevel.LOD_0 and lod.value not in fragment.summaries:
                target_length = self.config.max_summary_length // max(1, lod.value)
                content = self.summarize_fn(fragment.content, target_length)

            # 检查 token 预算
            estimated_tokens = max(1, len(content) // 4)
            if estimated_tokens > remaining_budget:
                # 预算不足，进一步压缩
                content = self.summarize_fn(content, remaining_budget * 4)
                estimated_tokens = max(1, len(content) // 4)

            if remaining_budget <= 0:
                break

            remaining_budget -= estimated_tokens

            output.append({
                "fragment_id": fragment.fragment_id,
                "content": content,
                "lod": lod.value,
                "score": round(score, 4),
                "shader_mode": mode.value,
                "tokens": estimated_tokens,
                "importance": round(fragment.decay_importance(), 4),
            })

        return output


# ============================================================
# 5. 记忆平面 —— 整合存储、检索、着色器管线
# ============================================================

@dataclass
class MemoryPlane:
    """
    记忆平面 —— 数据平面的完整实现
    整合 MemoryStore + MemoryRetriever + ShaderPipeline
    """
    name: str = "memory-plane"
    store: MemoryStore = field(default_factory=MemoryStore)
    retriever: MemoryRetriever = field(default_factory=MemoryRetriever)
    pipeline: ShaderPipeline = field(default_factory=ShaderPipeline)

    def __post_init__(self):
        # 确保检索器和管线共享同一个存储
        self.retriever.store = self.store

    def ingest(self, content: str, memory_type: str = "short_term",
               importance: float = 0.5, task_tags: Optional[List[str]] = None,
               summaries: Optional[Dict[int, str]] = None) -> MemoryFragment:
        """摄入新记忆"""
        if memory_type == "short_term":
            return self.store.add_short_term(content, importance, task_tags)
        elif memory_type == "long_term":
            return self.store.add_long_term(content, importance, summaries, task_tags)
        elif memory_type == "working":
            return self.store.add_working(content, importance, task_tags)
        else:
            raise AgentError(
                code=ErrorCode.MEMORY_ERROR,
                message=f"未知记忆类型: {memory_type}",
            )

    def query(self, query: str, task_type: TaskType = TaskType.GENERAL_CHAT,
              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        查询记忆 —— 完整的检索+着色器处理流程
        1. 检索相关记忆片段（遮挡剔除）
        2. 通过着色器管线处理（LOD + 压缩）
        3. 返回处理后的记忆内容
        """
        # 检索
        retrieved = self.retriever.retrieve(
            query, top_k=top_k, task_type=task_type,
            min_importance=self.pipeline.config.importance_threshold,
        )

        # 着色器处理
        processed = self.pipeline.process(retrieved, task_type)

        return processed

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆平面统计信息"""
        return {
            "short_term_count": len(self.store.short_term),
            "long_term_count": len(self.store.long_term),
            "working_count": len(self.store.working),
            "total_count": self.store.total_count,
            "shader_mode": self.pipeline.config.mode.value,
            "budget_usage": round(self.pipeline.budget.usage_ratio, 4),
        }