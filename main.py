"""
OpenINTJ —— 自研 claw-like + Claude Code-like Agent
=====================================================
以 OpenClaw 为基础架构，融合 pi-mono 极简设计和 3D 引擎 Shader 思想的
"记忆着色器"上下文管理创新机制。

四平面分层架构：
  控制平面 (Control Plane)   → 目标解析、任务规划、命令分发
  执行平面 (Execution Plane) → 工作流编排、工具调用、状态机
  记忆平面 (Memory Plane)    → 记忆存储、检索、着色器管线
  治理平面 (Governance Plane) → 策略引擎、审计追踪、配额限制

核心创新：记忆着色器 (Memory Shader)
  借鉴 3D 引擎 Shader 管线思想，根据任务类型和上下文 token 预算
  动态选择高保真/低保真/混合模式处理记忆内容。
"""
from __future__ import annotations

import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from framework_core import (
    AgentError,
    ContextBudget,
    ShaderMode,
    TaskType,
)
from memory_plane import MemoryPlane
from context_engine import ContextEngine
from control_plane import ControlPlane
from execution_plane import Executor
from governance_plane import GovernancePlane
from agent_loop import AgentLoop, OpenINTJFramework
from llm_client import get_hunyuan_client


# ============================================================
# 全局 Agent 实例
# ============================================================

framework = OpenINTJFramework()


def _preload_memories() -> None:
    """预加载演示记忆片段"""
    memory = framework.agent_loop.memory_plane

    memory.ingest(
        "OpenINTJ 框架采用四平面分层架构：控制平面、执行平面、记忆平面、治理平面。",
        "long_term", importance=0.9,
        task_tags=["architecture"],
        summaries={
            1: "OpenINTJ 四平面架构：控制、执行、记忆、治理",
            2: "四平面分层架构",
            3: "架构",
        },
    )
    memory.ingest(
        "记忆着色器是核心创新，借鉴 3D 引擎 Shader 管线，根据任务类型动态调整记忆细节级别。"
        "高保真模式保留原始细节，低保真模式激进压缩，混合模式部分高保真+部分压缩。"
        "自适应模式根据 token 预算动态切换。",
        "long_term", importance=0.95,
        task_tags=["shader", "innovation"],
        summaries={
            1: "记忆着色器：借鉴3D Shader，动态调整记忆细节（高保真/低保真/混合/自适应）",
            2: "记忆着色器：动态LOD调整",
            3: "着色器",
        },
    )
    memory.ingest(
        "Agent Loop 实现感知→决策→行动→观察→反馈的完整闭环，"
        "参考 OpenClaw 的 Lobster 循环和 pi-mono 的极简设计。",
        "long_term", importance=0.85,
        task_tags=["agent_loop"],
        summaries={
            1: "Agent Loop：感知→决策→行动→观察→反馈闭环",
            2: "闭环循环机制",
            3: "循环",
        },
    )
    memory.ingest(
        "上下文引擎负责 token 预算监控和 JIT 加载，类比 3D 引擎的资源调度器。"
        "当上下文接近上限时自动触发会话压缩（session compaction）。",
        "long_term", importance=0.8,
        task_tags=["context", "budget"],
        summaries={
            1: "上下文引擎：token预算监控 + JIT加载 + 自动压缩",
            2: "token预算与压缩",
            3: "预算",
        },
    )


# 初始化时预加载
_preload_memories()


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="OpenINTJ — Agent 框架",
    description="融合记忆着色器机制的自研 Agent，对接腾讯混元大模型",
    version="2.0.0",
)


# ============================================================
# 请求/响应模型
# ============================================================

class ChatRequest(BaseModel):
    query: str
    shader_mode: Optional[str] = None
    image: Optional[str] = None
    image_mime_type: Optional[str] = None


class ChatResponse(BaseModel):
    trace_id: str
    status: str
    response: str
    task_type: str
    shader_mode: str
    plan_progress: float
    duration_ms: float
    metrics: Dict[str, Any]
    events_count: int


class StatsResponse(BaseModel):
    state: str
    total_runs: int
    total_iterations: int
    context: Dict[str, Any]
    memory: Dict[str, Any]
    governance: Dict[str, Any]
    tools: List[Dict[str, Any]]


# ============================================================
# API 路由
# ============================================================

@app.get("/")
async def root():
    """根路径重定向到静态页面"""
    return RedirectResponse(url="/static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """运行 Agent Loop 处理用户查询"""
    try:
        # 图片数据校验
        image_data = None
        if request.image:
            # 校验 MIME 类型
            allowed_mimes = {"image/jpeg", "image/png", "image/gif", "image/webp"}
            mime = request.image_mime_type or "image/png"
            if mime not in allowed_mimes:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的图片格式: {mime}，支持: {', '.join(allowed_mimes)}"
                )
            # 校验 Base64 合法性和大小
            try:
                raw = base64.b64decode(request.image)
                if len(raw) > 5 * 1024 * 1024:  # 5MB 限制
                    raise HTTPException(
                        status_code=400,
                        detail="图片过大，请选择 5MB 以内的图片"
                    )
                image_data = {
                    "base64": request.image,
                    "mime_type": mime,
                    "size_bytes": len(raw),
                }
            except (base64.binascii.Error, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail="图片数据格式无效，请提供合法的 Base64 编码"
                )

        original_mode = None
        if request.shader_mode:
            original_mode = framework.agent_loop.context_engine.shader_config.mode
            try:
                framework.agent_loop.context_engine.shader_config.mode = ShaderMode(request.shader_mode)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的着色器模式: {request.shader_mode}，可选: {[m.value for m in ShaderMode]}"
                )

        result = framework.run(request.query, image_data=image_data)

        if original_mode is not None:
            framework.agent_loop.context_engine.shader_config.mode = original_mode

        return ChatResponse(**result)

    except AgentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """获取 Agent 框架运行统计"""
    try:
        stats = framework.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/stats")
async def memory_stats():
    """获取记忆平面统计"""
    try:
        return framework.agent_loop.memory_plane.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/governance/audit")
async def audit_trail():
    """获取治理审计记录"""
    try:
        governance = framework.agent_loop.governance
        return {
            "stats": governance.get_stats(),
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "action": e.action,
                    "target": e.target,
                    "result": e.result,
                    "risk_level": e.risk_level,
                    "timestamp": e.timestamp,
                }
                for e in governance.audit_trail.events[-20:]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """健康检查"""
    hunyuan = get_hunyuan_client()
    return {
        "status": "healthy",
        "framework": "OpenINTJ",
        "version": "2.0.0",
        "agent_state": framework.agent_loop.state.value,
        "total_runs": framework.agent_loop.total_runs,
        "llm": hunyuan.get_status(),
    }


@app.get("/api/llm/status")
async def llm_status():
    """获取 LLM 连接状态"""
    hunyuan = get_hunyuan_client()
    return hunyuan.get_status()


# ============================================================
# 静态文件服务（必须在最后 mount）
# ============================================================
app.mount("/static", StaticFiles(directory="static", html=True), name="static")