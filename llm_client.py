"""
LLM 客户端 —— 腾讯混元大模型对接
====================================
通过 OpenAI 兼容接口调用腾讯混元大模型。
混元 API 完全兼容 OpenAI SDK，仅需替换 base_url 和 api_key。

配置优先级：环境变量 > 代码内默认值

环境变量：
  HUNYUAN_API_KEY   - 混元 API Key（必须）
  HUNYUAN_BASE_URL  - 混元 API 地址（默认: https://api.hunyuan.cloud.tencent.com/v1）
  HUNYUAN_MODEL     - 模型名称（默认: hunyuan-turbos-latest）
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class HunyuanConfig:
    """腾讯混元配置（OpenAI 兼容端点）"""
    api_key: str = ""
    base_url: str = "https://api.hunyuan.cloud.tencent.com/v1"
    model: str = "hunyuan-turbos-latest"
    vision_model: str = "hunyuan-vision"  # Vision 多模态模型
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False  # 当前使用非流式，后续可扩展为流式

    @classmethod
    def from_env(cls) -> "HunyuanConfig":
        """从环境变量加载配置"""
        return cls(
            api_key=os.environ.get("HUNYUAN_API_KEY", ""),
            base_url=os.environ.get("HUNYUAN_BASE_URL", "https://api.hunyuan.cloud.tencent.com/v1"),
            model=os.environ.get("HUNYUAN_MODEL", "hunyuan-turbos-latest"),
            vision_model=os.environ.get("HUNYUAN_VISION_MODEL", "hunyuan-vision"),
        )


# ============================================================
# 混元 LLM 客户端
# ============================================================

@dataclass
class HunyuanClient:
    """
    腾讯混元大模型客户端
    
    基于 OpenAI 兼容接口，使用 openai SDK 直接调用。
    支持同步调用和流式调用（后续扩展）。
    """
    config: HunyuanConfig = field(default_factory=HunyuanConfig.from_env)
    _client: Optional[OpenAI] = field(default=None, init=False, repr=False)
    _last_error_message: str = field(default="", init=False, repr=False)
    _last_error_code: str = field(default="", init=False, repr=False)
    _last_error_type: str = field(default="", init=False, repr=False)
    _authorization_failed: bool = field(default=False, init=False, repr=False)

    @property
    def vision_model(self) -> str:
        """当前 Vision 模型名称，优先使用显式配置值"""
        return self.config.vision_model or self.config.model

    def __post_init__(self):
        if not self.config.api_key:
            logger.warning(
                "未设置 HUNYUAN_API_KEY 环境变量，LLM 调用将使用 mock 模式。"
                "请设置环境变量或在 HunyuanConfig 中提供 api_key。"
            )
            return

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        logger.info(
            f"混元客户端已初始化 | model={self.config.model} | "
            f"base_url={self.config.base_url}"
        )

    @property
    def is_available(self) -> bool:
        """检查客户端是否可用（已配置 API Key 且最近未出现鉴权失败）"""
        return (
            self._client is not None
            and bool(self.config.api_key)
            and not self._authorization_failed
        )

    def _reset_runtime_status(self) -> None:
        """在成功调用后重置运行时错误状态"""
        self._last_error_message = ""
        self._last_error_code = ""
        self._last_error_type = ""
        self._authorization_failed = False

    def _extract_error_text(self, error: Exception) -> str:
        """提取适合展示的错误文本"""
        message = str(error).strip()
        return message or error.__class__.__name__

    def _mark_runtime_error(self, error: Exception) -> str:
        """记录最近一次运行时错误，并识别鉴权失败"""
        error_text = self._extract_error_text(error)
        lowered = error_text.lower()

        self._last_error_message = error_text
        self._last_error_code = ""
        self._last_error_type = ""

        if "not_authorized" in lowered or "not authorized" in lowered or "401" in lowered:
            self._authorization_failed = True
            self._last_error_code = "not_authorized"
            self._last_error_type = "authentication"
        elif "invalid model" in lowered:
            self._last_error_code = "invalid_model"
            self._last_error_type = "configuration"
        else:
            self._last_error_code = "runtime_error"
            self._last_error_type = "runtime"

        return error_text

    def _mock_response(self, messages: List[Dict[str, str]], error: Optional[str] = None) -> str:
        """文本对话的本地降级响应"""
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user = content
                else:
                    last_user = str(content)
                break

        prefix = "[OpenINTJ Mock 模式] 当前使用本地模拟响应。"
        detail = "当前预览环境未连接可用的混元文本模型，因此返回本地演示响应。"
        if error:
            prefix = f"[混元 API 调用异常: {error}] 已降级为本地响应。"
            if self._authorization_failed:
                detail = (
                    "当前提供的 HUNYUAN_API_KEY 未通过鉴权或没有所选模型权限，"
                    "请更换有效密钥，或确认该密钥已开通当前模型访问权限。"
                )
            elif self._last_error_code == "invalid_model":
                detail = (
                    "当前配置的模型名称不可用，请检查 HUNYUAN_MODEL / HUNYUAN_VISION_MODEL "
                    "是否为该密钥已开通的有效模型。"
                )

        return (
            f"{prefix}\n\n"
            f"您的输入: {last_user[:200]}\n\n"
            f"{detail}"
        )

    def chat(self, messages: List[Dict[str, str]],
             model: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """
        调用混元大模型进行对话
        
        参数:
            messages: OpenAI 格式的消息列表 [{"role": "...", "content": "..."}]
                      content 可以是字符串或数组（多模态格式）
            model: 可选，覆盖默认模型
            temperature: 可选，覆盖默认温度
            max_tokens: 可选，覆盖默认最大 token 数
            
        返回:
            模型生成的文本响应
        """
        if not self.is_available:
            return self._mock_response(messages)

        try:
            response = self._client.chat.completions.create(
                model=model or self.config.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
            )

            self._reset_runtime_status()
            content = response.choices[0].message.content
            
            # 记录 token 使用情况
            usage = response.usage
            if usage:
                logger.info(
                    f"混元调用完成 | prompt_tokens={usage.prompt_tokens} | "
                    f"completion_tokens={usage.completion_tokens} | "
                    f"total_tokens={usage.total_tokens}"
                )

            return content or ""

        except Exception as e:
            error_text = self._mark_runtime_error(e)
            logger.error(f"混元 API 调用失败: {error_text}")
            return self._mock_response(messages, error_text)

    def vision_chat(self, messages: List[Dict[str, str]], image_base64: str,
                    image_mime_type: str = "image/png",
                    model: Optional[str] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> str:
        """调用多模态模型进行图文对话；当前演示环境下优先兼容为文本降级。"""
        if not self.is_available:
            return self._mock_vision_response(messages, image_mime_type)

        try:
            content_blocks = []
            for msg in messages:
                if msg.get("role") == "user":
                    content_blocks.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg.get("content", "")},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_mime_type};base64,{image_base64}"
                                },
                            },
                        ],
                    })
                else:
                    content_blocks.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })

            response = self._client.chat.completions.create(
                model=model or self.vision_model,
                messages=content_blocks,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
            )
            self._reset_runtime_status()
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            error_text = self._mark_runtime_error(e)
            logger.warning(f"混元 Vision 调用失败，降级为文本响应: {error_text}")
            return self._mock_vision_response(messages, image_mime_type)

    def _mock_vision_response(self, messages: List[Dict[str, str]], image_mime_type: str) -> str:
        """Vision Mock 响应"""
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        return (
            "[OpenINTJ Vision Mock 模式] 当前使用模拟图片理解响应。\n\n"
            f"图片类型: {image_mime_type}\n"
            f"您的输入: {last_user[:200]}\n\n"
            + (
                "当前提供的 HUNYUAN_API_KEY 未通过鉴权或没有 Vision 模型权限。"
                if self._authorization_failed
                else "请设置可用的 HUNYUAN_API_KEY 以启用真实多模态能力。"
            )
        )

    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态信息"""
        mode = "live" if self.is_available else "mock"
        status_label = "connected" if self.is_available else "degraded"
        if self._authorization_failed:
            mode = "unauthorized"
            status_label = "unauthorized"
        elif self._client is None or not self.config.api_key:
            mode = "mock"
            status_label = "missing_api_key"

        return {
            "provider": "腾讯混元",
            "model": self.config.model,
            "vision_model": self.vision_model,
            "base_url": self.config.base_url,
            "available": self.is_available,
            "mode": mode,
            "status": status_label,
            "last_error": self._last_error_message,
            "last_error_code": self._last_error_code,
            "last_error_type": self._last_error_type,
            "vision_supported": True,
        }


# ============================================================
# 全局单例
# ============================================================

_global_client: Optional[HunyuanClient] = None


def get_hunyuan_client() -> HunyuanClient:
    """获取全局混元客户端单例"""
    global _global_client
    if _global_client is None:
        _global_client = HunyuanClient()
    return _global_client


def create_llm_fn():
    """
    创建可注入 AgentLoop 的 LLM 调用函数
    
    返回一个签名为 (messages: List[Dict[str, Any]]) -> str 的函数，
    可直接赋值给 AgentLoop.llm_fn
    """
    client = get_hunyuan_client()

    def llm_fn(messages: List[Dict[str, str]]) -> str:
        return client.chat(messages)

    return llm_fn


def create_vision_llm_fn():
    """创建可注入 AgentLoop 的 Vision LLM 调用函数"""
    client = get_hunyuan_client()

    def vision_llm_fn(messages: List[Dict[str, str]], image_base64: str,
                      image_mime_type: str = "image/png") -> str:
        return client.vision_chat(messages, image_base64, image_mime_type)

    return vision_llm_fn


__all__ = [
    "HunyuanConfig", "HunyuanClient",
    "get_hunyuan_client", "create_llm_fn", "create_vision_llm_fn",
]