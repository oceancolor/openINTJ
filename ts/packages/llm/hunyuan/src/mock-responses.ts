/**
 * Mock 响应库 —— 与 Python llm_client._mock_response 行为对齐。
 * 在没有 API key 或鉴权失败降级时返回这些占位响应。
 */
import type { ChatMessage, ChatMessageContent } from "@openintj/core";

const stringifyContent = (content: ChatMessageContent): string => {
  if (typeof content === "string") return content;
  return content.map((c) => (c.type === "text" ? c.text : `[image:${c.image_url.url}]`)).join(" ");
};

export const MOCK_RESPONSES: Record<string, string> = {
  greet: "你好！我是 OpenINTJ Agent（mock 模式）。配置 LLM_API_KEY 启用真实模型。",
  help: "OpenINTJ 是一个本地优先的 Agent 框架，支持 TAO/ReAct 循环、4 平面架构、混合记忆、函数钩子。请配置 LLM 凭据以启用智能对话。",
  default: "[mock] 已收到您的请求，但当前运行在 mock 模式（无凭据/鉴权失败）。",
};

export const generateMockResponse = (messages: ChatMessage[]): string => {
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const text = lastUser ? stringifyContent(lastUser.content) : "";
  const lower = text.toLowerCase();
  if (
    lower.includes("hello") ||
    lower.includes("hi") ||
    text.includes("你好") ||
    text.includes("您好")
  ) {
    return MOCK_RESPONSES["greet"]!;
  }
  if (
    lower.includes("help") ||
    text.includes("帮助") ||
    text.includes("是什么") ||
    text.includes("介绍")
  ) {
    return MOCK_RESPONSES["help"]!;
  }
  return `${MOCK_RESPONSES["default"]} 您说: "${text.slice(0, 80)}${
    text.length > 80 ? "..." : ""
  }"`;
};
