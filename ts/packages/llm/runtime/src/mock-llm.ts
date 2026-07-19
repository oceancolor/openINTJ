import type { ChatMessage, ChatOptions, LlmClient, LlmStatus } from "@openintj/core";

/** 显式 mock LLM：不依赖 Hunyuan，状态 mode=mock 可识别。 */
export class MockLlmClient implements LlmClient {
  readonly name = "mock-llm";

  async chat(messages: ChatMessage[], _opts?: ChatOptions): Promise<string> {
    const last = [...messages].reverse().find((m) => m.role === "user");
    const q =
      typeof last?.content === "string"
        ? last.content
        : Array.isArray(last?.content)
          ? last.content.map((p) => (p.type === "text" ? p.text : "")).join("")
          : "";
    return `[mock] 收到：${q.slice(0, 200)}`;
  }

  async visionChat(
    messages: ChatMessage[],
    _image: { base64: string; mimeType: string },
    opts?: ChatOptions,
  ): Promise<string> {
    return this.chat(messages, opts);
  }

  getStatus(): LlmStatus {
    return {
      provider: "mock",
      model: "mock-template",
      available: true,
      mode: "mock",
      status: "connected",
      visionSupported: false,
    };
  }
}
