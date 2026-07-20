import React from "react";
import type { InputStructure, ModelProfile } from "../../shared/ipc-protocol.js";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  messageKind?: "message" | "answer" | "clarification";
  inputStructure?: InputStructure;
}

const UnderstandingCard: React.FC<{ value: InputStructure }> = ({ value }) => {
  if (value.mode === "pass-through") return null;
  const { structure } = value;
  const rows = [
    ["目标", structure.goal ? [structure.goal] : []],
    ["关系", structure.relations],
    ["约束", structure.constraints],
    ["交付物", structure.deliverables],
    ["依赖", structure.dependencies],
    ["假设", structure.assumptions],
  ] as const;
  return (
    <details
      open={value.action === "clarify"}
      className="mb-2 rounded border border-indigo-500/40 bg-indigo-950/30 p-2"
      data-testid="understanding-card"
    >
      <summary className="cursor-pointer text-xs font-medium text-indigo-200">
        任务理解
        <span className="ml-2 text-gray-400">
          {value.action === "clarify" ? "等待补充" : "已自动继续"}
        </span>
      </summary>
      <div className="mt-2 space-y-1 text-xs text-gray-300">
        {rows.map(([label, items]) =>
          items.length > 0 ? (
            <div key={label}>
              <span className="text-gray-500">{label}：</span>
              {items.join("；")}
            </div>
          ) : null,
        )}
        {value.questions.length > 0 && (
          <div className="mt-2 rounded bg-amber-950/40 p-2 text-amber-100">
            <div className="font-medium">请补充：</div>
            <ol className="list-decimal pl-5">
              {value.questions.map((question) => (
                <li key={question}>{question}</li>
              ))}
            </ol>
          </div>
        )}
        {value.mode === "fallback" && (
          <div className="text-amber-300">结构化未完成，已按原始输入继续执行。</div>
        )}
      </div>
    </details>
  );
};

export const ChatPanel: React.FC<{
  messages: ChatMessage[];
  onSend: (text: string) => void;
  pending: boolean;
  profiles?: ModelProfile[] | undefined;
  modelProfileId?: string | undefined;
  onModelChange?: ((profileId: string) => void) | undefined;
}> = ({ messages, onSend, pending, profiles = [], modelProfileId, onModelChange }) => {
  const [input, setInput] = React.useState("");
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages.length]);

  const submit = (): void => {
    const text = input.trim();
    if (!text || pending) return;
    onSend(text);
    setInput("");
  };

  return (
    <div className="flex flex-col h-full min-w-0 bg-[#1e1e2e]">
      <div className="h-10 px-3 border-b border-gray-800 flex items-center justify-between">
        <span className="text-xs text-gray-500">当前对话</span>
        <select
          aria-label="对话模型"
          value={modelProfileId ?? ""}
          disabled={pending}
          onChange={(event) => onModelChange?.(event.target.value)}
          className="bg-gray-800 text-gray-200 rounded px-2 py-1 text-xs"
        >
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id} disabled={!profile.hasCredential}>
              {profile.name}
              {profile.hasCredential ? "" : "（需密钥）"}
            </option>
          ))}
        </select>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <div className="text-gray-500 text-sm text-center mt-12">👋 输入消息开始与 Hy3 对话</div>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              data-message-kind={m.messageKind ?? "message"}
              className={`max-w-[80%] px-3 py-2 rounded-lg whitespace-pre-wrap text-sm ${
                m.role === "user"
                  ? "ml-auto bg-blue-600 text-white"
                  : m.role === "assistant"
                    ? "bg-[#313244] text-gray-100"
                    : "mx-auto text-xs text-gray-500"
              }`}
            >
              {m.inputStructure && <UnderstandingCard value={m.inputStructure} />}
              {m.messageKind !== "clarification" && m.content}
            </div>
          ))
        )}
        {pending && (
          <div className="bg-[#313244] text-gray-400 px-3 py-2 rounded-lg text-sm max-w-[80%]">
            <span className="inline-block w-2 h-2 bg-gray-500 rounded-full animate-pulse" /> Agent
            思考中...
          </div>
        )}
      </div>
      <div className="border-t border-gray-800 p-3 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={2}
          placeholder="说点什么... (Enter 发送, Shift+Enter 换行)"
          className="flex-1 bg-[#11111b] border border-gray-800 rounded-md px-3 py-2 text-sm text-gray-200 resize-none focus:outline-none focus:border-blue-500"
          disabled={pending}
        />
        <button
          type="button"
          onClick={submit}
          disabled={pending || !input.trim()}
          className="px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium"
        >
          发送
        </button>
      </div>
    </div>
  );
};
