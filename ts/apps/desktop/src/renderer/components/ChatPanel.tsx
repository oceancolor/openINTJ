import React from "react";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export const ChatPanel: React.FC<{
  messages: ChatMessage[];
  onSend: (text: string) => void;
  pending: boolean;
}> = ({ messages, onSend, pending }) => {
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
    <div className="flex flex-col h-full bg-[#1e1e2e]">
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <div className="text-gray-500 text-sm text-center mt-12">
            👋 输入消息开始对话（默认 mock 模式，无网络也可用）
          </div>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={`max-w-[80%] px-3 py-2 rounded-lg whitespace-pre-wrap text-sm ${
                m.role === "user"
                  ? "ml-auto bg-blue-600 text-white"
                  : m.role === "assistant"
                    ? "bg-[#313244] text-gray-100"
                    : "mx-auto text-xs text-gray-500"
              }`}
            >
              {m.content}
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
