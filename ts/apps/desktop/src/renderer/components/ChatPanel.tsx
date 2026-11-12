import React from "react";
import type { ContextUsage, InputStructure, ModelProfile } from "../../shared/ipc-protocol.js";
import { splitWorkspacePathMarkup } from "../../shared/workspace-view.js";
import appIcon from "../assets/app-icon.png";
import { useVoiceInput } from "./useVoiceInput.js";

/** openINTJ 头像：与程序图标/顶栏 logo 同一张（assets/app-icon.png）。 */
const AssistantAvatar: React.FC<{ size: "sm" | "lg" }> = ({ size }) =>
  size === "lg" ? (
    <img
      src={appIcon}
      alt="openINTJ"
      draggable={false}
      className="h-12 w-12 shrink-0 rounded-[14px] shadow-raise ring-1 ring-[#c9bdf0]"
    />
  ) : (
    <img
      src={appIcon}
      alt="openINTJ"
      draggable={false}
      className="h-7 w-7 shrink-0 rounded-[9px] ring-1 ring-[#c9bdf0]"
    />
  );

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  messageKind?: "message" | "answer" | "clarification";
  inputStructure?: InputStructure;
  /** RFC-010 ASSISTANT_DELTA placeholder; replaced by the turn's final answer. */
  streaming?: boolean;
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
      className="mb-3 rounded-xl border border-line bg-inset p-3"
      data-testid="understanding-card"
    >
      <summary className="cursor-pointer text-xs font-semibold text-ink">
        任务理解
        <span className="ml-2 text-[10px] font-normal text-muted">
          {value.action === "clarify" ? "等待补充" : "已自动继续"}
        </span>
      </summary>
      <div className="mt-2 space-y-1 text-xs leading-5 text-muted">
        {rows.map(([label, items]) =>
          items.length > 0 ? (
            <div key={label}>
              <span className="font-medium text-ink">{label}：</span>
              {items.join("；")}
            </div>
          ) : null,
        )}
        {value.questions.length > 0 && (
          <div className="mt-2 rounded-lg border border-amberborder bg-amberwash p-2.5 text-ink">
            <div className="font-semibold text-amberdeep">请补充：</div>
            <ol className="list-decimal pl-5">
              {value.questions.map((question) => (
                <li key={question}>{question}</li>
              ))}
            </ol>
          </div>
        )}
        {value.mode === "fallback" && (
          <div className="text-signal">结构化未完成，已按原始输入继续执行。</div>
        )}
      </div>
    </details>
  );
};

const estimateTokens = (text: string): number => Math.max(0, Math.floor(text.length / 4));

const ContextUsageRing: React.FC<{ used: number; max: number }> = ({ used, max }) => {
  const ratio = Math.min(1, used / Math.max(1, max));
  const radius = 11;
  const circumference = 2 * Math.PI * radius;
  const dash = `${(ratio * circumference).toFixed(2)} ${circumference.toFixed(2)}`;
  const tone = ratio >= 0.9 ? "#ca5b3f" : ratio >= 0.7 ? "#d39b42" : "#425d48";
  const usedLabel = used >= 1000 ? `${(used / 1000).toFixed(1)}k` : String(used);
  const maxLabel = max >= 1000 ? `${Math.round(max / 1000)}k` : String(max);
  return (
    <div
      className="flex items-center gap-1.5"
      data-testid="context-usage-ring"
      aria-label={`上下文用量 ${used} / ${max}`}
      title={`上下文 ${usedLabel} / ${maxLabel} tokens`}
    >
      <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
        <circle cx="14" cy="14" r={radius} fill="none" stroke="#e8e9e3" strokeWidth="3" />
        <circle
          cx="14"
          cy="14"
          r={radius}
          fill="none"
          stroke={tone}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={dash}
          transform="rotate(-90 14 14)"
        />
      </svg>
      <span className="text-[10px] leading-tight text-muted tabular-nums">
        {Math.round(ratio * 100)}%
      </span>
    </div>
  );
};

const MessageBody: React.FC<{
  content: string;
  onOpenPath?: ((path: string) => void) | undefined;
}> = ({ content, onOpenPath }) => (
  <>
    {splitWorkspacePathMarkup(content).map((part, index) =>
      part.type === "path" && onOpenPath ? (
        <button
          key={`${part.value}-${index}`}
          type="button"
          className="underline text-moss hover:text-mossdeep break-all"
          onClick={() => onOpenPath(part.value)}
        >
          {part.value}
        </button>
      ) : (
        <React.Fragment key={index}>{part.value}</React.Fragment>
      ),
    )}
  </>
);

const HERO_SUGGESTIONS: readonly { title: string; desc: string; preset: string }[] = [
  {
    title: "审查这个工作区",
    desc: "找出风险与可改进点",
    preset: "请审查当前工作区代码，找出风险并给出修复建议",
  },
  {
    title: "规划一个实现",
    desc: "把目标拆成可执行步骤",
    preset: "请帮我规划一个实现方案，先拆解步骤再执行",
  },
  {
    title: "运行维护任务",
    desc: "检查、执行并核验结果",
    preset: "请检查当前工作区状态并运行必要的维护与核验",
  },
];

export const ChatPanel: React.FC<{
  messages: ChatMessage[];
  onSend: (text: string) => void;
  pending: boolean;
  profiles?: ModelProfile[] | undefined;
  modelProfileId?: string | undefined;
  onModelChange?: ((profileId: string) => void) | undefined;
  context?: ContextUsage | undefined;
  onOpenWorkspacePath?: ((path: string) => void) | undefined;
  /** G5.3：对最近一轮回答的用户反馈（技能验证器的反馈通道）。 */
  onSkillFeedback?: ((sentiment: "positive" | "negative") => void) | undefined;
}> = ({
  messages,
  onSend,
  pending,
  profiles = [],
  modelProfileId,
  onModelChange,
  context,
  onOpenWorkspacePath,
  onSkillFeedback,
}) => {
  const [input, setInput] = React.useState("");
  const [feedbackGivenAt, setFeedbackGivenAt] = React.useState<number | undefined>();
  const scrollRef = React.useRef<HTMLDivElement>(null);
  // RFC-019: transcripts are appended to the input for review, never auto-sent.
  const voice = useVoiceInput({
    onText: (text) =>
      setInput((prev) => (prev ? `${prev}${/\s$/.test(prev) ? "" : " "}${text}` : text)),
  });

  React.useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages.length]);

  const submit = (): void => {
    const text = input.trim();
    if (!text || pending) return;
    onSend(text);
    setInput("");
  };

  const localUsed = messages.reduce((sum, message) => sum + estimateTokens(message.content), 0);
  const used = Math.max(localUsed + estimateTokens(input), context?.used ?? 0);
  const max = context?.max ?? 128_000;
  const lastAssistantIdx = messages.reduce(
    (acc, m, idx) => (m.role === "assistant" && !m.streaming ? idx : acc),
    -1,
  );

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col bg-panel" data-testid="chat-panel">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 py-5">
        {messages.length === 0 ? (
          <div className="mx-auto flex max-w-[430px] flex-col items-center pt-10">
            <AssistantAvatar size="lg" />
            <h2 className="mt-4 text-center text-[22px] font-bold leading-8 text-ink">
              接下来推进什么？
            </h2>
            <p className="mt-1 text-center text-[13px] leading-6 text-muted">
              描述你想达成的结果，openINTJ 会先提出计划，再执行并核验。
            </p>
            <span className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-mossmist px-2.5 py-1 font-mono text-[9px] font-semibold text-moss">
              <span className="h-1.5 w-1.5 rounded-full bg-moss" />
              本地运行 · 工作区上下文
            </span>
            <div className="mt-6 flex w-full flex-col gap-2">
              {HERO_SUGGESTIONS.map((s) => (
                <button
                  key={s.title}
                  type="button"
                  className="rounded-xl border border-line bg-white p-3 text-left shadow-card transition hover:border-linedark hover:shadow-raise"
                  onClick={() => setInput(s.preset)}
                >
                  <span className="block text-[13px] font-semibold text-ink">{s.title}</span>
                  <span className="mt-0.5 block text-xs text-muted">{s.desc}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => {
            if (m.role === "system") {
              return (
                <div key={i} className="flex justify-center">
                  <div
                    data-message-kind={m.messageKind ?? "message"}
                    className="rounded-full bg-inset px-3 py-1 text-[11px] text-muted"
                  >
                    {m.messageKind !== "clarification" && (
                      <MessageBody content={m.content} onOpenPath={onOpenWorkspacePath} />
                    )}
                  </div>
                </div>
              );
            }
            const isUser = m.role === "user";
            const isStreaming = m.streaming === true;
            const isError =
              m.content.startsWith("错误：") ||
              m.content.startsWith("llmCall failed:") ||
              m.content.startsWith("Run halted:");
            return (
              <div key={i} className="flex gap-3">
                {isUser ? (
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-line bg-white text-[10px] font-semibold text-muted">
                    你
                  </div>
                ) : (
                  <AssistantAvatar size="sm" />
                )}
                <div className="min-w-0 flex-1" data-message-kind={m.messageKind ?? "message"}>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-semibold text-muted">
                      {isUser ? "你" : "openINTJ"}
                    </span>
                    {!isUser && isStreaming ? (
                      <span className="inline-flex items-center gap-1 font-mono text-[9px] font-semibold text-moss">
                        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-moss" />
                        工作中
                      </span>
                    ) : null}
                    {!isUser && isError && m.content ? (
                      <span className="inline-flex rounded-full bg-signaltint px-2 py-0.5 text-[10px] font-semibold text-signaldeep">
                        出现问题
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-1 whitespace-pre-wrap break-words text-sm leading-6 text-ink">
                    {m.inputStructure && <UnderstandingCard value={m.inputStructure} />}
                    {m.messageKind !== "clarification" && (
                      <MessageBody content={m.content} onOpenPath={onOpenWorkspacePath} />
                    )}
                  </div>
                  {!isUser && i === lastAssistantIdx && onSkillFeedback ? (
                    feedbackGivenAt === i ? (
                      <div className="mt-1 text-[10px] text-muted">已记录反馈，谢谢。</div>
                    ) : (
                      <div className="mt-1 flex items-center gap-2 text-[10px] text-muted">
                        <span>这条回答：</span>
                        <button
                          type="button"
                          className="rounded border border-linedark px-2 py-0.5 hover:text-ink"
                          onClick={() => {
                            onSkillFeedback("positive");
                            setFeedbackGivenAt(i);
                          }}
                        >
                          有帮助
                        </button>
                        <button
                          type="button"
                          className="rounded border border-linedark px-2 py-0.5 hover:text-ink"
                          onClick={() => {
                            onSkillFeedback("negative");
                            setFeedbackGivenAt(i);
                          }}
                        >
                          没帮助
                        </button>
                      </div>
                    )
                  ) : null}
                </div>
              </div>
            );
          })
        )}
        {pending && !(messages.at(-1)?.streaming && messages.at(-1)?.content) && (
          <div className="flex gap-3">
            <AssistantAvatar size="sm" />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-semibold text-muted">openINTJ</span>
                <span className="inline-flex items-center gap-1 font-mono text-[9px] font-semibold text-moss">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-moss" />
                  工作中
                </span>
              </div>
              <div className="mt-1 text-sm text-muted">思考中…</div>
            </div>
          </div>
        )}
      </div>
      <div className="shrink-0 p-3 pt-1">
        {voice.error ? <div className="pb-2 text-xs text-signal">{voice.error}</div> : null}
        <div className="rounded-2xl border border-linedark bg-white p-2 shadow-composer">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={3}
            placeholder="描述你想达成的结果，或粘贴一个本地路径…"
            className="w-full resize-none bg-transparent px-2 pt-1.5 text-sm leading-6 text-ink placeholder:text-faint focus:outline-none"
            disabled={pending}
          />
          <div className="mt-1 flex items-center gap-2 border-t border-[#eef0e9] px-1 pt-2">
            <button
              type="button"
              onClick={voice.toggle}
              disabled={!voice.supported || pending || voice.state === "transcribing"}
              aria-label={voice.state === "recording" ? "停止录音" : "语音输入"}
              title={
                voice.supported
                  ? voice.state === "recording"
                    ? "停止录音并转写"
                    : "语音输入（转写后填入输入框）"
                  : "当前环境不支持录音"
              }
              className={`rounded-lg px-2.5 py-1.5 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-40 ${
                voice.state === "recording"
                  ? "animate-pulse bg-signal text-white hover:bg-signaldeep"
                  : "text-muted hover:bg-inset hover:text-ink"
              }`}
            >
              {voice.state === "recording"
                ? `● ${voice.elapsedS}s`
                : voice.state === "transcribing"
                  ? "转写中…"
                  : "🎤 语音"}
            </button>
            <span className="ml-auto flex items-center gap-2">
              <ContextUsageRing used={used} max={max} />
              <select
                aria-label="对话模型"
                value={modelProfileId ?? ""}
                disabled={pending}
                onChange={(event) => onModelChange?.(event.target.value)}
                className="rounded-lg border border-line bg-white px-2 py-1 text-xs text-ink focus:border-moss focus:outline-none"
              >
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id} disabled={!profile.hasCredential}>
                    {profile.name}
                    {profile.hasCredential ? "" : "（需密钥）"}
                  </option>
                ))}
              </select>
              <span className="hidden font-mono text-[9px] text-faint xl:inline">
                ENTER 发送 · SHIFT+ENTER 换行
              </span>
            </span>
            <button
              type="button"
              onClick={submit}
              disabled={pending || !input.trim()}
              className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white hover:bg-inksoft disabled:cursor-not-allowed disabled:opacity-40"
            >
              发送
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
