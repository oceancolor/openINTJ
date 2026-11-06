import {
  type ChatMessage,
  type Journal,
  type JournalEvent,
  type LlmClient,
  activeJournalEvents,
  appendSessionCompacted,
  deriveHistory,
} from "@openintj/core";

export type CompactTrigger = "pressure" | "overflow" | "command";

export interface CompactorOpts {
  llm: LlmClient;
  /** Rough chars-per-token heuristic for pressure estimation (default 2). */
  charsPerToken?: number;
  /** Estimated-token threshold for the pressure trigger (default 24_000). */
  pressureTokens?: number;
  /** Most recent completed turns always kept verbatim (default 3). */
  keepRecentTurns?: number;
  /** Cap on the serialized history fed to the summarizer (default 60_000 chars). */
  maxSummaryInputChars?: number;
}

export interface CompactOutcome {
  compacted: boolean;
  reason?: "too_short" | "below_pressure" | "llm_failed" | "empty_summary";
  coveredTurns?: number;
  /** Post-compaction derived history (unchanged input history when skipped). */
  history: ChatMessage[];
}

export interface Compactor {
  compactIfNeeded(input: {
    journal: Journal;
    conversationId: string;
    trigger: CompactTrigger;
    llm?: LlmClient;
    /** Per-call summarizer input cap (overflow recovery tightens this so a small-window model can still summarize). */
    maxSummaryInputChars?: number;
  }): Promise<CompactOutcome>;
  estimateTokens(messages: readonly ChatMessage[]): number;
}

const SUMMARY_SYSTEM_PROMPT = [
  "你负责把一段对话历史压缩成摘要，供后续对话替代原始历史使用。",
  "保留：用户的目标与约束、已经做出的决定、关键事实与数字、人名/文件路径、未完成的事项。",
  "丢弃：寒暄、重复内容、工具调用噪声、已失效的中间尝试。",
  "用中文要点列表输出，只输出摘要正文，不要寒暄或解释。",
].join("\n");

const ROLE_LABELS: Record<string, string> = {
  user: "用户",
  assistant: "助手",
  tool: "工具",
  system: "系统",
};

const textOf = (message: ChatMessage): string =>
  typeof message.content === "string"
    ? message.content
    : message.content.map((part) => (part.type === "text" ? part.text : "")).join("");

const serializeForSummary = (messages: readonly ChatMessage[], cap: number): string => {
  const lines: string[] = [];
  let used = 0;
  for (const message of messages) {
    const label = ROLE_LABELS[message.role] ?? message.role;
    const name = message.role === "tool" && message.name ? `(${message.name})` : "";
    const raw = textOf(message);
    const content = raw.length > 3_000 ? `${raw.slice(0, 3_000)}…` : raw;
    const line = `${label}${name}：${content}`;
    if (used + line.length > cap) break;
    lines.push(line);
    used += line.length + 1;
  }
  return lines.join("\n");
};

interface CompletedTurn {
  turnId: string;
  messages: ChatMessage[];
}

/** Completed turns (user + assistant) in order, mirroring deriveHistory. */
const completedTurns = (events: readonly JournalEvent[]): CompletedTurn[] => {
  const byTurn = new Map<string, JournalEvent[]>();
  const order: string[] = [];
  for (const event of events) {
    if (!event.turnId) continue;
    let bucket = byTurn.get(event.turnId);
    if (!bucket) {
      bucket = [];
      byTurn.set(event.turnId, bucket);
      order.push(event.turnId);
    }
    bucket.push(event);
  }
  const turns: CompletedTurn[] = [];
  for (const turnId of order) {
    const turn = byTurn.get(turnId) ?? [];
    if (!turn.some((event) => event.type === "turn.end")) continue;
    const messages: ChatMessage[] = [];
    const user = turn.find((event) => event.type === "user");
    const assistant = [...turn].reverse().find((event) => event.type === "assistant");
    if (user?.type === "user") messages.push({ role: "user", content: user.content });
    if (assistant?.type === "assistant") {
      messages.push({ role: "assistant", content: assistant.content });
    }
    if (messages.length > 0) turns.push({ turnId, messages });
  }
  return turns;
};

/** Shared /compact command surface: run a command-triggered compaction and map the outcome to a user message. */
export const runCompactCommand = async (
  compactor: Compactor,
  journal: Journal,
  conversationId: string,
  llm?: LlmClient,
): Promise<{ compacted: boolean; message: string }> => {
  const out = await compactor.compactIfNeeded({
    journal,
    conversationId,
    trigger: "command",
    ...(llm ? { llm } : {}),
  });
  if (out.compacted) {
    return {
      compacted: true,
      message: `已把此前 ${out.coveredTurns ?? 0} 轮对话压缩为摘要，后续将以摘要替代原始历史（原始记录仍保留在日志中）。`,
    };
  }
  return {
    compacted: false,
    message:
      out.reason === "below_pressure"
        ? "当前上下文尚未达到压缩压力，无需压缩。"
        : out.reason === "llm_failed"
          ? "压缩失败：摘要生成出错，历史保持不变。"
          : "历史还不够长，没有可压缩的内容。",
  };
};

/**
 * RFC-012 compaction driver. Compaction appends a `session.compacted`
 * projection event to the journal; committed events are never rewritten.
 * Any LLM failure is fail-soft: the run proceeds with the full history.
 */
export const createCompactor = (opts: CompactorOpts): Compactor => {
  const charsPerToken = opts.charsPerToken ?? 2;
  const pressureTokens = opts.pressureTokens ?? 24_000;
  const keepRecentTurns = opts.keepRecentTurns ?? 3;
  const maxSummaryInputChars = opts.maxSummaryInputChars ?? 60_000;

  const estimateTokens = (messages: readonly ChatMessage[]): number =>
    Math.ceil(messages.reduce((sum, message) => sum + textOf(message).length, 0) / charsPerToken);

  return {
    estimateTokens,
    async compactIfNeeded({
      journal,
      conversationId,
      trigger,
      llm,
      maxSummaryInputChars: capOverride,
    }) {
      const events = activeJournalEvents(journal.list(conversationId));
      const history = deriveHistory(events);
      const turns = completedTurns(events);
      // Need at least two foldable turns beyond the verbatim tail.
      if (turns.length <= keepRecentTurns + 1) {
        return { compacted: false, reason: "too_short", history };
      }
      if (trigger === "pressure" && estimateTokens(history) < pressureTokens) {
        return { compacted: false, reason: "below_pressure", history };
      }
      const fold = turns.slice(0, turns.length - keepRecentTurns);
      const toSummarize = fold.flatMap((turn) => turn.messages);
      let summary: string;
      try {
        summary = (
          await (llm ?? opts.llm).chat([
            { role: "system", content: SUMMARY_SYSTEM_PROMPT },
            {
              role: "user",
              content: serializeForSummary(toSummarize, capOverride ?? maxSummaryInputChars),
            },
          ])
        ).trim();
      } catch {
        return { compacted: false, reason: "llm_failed", history };
      }
      if (!summary) return { compacted: false, reason: "empty_summary", history };
      appendSessionCompacted(journal, conversationId, {
        summary,
        coveredTurns: fold.length,
        upToTurnId: fold.at(-1)!.turnId,
        trigger,
      });
      return {
        compacted: true,
        coveredTurns: fold.length,
        history: deriveHistory(journal.list(conversationId)),
      };
    },
  };
};
