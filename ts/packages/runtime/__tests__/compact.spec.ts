import { type Journal, createInMemoryJournal, deriveHistory } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { createCompactor } from "../src/index.js";

const cid = "conv-compact";

const seedTurns = (journal: Journal, count: number): void => {
  for (let i = 1; i <= count; i += 1) {
    const turnId = `t${i}`;
    journal.append({ type: "turn.start", conversationId: cid, turnId });
    journal.append({
      type: "user",
      conversationId: cid,
      turnId,
      content: `问题 ${i}：讨论一个足够长的主题 ${"细".repeat(20)}`,
    });
    journal.append({
      type: "assistant",
      conversationId: cid,
      turnId,
      content: `回答 ${i}：这是第 ${i} 轮的结论 ${"节".repeat(20)}`,
      messageKind: "answer",
    });
    journal.append({ type: "turn.end", conversationId: cid, turnId, status: "completed" });
  }
};

const fakeLlm = {
  chat: async () => "此前讨论了若干主题并得出结论。",
} as never;

describe("Compactor (RFC-012)", () => {
  it("command trigger folds old turns into a summary event", async () => {
    const journal = createInMemoryJournal();
    seedTurns(journal, 6); // 12 history messages
    const compactor = createCompactor({ llm: fakeLlm, keepRecentTurns: 2 });
    const out = await compactor.compactIfNeeded({
      journal,
      conversationId: cid,
      trigger: "command",
    });
    expect(out.compacted).toBe(true);
    expect(out.coveredTurns).toBe(4);
    expect(out.history).toHaveLength(5); // summary + last 2 turns
    expect(out.history[0]!.content).toContain("[历史摘要]");
    const events = journal.list(cid);
    expect(events.some((e) => e.type === "session.compacted")).toBe(true);
    // Second immediate compact: history now below the foldable minimum split.
    const again = await compactor.compactIfNeeded({
      journal,
      conversationId: cid,
      trigger: "command",
    });
    expect(again.history[0]!.content).toContain("[历史摘要]");
  });

  it("pressure trigger skips below the threshold", async () => {
    const journal = createInMemoryJournal();
    seedTurns(journal, 6);
    const compactor = createCompactor({ llm: fakeLlm, pressureTokens: 1_000_000 });
    const out = await compactor.compactIfNeeded({
      journal,
      conversationId: cid,
      trigger: "pressure",
    });
    expect(out).toMatchObject({ compacted: false, reason: "below_pressure" });
    expect(journal.list(cid).some((e) => e.type === "session.compacted")).toBe(false);
  });

  it("per-call cap override tightens the summarizer input (overflow recovery)", async () => {
    const journal = createInMemoryJournal();
    seedTurns(journal, 8);
    const seen: number[] = [];
    const recording = {
      chat: async (messages: Array<{ content: string }>) => {
        seen.push(messages[1]!.content.length);
        return "摘要。";
      },
    } as never;
    const compactor = createCompactor({ llm: recording, keepRecentTurns: 2 });
    const out = await compactor.compactIfNeeded({
      journal,
      conversationId: cid,
      trigger: "overflow",
      maxSummaryInputChars: 200,
    });
    expect(out.compacted).toBe(true);
    expect(seen[0]).toBeLessThanOrEqual(210);
  });

  it("short histories are left alone", async () => {
    const journal = createInMemoryJournal();
    seedTurns(journal, 2);
    const compactor = createCompactor({ llm: fakeLlm });
    const out = await compactor.compactIfNeeded({
      journal,
      conversationId: cid,
      trigger: "command",
    });
    expect(out).toMatchObject({ compacted: false, reason: "too_short" });
    expect(deriveHistory(journal.list(cid))).toHaveLength(4);
  });

  it("LLM failure is fail-soft and appends nothing", async () => {
    const journal = createInMemoryJournal();
    seedTurns(journal, 6);
    const failing = {
      chat: async () => {
        throw new Error("provider down");
      },
    } as never;
    const compactor = createCompactor({ llm: failing });
    const out = await compactor.compactIfNeeded({
      journal,
      conversationId: cid,
      trigger: "command",
    });
    expect(out).toMatchObject({ compacted: false, reason: "llm_failed" });
    expect(journal.list(cid).some((e) => e.type === "session.compacted")).toBe(false);
    expect(out.history).toHaveLength(12);
  });
});
