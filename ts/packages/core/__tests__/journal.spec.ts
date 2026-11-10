import { describe, expect, it } from "vitest";
import {
  appendSessionCleared,
  appendSessionCompacted,
  commitHostTurn,
  createInMemoryJournal,
  deriveChatMessages,
  deriveHistory,
  deriveTrajectory,
  journalTurnStart,
  journalUser,
} from "../src/index.js";

const cid = "conv-1";

describe("Session Journal", () => {
  it("assigns per-conversation seq and deriveHistory is user+assistant only", () => {
    const journal = createInMemoryJournal({ now: () => 1_000, idFactory: () => "id" });
    journal.append({ type: "turn.start", conversationId: cid, turnId: "t1", queryPreview: "hi" });
    journal.append({
      type: "user",
      conversationId: cid,
      turnId: "t1",
      content: "execution",
      originalInput: "hi there",
    });
    journal.append({
      type: "react.thought",
      conversationId: cid,
      turnId: "t1",
      step: 1,
      content: "thinking",
    });
    journal.append({
      type: "tool.call",
      conversationId: cid,
      turnId: "t1",
      step: 1,
      tool: "read_file",
      params: { path: "a.ts" },
    });
    journal.append({
      type: "tool.result",
      conversationId: cid,
      turnId: "t1",
      step: 1,
      tool: "read_file",
      ok: true,
      observation: "file body",
    });
    journal.append({
      type: "assistant",
      conversationId: cid,
      turnId: "t1",
      content: "done",
      messageKind: "answer",
    });
    journal.append({ type: "turn.end", conversationId: cid, turnId: "t1", status: "completed" });

    const events = journal.list(cid);
    expect(events.map((event) => event.seq)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(deriveHistory(events)).toEqual([
      { role: "user", content: "execution" },
      { role: "assistant", content: "done", metadata: { messageKind: "answer" } },
    ]);
    expect(deriveHistory(events).some((message) => message.role === "tool")).toBe(false);
    expect(deriveChatMessages(events)).toMatchObject([
      { role: "user", content: "hi there", messageKind: "message" },
      { role: "assistant", content: "done", messageKind: "answer" },
    ]);
    expect(deriveTrajectory(events).map((entry) => entry.type)).toEqual([
      "react.thought",
      "tool.call",
      "tool.result",
      "assistant",
    ]);
  });

  it("session.compacted folds completed turns into one summary message (RFC-012)", () => {
    const journal = createInMemoryJournal();
    const turn = (turnId: string, user: string, assistant: string): void => {
      journal.append({ type: "turn.start", conversationId: cid, turnId });
      journal.append({ type: "user", conversationId: cid, turnId, content: user });
      journal.append({
        type: "assistant",
        conversationId: cid,
        turnId,
        content: assistant,
        messageKind: "answer",
      });
      journal.append({ type: "turn.end", conversationId: cid, turnId, status: "completed" });
    };
    turn("t1", "u1", "a1");
    turn("t2", "u2", "a2");
    appendSessionCompacted(journal, cid, {
      summary: "聊了两件事",
      coveredTurns: 2,
      upToTurnId: "t2",
      trigger: "command",
    });
    turn("t3", "u3", "a3");

    const events = journal.list(cid);
    const history = deriveHistory(events);
    expect(history).toHaveLength(3);
    expect(history[0]).toMatchObject({ role: "user" });
    expect(history[0]!.content).toContain("[历史摘要]");
    expect(history[0]!.content).toContain("聊了两件事");
    expect(history.slice(1).map((m) => m.content)).toEqual(["u3", "a3"]);
    // The UI projection is untouched by compaction.
    expect(deriveChatMessages(events).map((m) => m.content)).toEqual([
      "u1",
      "a1",
      "u2",
      "a2",
      "u3",
      "a3",
    ]);
    // A later /clear wipes the folded summary too.
    appendSessionCleared(journal, cid);
    expect(deriveHistory(journal.list(cid))).toEqual([]);
  });

  it("repeated compaction unions coverage across compacted events (inheritedCoverage)", () => {
    const journal = createInMemoryJournal();
    const turn = (turnId: string, user: string, assistant: string): void => {
      journal.append({ type: "turn.start", conversationId: cid, turnId });
      journal.append({ type: "user", conversationId: cid, turnId, content: user });
      journal.append({
        type: "assistant",
        conversationId: cid,
        turnId,
        content: assistant,
        messageKind: "answer",
      });
      journal.append({ type: "turn.end", conversationId: cid, turnId, status: "completed" });
    };
    turn("t1", "u1", "a1");
    turn("t2", "u2", "a2");
    turn("t3", "u3", "a3");
    appendSessionCompacted(journal, cid, {
      summary: "第一次摘要",
      coveredTurns: 3,
      upToTurnId: "t3",
      trigger: "pressure",
      includedTurnIds: ["t1", "t2", "t3"],
    });
    turn("t4", "u4", "a4");
    // Second compaction covers only t4 (fold window/cap changed): the union
    // must keep t1–t3 hidden instead of resurrecting them, and only the
    // latest summary is shown.
    appendSessionCompacted(journal, cid, {
      summary: "第二次摘要",
      coveredTurns: 1,
      upToTurnId: "t4",
      trigger: "pressure",
      includedTurnIds: ["t4"],
    });
    turn("t5", "u5", "a5");

    const history = deriveHistory(journal.list(cid));
    expect(history).toHaveLength(3);
    expect(history[0]!.content).toContain("此前 4 轮对话已压缩");
    expect(history[0]!.content).toContain("第二次摘要");
    expect(history[0]!.content).not.toContain("第一次摘要");
    expect(history.slice(1).map((m) => m.content)).toEqual(["u5", "a5"]);
  });

  it("compaction union mixes legacy prefix events with exact-coverage events", () => {
    const journal = createInMemoryJournal();
    const turn = (turnId: string, user: string, assistant: string): void => {
      journal.append({ type: "turn.start", conversationId: cid, turnId });
      journal.append({ type: "user", conversationId: cid, turnId, content: user });
      journal.append({
        type: "assistant",
        conversationId: cid,
        turnId,
        content: assistant,
        messageKind: "answer",
      });
      journal.append({ type: "turn.end", conversationId: cid, turnId, status: "completed" });
    };
    turn("t1", "u1", "a1");
    turn("t2", "u2", "a2");
    // Legacy event (pre-F09, no includedTurnIds): prefix coverage up to t2.
    appendSessionCompacted(journal, cid, {
      summary: "旧格式摘要",
      coveredTurns: 2,
      upToTurnId: "t2",
      trigger: "command",
    });
    turn("t3", "u3", "a3");
    turn("t4", "u4", "a4");
    appendSessionCompacted(journal, cid, {
      summary: "新格式摘要",
      coveredTurns: 1,
      upToTurnId: "t4",
      trigger: "pressure",
      includedTurnIds: ["t4"],
    });

    const history = deriveHistory(journal.list(cid));
    // t1/t2 (legacy prefix) + t4 (exact) hidden; t3 was never covered.
    expect(history.map((m) => m.content)).toEqual([
      expect.stringContaining("新格式摘要"),
      "u3",
      "a3",
    ]);
    expect(history[0]!.content).toContain("此前 3 轮对话已压缩");
  });

  it("session.cleared truncates all projections", () => {
    const journal = createInMemoryJournal();
    journal.append({ type: "turn.start", conversationId: cid, turnId: "t1" });
    journal.append({ type: "user", conversationId: cid, turnId: "t1", content: "old" });
    journal.append({
      type: "assistant",
      conversationId: cid,
      turnId: "t1",
      content: "old-a",
      messageKind: "answer",
    });
    journal.append({ type: "turn.end", conversationId: cid, turnId: "t1", status: "completed" });
    appendSessionCleared(journal, cid, { command: "/clear" });
    journal.append({ type: "turn.start", conversationId: cid, turnId: "t2" });
    journal.append({ type: "user", conversationId: cid, turnId: "t2", content: "new" });
    journal.append({
      type: "assistant",
      conversationId: cid,
      turnId: "t2",
      content: "new-a",
      messageKind: "answer",
    });
    journal.append({ type: "turn.end", conversationId: cid, turnId: "t2", status: "completed" });

    const events = journal.list(cid);
    expect(deriveHistory(events).map((message) => message.content)).toEqual(["new", "new-a"]);
    expect(deriveChatMessages(events)).toHaveLength(2);
    expect(deriveTrajectory(events).every((entry) => entry.turnId === "t2")).toBe(true);
    expect(journal.gc(cid)).toBeGreaterThan(0);
    expect(
      journal
        .list(cid)
        .every((event) => event.seq > 4 || event.type !== "user" || event.content === "new"),
    ).toBe(true);
  });

  it("in-progress turn exposes tool observations but not thought", () => {
    const journal = createInMemoryJournal();
    journal.append({ type: "turn.start", conversationId: cid, turnId: "t1" });
    journal.append({ type: "user", conversationId: cid, turnId: "t1", content: "q" });
    journal.append({
      type: "react.thought",
      conversationId: cid,
      turnId: "t1",
      step: 1,
      content: "secret thought",
    });
    journal.append({
      type: "tool.result",
      conversationId: cid,
      turnId: "t1",
      step: 1,
      tool: "search",
      ok: true,
      observation: "hit",
    });
    const history = deriveHistory(journal.list(cid));
    expect(history).toEqual([
      { role: "user", content: "q" },
      { role: "tool", name: "search", content: "hit" },
    ]);
    expect(history.some((message) => String(message.content).includes("secret"))).toBe(false);
  });

  it("copies user inputStructure onto the clarification assistant for ChatPanel", () => {
    const journal = createInMemoryJournal();
    const structure = { action: "clarify", questions: ["哪个环境？"] };
    journal.append({ type: "turn.start", conversationId: cid, turnId: "t1" });
    journal.append({
      type: "user",
      conversationId: cid,
      turnId: "t1",
      content: "部署到生产",
      inputStructure: structure,
    });
    journal.append({
      type: "assistant",
      conversationId: cid,
      turnId: "t1",
      content: "部署到哪个环境？",
      messageKind: "clarification",
    });
    journal.append({ type: "turn.end", conversationId: cid, turnId: "t1", status: "clarify" });
    expect(deriveChatMessages(journal.list(cid))[1]).toMatchObject({
      role: "assistant",
      messageKind: "clarification",
      inputStructure: structure,
    });
  });
});

describe("RFC-021 F01 commitHostTurn (T01/T02)", () => {
  const ctx = (journal: ReturnType<typeof createInMemoryJournal>) => ({
    conversationId: cid,
    turnId: "t-host",
    sink: journal,
  });

  it("T01: partial process events do not block the host final; committed text is the shown answer", () => {
    const journal = createInMemoryJournal();
    const c = ctx(journal);
    journalTurnStart(c, "q");
    journalUser(c, "q");
    journal.append({
      type: "react.thought",
      conversationId: cid,
      turnId: "t-host",
      step: 1,
      content: "thinking",
    });
    const receipt = commitHostTurn(journal, c, {
      userContent: "q",
      assistantContent: "post-processed answer",
      endStatus: "completed",
    });
    expect(receipt.outcome).toBe("committed");
    const history = deriveHistory(journal.list(cid));
    expect(history).toEqual([
      { role: "user", content: "q" },
      { role: "assistant", content: "post-processed answer", metadata: { messageKind: "answer" } },
    ]);
    const ends = journal.list(cid).filter((e) => e.type === "turn.end");
    expect(ends).toHaveLength(1);
  });

  it("T01: a clean turn with no prior events commits the full closed turn", () => {
    const journal = createInMemoryJournal();
    const receipt = commitHostTurn(journal, ctx(journal), {
      userContent: "hello",
      assistantContent: "hi",
      endStatus: "clarify",
      messageKind: "clarification",
    });
    expect(receipt.outcome).toBe("committed");
    expect(journal.list(cid).map((e) => e.type)).toEqual([
      "turn.start",
      "user",
      "assistant",
      "turn.end",
    ]);
  });

  it("T02: recommitting the same final replays; a different final conflicts and writes nothing", () => {
    const journal = createInMemoryJournal();
    const c = ctx(journal);
    const input = {
      userContent: "q",
      assistantContent: "answer",
      endStatus: "completed" as const,
    };
    expect(commitHostTurn(journal, c, input).outcome).toBe("committed");
    const countAfterCommit = journal.list(cid).length;
    expect(commitHostTurn(journal, c, input).outcome).toBe("replayed");
    expect(
      commitHostTurn(journal, c, { ...input, assistantContent: "rewritten" }).outcome,
    ).toBe("conflict");
    expect(
      commitHostTurn(journal, c, { ...input, endStatus: "failed" }).outcome,
    ).toBe("conflict");
    expect(journal.list(cid).length).toBe(countAfterCommit);
  });

  it("T02: a throwing sink surfaces the failure instead of swallowing it", () => {
    const journal = createInMemoryJournal();
    const broken = {
      ...journal,
      append: () => {
        throw new Error("disk full");
      },
    };
    expect(() =>
      commitHostTurn(broken as typeof journal, ctx(journal), {
        userContent: "q",
        assistantContent: "a",
        endStatus: "completed",
      }),
    ).toThrow("disk full");
  });
});
