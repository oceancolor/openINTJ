/**
 * RFC-021 F04 (T07/T08 entry-level): scope eligibility runs before ranking
 * truncation. A canary fragment in another workspace/conversation must
 * never enter the prompt, never take a topK slot, never accrue access.
 */
import { TaskType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import {
  ContextEngine,
  MemoryRetriever,
  MemoryStore,
  isFragmentVisibleToScope,
  scopeFromContextTags,
} from "../src/index.js";

describe("RFC-021 F04 scope eligibility", () => {
  it("tag visibility is fail-closed per level; untagged is user-level", () => {
    const store = new MemoryStore();
    const conv = store.addLongTerm("x", {
      taskTags: ["workspace:w1", "task:t1", "conversation:c1"],
    });
    const wsOnly = store.addLongTerm("y", { taskTags: ["workspace:w1"] });
    const plain = store.addLongTerm("z");
    const scope = { workspaceId: "w1", taskId: "t1", conversationId: "c1" };
    expect(isFragmentVisibleToScope(conv, scope)).toBe(true);
    expect(isFragmentVisibleToScope(conv, { ...scope, conversationId: "c2" })).toBe(false);
    expect(isFragmentVisibleToScope(conv, { ...scope, taskId: "t2" })).toBe(false);
    expect(isFragmentVisibleToScope(wsOnly, { workspaceId: "w1" })).toBe(true);
    expect(isFragmentVisibleToScope(wsOnly, { workspaceId: "w2" })).toBe(false);
    expect(isFragmentVisibleToScope(wsOnly, {})).toBe(false);
    expect(isFragmentVisibleToScope(plain, { workspaceId: "w2" })).toBe(true);
    expect(isFragmentVisibleToScope(conv, undefined)).toBe(true);
  });

  it("scopeFromContextTags parses verified host tags", () => {
    expect(
      scopeFromContextTags(["workspace:w1", "task:t1", "conversation:c1", "user_input"]),
    ).toEqual({ workspaceId: "w1", taskId: "t1", conversationId: "c1" });
    expect(scopeFromContextTags(undefined)).toEqual({});
  });

  it("T08: invisible high-importance fragments do not consume topK slots", async () => {
    const store = new MemoryStore();
    for (let i = 0; i < 10; i++) {
      store.addLongTerm(`canary secret token ${i}`, {
        importance: 1,
        taskTags: ["workspace:other", "conversation:other"],
      });
    }
    const visible = store.addLongTerm("canary visible token", {
      importance: 0.9,
      taskTags: ["workspace:mine", "conversation:mine"],
    });
    const retriever = new MemoryRetriever(store);
    const hits = await retriever.retrieveAsync("canary token", {
      topK: 3,
      scope: { workspaceId: "mine", conversationId: "mine" },
    });
    expect(hits.map((h) => h.fragment.fragmentId)).toEqual([visible.fragmentId]);
    // Invisible fragments never accrued access statistics.
    const canary = store.all.find((f) => f.content.includes("secret"));
    expect(canary?.accessCount).toBe(0);
  });

  it("T07: ContextEngine prompt contains only in-scope memories", async () => {
    const store = new MemoryStore();
    store.addLongTerm("canary from another conversation about apples", {
      importance: 1,
      taskTags: ["workspace:w1", "conversation:c-other"],
    });
    store.addLongTerm("note from this conversation about apples", {
      importance: 0.9,
      taskTags: ["workspace:w1", "conversation:c-mine"],
    });
    const engine = new ContextEngine({ store });
    const win = await engine.build({
      query: "apples",
      history: [],
      taskType: TaskType.GENERAL_CHAT,
      systemPrompt: "p",
      scope: { workspaceId: "w1", conversationId: "c-mine" },
    });
    expect(win.systemPrompt).toContain("note from this conversation");
    expect(win.systemPrompt).not.toContain("canary from another conversation");
  });
});
