import { describe, expect, it } from "vitest";
import {
  HookBus,
  createAgentHandle,
  createBuiltinSlashCommands,
  createInMemoryJournal,
  createSlashCommandRegistry,
  deriveHistory,
  invokeSlashCommand,
} from "../src/index.js";

describe("SlashCommand registry", () => {
  it("matches /clear and aliases without calling runTurn", async () => {
    const journal = createInMemoryJournal();
    const hooks = new HookBus();
    let ran = 0;
    const handle = createAgentHandle({
      id: "cli",
      journal,
      hooks,
      runTurn: async () => {
        ran += 1;
        throw new Error("should not run");
      },
    });
    const registry = createSlashCommandRegistry(createBuiltinSlashCommands());
    const invoked = await invokeSlashCommand(registry, "/clear", {
      journal,
      handle,
      conversationId: "cli",
    });
    expect(invoked?.command.name).toBe("/clear");
    expect(invoked?.result).toEqual({ kind: "ok", message: "对话已清空。" });
    expect(ran).toBe(0);
    expect(journal.list("cli").at(-1)).toMatchObject({
      type: "session.cleared",
      command: "/clear",
    });
    expect(deriveHistory(journal.list("cli"))).toEqual([]);
  });

  it("matches /clear history as an alias", async () => {
    const journal = createInMemoryJournal();
    const handle = createAgentHandle({
      id: "c",
      journal,
      hooks: new HookBus(),
      runTurn: async () => {
        throw new Error("no");
      },
    });
    const registry = createSlashCommandRegistry(createBuiltinSlashCommands());
    expect(registry.match("/clear history")?.name).toBe("/clear");
    await invokeSlashCommand(registry, "/CLEAR HISTORY", {
      journal,
      handle,
      conversationId: "c",
    });
    expect(journal.list("c").at(-1)?.type).toBe("session.cleared");
  });

  it("returns a stub error for /compact and does not match unknown /foo", async () => {
    const journal = createInMemoryJournal();
    const handle = createAgentHandle({
      id: "c",
      journal,
      hooks: new HookBus(),
      runTurn: async () => {
        throw new Error("no");
      },
    });
    const registry = createSlashCommandRegistry(createBuiltinSlashCommands());
    expect(registry.match("/foo")).toBeUndefined();
    const compact = await invokeSlashCommand(registry, "/compact", {
      journal,
      handle,
      conversationId: "c",
    });
    expect(compact?.command.name).toBe("/compact");
    expect(compact?.result.kind).toBe("error");
    expect(await invokeSlashCommand(registry, "/foo", { journal, handle })).toBeUndefined();
  });
});
