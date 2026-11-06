import type { SlashCommand, SlashCommandRegistry } from "./types.js";

const defaultMatch = (command: SlashCommand, text: string): boolean => {
  const normalized = text.trim().toLowerCase();
  if (!normalized) return false;
  if (normalized === command.name.toLowerCase()) return true;
  return (command.aliases ?? []).some((alias) => alias.toLowerCase() === normalized);
};

export const createSlashCommandRegistry = (
  commands: readonly SlashCommand[] = [],
): SlashCommandRegistry => {
  const list: SlashCommand[] = [...commands];
  return {
    register(command) {
      const idx = list.findIndex((item) => item.name === command.name);
      if (idx >= 0) list[idx] = command;
      else list.push(command);
    },
    list() {
      return list;
    },
    match(text) {
      return list.find((command) => (command.match ?? defaultMatch.bind(null, command))(text));
    },
  };
};
