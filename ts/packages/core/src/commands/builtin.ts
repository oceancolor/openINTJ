import { appendSessionCleared } from "../journal/memory.js";
import type { Journal } from "../journal/types.js";
import type { SlashCommand } from "./types.js";

export interface BuiltinSlashCommandOpts {
  /** Desktop override: also delete the messages cache. Default appends session.cleared. */
  clearConversation?: (conversationId: string, journal: Journal) => void;
}

export const createBuiltinSlashCommands = (opts: BuiltinSlashCommandOpts = {}): SlashCommand[] => [
  {
    name: "/clear",
    aliases: ["/clear history"],
    description: "清空当前会话（journal 追加 session.cleared，不调用模型）",
    async run(ctx) {
      const conversationId = ctx.conversationId ?? ctx.handle.id;
      if (opts.clearConversation) {
        opts.clearConversation(conversationId, ctx.journal);
      } else {
        appendSessionCleared(ctx.journal, conversationId, { by: "command", command: "/clear" });
      }
      ctx.handle.cancel("command");
      return { kind: "ok", message: "对话已清空。" };
    },
  },
  {
    name: "/compact",
    description: "压缩会话上下文（把较早轮次总结为摘要，原始 journal 事件不变）",
    async run(ctx) {
      if (!ctx.compact) {
        return { kind: "error", message: "/compact 不可用：宿主未提供压缩实现。" };
      }
      const out = await ctx.compact();
      return { kind: "ok", message: out.message };
    },
  },
];
