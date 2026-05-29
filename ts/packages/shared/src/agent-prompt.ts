/**
 * Agent 层共享的系统提示与搜索来源辅助。
 *
 * - `DEFAULT_AGENT_SYSTEM_PROMPT`：引导模型对时效/事实类问题先调用 `search` 工具再作答。
 * - `collectSearchSources` / `appendSourcesFooter`：从 TAO 轨迹里抽取 search 工具命中的
 *   联网来源，追加到最终答案末尾（从而自动随 recordAssistantOutput / dormant.record 入库）。
 *
 * 这里对轨迹与工具结果用**结构化鸭子类型**，不直接依赖 @openintj/core，避免 shared 反向耦合。
 */

export const DEFAULT_AGENT_SYSTEM_PROMPT = [
  "你是 OpenINTJ 智能助手，运行在本地优先的 TAO/ReAct Agent 框架上。",
  "当用户的问题涉及实时信息、最新事件、事实核查、具体数据，或你不确定答案时，",
  "优先调用 `search` 工具联网查证，再基于搜索结果作答，不要凭空编造。",
  "回答力求简洁准确；当你引用了搜索结果时，无需自行罗列链接，系统会自动在末尾附上「参考来源」。",
].join("");

export interface SearchSourceLike {
  title?: string;
  url?: string;
}

interface TrajectoryEntryLike {
  state?: {
    type?: string;
    toolResult?: {
      toolName?: string;
      success?: boolean;
      output?: unknown;
    };
  };
}

/** 从 TAO 轨迹里抽取所有 search 工具命中的联网来源（按 url 去重，保序）。 */
export const collectSearchSources = (
  trajectory: ReadonlyArray<TrajectoryEntryLike> = [],
): SearchSourceLike[] => {
  const seen = new Set<string>();
  const out: SearchSourceLike[] = [];
  for (const entry of trajectory) {
    const st = entry?.state;
    if (!st || st.type !== "observation") continue;
    const r = st.toolResult;
    if (!r || r.toolName !== "search" || r.success !== true) continue;
    const output = r.output as { sources?: unknown } | undefined;
    const sources = Array.isArray(output?.sources) ? (output.sources as SearchSourceLike[]) : [];
    for (const s of sources) {
      const url = typeof s?.url === "string" ? s.url : undefined;
      const title = typeof s?.title === "string" ? s.title : undefined;
      const key = url ?? title ?? "";
      if (!key || seen.has(key)) continue;
      seen.add(key);
      out.push({ ...(title ? { title } : {}), ...(url ? { url } : {}) });
    }
  }
  return out;
};

/** 把来源列表格式化成「参考来源」脚注（最多 max 条）。空列表返回空串。 */
export const formatSourcesFooter = (
  sources: ReadonlyArray<SearchSourceLike>,
  max = 5,
): string => {
  if (sources.length === 0) return "";
  const lines = sources.slice(0, max).map((s, i) => {
    const label = (s.title?.trim() || s.url || `来源 ${i + 1}`).trim();
    return s.url ? `${i + 1}. ${label} — ${s.url}` : `${i + 1}. ${label}`;
  });
  return `\n\n参考来源：\n${lines.join("\n")}`;
};

/**
 * 若答案基于 search 工具命中了来源，则在末尾追加「参考来源」脚注。
 * 已包含「参考来源」字样时不重复追加。
 */
export const appendSourcesFooter = (
  answer: string,
  trajectory: ReadonlyArray<TrajectoryEntryLike> = [],
): string => {
  if (answer.includes("参考来源")) return answer;
  const footer = formatSourcesFooter(collectSearchSources(trajectory));
  return footer ? `${answer}${footer}` : answer;
};
