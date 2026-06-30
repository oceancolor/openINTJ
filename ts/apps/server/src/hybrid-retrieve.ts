import type { HybridConfig, MemoryHybridHit } from "@openintj/taskpool";
import type { ServerAgent } from "./agent.js";

/**
 * RFC-003 方向 2：HybridRetriever 装配。
 *
 * 设计（A1 起改为增量）：
 *  - 复用 agent 上 session 级 `MemoryHybridIndex`（开局 seed + 订阅 change-feed 增量维护），
 *    不再每次查询全量重建 → 检索随对话「越用越好」且省去重建成本。
 *  - 与默认 MemoryRetriever 并存：用户按需调 retrieveHybrid，不替换默认路径。
 *  - 超大规模仍可换 LanceDB 内建 FTS（roadmap #10 余下部分）。
 */
export type HybridMemoryHit = MemoryHybridHit;

export interface RetrieveHybridOpts {
  topK?: number;
  /** 按查询覆盖融合配置（alpha/beta/useRRF/...）。 */
  config?: Partial<HybridConfig>;
  /** 仅检索这些 memoryType。 */
  memoryTypes?: readonly string[];
  /** 任意一个标签命中即保留。 */
  taskTags?: readonly string[];
  /** 显式传入的 query embedding；不传则用 store.embedder 生成。 */
  queryEmbedding?: readonly number[];
}

export const retrieveHybrid = async (
  agent: ServerAgent,
  query: string,
  opts: RetrieveHybridOpts = {},
): Promise<HybridMemoryHit[]> => {
  if (agent.hybridIndex.size === 0) return [];

  let qVec: readonly number[] | undefined = opts.queryEmbedding;
  if (qVec === undefined) {
    const r = agent.memory.store.embedder.embed(query);
    qVec = r instanceof Promise ? await r : r;
  }

  return agent.hybridIndex.search(query, qVec, {
    topK: opts.topK ?? 10,
    ...(opts.config ? { config: opts.config } : {}),
    ...(opts.memoryTypes ? { memoryTypes: opts.memoryTypes } : {}),
    ...(opts.taskTags ? { taskTags: opts.taskTags } : {}),
  });
};
