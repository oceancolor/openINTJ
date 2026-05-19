import {
  type HybridConfig,
  type HybridDoc,
  HybridRetriever,
  type HybridScored,
} from "@openintj/taskpool";
import type { ServerAgent } from "./agent.js";

/**
 * RFC-003 方向 2：HybridRetriever 装配。
 *
 * 设计取舍：
 *  - 不维护持久化的混合索引；每次查询时按当前 MemoryStore 快照重建 BM25 + cosine 双路索引
 *  - 适合"中等量级（≤ 几千 fragments）"的本地 agent；超大规模可换 LanceDB 内建 FTS
 *  - 与现有 MemoryRetriever 并存：用户按需调 retrieveHybrid，不替换默认路径
 */
export type HybridMemoryHit = HybridScored<
  HybridDoc & {
    metadata: {
      memoryType: string;
      taskTags: readonly string[];
      importance: number;
    };
  }
>;

export interface RetrieveHybridOpts {
  topK?: number;
  /** 透传给 HybridRetriever 的配置（alpha/beta/useRRF/...）。 */
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
  let fragments = agent.memory.store.all;
  if (opts.memoryTypes && opts.memoryTypes.length > 0) {
    const set = new Set(opts.memoryTypes);
    fragments = fragments.filter((f) => set.has(f.memoryType));
  }
  if (opts.taskTags && opts.taskTags.length > 0) {
    const tagSet = new Set(opts.taskTags);
    fragments = fragments.filter((f) => f.taskTags.some((t) => tagSet.has(t)));
  }
  if (fragments.length === 0) return [];

  const docs: HybridMemoryHit["doc"][] = fragments.map((f) => ({
    id: f.fragmentId,
    text: f.content,
    vector: f.embedding,
    metadata: {
      memoryType: f.memoryType,
      taskTags: f.taskTags,
      importance: f.importance,
    },
  }));

  let qVec: readonly number[] | undefined = opts.queryEmbedding;
  if (qVec === undefined) {
    const r = agent.memory.store.embedder.embed(query);
    qVec = r instanceof Promise ? await r : r;
  }

  const retriever = new HybridRetriever<HybridMemoryHit["doc"]>({
    ...(opts.config ? { config: opts.config } : {}),
  });
  retriever.index(docs);
  return retriever.search(query, qVec, opts.topK ?? 10);
};
