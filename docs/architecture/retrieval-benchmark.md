# 检索 / 嵌入基准（roadmap B：#3 嵌入三方对比 + #10 LanceDB 原生 FTS）

> 归档位置：`docs/architecture/`，与 `phase-flywheel-design.md` / `phase-skills-design.md` /
> `next-session.md` 同侧。实现代码在 `ts/packages/planes/memory/src/eval/` 与 `ts/packages/storage/lance/src/`。

本文回答两个问题，并给出**可复现**的跑法：

1. **#3**：`simple` / `xenova` / `ollama` 三个 embedder 在固定语料上的检索质量（nDCG/recall/precision/MRR），为「默认嵌入器选型」提供数据支撑。
2. **#10**：为什么、以及如何在 fragment 规模变大（N>10k）时把词法检索从「内存 BM25 全表扫描」切到「LanceDB 原生 FTS」。

---

## 一、评测方法（harness）

代码：`ts/packages/planes/memory/src/eval/retrieval-benchmark.ts` + `retrieval-metrics.ts`

- **语料**：`BENCHMARK_CORPUS` —— 3 主题（database / cooking / astronomy）× 4 文档 = 12 篇。
- **查询**：`BENCHMARK_QUERIES` —— 每主题 2 条，共 6 条。相关性按主题二值判定（同主题 4 篇为相关）。
- **指标**：`evaluateRanker` 对 6 条 query 宏平均 nDCG@k / recall@k / precision@k / MRR（默认 k=4）。
- **两条评测路径**（关键区别）：
  | 函数 | 排序依据 | 衡量的是 |
  | --- | --- | --- |
  | `benchmarkRetrieval` | `MemoryStore` + `MemoryRetriever`（cosine + 关键词重叠 + 时间衰减） | **产品实际检索路径**的端到端质量 |
  | `benchmarkEmbedderCosine` | **纯 cosine**（只用 embedder 向量排序） | **隔离出的 embedder 语义能力** |

  之所以要两条：`MemoryRetriever` 里的关键词重叠项会「补偿」弱 embedder，掩盖嵌入器之间的真实差距。
  纯 cosine 才是「换更好的 embedder 能带来多少语义召回收益」的直接证据。

---

## 二、已测结果（本机实跑）

### `simple`（SimpleEmbedder，SHA-256 词袋哈希）

| 路径 | nDCG@4 | recall@4 | precision@4 | MRR |
| --- | --- | --- | --- | --- |
| MemoryRetriever（产品路径） | **0.773** | 0.708 | 0.708 | 1.000 |
| 纯 cosine（隔离语义） | **0.396** | 0.375 | 0.375 | 0.617 |

- **维度无关**：`dim ∈ {32,64,128,256}` 两条路径的分数完全一致 —— SHA-256 词袋哈希不含真正语义，
  增大维度只降低哈希碰撞概率，不引入近义/同义泛化能力。
- **两条路径的落差（0.773 → 0.396）**量化了当前默认 embedder 的短板：产品路径能拿 0.773，
  几乎全靠 `MemoryRetriever` 的**关键词重叠**兜底；embedder 自身语义召回仅 0.396。
  → 这正是引入神经嵌入器（xenova / ollama）的收益空间。

### `xenova` / `ollama`

- 现状：本机 **未安装** `@xenova/transformers`（可选 peer dep），**未运行** ollama 服务，故暂无实测数字。
- 预期：两者为真正的神经句向量（MiniLM 384 维 / nomic-embed-text 768 维），在**纯 cosine**路径上
  应显著高于 `simple` 的 0.396；产品路径因已被关键词项抬高，提升幅度会小一些但仍应 ≥ simple。
- 待补：安装 / 起服务后按下节命令实跑，把数字回填到本表。

---

## 三、复现方式

```bash
# 仅 simple 基线（始终可跑，CI 默认）
pnpm --filter @openintj/plane-memory test retrieval-benchmark

# 三方对比（simple vs xenova vs ollama）——需先满足各自前置条件
#   xenova：pnpm --filter @openintj/embed-xenova add @xenova/transformers（首跑会下载 ~90MB 模型）
#   ollama：本地起服务并拉模型  ollama pull nomic-embed-text
RUN_EMBED_COMPARE=1 pnpm --filter @openintj/plane-memory test retrieval-benchmark
```

`RUN_EMBED_COMPARE=1` 时会分别打印「MemoryRetriever 路径」与「纯 cosine」两套评分表；缺失的
embedder（未装 / 服务未起）会被 try/catch 跳过并打印 warn，不影响 simple 基线通过。

---

## 四、#10：LanceDB 原生 FTS

### 动机

`MemoryHybridIndex`（`ts/packages/taskpool`）持有一个 session 级 `HybridRetriever`，query 时对
**全部内存文档**逐条算 BM25 + cosine（O(N)/query）。中等规模够用，但 N>10k fragment 时全表扫描
成本上升。LanceDB 自带基于 BM25 的**原生 FTS 索引**，可把词法检索下推到存储层。

### 设计（存储层混合检索 + RRF 融合）

代码：`ts/packages/storage/lance/src/{types,lancedb,in-memory,fusion}.ts`

- `VectorStore` 新增可选能力：
  - `supportsFts`：是否支持原生 FTS。
  - `ensureFtsIndex()`：在 `content` 列建 FTS 索引（幂等；旧版 / 不支持时静默降级）。
  - `searchText(query, opts)`：原生 BM25 词法检索，过滤语义（memoryType/taskTags/minImportance）与
    向量检索一致。
- `LanceDBVectorStore`：用 `table.createIndex("content", { config: Index.fts() })` 建索引，
  `table.search(query, "fts")` 查询；探测失败 / 版本不支持时 `supportsFts=false`，`searchText` 返回空。
- `InMemoryVectorStore`：同样实现 `searchText`（BM25-lite），让融合逻辑在**不装 LanceDB** 时也能
  端到端单测。
- `hybridVectorSearch(store, {query, queryEmbedding, topK, ...})`：向量榜 + FTS 榜各自召回，用
  **RRF（Reciprocal Rank Fusion）** 融合。RRF 只依赖名次、不依赖分数量纲，天然适配「cosine 分」
  与「BM25 分」这类异构分数（无需先 min-max 归一）。`searchText` 缺失 / 返回空时自动降级为纯向量。

### 接入（opt-in，不改默认路径）

`ts/apps/server/src/hybrid-retrieve.ts`：

- 默认仍走内存 `MemoryHybridIndex`（行为不变）。
- 传 `useLanceFts: true` 或设 env `OPENINTJ_LANCE_FTS=1` → 走 `hybridVectorSearch(persistentStore.vectorStore, …)`，
  把词法检索下推到持久层（真实模式=LanceDB 原生 FTS；mock 模式=InMemory 的 BM25-lite，用于测试）。
- 结果映射回 `MemoryHybridHit`，RRF 分记入 `components.rrf`。

### 何时开

- N 较小（数千内）：默认内存路径足够，省一次存储层查询。
- N>10k：开 `OPENINTJ_LANCE_FTS=1`，用 LanceDB 原生 FTS 避免全表 BM25 扫描。
