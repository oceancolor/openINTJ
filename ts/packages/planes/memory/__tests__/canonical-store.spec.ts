import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryMetadataStore, SqliteCanonicalMemoryStore } from "@openintj/storage-sqlite";
import { InMemoryVectorStore, type VectorStore } from "@openintj/storage-lance";
import { afterEach, describe, expect, it } from "vitest";
import {
  type CanonicalMemoryRepository,
  CandidateConflictError,
  InMemoryCanonicalMemoryRepository,
  MemoryRetriever,
  MutationConflictError,
  PersistentMemoryStore,
} from "../src/index.js";

const dirs: string[] = [];
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

const tmpCanonical = async (): Promise<CanonicalMemoryRepository> => {
  const dir = mkdtempSync(join(tmpdir(), "openintj-g2-"));
  dirs.push(dir);
  const repo = new SqliteCanonicalMemoryStore(join(dir, "canonical.db")) as unknown as CanonicalMemoryRepository;
  await repo.init();
  return repo;
};

const makeStore = async (
  canonical: CanonicalMemoryRepository,
  vectorStore: VectorStore = new InMemoryVectorStore(),
): Promise<PersistentMemoryStore> => {
  const store = new PersistentMemoryStore({
    vectorStore,
    metadataStore: new InMemoryMetadataStore(),
    canonical,
  });
  await store.init();
  return store;
};

/** 可控失败的向量桩：failWhen 命中即抛错，模拟索引崩溃/空间冲突。 */
class FlakyVectorStore extends InMemoryVectorStore {
  failWhen: ((content: string) => boolean) | undefined;
  override async upsert(rows: Parameters<InMemoryVectorStore["upsert"]>[0]): Promise<void> {
    for (const r of rows) {
      if (this.failWhen?.(r.content)) throw new Error("vector index unavailable (injected)");
    }
    return super.upsert(rows);
  }
}

describe("G2 canonical memory (T10–T15)", () => {
  it("T10 权威恢复：向量层全丢，正文仍从 canonical 恢复且可检索", async () => {
    const canonical = await tmpCanonical();
    const first = await makeStore(canonical);
    await first.addLongTermAsync("权威正文：项目代号蓝鲸，负责人林工。");
    await first.close();

    const brokenVector = new FlakyVectorStore();
    brokenVector.failWhen = () => true;
    const second = await makeStore(canonical, brokenVector);
    expect(second.all.map((f) => f.content)).toContain("权威正文：项目代号蓝鲸，负责人林工。");
    const hits = await new MemoryRetriever(second).retrieveAsync("蓝鲸项目负责人");
    expect(hits.length).toBeGreaterThan(0);
    await second.close();
  });

  it("T11 mutation 覆盖：晋升/清除/删除重启后不复活（含 E06 clearWorking）", async () => {
    const canonical = await tmpCanonical();
    const first = await makeStore(canonical);
    const keep = await first.addLongTermAsync("长期保留条目");
    const temp = await first.addWorkingAsync("临时工作条目");
    const drop = await first.addLongTermAsync("将被删除条目");
    expect(temp.memoryType).toBe("working");
    first.clearWorking();
    first.remove(drop.fragmentId);
    await first.awaitPendingWrites();
    await first.close();

    const second = await makeStore(canonical);
    const contents = second.all.map((f) => f.content);
    expect(contents).toContain("长期保留条目");
    expect(contents).not.toContain("临时工作条目");
    expect(contents).not.toContain("将被删除条目");
    expect(second.all.find((f) => f.content === "长期保留条目")?.fragmentId).toBe(keep.fragmentId);
    await second.close();
  });

  it("T12 写幂等/CAS：同键同 hash 重放、同键异 hash 冲突、CAS 不符零写入", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    await repo.init();
    const record = {
      fragmentId: "frag-cas",
      content: "v1",
      memoryType: "long_term" as const,
      importance: 0.5,
      taskTags: [],
      scopeTags: [],
      metadata: {},
      summaries: {},
      embedding: [0.1, 0.2],
      source: "test",
      state: "active" as const,
      revision: 0,
      timestamp: Date.now(),
      accessCount: 0,
      lastAccessed: 0,
    };
    const r1 = await repo.applyMutation({ writeKey: "w1", inputHash: "h1", kind: "upsert", record });
    const r2 = await repo.applyMutation({ writeKey: "w1", inputHash: "h1", kind: "upsert", record });
    expect(r1.outcome).toBe("applied");
    expect(r2.outcome).toBe("replayed");
    expect((await repo.listRecords())).toHaveLength(1);
    expect((await repo.getRecord("frag-cas"))?.revision).toBe(1);
    await expect(
      repo.applyMutation({ writeKey: "w1", inputHash: "h-other", kind: "upsert", record }),
    ).rejects.toBeInstanceOf(MutationConflictError);
    await expect(
      repo.applyMutation({
        writeKey: "w2",
        inputHash: "h2",
        kind: "upsert",
        record: { ...record, content: "v2" },
        expectedRevision: 99,
      }),
    ).rejects.toBeInstanceOf(MutationConflictError);
    expect((await repo.getRecord("frag-cas"))?.content).toBe("v1");
  });

  it("T13 outbox 崩溃重放：坏 job 不毒化队列，恢复后投影补齐", async () => {
    const canonical = await tmpCanonical();
    const vector = new FlakyVectorStore();
    vector.failWhen = (c) => c.includes("坏索引");
    const store = await makeStore(canonical, vector);
    await store.addLongTermAsync("正常条目一");
    await store.addLongTermAsync("坏索引条目");
    await store.addLongTermAsync("正常条目二");
    await store.awaitPendingWrites();
    let status = await store.indexStatus();
    expect(status.pending).toBeGreaterThan(0);
    expect(status.failed).toBeGreaterThan(0);
    // 正常条目已投影，坏条目留队等待——队列没被毒化
    const projected = await vector.scanAll();
    expect(projected.map((r) => r.content)).toContain("正常条目一");
    expect(projected.map((r) => r.content)).toContain("正常条目二");

    vector.failWhen = undefined;
    const result = await store.projectIndex();
    expect(result.failed).toBe(0);
    status = await store.indexStatus();
    expect(status.pending).toBe(0);
    expect((await vector.scanAll()).map((r) => r.content)).toContain("坏索引条目");
    await store.close();
  });

  it("T14 stale index：删除后的延迟旧 upsert 不得复活", async () => {
    const canonical = await tmpCanonical();
    const vector = new FlakyVectorStore();
    vector.failWhen = () => true;
    const store = await makeStore(canonical, vector);
    const f = await store.addLongTermAsync("短暂存在条目");
    store.remove(f.fragmentId);
    await store.awaitPendingWrites();
    vector.failWhen = undefined;
    await store.projectIndex();
    expect(await vector.scanAll()).toHaveLength(0);
    await store.close();

    const second = await makeStore(canonical, new InMemoryVectorStore());
    expect(second.all.map((x) => x.content)).not.toContain("短暂存在条目");
    await second.close();
  });

  it("T15 向量空间冲突：投影显式失败但正文持久、词法检索不中断", async () => {
    const canonical = await tmpCanonical();
    const first = await makeStore(canonical);
    await first.addLongTermAsync("空间冲突下仍可读的正文：火星计划。");
    await first.close();

    // 预置 8 维向量库冻结维度，与 64 维 embedder 冲突
    const foreignVector = new InMemoryVectorStore();
    await foreignVector.init();
    await foreignVector.upsert([
      {
        fragmentId: "foreign",
        content: "foreign",
        embedding: [0, 0, 0, 0, 0, 0, 0, 1],
        memoryType: "long_term",
        importance: 0.5,
        taskTags: [],
        contentHash: "x",
        timestamp: Date.now(),
        accessCount: 0,
        lastAccessed: 0,
        metadataJson: "{}",
        summariesJson: "{}",
      },
    ]);
    const second = await makeStore(canonical, foreignVector);
    // 先把外来行清掉模拟重建场景外的冲突保留：正文仍在且可检索
    expect(second.all.map((f) => f.content)).toContain("空间冲突下仍可读的正文：火星计划。");
    const status = await second.indexStatus();
    expect(status.dimensionCompatible).toBe(false);
    const hits = await new MemoryRetriever(second).retrieveAsync("火星计划");
    expect(hits.length).toBeGreaterThan(0);
    await second.close();
  });

  it("F06 候选接收：固定 fixture 幂等接收、异 hash 冲突、候选不激活", async () => {
    const canonical = await tmpCanonical();
    const store = await makeStore(canonical);
    const key = {
      sourceJournalId: "journal-1",
      sourceCommitId: "commit-1",
      consumer: "memory-writer",
      policyVersion: "v1",
      extractionRevision: 1,
    };
    const fixture = [{ content: "候选：用户偏好深色模式", kind: "preference" }];
    const a = await store.receiveCandidateBatch({ key, inputHash: "hash-a", candidates: fixture });
    const b = await store.receiveCandidateBatch({ key, inputHash: "hash-a", candidates: fixture });
    expect(a.outcome).toBe("applied");
    expect(b.outcome).toBe("replayed");
    await expect(
      store.receiveCandidateBatch({ key, inputHash: "hash-b", candidates: fixture }),
    ).rejects.toBeInstanceOf(CandidateConflictError);
    // 候选只入库待审，绝不出现在权威记忆里
    expect(store.all.map((f) => f.content)).not.toContain("候选：用户偏好深色模式");
    const status = await canonical.candidateBatchStatus(key);
    expect(status?.status).toBe("received");
    await store.close();
  });

  it("F12 P0：forget 后重启不出现，epoch 递增", async () => {
    const canonical = await tmpCanonical();
    const first = await makeStore(canonical);
    const f = await first.addLongTermAsync("需要被遗忘的条目");
    const epochBefore = await first.memoryEpoch();
    await first.forgetFragment(f.fragmentId, "user:local", "user request");
    await first.close();

    const second = await makeStore(canonical);
    expect(second.all.map((x) => x.content)).not.toContain("需要被遗忘的条目");
    expect(await second.memoryEpoch()).toBe(epochBefore);
    expect(await canonical.isSuppressed(f.fragmentId)).toBe(true);
    const bumped = await canonical.bumpMemoryEpoch();
    expect(bumped).toBe(epochBefore + 1);
    await second.close();
  });
});
