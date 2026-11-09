import { describe, expect, it } from "vitest";
import {
  CanonicalCandidateConflictError,
  CanonicalMutationConflictError,
  SqliteCanonicalMemoryStore,
  type CanonicalMemoryRecordRow,
} from "../src/index.js";

const makeRepo = async (): Promise<SqliteCanonicalMemoryStore> => {
  const repo = new SqliteCanonicalMemoryStore(":memory:");
  await repo.init();
  return repo;
};

const record = (id: string, content: string): CanonicalMemoryRecordRow => ({
  fragmentId: id,
  content,
  memoryType: "long_term",
  importance: 0.5,
  taskTags: ["workspace:demo"],
  scopeTags: ["workspace:demo"],
  metadata: {},
  summaries: {},
  embedding: [0.1, 0.2],
  source: "test",
  state: "active",
  revision: 0,
  timestamp: Date.now(),
  accessCount: 0,
  lastAccessed: 0,
});

describe("SqliteCanonicalMemoryStore", () => {
  it("mutation 幂等/CAS/冲突（T12 的 SQLite 侧）", async () => {
    const repo = await makeRepo();
    const r1 = await repo.applyMutation({ writeKey: "w1", inputHash: "h1", kind: "upsert", record: record("a", "v1") });
    const r2 = await repo.applyMutation({ writeKey: "w1", inputHash: "h1", kind: "upsert", record: record("a", "v1") });
    expect(r1.outcome).toBe("applied");
    expect(r2.outcome).toBe("replayed");
    expect((await repo.listRecords())).toHaveLength(1);
    await expect(
      repo.applyMutation({ writeKey: "w1", inputHash: "hx", kind: "upsert", record: record("a", "v1") }),
    ).rejects.toBeInstanceOf(CanonicalMutationConflictError);
    await expect(
      repo.applyMutation({
        writeKey: "w2",
        inputHash: "h2",
        kind: "upsert",
        record: record("a", "v2"),
        expectedRevision: 42,
      }),
    ).rejects.toBeInstanceOf(CanonicalMutationConflictError);
    expect((await repo.getRecord("a"))?.content).toBe("v1");
    await repo.close();
  });

  it("outbox 与 mutation 同事务：每次变更都有待投影 job", async () => {
    const repo = await makeRepo();
    await repo.applyMutation({ writeKey: "w1", inputHash: "h1", kind: "upsert", record: record("a", "x") });
    await repo.applyMutation({ writeKey: "w2", inputHash: "h2", kind: "delete", fragmentId: "a" });
    const jobs = await repo.pendingIndexJobs();
    expect(jobs.map((j) => j.kind)).toEqual(["upsert", "delete"]);
    expect((await repo.listRecords())).toHaveLength(0);
    expect(await repo.listRecords({ includeInactive: true })).toHaveLength(1);
    await repo.close();
  });

  it("候选批次接收键幂等 + 异 hash 冲突", async () => {
    const repo = await makeRepo();
    const key = {
      sourceJournalId: "j1",
      sourceCommitId: "c1",
      consumer: "memory-writer",
      policyVersion: "v1",
      extractionRevision: 1,
    };
    const a = await repo.receiveCandidates({ key, inputHash: "h", candidates: [{ content: "x" }] });
    const b = await repo.receiveCandidates({ key, inputHash: "h", candidates: [{ content: "x" }] });
    expect(a.outcome).toBe("applied");
    expect(b.outcome).toBe("replayed");
    await expect(repo.receiveCandidates({ key, inputHash: "h2", candidates: [] })).rejects.toBeInstanceOf(
      CanonicalCandidateConflictError,
    );
    expect((await repo.candidateBatchStatus(key))?.status).toBe("received");
    await repo.close();
  });

  it("control ledger：epoch 递增与 suppression", async () => {
    const repo = await makeRepo();
    expect(await repo.memoryEpoch()).toBe(1);
    expect(await repo.bumpMemoryEpoch()).toBe(2);
    await repo.addSuppression("frag-1", "user:local", "forget");
    expect(await repo.isSuppressed("frag-1")).toBe(true);
    expect(await repo.listSuppressions()).toHaveLength(1);
    await repo.close();
  });
});
