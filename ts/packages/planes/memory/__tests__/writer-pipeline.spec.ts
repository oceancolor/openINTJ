import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SqliteCanonicalMemoryStore } from "@openintj/storage-sqlite";
import { afterEach, describe, expect, it } from "vitest";
import {
  type CandidateReceiveKey,
  type CanonicalMemoryRecord,
  type CanonicalMemoryRepository,
  type CandidateValidationContext,
  type MemoryWriter,
  type WriterInput,
  InMemoryCanonicalMemoryRepository,
  ShadowWriterPipeline,
  createFixtureWriter,
  hashEvidenceText,
} from "../src/index.js";

const dirs: string[] = [];
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

const span = (text: string) => ({
  evidenceId: "ev1",
  journalId: "journal-1",
  commitId: "commit-1",
  turnId: "t1",
  contentHash: hashEvidenceText(text),
  text,
});

const makeInput = (text: string, over: Partial<WriterInput> = {}): WriterInput => ({
  committedTurnRef: "commit-1",
  evidence: [span(text)],
  scopedCurrentRecords: [],
  allowedKinds: ["fact", "preference"],
  policyVersion: "v1",
  inputHash: "input-hash-1",
  memoryEpoch: 1,
  extractionRevision: 1,
  ...over,
});

const key: CandidateReceiveKey = {
  sourceJournalId: "journal-1",
  sourceCommitId: "commit-1",
  consumer: "memory-writer",
  policyVersion: "v1",
  extractionRevision: 1,
};

const buildContext =
  (epochShift = 0) =>
  (input: WriterInput): CandidateValidationContext => ({
    scope: { workspaceId: "ws1" },
    currentEpoch: input.memoryEpoch + epochShift,
    allowedKinds: [...input.allowedKinds],
    resolveEvidence: (id) => input.evidence.find((s) => s.evidenceId === id),
    recordsById: new Map(input.scopedCurrentRecords.map((r) => [r.fragmentId, r])),
    userOwnedKeys: new Set(),
    isSuppressed: () => false,
  });

const seedRecord = async (repo: CanonicalMemoryRepository): Promise<void> => {
  const record: CanonicalMemoryRecord = {
    fragmentId: "frag-seed",
    content: "既有记忆",
    memoryType: "long_term",
    importance: 0.5,
    taskTags: [],
    scopeTags: ["workspace:ws1"],
    metadata: {},
    summaries: {},
    embedding: [],
    source: "user",
    state: "active",
    revision: 1,
    timestamp: 1,
    accessCount: 0,
    lastAccessed: 0,
    kind: "fact",
    assertion: "user",
  };
  await repo.applyMutation({
    writeKey: "seed-1",
    inputHash: "seed-hash",
    kind: "upsert",
    fragmentId: record.fragmentId,
    record,
  });
};

describe("G5.1 F13 影子管线 (T27)", () => {
  it("影子零影响：validated 只落 shadow_only，canonical 记录/索引/epoch 全不变", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    await seedRecord(repo);
    const before = {
      records: await repo.listRecords({ includeInactive: true }),
      index: await repo.indexStats(),
      epoch: await repo.memoryEpoch(),
    };
    const pipeline = new ShadowWriterPipeline({
      repository: repo,
      writer: createFixtureWriter(),
      buildContext: buildContext(),
    });
    const result = await pipeline.enqueue({ key, input: makeInput("记住：项目截止日是周五") });
    expect(result).toMatchObject({ status: "shadow_recorded", validated: 1, rejected: 0 });

    expect(await repo.listRecords({ includeInactive: true })).toEqual(before.records);
    expect(await repo.indexStats()).toEqual(before.index);
    expect(await repo.memoryEpoch()).toBe(before.epoch);

    const outcomes = await repo.listCandidateOutcomes();
    expect(outcomes).toHaveLength(1);
    expect(outcomes[0]).toMatchObject({ status: "shadow_only", receiveKey: expect.stringContaining("commit-1") });
    expect(JSON.parse(outcomes[0]!.candidateJson).content).toBe("项目截止日是周五");
    expect(await repo.candidateBatchStatus(key)).toMatchObject({ status: "received", candidateCount: 1 });
  });

  it("ACK 只确认完整接收：被拒候选与空批次同样 shadow_recorded，不代表生效", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    const garbage: MemoryWriter = {
      writerId: "garbage",
      async propose() {
        return { candidates: [{ operation: "add", kind: "fact", content: "x", sourceRefs: ["ev1"], rationale: "", scope: "evil" }] };
      },
    };
    const pipeline = new ShadowWriterPipeline({ repository: repo, writer: garbage, buildContext: buildContext() });
    const result = await pipeline.enqueue({ key, input: makeInput("任意文本") });
    expect(result).toMatchObject({ status: "shadow_recorded", validated: 0, rejected: 1 });
    expect((await repo.listCandidateOutcomes())[0]).toMatchObject({
      status: "rejected",
      reasons: ["SCHEMA_INVALID"],
    });
    expect(await repo.listRecords({ includeInactive: true })).toHaveLength(0);

    const empty = new ShadowWriterPipeline({
      repository: repo,
      writer: createFixtureWriter(),
      buildContext: buildContext(),
    });
    const emptyResult = await empty.enqueue({
      key: { ...key, sourceCommitId: "commit-2" },
      input: makeInput("没有任何标记的闲聊", { committedTurnRef: "commit-2", inputHash: "input-hash-2" }),
    });
    expect(emptyResult).toMatchObject({ status: "shadow_recorded", validated: 0, rejected: 0, conflicted: 0 });
  });

  it("重放幂等：同 job 二次 enqueue 返回 replayed，结局不重复记", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    const pipeline = new ShadowWriterPipeline({
      repository: repo,
      writer: createFixtureWriter(),
      buildContext: buildContext(),
    });
    const job = { key, input: makeInput("记住：部署在周五") };
    expect((await pipeline.enqueue(job)).status).toBe("shadow_recorded");
    expect(await pipeline.enqueue(job)).toMatchObject({ status: "replayed" });
    expect(await repo.listCandidateOutcomes()).toHaveLength(1);
  });

  it("同接收键异 inputHash → 死信 RECEIVE_CONFLICT，不覆盖已收批次", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    const pipeline = new ShadowWriterPipeline({
      repository: repo,
      writer: createFixtureWriter(),
      buildContext: buildContext(),
    });
    await pipeline.enqueue({ key, input: makeInput("记住：甲") });
    const conflict = await pipeline.enqueue({
      key,
      input: makeInput("记住：乙", { inputHash: "another-hash" }),
    });
    expect(conflict).toMatchObject({ status: "dead_letter", reason: "RECEIVE_CONFLICT" });
    expect(await repo.listCandidateOutcomes()).toHaveLength(1);
  });

  it("writer 异常同版本重试：前两次失败第三次成功；始终失败死信 WRITER_FAILED 且零落盘", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    let calls = 0;
    const revisions: number[] = [];
    const flaky: MemoryWriter = {
      writerId: "flaky",
      async propose(input) {
        calls += 1;
        revisions.push(input.extractionRevision);
        if (calls < 3) throw new Error("model unavailable");
        return createFixtureWriter().propose(input);
      },
    };
    const pipeline = new ShadowWriterPipeline({ repository: repo, writer: flaky, buildContext: buildContext() });
    const ok = await pipeline.enqueue({ key, input: makeInput("记住：周五发布") });
    expect(ok).toMatchObject({ status: "shadow_recorded", attempts: 3 });
    expect(revisions).toEqual([1, 1, 1]);

    const deadRepo = new InMemoryCanonicalMemoryRepository();
    const dead = new ShadowWriterPipeline({
      repository: deadRepo,
      writer: {
        writerId: "down",
        async propose() {
          throw new Error("always down");
        },
      },
      buildContext: buildContext(),
      maxAttempts: 2,
    });
    const deadResult = await dead.enqueue({ key, input: makeInput("记住：不会落盘") });
    expect(deadResult).toMatchObject({ status: "dead_letter", reason: "WRITER_FAILED", attempts: 2 });
    expect(await deadRepo.candidateBatchStatus(key)).toBeUndefined();
    expect(dead.listDeadLetters()).toHaveLength(1);
  });

  it("超预算死信 OVER_BUDGET：候选数超限与证据过大（writer 不被调用）", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    const spammy: MemoryWriter = {
      writerId: "spammy",
      async propose() {
        return {
          candidates: Array.from({ length: 21 }, () => ({
            operation: "add",
            kind: "fact",
            content: "x",
            sourceRefs: ["ev1"],
            rationale: "",
          })),
        };
      },
    };
    const pipeline = new ShadowWriterPipeline({ repository: repo, writer: spammy, buildContext: buildContext() });
    expect(await pipeline.enqueue({ key, input: makeInput("普通文本") })).toMatchObject({
      status: "dead_letter",
      reason: "OVER_BUDGET",
    });

    let called = 0;
    const counting: MemoryWriter = {
      writerId: "counting",
      async propose(input) {
        called += 1;
        return createFixtureWriter().propose(input);
      },
    };
    const tight = new ShadowWriterPipeline({
      repository: repo,
      writer: counting,
      buildContext: buildContext(),
      maxEvidenceChars: 4,
    });
    expect(await tight.enqueue({ key: { ...key, sourceCommitId: "c-big" }, input: makeInput("这段证据超过四个字符") })).toMatchObject({
      status: "dead_letter",
      reason: "OVER_BUDGET",
      attempts: 0,
    });
    expect(called).toBe(0);
  });

  it("过期 epoch 的迟到 writer：整批 STALE_EPOCH 拒绝（T24 联动）", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    const pipeline = new ShadowWriterPipeline({
      repository: repo,
      writer: createFixtureWriter(),
      buildContext: buildContext(1),
    });
    const result = await pipeline.enqueue({ key, input: makeInput("记住：旧 epoch 的提议") });
    expect(result).toMatchObject({ status: "shadow_recorded", validated: 0, rejected: 1 });
    expect((await repo.listCandidateOutcomes())[0]).toMatchObject({
      status: "rejected",
      reasons: ["STALE_EPOCH"],
    });
  });

  it("单 worker 串行：并发 enqueue 按入队顺序结算", async () => {
    const repo = new InMemoryCanonicalMemoryRepository();
    const pipeline = new ShadowWriterPipeline({
      repository: repo,
      writer: createFixtureWriter(),
      buildContext: buildContext(),
    });
    const order: string[] = [];
    const p1 = pipeline
      .enqueue({ key: { ...key, sourceCommitId: "c1" }, input: makeInput("记住：一", { inputHash: "h1" }) })
      .then((r) => order.push(r.status === "shadow_recorded" ? "c1" : "?"));
    const p2 = pipeline
      .enqueue({ key: { ...key, sourceCommitId: "c2" }, input: makeInput("记住：二", { inputHash: "h2" }) })
      .then((r) => order.push(r.status === "shadow_recorded" ? "c2" : "?"));
    await Promise.all([p1, p2]);
    expect(order).toEqual(["c1", "c2"]);
    expect(await repo.listCandidateOutcomes()).toHaveLength(2);
  });

  it("真实 SQLite：影子结局重启后仍可读（审核台数据源）", async () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-g51-"));
    dirs.push(dir);
    const dbPath = join(dir, "canonical.db");
    const repo1 = new SqliteCanonicalMemoryStore(dbPath) as unknown as CanonicalMemoryRepository;
    await repo1.init();
    const pipeline = new ShadowWriterPipeline({
      repository: repo1,
      writer: createFixtureWriter(),
      buildContext: buildContext(),
    });
    await pipeline.enqueue({ key, input: makeInput("记住：重启后仍在") });
    await repo1.close();

    const repo2 = new SqliteCanonicalMemoryStore(dbPath) as unknown as CanonicalMemoryRepository;
    await repo2.init();
    const outcomes = await repo2.listCandidateOutcomes({ status: "shadow_only" });
    expect(outcomes).toHaveLength(1);
    expect(JSON.parse(outcomes[0]!.candidateJson).content).toBe("重启后仍在");
    await repo2.close();
  });
});
