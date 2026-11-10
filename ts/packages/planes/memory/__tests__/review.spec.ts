import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryMetadataStore, SqliteCanonicalMemoryStore } from "@openintj/storage-sqlite";
import { InMemoryVectorStore } from "@openintj/storage-lance";
import { afterEach, describe, expect, it } from "vitest";
import {
  type CanonicalMemoryRepository,
  CandidateReviewError,
  PersistentMemoryStore,
  ShadowWriterPipeline,
  buildCanonicalValidationContext,
  createFixtureWriter,
  hashEvidenceText,
  type WriterInput,
} from "../src/index.js";

const dirs: string[] = [];
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

const setup = async (): Promise<{ canonical: CanonicalMemoryRepository; store: PersistentMemoryStore }> => {
  const dir = mkdtempSync(join(tmpdir(), "openintj-g52-"));
  dirs.push(dir);
  const canonical = new SqliteCanonicalMemoryStore(
    join(dir, "canonical.db"),
  ) as unknown as CanonicalMemoryRepository;
  await canonical.init();
  const store = new PersistentMemoryStore({
    vectorStore: new InMemoryVectorStore(),
    metadataStore: new InMemoryMetadataStore(),
    canonical,
  });
  await store.init();
  return { canonical, store };
};

const shadowOnce = async (
  canonical: CanonicalMemoryRepository,
  text: string,
  commitId: string,
): Promise<void> => {
  const input: WriterInput = {
    committedTurnRef: commitId,
    evidence: [
      {
        evidenceId: `ev:${commitId}`,
        journalId: "journal-1",
        commitId,
        turnId: commitId,
        contentHash: hashEvidenceText(text),
        text,
      },
    ],
    scopedCurrentRecords: [],
    allowedKinds: ["fact", "preference"],
    policyVersion: "v1",
    inputHash: `hash:${commitId}`,
    memoryEpoch: await canonical.memoryEpoch(),
    extractionRevision: 1,
  };
  const pipeline = new ShadowWriterPipeline({
    repository: canonical,
    writer: createFixtureWriter(),
    buildContext: (inp, job) => buildCanonicalValidationContext(canonical, inp, job.scope ?? {}),
  });
  const result = await pipeline.enqueue({
    key: {
      sourceJournalId: "journal-1",
      sourceCommitId: commitId,
      consumer: "memory-writer",
      policyVersion: "v1",
      extractionRevision: 1,
    },
    input,
  });
  expect(result.status).toBe("shadow_recorded");
};

describe("G5.2 候选审核台后端 (F14/approve)", () => {
  it("approve(add)：host 执行 mutation 生效，结局转 applied；重复审核被拒", async () => {
    const { canonical, store } = await setup();
    await shadowOnce(canonical, "记住：项目截止日是周五", "c1");
    const pending = await store.listMemoryCandidates({ status: "shadow_only" });
    expect(pending).toHaveLength(1);
    const candidateId = pending[0]!.candidateId;

    const applied = await store.reviewMemoryCandidate(candidateId, "approve", "user-1");
    expect(applied.status).toBe("applied");
    expect(applied.fragmentId).toBeTruthy();

    const records = await canonical.listRecords();
    expect(records.some((r) => r.state === "active" && r.content === "项目截止日是周五")).toBe(true);
    expect((await canonical.candidateOutcome(candidateId))?.status).toBe("applied");

    await expect(store.reviewMemoryCandidate(candidateId, "approve", "user-1")).rejects.toMatchObject({
      code: "candidate_not_reviewable",
    });
    await store.close();
  });

  it("reject：结局转 dismissed，不产生任何记录", async () => {
    const { canonical, store } = await setup();
    await shadowOnce(canonical, "记住：不要保存这条", "c2");
    const [pending] = await store.listMemoryCandidates({ status: "shadow_only" });
    const out = await store.reviewMemoryCandidate(pending!.candidateId, "reject", "user-1");
    expect(out.status).toBe("dismissed");
    expect(await canonical.listRecords()).toHaveLength(0);
    await store.close();
  });

  it("approve(supersede)：旧值 superseded 新值生效；目标 revision 漂移后保持 shadow_only 报冲突", async () => {
    const { canonical, store } = await setup();
    await shadowOnce(canonical, "记住：截止日是周五", "c3");
    const [first] = await store.listMemoryCandidates({ status: "shadow_only" });
    const { fragmentId } = await store.reviewMemoryCandidate(first!.candidateId, "approve", "user-1");
    const target = await canonical.getRecord(fragmentId!);
    expect(target?.revision).toBe(1);

    const manual = {
      candidateId: "cand_manual_supersede",
      operation: "supersede",
      kind: "fact",
      content: "截止日改为下周一",
      sourceRefs: ["ev:c3"],
      targetRecordId: fragmentId,
      expectedRevision: 1,
      rationale: "user corrected",
    };
    await canonical.upsertCandidateOutcomes([
      {
        candidateId: manual.candidateId,
        receiveKey: "rk-manual",
        status: "shadow_only",
        reasons: [],
        candidateJson: JSON.stringify(manual),
        updatedAt: Date.now(),
      },
    ]);
    const applied = await store.reviewMemoryCandidate(manual.candidateId, "approve", "user-1");
    expect(applied.status).toBe("applied");
    const after = await canonical.listRecords({ includeInactive: true });
    expect(after.find((r) => r.fragmentId === fragmentId)?.state).toBe("superseded");
    expect(after.some((r) => r.state === "active" && r.content === "截止日改为下周一")).toBe(true);

    // 漂移场景：目标被其他路径改过（revision 前进），旧候选 approve 应冲突且不生效。
    const active = after.find((r) => r.state === "active")!;
    const stale = { ...manual, candidateId: "cand_manual_stale", content: "截止日改为周三", targetRecordId: active.fragmentId, expectedRevision: 1 };
    await store.correctMemory(active.fragmentId, "截止日改为周二", {});
    await canonical.upsertCandidateOutcomes([
      {
        candidateId: stale.candidateId,
        receiveKey: "rk-manual-2",
        status: "shadow_only",
        reasons: [],
        candidateJson: JSON.stringify(stale),
        updatedAt: Date.now(),
      },
    ]);
    await expect(store.reviewMemoryCandidate(stale.candidateId, "approve", "user-1")).rejects.toMatchObject({
      code: "candidate_revision_conflict",
    });
    expect((await canonical.candidateOutcome(stale.candidateId))?.status).toBe("shadow_only");
    await store.close();
  });

  it("approve(retract)：走有效遗忘（tombstone + suppression），结局 applied", async () => {
    const { canonical, store } = await setup();
    await shadowOnce(canonical, "记住：临时口令贴在显示器上", "c4");
    const [first] = await store.listMemoryCandidates({ status: "shadow_only" });
    const { fragmentId } = await store.reviewMemoryCandidate(first!.candidateId, "approve", "user-1");
    const epochBefore = await canonical.memoryEpoch();

    const retract = {
      candidateId: "cand_manual_retract",
      operation: "retract",
      kind: "fact",
      content: "",
      sourceRefs: ["ev:c4"],
      targetRecordId: fragmentId,
      expectedRevision: 1,
      rationale: "user asked to forget",
    };
    // retract 候选 content 为空仅为本测试手工构造（校验器要求 content 非空，
    // 真实管线不会产出此形状）；审核 apply 不依赖 content。
    await canonical.upsertCandidateOutcomes([
      {
        candidateId: retract.candidateId,
        receiveKey: "rk-manual-3",
        status: "shadow_only",
        reasons: [],
        candidateJson: JSON.stringify(retract),
        updatedAt: Date.now(),
      },
    ]);
    const applied = await store.reviewMemoryCandidate(retract.candidateId, "approve", "user-1");
    expect(applied.status).toBe("applied");
    expect((await canonical.getRecord(fragmentId!))?.state).toBe("tombstoned");
    expect(await canonical.isSuppressed(fragmentId!)).toBe(true);
    expect(await canonical.memoryEpoch()).toBe(epochBefore + 1);
    await store.close();
  });

  it("无 canonical：列表为空、审核明确报错（不伪造）", async () => {
    const store = new PersistentMemoryStore({
      vectorStore: new InMemoryVectorStore(),
      metadataStore: new InMemoryMetadataStore(),
    });
    await store.init();
    expect(await store.listMemoryCandidates()).toEqual([]);
    await expect(store.reviewMemoryCandidate("cand_x", "approve", "user-1")).rejects.toMatchObject({
      code: "canonical_unavailable",
    });
    expect(CandidateReviewError).toBeDefined();
    await store.close();
  });
});
