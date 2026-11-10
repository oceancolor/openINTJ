import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemoryMetadataStore, SqliteCanonicalMemoryStore } from "@openintj/storage-sqlite";
import { InMemoryVectorStore } from "@openintj/storage-lance";
import { afterEach, describe, expect, it } from "vitest";
import {
  type CanonicalMemoryRepository,
  MemoryEpochStaleError,
  PersistentMemoryStore,
} from "../src/index.js";

const dirs: string[] = [];
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

const setup = async (): Promise<{
  canonical: CanonicalMemoryRepository;
  store: PersistentMemoryStore;
  vector: InMemoryVectorStore;
}> => {
  const dir = mkdtempSync(join(tmpdir(), "openintj-g4-"));
  dirs.push(dir);
  const canonical = new SqliteCanonicalMemoryStore(
    join(dir, "canonical.db"),
  ) as unknown as CanonicalMemoryRepository;
  await canonical.init();
  const vector = new InMemoryVectorStore();
  const store = new PersistentMemoryStore({
    vectorStore: vector,
    metadataStore: new InMemoryMetadataStore(),
    canonical,
  });
  await store.init();
  return { canonical, store, vector };
};

describe("G4 F12 有效遗忘与 purge (T24/T25)", () => {
  it("T24 forget 原子性：tombstone+suppression+epoch 同步生效，迟到投影不复活", async () => {
    const { canonical, store, vector } = await setup();
    const f = await store.addLongTermAsync("敏感条目：银行卡密码是 123456");
    expect((await vector.scanAll()).map((r) => r.fragmentId)).toContain(f.fragmentId);
    const epochBefore = await store.memoryEpoch();

    const newEpoch = await store.forgetFragment(f.fragmentId, "user:local", "user forget");
    expect(newEpoch).toBe(epochBefore + 1);
    expect(store.all.map((x) => x.fragmentId)).not.toContain(f.fragmentId);
    expect((await vector.scanAll()).map((r) => r.fragmentId)).not.toContain(f.fragmentId);
    const record = await canonical.getRecord(f.fragmentId);
    expect(record?.state).toBe("tombstoned");
    expect(await canonical.isSuppressed(f.fragmentId)).toBe(true);

    // 旧 epoch 的 run 在授权点被拒
    await expect(store.assertMemoryEpoch(epochBefore)).rejects.toBeInstanceOf(MemoryEpochStaleError);
    await store.assertMemoryEpoch(newEpoch);

    // 重启后仍不可见
    const second = new PersistentMemoryStore({
      vectorStore: new InMemoryVectorStore(),
      metadataStore: new InMemoryMetadataStore(),
      canonical,
    });
    await second.init();
    expect(second.all.map((x) => x.content)).not.toContain("敏感条目：银行卡密码是 123456");
    await second.close();
    await store.close();
  });

  it("T25 purge 物理删除：记录本体消失、barrier 保留、备份如实标注 unsupported", async () => {
    const { canonical, store } = await setup();
    const f = await store.addLongTermAsync("要物理删除的条目");
    const report = await store.purgeMemory(f.fragmentId, "user:local");
    expect(report.recordDeleted).toBe(true);
    expect(report.vectorProjectionDeleted).toBe(true);
    expect(report.barrierRetained).toBe(true);
    expect(report.backups).toBe("unsupported");
    // 本体不在了（区别于 tombstone：连审计行都删）
    expect(await canonical.getRecord(f.fragmentId)).toBeUndefined();
    expect(await canonical.listRecords({ includeInactive: true })).toHaveLength(0);
    // barrier 还在：同 id 不能回潮
    expect(await canonical.isSuppressed(f.fragmentId)).toBe(true);
    expect(store.all).toHaveLength(0);
    await store.close();
  });
});
