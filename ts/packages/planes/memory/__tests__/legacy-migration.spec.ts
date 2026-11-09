import { InMemoryMetadataStore } from "@openintj/storage-sqlite";
import { InMemoryVectorStore, type VectorRow } from "@openintj/storage-lance";
import { describe, expect, it } from "vitest";
import { auditLegacyMemory, InMemoryCanonicalMemoryRepository, migrateLegacyMemory } from "../src/index.js";

const row = (id: string, content: string, tags: string[]): VectorRow => ({
  fragmentId: id,
  content,
  embedding: [0.1, 0.2],
  memoryType: "long_term",
  importance: 0.5,
  taskTags: tags,
  contentHash: `h-${id}`,
  timestamp: Date.now(),
  accessCount: 0,
  lastAccessed: 0,
  metadataJson: "{}",
  summariesJson: "{}",
});

describe("legacy migration (dry-run 门)", () => {
  it("对账报告 + dry-run 零写入 + 正式迁移时 scope 不明进 quarantine", async () => {
    const vectorStore = new InMemoryVectorStore();
    await vectorStore.init();
    await vectorStore.upsert([
      row("scoped", "有归属的记忆", ["workspace:demo"]),
      row("unscoped", "归属不明的记忆", ["misc"]),
    ]);
    const metadataStore = new InMemoryMetadataStore();
    await metadataStore.init();
    await metadataStore.putFragmentMeta([
      {
        fragmentId: "scoped",
        memoryType: "long_term",
        importance: 0.5,
        contentHash: "h-scoped",
        taskTagsCsv: "workspace:demo",
        metadataJson: "{}",
        summariesJson: "{}",
        timestamp: Date.now(),
        accessCount: 0,
        lastAccessed: 0,
      },
    ]);
    const canonical = new InMemoryCanonicalMemoryRepository();
    await canonical.init();

    const audit = await auditLegacyMemory({ vectorStore, metadataStore });
    expect(audit.vectorRows).toBe(2);
    expect(audit.matched).toBe(1);
    expect(audit.missingMeta).toEqual(["unscoped"]);
    expect(audit.scopeUnknown).toEqual(["unscoped"]);

    const dry = await migrateLegacyMemory({ vectorStore, metadataStore, canonical });
    expect(dry.dryRun).toBe(true);
    expect(await canonical.listRecords({ includeInactive: true })).toHaveLength(0);

    const done = await migrateLegacyMemory({ vectorStore, metadataStore, canonical }, { dryRun: false });
    expect(done.migratedActive).toBe(1);
    expect(done.migratedQuarantined).toBe(1);
    const active = await canonical.listRecords();
    expect(active.map((r) => r.fragmentId)).toEqual(["scoped"]);
    const all = await canonical.listRecords({ includeInactive: true });
    expect(all.find((r) => r.fragmentId === "unscoped")?.state).toBe("quarantined");
  });
});
