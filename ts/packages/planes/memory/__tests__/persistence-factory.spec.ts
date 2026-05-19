/**
 * createPersistentMemoryStore 工厂测试。
 *
 * 真实磁盘部分（LanceDB + SQLite）由 OPENINTJ_E2E=1 控制；CI 默认仅跑 in-memory 路径。
 */
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import {
  buildPersistenceBackends,
  createPersistentMemoryStore,
} from "../src/persistence-factory.js";

const fixtures: string[] = [];
const makeDir = (label: string): string => {
  const d = mkdtempSync(join(tmpdir(), `openintj-fac-${label}-`));
  fixtures.push(d);
  return d;
};
afterAll(() => {
  for (const d of fixtures) {
    try {
      rmSync(d, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
});

describe("createPersistentMemoryStore (in-memory path)", () => {
  it("默认无 dataDir 走 memory 模式", async () => {
    const backends = await buildPersistenceBackends();
    expect(backends.mode).toBe("memory");
    expect(backends.dataDir).toBeUndefined();
    const store = await createPersistentMemoryStore();
    store.addLongTerm("hi", { importance: 0.5 });
    await store.awaitPendingWrites();
    const meta = await store.metadataStore.listFragmentMeta({});
    expect(meta.length).toBe(1);
    await store.close();
  });

  it("mode='memory' 显式覆盖 dataDir", async () => {
    const backends = await buildPersistenceBackends({
      mode: "memory",
      dataDir: "/tmp/should-not-be-used",
    });
    expect(backends.mode).toBe("memory");
  });

  it("mode='real' 但缺 dataDir 抛错", async () => {
    await expect(buildPersistenceBackends({ mode: "real" })).rejects.toThrow(/dataDir/);
  });
});

const RUN_E2E = process.env["OPENINTJ_E2E"] === "1";
const describeReal = RUN_E2E ? describe : describe.skip;

describeReal("createPersistentMemoryStore (real disk)", { timeout: 30_000 }, () => {
  it("dataDir 自动创建子目录 + 文件", async () => {
    const dir = makeDir("auto");
    const backends = await buildPersistenceBackends({ dataDir: dir });
    expect(backends.mode).toBe("real");
    expect(backends.dataDir).toBe(dir);
    expect(existsSync(join(dir, "lancedb"))).toBe(true);
    await backends.vectorStore.init();
    await backends.metadataStore.init();
    await backends.metadataStore.migrate();
    expect(existsSync(join(dir, "metadata.db"))).toBe(true);
    await backends.vectorStore.close();
    await backends.metadataStore.close();
  });

  it("write → close → reopen → read back", async () => {
    const dir = makeDir("rw");

    const s1 = await createPersistentMemoryStore({
      dataDir: dir,
      embeddingDim: 32,
    });
    s1.addLongTerm("绿茶 是健康饮品", {
      taskTags: ["health"],
      importance: 0.8,
    });
    s1.addWorking("项目代号 Atlas", { importance: 0.6 });
    await s1.awaitPendingWrites();
    expect(await s1.vectorStore.count()).toBe(2);
    await s1.close();

    const s2 = await createPersistentMemoryStore({
      dataDir: dir,
      embeddingDim: 32,
    });
    expect(await s2.vectorStore.count()).toBe(2);

    const meta = await s2.metadataStore.listFragmentMeta({});
    expect(meta.length).toBe(2);
    expect(meta.some((m) => m.taskTagsCsv === "health")).toBe(true);

    const vec = s2.embedder.embed("茶");
    const queryVec = Array.isArray(vec) ? vec : await vec;
    const hits = await s2.vectorSearch(queryVec, { topK: 5 });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.some((h) => h.row.content.includes("绿茶"))).toBe(true);

    await s2.close();
  });
});
