/**
 * Desktop agent 持久化模式 e2e。
 *
 * 不启动 Electron；只验证 assembleDesktopAgent 在 dataDir 下能正确装配真实 store
 * 并完成一轮 write → close → reopen → read。
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { assembleDesktopAgent } from "../src/main/agent.js";

const RUN_E2E = process.env["OPENINTJ_E2E"] === "1";
const describeE2E = RUN_E2E ? describe : describe.skip;

const fixtures: string[] = [];
const makeDir = (label: string): string => {
  const d = mkdtempSync(join(tmpdir(), `openintj-desk-${label}-`));
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

describe("desktop agent (memory mode default)", () => {
  it("不传 dataDir → memory 模式", async () => {
    const a = await assembleDesktopAgent({ llmProvider: "mock" });
    expect(a.persistenceInfo.mode).toBe("memory");
    expect(a.persistenceInfo.dataDir).toBeUndefined();
    await a.close();
  });

  it("OPENINTJ_DESKTOP_NO_PERSIST=1 强制 memory 模式（即使有 dataDir）", async () => {
    const dir = makeDir("nopersist");
    const prev = process.env["OPENINTJ_DESKTOP_NO_PERSIST"];
    process.env["OPENINTJ_DESKTOP_NO_PERSIST"] = "1";
    try {
      const a = await assembleDesktopAgent({
        llmProvider: "mock",
        dataDir: dir,
      });
      expect(a.persistenceInfo.mode).toBe("memory");
      await a.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_DESKTOP_NO_PERSIST"];
      else process.env["OPENINTJ_DESKTOP_NO_PERSIST"] = prev;
    }
  });

  it("status() 返回 persistence 信息", async () => {
    const a = await assembleDesktopAgent({ llmProvider: "mock" });
    const s = a.status();
    expect(s.persistence.mode).toBe("memory");
    await a.close();
  });
});

describeE2E("desktop agent (real persistence)", { timeout: 30_000 }, () => {
  it("dataDir → 写入 → 重启 → 读回", async () => {
    const dir = makeDir("roundtrip");
    const a1 = await assembleDesktopAgent({
      llmProvider: "mock",
      dataDir: dir,
    });
    expect(a1.persistenceInfo.mode).toBe("real");
    expect(a1.persistenceInfo.dataDir).toBe(dir);
    a1.persistentStore.addLongTerm("我会下围棋", { importance: 0.7 });
    await a1.persistentStore.awaitPendingWrites();
    await a1.close();

    const a2 = await assembleDesktopAgent({
      llmProvider: "mock",
      dataDir: dir,
    });
    const stats = a2.memory.getStats();
    expect(stats.counts.long_term).toBeGreaterThanOrEqual(1);
    const all = await a2.persistentStore.metadataStore.listFragmentMeta({});
    expect(all.some((m) => m.memoryType === "long_term")).toBe(true);
    await a2.close();
  });
});
