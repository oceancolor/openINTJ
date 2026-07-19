import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import type { EmbeddingProvider } from "@openintj/core";
import {
  assertEmbeddingFingerprint,
  readEmbeddingFingerprint,
  writeEmbeddingFingerprint,
} from "./embedding-fingerprint.js";
import { resolveEmbedder } from "./resolve-embedder.js";
import type { EmbeddingFingerprint, ResolveEmbedOpts } from "./types.js";

export interface PreparedEmbed {
  embedder: EmbeddingProvider;
  dimension: number;
  fingerprint: EmbeddingFingerprint;
  status: Awaited<ReturnType<typeof resolveEmbedder>>["status"];
}

/** 解析 embedder 并生成指纹（装配前调用）。 */
export const prepareEmbedder = async (opts: ResolveEmbedOpts = {}): Promise<PreparedEmbed> => {
  const resolved = await resolveEmbedder(opts);
  const fingerprint: EmbeddingFingerprint = {
    schemaVersion: 1,
    provider: resolved.status.provider,
    model: resolved.status.model,
    dimension: resolved.dimension,
  };
  return {
    embedder: resolved.embedder,
    dimension: resolved.dimension,
    fingerprint,
    status: resolved.status,
  };
};

/**
 * 真盘模式：打开前校验指纹；新建库则在首次无片段时写入指纹。
 * `fragmentCount` 来自 metadata（hydrate 前或后均可，由调用方传入）。
 */
export const validateEmbeddingFingerprintForDataDir = async (
  dataDir: string,
  expected: EmbeddingFingerprint,
  fragmentCount?: number,
): Promise<void> => {
  const stored = await readEmbeddingFingerprint(dataDir);
  if (stored) {
    assertEmbeddingFingerprint(expected, stored);
    return;
  }
  let hasExistingData = (fragmentCount ?? 0) > 0;
  if (!hasExistingData) {
    const candidates = [path.join(dataDir, "lancedb"), path.join(dataDir, "metadata.db")];
    for (const candidate of candidates) {
      try {
        const info = await stat(candidate);
        if (info.isFile() ? info.size > 0 : (await readdir(candidate)).length > 0) {
          hasExistingData = true;
          break;
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
    }
  }
  if (hasExistingData) {
    throw new Error(
      "EMBEDDING_FINGERPRINT_MISSING: 持久化目录已有数据但缺少 embedding 指纹。请恢复原配置、使用新 OPENINTJ_DATA_DIR 或显式清空后重建。",
    );
  }
  await writeEmbeddingFingerprint(dataDir, expected);
};
