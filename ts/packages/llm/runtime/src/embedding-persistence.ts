import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import type { EmbeddingProvider, HookBus } from "@openintj/core";
import {
  assertEmbeddingFingerprint,
  canonicalEmbeddingFingerprint,
  readEmbeddingFingerprint,
  writeEmbeddingFingerprint,
} from "./embedding-fingerprint.js";
import { ModelRuntimeError } from "./errors.js";
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
  opts: { hooks?: HookBus } = {},
): Promise<void> => {
  const expectedValue = canonicalEmbeddingFingerprint(expected);
  let stored: EmbeddingFingerprint | undefined;
  try {
    stored = await readEmbeddingFingerprint(dataDir);
  } catch (error) {
    const structured =
      error instanceof ModelRuntimeError
        ? error
        : new ModelRuntimeError({
            code: "EMBEDDING_FINGERPRINT_MISSING",
            message: error instanceof Error ? error.message : String(error),
            retriable: false,
            cause: error,
          });
    await opts.hooks?.emit("model.embedding.fingerprint.rejected", {
      expected: expectedValue,
      code: "EMBEDDING_FINGERPRINT_MISSING",
    });
    throw structured;
  }
  if (stored) {
    const storedValue = canonicalEmbeddingFingerprint(stored);
    try {
      assertEmbeddingFingerprint(expected, stored);
    } catch (error) {
      const structured =
        error instanceof ModelRuntimeError
          ? error
          : new ModelRuntimeError({
              code: "EMBEDDING_FINGERPRINT_MISMATCH",
              message: error instanceof Error ? error.message : String(error),
              retriable: false,
              cause: error,
            });
      await opts.hooks?.emit("model.embedding.fingerprint.rejected", {
        expected: expectedValue,
        stored: storedValue,
        code: structured.code as "EMBEDDING_FINGERPRINT_MISMATCH",
      });
      throw structured;
    }
    await opts.hooks?.emit("model.embedding.fingerprint.checked", {
      expected: expectedValue,
      stored: storedValue,
      result: "matched",
    });
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
    const error = new ModelRuntimeError({
      code: "EMBEDDING_FINGERPRINT_MISSING",
      message:
        "EMBEDDING_FINGERPRINT_MISSING: 持久化目录已有数据但缺少 embedding 指纹。请恢复原配置、使用新 OPENINTJ_DATA_DIR 或显式清空后重建。",
      retriable: false,
    });
    await opts.hooks?.emit("model.embedding.fingerprint.rejected", {
      expected: expectedValue,
      code: "EMBEDDING_FINGERPRINT_MISSING",
    });
    throw error;
  }
  await writeEmbeddingFingerprint(dataDir, expected);
  await opts.hooks?.emit("model.embedding.fingerprint.checked", {
    expected: expectedValue,
    result: "created",
  });
};
