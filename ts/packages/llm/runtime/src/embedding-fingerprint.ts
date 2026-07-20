import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { ModelRuntimeError } from "./errors.js";
import type { EmbeddingFingerprint } from "./types.js";
import { EMBEDDING_FINGERPRINT_FILENAME } from "./types.js";

export const fingerprintPath = (dataDir: string): string =>
  path.join(dataDir, EMBEDDING_FINGERPRINT_FILENAME);

export const canonicalEmbeddingFingerprint = (fp: EmbeddingFingerprint): string =>
  `v${fp.schemaVersion}:${fp.provider}:${fp.model}:${fp.dimension}`;

export const readEmbeddingFingerprint = async (
  dataDir: string,
): Promise<EmbeddingFingerprint | undefined> => {
  try {
    const raw = await readFile(fingerprintPath(dataDir), "utf8");
    const parsed = JSON.parse(raw) as EmbeddingFingerprint;
    if (
      parsed.schemaVersion === 1 &&
      typeof parsed.provider === "string" &&
      typeof parsed.model === "string" &&
      typeof parsed.dimension === "number" &&
      parsed.dimension > 0
    ) {
      return parsed;
    }
    throw new ModelRuntimeError({
      code: "EMBEDDING_FINGERPRINT_MISSING",
      message: "EMBEDDING_FINGERPRINT_MISSING: embedding fingerprint is invalid",
      retriable: false,
    });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
};

export const writeEmbeddingFingerprint = async (
  dataDir: string,
  fp: EmbeddingFingerprint,
): Promise<void> => {
  await mkdir(dataDir, { recursive: true });
  await writeFile(fingerprintPath(dataDir), `${JSON.stringify(fp, null, 2)}\n`, "utf8");
};

/** 校验期望指纹与磁盘一致；旧库无指纹时若已有向量数据则拒绝（fail closed）。 */
export const assertEmbeddingFingerprint = (
  expected: EmbeddingFingerprint,
  stored: EmbeddingFingerprint | undefined,
  opts: { hasExistingVectors?: boolean } = {},
): void => {
  if (!stored) {
    if (opts.hasExistingVectors) {
      throw new ModelRuntimeError({
        code: "EMBEDDING_FINGERPRINT_MISSING",
        message:
          "EMBEDDING_FINGERPRINT_MISSING: 持久化目录缺少 embedding 指纹，但检测到已有向量数据。请使用新 OPENINTJ_DATA_DIR 或恢复创建该库时的 embed 配置后重试。",
        retriable: false,
      });
    }
    return;
  }
  if (
    stored.schemaVersion !== expected.schemaVersion ||
    stored.provider !== expected.provider ||
    stored.model !== expected.model ||
    stored.dimension !== expected.dimension
  ) {
    throw new ModelRuntimeError({
      code: "EMBEDDING_FINGERPRINT_MISMATCH",
      message:
        `EMBEDDING_FINGERPRINT_MISMATCH: 磁盘=${canonicalEmbeddingFingerprint(stored)}，` +
        `当前=${canonicalEmbeddingFingerprint(expected)}。请换新 dataDir 或清空后重建。`,
      retriable: false,
    });
  }
};
