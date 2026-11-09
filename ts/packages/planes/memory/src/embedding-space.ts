import type { EmbeddingProvider } from "@openintj/core";

/**
 * F07：embedding 空间身份。指纹 = provider/model/dimension/transformVersion，
 * 随索引代次一起记录；空间不兼容时走重建，正文与词法检索不中断。
 */
export interface EmbeddingSpace {
  provider: string;
  model: string;
  dimension: number;
  transformVersion: number;
  generation: number;
}

export const embeddingSpaceFingerprint = (s: Omit<EmbeddingSpace, "generation">): string =>
  `emb:v1:${s.provider}:${s.model}:${s.dimension}:t${s.transformVersion}`;

export const embeddingSpaceForEmbedder = (
  embedder: EmbeddingProvider,
  opts: { model?: string; transformVersion?: number; generation?: number } = {},
): EmbeddingSpace => {
  const space = {
    provider: embedder.name,
    model: opts.model ?? embedder.name,
    dimension: embedder.dimension,
    transformVersion: opts.transformVersion ?? 1,
  };
  return { ...space, generation: opts.generation ?? 1 };
};

export class EmbeddingSpaceMismatchError extends Error {
  readonly code = "EMBEDDING_SPACE_MISMATCH";
  constructor(
    readonly expected: string,
    readonly actual: string,
  ) {
    super(`embedding space mismatch: expected ${expected}, got ${actual}; rebuild the index generation`);
    this.name = "EmbeddingSpaceMismatchError";
  }
}

/** 维度是硬冲突：显式报错，绝不静默截断/填充。 */
export const assertEmbeddingCompatible = (space: EmbeddingSpace, embedder: EmbeddingProvider): void => {
  if (embedder.dimension !== space.dimension) {
    throw new EmbeddingSpaceMismatchError(
      embeddingSpaceFingerprint(space),
      embeddingSpaceFingerprint(embeddingSpaceForEmbedder(embedder)),
    );
  }
};
