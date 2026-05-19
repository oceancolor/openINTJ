import type { EmbeddingProvider } from "@openintj/core";
import { z } from "zod";

export const OllamaEmbedderConfigSchema = z.object({
  endpoint: z.string().default("http://127.0.0.1:11434"),
  model: z.string().default("nomic-embed-text"),
  /** 当 dimension 已知时设置；不设置则首次调用动态推断。 */
  dimension: z.number().int().positive().optional(),
  /** 单次请求超时（ms）；默认 30s。 */
  timeoutMs: z.number().int().positive().default(30_000),
});
export type OllamaEmbedderConfig = z.infer<typeof OllamaEmbedderConfigSchema>;

interface OllamaEmbedResponse {
  embedding: number[];
}

/**
 * OllamaEmbedder —— 通过 Ollama /api/embeddings 端点生成嵌入。
 *
 * - 默认模型：nomic-embed-text（768 维）
 * - 可选模型：mxbai-embed-large（1024 维）、all-minilm（384 维）
 * - 串行 batch（Ollama 单次调用一次性处理一个 prompt）
 *
 * 失败处理：抛错，由调用方决定是否回落到 SimpleEmbedder。
 */
export class OllamaEmbedder implements EmbeddingProvider {
  readonly name: string;
  private _dimension: number;
  readonly config: OllamaEmbedderConfig;

  constructor(config: Partial<OllamaEmbedderConfig> = {}) {
    this.config = OllamaEmbedderConfigSchema.parse(config);
    this.name = `ollama:${this.config.model}`;
    this._dimension = this.config.dimension ?? 0; // 0 表示尚未推断
  }

  get dimension(): number {
    return this._dimension;
  }

  async embed(text: string): Promise<number[]> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.config.timeoutMs);
    try {
      const res = await fetch(`${this.config.endpoint}/api/embeddings`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: this.config.model,
          prompt: text,
        }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`OllamaEmbedder ${res.status} ${res.statusText}: ${txt.slice(0, 200)}`);
      }
      const json = (await res.json()) as OllamaEmbedResponse;
      if (!Array.isArray(json.embedding) || json.embedding.length === 0) {
        throw new Error(`OllamaEmbedder: invalid embedding payload (model=${this.config.model})`);
      }
      if (this._dimension === 0) {
        this._dimension = json.embedding.length;
      } else if (this._dimension !== json.embedding.length) {
        throw new Error(
          `OllamaEmbedder dimension mismatch: expected ${this._dimension} got ${json.embedding.length}`,
        );
      }
      return json.embedding;
    } finally {
      clearTimeout(timer);
    }
  }

  async embedBatch(texts: readonly string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (const t of texts) {
      out.push(await this.embed(t));
    }
    return out;
  }

  /** 健康检查（quick ping，不消耗模型）。 */
  async healthCheck(): Promise<boolean> {
    try {
      const r = await fetch(`${this.config.endpoint}/api/tags`, {
        method: "GET",
      });
      return r.ok;
    } catch {
      return false;
    }
  }
}

export const loadOllamaEmbedderConfigFromEnv = (
  env: Record<string, string | undefined> = process.env,
): OllamaEmbedderConfig => {
  const cfg: Partial<OllamaEmbedderConfig> = {};
  if (env.OLLAMA_EMBED_ENDPOINT) cfg.endpoint = env.OLLAMA_EMBED_ENDPOINT;
  else if (env.OLLAMA_ENDPOINT) cfg.endpoint = env.OLLAMA_ENDPOINT;
  if (env.OLLAMA_EMBED_MODEL) cfg.model = env.OLLAMA_EMBED_MODEL;
  if (env.OLLAMA_EMBED_DIMENSION) {
    const d = Number.parseInt(env.OLLAMA_EMBED_DIMENSION, 10);
    if (Number.isFinite(d) && d > 0) cfg.dimension = d;
  }
  return OllamaEmbedderConfigSchema.parse(cfg);
};
