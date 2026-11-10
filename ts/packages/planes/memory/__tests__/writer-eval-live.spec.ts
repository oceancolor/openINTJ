import { describe, expect, it } from "vitest";
import { createLlmWriter, evaluateWriter } from "../src/index.js";

/**
 * G5.4 真实 writer 对照（手动执行，不进 CI）：
 * OPENINTJ_WRITER_EVAL=1 pnpm --filter @openintj/plane-memory test -- writer-eval-live
 * 默认模型 qwen2.5:7b（本机 Ollama，OPENINTJ_EVAL_MODEL 可换）。
 * 结果打印到 stdout，人工记入 RFC-021 的 G5.4 激活建议报告。
 */
const enabled = process.env["OPENINTJ_WRITER_EVAL"] === "1";
const model = process.env["OPENINTJ_EVAL_MODEL"] ?? "qwen2.5:7b";

const ollamaGenerate = async (prompt: string): Promise<string> => {
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      stream: false,
      messages: [{ role: "user", content: prompt }],
      options: { temperature: 0 },
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}`);
  const body = (await res.json()) as { message?: { content?: string } };
  return body.message?.content ?? "";
};

describe.skipIf(!enabled)("G5.4 live writer eval (manual)", () => {
  it(
    `通用 writer (${model}) 跑 §10 样本集`,
    async () => {
      const writer = createLlmWriter({ writerId: `llm:${model}`, generate: ollamaGenerate });
      const report = await evaluateWriter(writer);
      console.log(JSON.stringify(report, null, 2));
      expect(report.totals.samples).toBeGreaterThan(0);
    },
    600_000,
  );
});
