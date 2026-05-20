import { serve } from "@hono/node-server";
import { loadOpenintjEnv, summarizeLlmEnv } from "@openintj/shared";
import { assembleServerAgent } from "./agent.js";
import { buildApp } from "./routes.js";

loadOpenintjEnv({ logPrefix: "[OpenINTJ server env]" });

const PORT = Number.parseInt(process.env["PORT"] ?? "8788", 10);
const HOST = process.env["HOST"] ?? "127.0.0.1";

const main = async (): Promise<void> => {
  const provider =
    (process.env["LLM_PROVIDER"] as "ollama" | "hunyuan" | "mock" | undefined) ?? "mock";
  const envSummary = summarizeLlmEnv();
  console.log(`[OpenINTJ server] llm: ${envSummary.summary}`);
  if (provider === "hunyuan" && !envSummary.hunyuan.hasKey) {
    console.warn(
      "[OpenINTJ server] LLM_PROVIDER=hunyuan 但未读到 HUNYUAN_API_KEY —— 会自动降级 mock。",
    );
  }
  const agent = await assembleServerAgent({ llmProvider: provider });
  const app = buildApp(agent);

  serve({ fetch: app.fetch, hostname: HOST, port: PORT }, (info) => {
    console.log(
      `OpenINTJ server listening on http://${info.address}:${info.port} (provider=${provider})`,
    );
  });
};

main().catch((e) => {
  console.error("Server start failed:", e);
  process.exit(1);
});
