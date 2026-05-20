/**
 * server 装配端：验证 enableOtel / OPENINTJ_OTEL=1 真的把 OTel 接到 hooks，
 * 用 InMemorySpanExporter 抓 span，跑一次真实 chat 看到完整 span 树。
 */
import { trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { assembleServerAgent } from "../src/agent.js";

let exporter: InMemorySpanExporter;
let provider: BasicTracerProvider;

beforeEach(() => {
  exporter = new InMemorySpanExporter();
  provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  trace.setGlobalTracerProvider(provider);
});

afterEach(async () => {
  await provider.shutdown();
  trace.disable();
});

describe("ServerAgent enableOtel wiring", () => {
  it("with enableOtel=true: agent.otel is attached and run() produces spans", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock", enableOtel: true });
    expect(agent.otel).toBeDefined();

    await agent.run("hello");
    await agent.close();

    const names = exporter.getFinishedSpans().map((s) => s.name);
    expect(names).toContain("openintj.tao.iteration");
  });

  it("with OPENINTJ_OTEL=1 env: same effect", async () => {
    process.env["OPENINTJ_OTEL"] = "1";
    try {
      const agent = await assembleServerAgent({ llmProvider: "mock" });
      expect(agent.otel).toBeDefined();
      await agent.run("hi");
      await agent.close();
      expect(exporter.getFinishedSpans().length).toBeGreaterThan(0);
    } finally {
      delete process.env["OPENINTJ_OTEL"];
    }
  });

  it("without any otel opt: agent.otel === undefined; no spans produced", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    expect(agent.otel).toBeUndefined();
    await agent.run("hi");
    await agent.close();
    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });

  it("explicit enableOtel=false beats env", async () => {
    process.env["OPENINTJ_OTEL"] = "1";
    try {
      const agent = await assembleServerAgent({ llmProvider: "mock", enableOtel: false });
      expect(agent.otel).toBeUndefined();
      await agent.run("hi");
      await agent.close();
      expect(exporter.getFinishedSpans()).toHaveLength(0);
    } finally {
      delete process.env["OPENINTJ_OTEL"];
    }
  });
});
