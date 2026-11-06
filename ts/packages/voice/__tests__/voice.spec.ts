import { type Server, createServer } from "node:http";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createHttpSttProvider, createMockSttProvider, sttFromEnv } from "../src/index.js";

describe("voice providers", () => {
  let server: Server;
  let baseUrl = "";

  beforeAll(async () => {
    server = createServer((req, res) => {
      if (req.url === "/v1/audio/transcriptions" && req.method === "POST") {
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          const body = Buffer.concat(chunks).toString("utf8");
          const ok = body.includes("whisper-1") && req.headers.authorization === "Bearer k";
          res.writeHead(ok ? 200 : 400, { "content-type": "application/json" });
          res.end(JSON.stringify(ok ? { text: "转写文本" } : { error: "bad" }));
        });
        return;
      }
      res.writeHead(404).end();
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    const addr = server.address();
    baseUrl = `http://127.0.0.1:${typeof addr === "object" && addr !== null ? addr.port : 0}/v1`;
  });
  afterAll(async () => {
    await new Promise<void>((r) => server.close(() => r()));
  });

  it("mock provider returns fixed text", async () => {
    const p = createMockSttProvider("hi");
    await expect(
      p.transcribe({ audio: new Uint8Array([1]), mimeType: "audio/wav" }),
    ).resolves.toEqual({ text: "hi" });
  });

  it("http provider posts multipart and parses text", async () => {
    const p = createHttpSttProvider({ baseUrl, apiKey: "k" });
    const out = await p.transcribe({ audio: new Uint8Array([1, 2]), mimeType: "audio/webm" });
    expect(out.text).toBe("转写文本");
  });

  it("env selection: url wins, mock opt-in, unset undefined", () => {
    expect(sttFromEnv({})).toBeUndefined();
    expect(sttFromEnv({ OPENINTJ_STT: "mock" })?.id).toBe("mock");
    expect(sttFromEnv({ OPENINTJ_STT_URL: "http://x/v1" })?.id).toBe("http");
  });
});
