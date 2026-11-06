import { createServer } from "node:http";
import { describe, expect, it } from "vitest";
import {
  ChannelDispatcher,
  channelsFromEnv,
  createMemoryChannel,
  createWebhookChannel,
} from "../src/index.js";

describe("ChannelDispatcher (RFC-018)", () => {
  it("fans out to webhook + memory and isolates failures", async () => {
    const received: unknown[] = [];
    const server = createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
      });
      req.on("end", () => {
        received.push(JSON.parse(body));
        res.writeHead(204).end();
      });
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = (server.address() as { port: number }).port;
    try {
      const dispatcher = new ChannelDispatcher();
      const memory = createMemoryChannel({ id: "mem" });
      dispatcher.register(memory);
      dispatcher.register(
        createWebhookChannel({ id: "hook", url: `http://127.0.0.1:${port}/push` }),
      );
      dispatcher.register(createWebhookChannel({ id: "bad", url: "http://127.0.0.1:1/down" }));
      const results = await dispatcher.dispatch({
        kind: "schedule.completed",
        title: "任务完成",
        body: "日报已生成",
        at: 1,
      });
      expect(results).toEqual([
        { channelId: "mem", ok: true },
        { channelId: "hook", ok: true },
        { channelId: "bad", ok: false, error: expect.any(String) },
      ]);
      expect(memory.events).toHaveLength(1);
      expect(received).toHaveLength(1);
      expect(dispatcher.list().find((s) => s.id === "bad")?.ok).toBe(false);
      const tested = await dispatcher.test("mem");
      expect(tested).toEqual({ channelId: "mem", ok: true });
      expect(memory.events.at(-1)?.kind).toBe("test");
    } finally {
      server.close();
    }
  });

  it("skips channels whose kinds filter does not match", async () => {
    const dispatcher = new ChannelDispatcher();
    const memory = createMemoryChannel({ id: "only-failed", kinds: ["schedule.failed"] });
    dispatcher.register(memory);
    const results = await dispatcher.dispatch({
      kind: "schedule.completed",
      title: "t",
      body: "b",
      at: 1,
    });
    expect(results).toEqual([{ channelId: "only-failed", ok: true, skipped: true }]);
    expect(memory.events).toHaveLength(0);
  });

  it("builds webhook channels from env and ignores malformed config", () => {
    const dispatcher = channelsFromEnv({
      OPENINTJ_CHANNELS: JSON.stringify([
        { id: "ntfy", kind: "webhook", url: "https://ntfy.sh/example" },
        { id: "broken", kind: "webhook" },
      ]),
    });
    expect(dispatcher.list().map((s) => s.id)).toEqual(["ntfy"]);
    expect(channelsFromEnv({ OPENINTJ_CHANNELS: "not-json" }).list()).toEqual([]);
    expect(channelsFromEnv({}).list()).toEqual([]);
  });
});
