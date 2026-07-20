import type { ChatOptions, LlmClient } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { StrictLlmWrapper } from "../src/strict-wrapper.js";

const status = {
  provider: "test",
  model: "test",
  available: true,
  mode: "live" as const,
  status: "connected" as const,
  visionSupported: true,
};

describe("StrictLlmWrapper cancellation", () => {
  it.each(["chat", "visionChat"] as const)(
    "preserves the caller abort reason through %s",
    async (method) => {
      const inner: LlmClient = {
        chat: async (_messages, opts) => waitForAbort(opts),
        visionChat: async (_messages, _image, opts) => waitForAbort(opts),
        getStatus: () => status,
      };
      const wrapper = new StrictLlmWrapper(inner, "test");
      const controller = new AbortController();
      const reason = new Error("caller cancelled");
      const pending =
        method === "chat"
          ? wrapper.chat([{ role: "user", content: "wait" }], { signal: controller.signal })
          : wrapper.visionChat(
              [{ role: "user", content: "wait" }],
              { base64: "", mimeType: "image/png" },
              { signal: controller.signal },
            );

      controller.abort(reason);

      await expect(pending).rejects.toBe(reason);
    },
  );
});

const waitForAbort = async (opts?: ChatOptions): Promise<string> =>
  await new Promise<string>((_resolve, reject) => {
    opts?.signal?.addEventListener("abort", () => reject(opts.signal?.reason), { once: true });
  });
