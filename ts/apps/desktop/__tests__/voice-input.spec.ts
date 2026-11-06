import { describe, expect, it } from "vitest";
import {
  arrayBufferToBase64,
  describeVoiceError,
  pickRecorderMimeType,
} from "../src/renderer/components/useVoiceInput.js";

describe("voice input helpers (RFC-019)", () => {
  it("encodes ArrayBuffer to base64, including chunk boundaries", () => {
    const small = new TextEncoder().encode("hello 世界").buffer as ArrayBuffer;
    expect(arrayBufferToBase64(small)).toBe(Buffer.from("hello 世界", "utf8").toString("base64"));
    const large = new Uint8Array(100_000);
    for (let i = 0; i < large.length; i += 1) large[i] = i % 251;
    expect(arrayBufferToBase64(large.buffer)).toBe(Buffer.from(large).toString("base64"));
    expect(arrayBufferToBase64(new ArrayBuffer(0))).toBe("");
  });

  it("picks the first supported recorder mime type", () => {
    expect(pickRecorderMimeType((t) => t === "audio/webm")).toBe("audio/webm");
    expect(pickRecorderMimeType(() => true)).toBe("audio/webm;codecs=opus");
    expect(pickRecorderMimeType(() => false)).toBeUndefined();
    expect(pickRecorderMimeType(undefined)).toBeUndefined();
  });

  it("maps transcribe errors to actionable hints", () => {
    expect(describeVoiceError("stt_not_configured")).toContain("OPENINTJ_STT_URL");
    expect(describeVoiceError("audio_too_large")).toContain("太长");
    expect(describeVoiceError("boom")).toBe("转写失败：boom");
  });
});
