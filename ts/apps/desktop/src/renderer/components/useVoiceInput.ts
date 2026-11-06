import React from "react";
import type { VoiceTranscribeRequest } from "../../shared/ipc-protocol.js";

export type VoiceInputState = "idle" | "recording" | "transcribing";

/** Hard cap per take; the main process also caps payload size (~25MB). */
export const MAX_RECORDING_MS = 120_000;

const RECORDER_MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
] as const;

export const pickRecorderMimeType = (
  isTypeSupported: ((mimeType: string) => boolean) | undefined,
): string | undefined => {
  if (!isTypeSupported) return undefined;
  return RECORDER_MIME_CANDIDATES.find((candidate) => isTypeSupported(candidate));
};

/** Chunked base64 (spread on large buffers would blow the call stack). */
export const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
};

export const describeVoiceError = (error: string): string => {
  if (error === "stt_not_configured") {
    return "语音转写未配置：在启动环境设置 OPENINTJ_STT_URL（或 OPENINTJ_STT=mock）后重启应用";
  }
  if (error === "audio_too_large") return "录音太长了，请缩短后再试";
  return `转写失败：${error}`;
};

type TranscribeFn = (
  req: VoiceTranscribeRequest,
) => Promise<{ text: string; durationMs?: number } | { error: string }>;

const defaultTranscribe: TranscribeFn = (req) => {
  const api = window.openintj;
  if (!api?.voiceTranscribe) return Promise.resolve({ error: "voice_unavailable" });
  return api.voiceTranscribe(req);
};

export interface VoiceInput {
  state: VoiceInputState;
  elapsedS: number;
  error: string | undefined;
  supported: boolean;
  toggle: () => void;
}

/**
 * RFC-019 renderer recording: getUserMedia + MediaRecorder, toggle to
 * start/stop, transcript is handed to `onText` (the caller appends it to
 * the input box; nothing is auto-sent).
 */
export const useVoiceInput = (opts: {
  onText: (text: string) => void;
  transcribe?: TranscribeFn;
}): VoiceInput => {
  const [state, setState] = React.useState<VoiceInputState>("idle");
  const [elapsedS, setElapsedS] = React.useState(0);
  const [error, setError] = React.useState<string | undefined>();
  const recorderRef = React.useRef<MediaRecorder | undefined>(undefined);
  const streamRef = React.useRef<MediaStream | undefined>(undefined);
  const chunksRef = React.useRef<Blob[]>([]);
  const timerRef = React.useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const startedAtRef = React.useRef(0);
  const onTextRef = React.useRef(opts.onText);
  onTextRef.current = opts.onText;
  const transcribeRef = React.useRef<TranscribeFn>(opts.transcribe ?? defaultTranscribe);
  transcribeRef.current = opts.transcribe ?? defaultTranscribe;

  const supported =
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function" &&
    typeof MediaRecorder !== "undefined";

  const cleanup = React.useCallback((): void => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = undefined;
    for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    streamRef.current = undefined;
    recorderRef.current = undefined;
  }, []);

  React.useEffect(() => cleanup, [cleanup]);

  const stop = React.useCallback((): void => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  const start = React.useCallback(async (): Promise<void> => {
    setError(undefined);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("无法访问麦克风（权限被拒绝或没有可用设备）");
      return;
    }
    streamRef.current = stream;
    chunksRef.current = [];
    const mimeType = pickRecorderMimeType((t) => MediaRecorder.isTypeSupported(t));
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorderRef.current = recorder;
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      const blob = new Blob(chunksRef.current, {
        type: recorder.mimeType || mimeType || "audio/webm",
      });
      cleanup();
      setElapsedS(0);
      if (blob.size === 0) {
        setState("idle");
        return;
      }
      setState("transcribing");
      void (async () => {
        try {
          const audioBase64 = arrayBufferToBase64(await blob.arrayBuffer());
          const result = await transcribeRef.current({
            audioBase64,
            mimeType: blob.type.split(";")[0] || "audio/webm",
          });
          if ("error" in result) setError(describeVoiceError(result.error));
          else if (result.text.trim()) onTextRef.current(result.text.trim());
        } catch (e) {
          setError(`转写失败：${(e as Error).message}`);
        } finally {
          setState("idle");
        }
      })();
    });
    startedAtRef.current = Date.now();
    setElapsedS(0);
    setState("recording");
    recorder.start(250);
    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startedAtRef.current;
      setElapsedS(Math.floor(elapsed / 1000));
      if (elapsed >= MAX_RECORDING_MS) stop();
    }, 250);
  }, [cleanup, stop]);

  const toggle = React.useCallback((): void => {
    if (state === "recording") stop();
    else if (state === "idle") void start();
  }, [state, start, stop]);

  return { state, elapsedS, error, supported, toggle };
};
