export interface SttTranscribeInput {
  audio: Uint8Array;
  /** e.g. audio/webm, audio/wav, audio/mpeg */
  mimeType: string;
  language?: string;
}

export interface SttResult {
  text: string;
  durationMs?: number;
}

export interface SttStatus {
  id: string;
  ok: boolean;
  detail?: string;
}

/** Speech-to-text provider (RFC-019). Audio only goes where env points it. */
export interface SpeechToTextProvider {
  id: string;
  transcribe(input: SttTranscribeInput): Promise<SttResult>;
  status(): SttStatus;
}
