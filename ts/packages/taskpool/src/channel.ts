import type { ZodType } from "zod";

export interface ChannelMessage<T> {
  readonly sequence: number;
  readonly value: T;
}

export type ChannelReducer<T, State> = (
  state: Readonly<State>,
  message: Readonly<ChannelMessage<T>>,
) => State;

/** Typed in-process channel with runtime validation and deterministic reduction. */
export class Channel<T, State = readonly T[]> {
  private sequence = 0;
  private current: State;
  private readonly listeners = new Set<(message: ChannelMessage<T>) => void>();

  constructor(
    private readonly schema: ZodType<T>,
    initial: State = [] as unknown as State,
    private readonly reducer: ChannelReducer<T, State> = ((state, message) => [
      ...(state as unknown as readonly T[]),
      message.value,
    ]) as ChannelReducer<T, State>,
  ) {
    this.current = initial;
  }

  send(input: unknown): ChannelMessage<T> {
    const value = this.schema.parse(input);
    const message = Object.freeze({ sequence: this.sequence++, value });
    this.current = this.reducer(this.current, message);
    for (const listener of this.listeners) listener(message);
    return message;
  }

  subscribe(listener: (message: ChannelMessage<T>) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  state(): Readonly<State> {
    return this.current;
  }
}
