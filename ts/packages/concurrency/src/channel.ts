/**
 * Channel —— 缓冲通道（Go-style）。
 *
 * - bufferSize=0：无缓冲（同步握手），send 阻塞直到 recv
 * - bufferSize>0：环形缓冲；满时 send 阻塞
 * - 关闭后 send 抛错；recv 拿到 done=true（剩余项耗尽后）
 */
export interface ChannelMessage<T> {
  value: T | undefined;
  done: boolean;
}

export class Channel<T> {
  private buffer: T[] = [];
  readonly capacity: number;
  private closed = false;
  private sendWaiters: Array<{ value: T; resolve: () => void; reject: (e: Error) => void }> = [];
  private recvWaiters: Array<(msg: ChannelMessage<T>) => void> = [];

  constructor(capacity = 0) {
    this.capacity = Math.max(0, Math.floor(capacity));
  }

  async send(value: T): Promise<void> {
    if (this.closed) throw new Error("Channel: cannot send on closed channel");

    // 有等待 recv 的：直接交付
    const recv = this.recvWaiters.shift();
    if (recv) {
      recv({ value, done: false });
      return;
    }

    // buffer 还有空：入队
    if (this.buffer.length < this.capacity) {
      this.buffer.push(value);
      return;
    }

    // 否则阻塞
    return new Promise<void>((resolve, reject) => {
      this.sendWaiters.push({ value, resolve, reject });
    });
  }

  async recv(): Promise<ChannelMessage<T>> {
    // buffer 有：先消费
    if (this.buffer.length > 0) {
      const value = this.buffer.shift();
      // 唤醒一个被阻塞的 send 把它的值放进 buffer
      const sender = this.sendWaiters.shift();
      if (sender) {
        this.buffer.push(sender.value);
        sender.resolve();
      }
      return { value, done: false };
    }

    // buffer 空 + 有 sender 等：直接握手
    const sender = this.sendWaiters.shift();
    if (sender) {
      sender.resolve();
      return { value: sender.value, done: false };
    }

    // 关闭且空：done
    if (this.closed) {
      return { value: undefined, done: true };
    }

    // 否则阻塞
    return new Promise<ChannelMessage<T>>((resolve) => {
      this.recvWaiters.push(resolve);
    });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const w of this.recvWaiters) w({ value: undefined, done: true });
    this.recvWaiters = [];
    for (const s of this.sendWaiters) s.reject(new Error("Channel: closed during send"));
    this.sendWaiters = [];
  }

  get isClosed(): boolean {
    return this.closed;
  }

  get pendingCount(): number {
    return this.buffer.length;
  }

  /** AsyncIterable 适配，可用 for-await-of。 */
  async *[Symbol.asyncIterator](): AsyncIterableIterator<T> {
    while (true) {
      const m = await this.recv();
      if (m.done) return;
      yield m.value as T;
    }
  }
}
