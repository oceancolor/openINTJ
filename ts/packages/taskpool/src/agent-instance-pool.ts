export interface AgentInstance<Role extends string = string> {
  readonly id: string;
  readonly role: Role;
}

export interface AgentLease<A extends AgentInstance> {
  readonly agent: A;
  release(): void;
}

/**
 * Role-aware bounded pool for expensive agent instances. This deliberately
 * does not reuse ObjectPool: role affinity and waiter fairness are semantic.
 */
export class AgentInstancePool<A extends AgentInstance> {
  private readonly available = new Map<A["role"], A[]>();
  private readonly counts = new Map<A["role"], number>();
  private readonly waiters = new Map<A["role"], Array<(lease: AgentLease<A>) => void>>();

  constructor(
    private readonly factory: (role: A["role"]) => Promise<A>,
    private readonly maxPerRole = 1,
  ) {
    if (!Number.isInteger(maxPerRole) || maxPerRole < 1) {
      throw new RangeError("maxPerRole must be a positive integer");
    }
  }

  async acquire(role: A["role"], signal?: AbortSignal): Promise<AgentLease<A>> {
    if (signal?.aborted) throw signal.reason ?? new Error("acquisition cancelled");
    const idle = this.available.get(role)?.shift();
    if (idle) return this.lease(idle);
    const count = this.counts.get(role) ?? 0;
    if (count < this.maxPerRole) {
      this.counts.set(role, count + 1);
      try {
        return this.lease(await this.factory(role));
      } catch (error) {
        this.counts.set(role, count);
        throw error;
      }
    }
    return new Promise<AgentLease<A>>((resolve, reject) => {
      const queue = this.waiters.get(role) ?? [];
      const waiter = (lease: AgentLease<A>): void => {
        signal?.removeEventListener("abort", onAbort);
        resolve(lease);
      };
      const onAbort = (): void => {
        const index = queue.indexOf(waiter);
        if (index >= 0) queue.splice(index, 1);
        reject(signal?.reason ?? new Error("acquisition cancelled"));
      };
      queue.push(waiter);
      this.waiters.set(role, queue);
      signal?.addEventListener("abort", onAbort, { once: true });
    });
  }

  private lease(agent: A): AgentLease<A> {
    let released = false;
    return {
      agent,
      release: () => {
        if (released) return;
        released = true;
        const waiter = this.waiters.get(agent.role)?.shift();
        if (waiter) waiter(this.lease(agent));
        else {
          const idle = this.available.get(agent.role) ?? [];
          idle.push(agent);
          this.available.set(agent.role, idle);
        }
      },
    };
  }

  stats(role: A["role"]): { created: number; available: number; waiting: number } {
    return {
      created: this.counts.get(role) ?? 0,
      available: this.available.get(role)?.length ?? 0,
      waiting: this.waiters.get(role)?.length ?? 0,
    };
  }
}
