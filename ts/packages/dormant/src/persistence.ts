import type { InternalizationProposal, PassiveEvent, PersonaConfig } from "./types.js";

/**
 * DormantPersistenceAdapter —— Dormant 子系统的持久化插槽。
 *
 * 接口契约：
 *  - `loadAll()` 在 `DormantRuntime.hydrate()` 中调用一次，把历史 events / proposals / persona 全量
 *    导回内存（PassiveStore + InternalizationManager）。
 *  - 热路径写入（`recordEvent` / `upsertProposal` / `savePersona` / `clearAll`）必须同步、不抛错；
 *    底层 better-sqlite3 同步即可，远端 IO 适配器请自行做 fire-and-forget 缓冲。
 *  - `close()` 仅释放底层句柄，不做最终 flush（热路径已经写穿了）。
 *
 * dormant 包本身不依赖任何具体存储，由 `@openintj/storage-sqlite` 等下游实现。
 */
export interface DormantPersistenceAdapter {
  readonly name: string;
  loadAll(): Promise<DormantSnapshot>;
  recordEvent(event: PassiveEvent): void;
  upsertProposal(proposal: InternalizationProposal): void;
  savePersona(persona: PersonaConfig): void;
  clearAll(): void;
  close(): Promise<void>;
}

export interface DormantSnapshot {
  events: PassiveEvent[];
  proposals: InternalizationProposal[];
  /** 未持久化过 persona 时返回 undefined，由 runtime 用默认值兜底。 */
  persona: PersonaConfig | undefined;
}

/**
 * 内存版适配器 —— 仅用于测试、CLI 短会话或显式选择"不落盘但保留接口对称"的场景。
 *
 * 行为等价于不挂适配器 + 自己管 events/proposals/persona 的 in-memory 镜像。
 */
export class InMemoryDormantStore implements DormantPersistenceAdapter {
  readonly name = "in-memory-dormant";
  private events: PassiveEvent[] = [];
  private proposals = new Map<string, InternalizationProposal>();
  private persona: PersonaConfig | undefined;

  async loadAll(): Promise<DormantSnapshot> {
    return {
      events: this.events.map((e) => ({ ...e, metadata: { ...e.metadata } })),
      proposals: [...this.proposals.values()].map((p) => ({ ...p })),
      persona: this.persona
        ? (JSON.parse(JSON.stringify(this.persona)) as PersonaConfig)
        : undefined,
    };
  }

  recordEvent(event: PassiveEvent): void {
    this.events.push({ ...event, metadata: { ...event.metadata } });
  }

  upsertProposal(proposal: InternalizationProposal): void {
    this.proposals.set(proposal.proposalId, { ...proposal });
  }

  savePersona(persona: PersonaConfig): void {
    this.persona = JSON.parse(JSON.stringify(persona)) as PersonaConfig;
  }

  clearAll(): void {
    this.events.length = 0;
    this.proposals.clear();
    this.persona = undefined;
  }

  async close(): Promise<void> {
    this.clearAll();
  }
}
