import { InternalizationManager, type InternalizationOpts } from "./internalization-manager.js";
import { PassiveStore } from "./passive-store.js";
import { PatternMiner, type PatternMinerOpts } from "./pattern-miner.js";
import type { DormantPersistenceAdapter } from "./persistence.js";
import type {
  DormantPattern,
  InternalizationProposal,
  PassiveEvent,
  PersonaConfig,
} from "./types.js";

/**
 * DormantRuntime —— Dormant Memory Learning 的三件套门面。
 *
 * 把 PassiveStore（捕获）/ PatternMiner（挖掘）/ InternalizationManager（审批写入）
 * 串成一个可由 server / desktop / cli 复用的运行时。
 *
 * 业务装配点（典型用法）：
 *  1. record(text, source)：每次用户输入 / agent 输出都喂进去
 *  2. mine()：周期性 / 用户触发；产出待审批 proposals
 *  3. listProposals / approve / reject：UI 与用户协作
 *  4. snapshot()：当前 PersonaConfig
 *
 * 持久化（Phase 3.4 #9.A）：可选注入 {@link DormantPersistenceAdapter}：
 *  - 启动后调用 `await runtime.hydrate()` 把历史 events / proposals / persona 拉回内存
 *  - 之后 record / propose / approve / reject / reset 自动写穿；adapter 同步路径
 *  - 关停调 `await runtime.close()` 释放底层句柄
 */
export interface DormantRuntimeOpts {
  /** PassiveStore 最大事件数，超过环形丢弃。 */
  maxPassiveEvents?: number;
  /** PatternMiner 配置。 */
  minerOpts?: Partial<PatternMinerOpts>;
  /** InternalizationManager 配置。 */
  internalizationOpts?: InternalizationOpts;
  /** PersonaConfig 初始值（持久化恢复时传入）。 */
  initialPersona?: Partial<PersonaConfig>;
  /** 每次 record 时自动生成 eventId 的前缀。 */
  eventIdPrefix?: string;
  /** 可选持久化适配器；不传则纯内存。 */
  adapter?: DormantPersistenceAdapter;
}

export interface DormantMineResult {
  patterns: DormantPattern[];
  proposals: InternalizationProposal[];
  /** 本次 mine 时 PassiveStore 中的事件总数。 */
  scannedEvents: number;
}

export class DormantRuntime {
  readonly passive: PassiveStore;
  readonly miner: PatternMiner;
  readonly internalization: InternalizationManager;
  readonly adapter?: DormantPersistenceAdapter;
  private readonly eventIdPrefix: string;
  private eventSeq = 0;

  constructor(opts: DormantRuntimeOpts = {}) {
    this.passive = new PassiveStore(opts.maxPassiveEvents ?? 10_000);
    this.miner = new PatternMiner(opts.minerOpts ?? {});
    this.internalization = new InternalizationManager(
      opts.initialPersona,
      opts.internalizationOpts ?? {},
    );
    this.eventIdPrefix = opts.eventIdPrefix ?? "evt";
    if (opts.adapter) this.adapter = opts.adapter;
  }

  /**
   * 从持久化层恢复历史状态。装配点应在构造完后、accept 第一条 record 前调用一次。
   * 无 adapter 时为 no-op。多次调用安全（每次都全量覆写）。
   */
  async hydrate(): Promise<void> {
    if (!this.adapter) return;
    const snap = await this.adapter.loadAll();
    this.passive.clear();
    this.passive.recordBulk(snap.events);
    this.internalization.restoreState(snap.proposals, snap.persona);
    this.eventSeq = snap.events.length;
  }

  /**
   * 记录一条被动事件。eventId 自动生成（前缀_序号_时间戳）。
   * metadata 可以塞 traceId / tool name 等便于事后定位。
   */
  record(
    text: string,
    source: PassiveEvent["source"],
    metadata: Record<string, unknown> = {},
  ): PassiveEvent {
    this.eventSeq += 1;
    const ts = Date.now();
    const event: PassiveEvent = {
      eventId: `${this.eventIdPrefix}_${this.eventSeq}_${ts}`,
      ts,
      source,
      text,
      metadata,
    };
    this.passive.record(event);
    this.adapter?.recordEvent(event);
    return event;
  }

  /** 触发一次挖掘 + 提案生成。 */
  async mine(): Promise<DormantMineResult> {
    const events = this.passive.exportAll();
    const patterns = await this.miner.mine(events);
    const proposals = this.internalization.proposeBatch(patterns.map((p) => ({ ...p })));
    for (const p of proposals) this.adapter?.upsertProposal(p);
    return { patterns, proposals, scannedEvents: events.length };
  }

  listProposals(status?: InternalizationProposal["status"]): InternalizationProposal[] {
    return this.internalization.listProposals(status);
  }

  approve(proposalId: string): InternalizationProposal | undefined {
    const p = this.internalization.approve(proposalId);
    if (p && this.adapter) {
      this.adapter.upsertProposal(p);
      this.adapter.savePersona(this.internalization.snapshot());
    }
    return p;
  }

  reject(proposalId: string): InternalizationProposal | undefined {
    const p = this.internalization.reject(proposalId);
    if (p) this.adapter?.upsertProposal(p);
    return p;
  }

  snapshot(): PersonaConfig {
    return this.internalization.snapshot();
  }

  /** 调试 / 测试用：当前 PassiveStore 事件数。 */
  passiveSize(): number {
    return this.passive.size();
  }

  /** 清空全部（PassiveStore + proposals + persona），仅测试或用户主动重置。 */
  reset(initialPersona?: Partial<PersonaConfig>): void {
    this.passive.clear();
    this.internalization.reset(initialPersona);
    this.eventSeq = 0;
    this.adapter?.clearAll();
  }

  /** 关停 —— 释放 adapter 句柄；无 adapter 时为 no-op。 */
  async close(): Promise<void> {
    if (this.adapter) await this.adapter.close();
  }
}
