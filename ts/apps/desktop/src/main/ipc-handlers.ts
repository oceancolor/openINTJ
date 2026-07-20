import { type FSWatcher, watch } from "node:fs";
import { createToolCallGate } from "@openintj/plane-governance";
import { withRootSpan } from "@openintj/telemetry-otel";
import { type IpcMain, type WebContents, ipcMain } from "electron";
import {
  AppConfigPatchSchema,
  ChatRequestSchema,
  DEFAULT_MODEL_PROFILES,
  DormantListRequestSchema,
  DormantProposalDecisionSchema,
  IPC,
  MemoryQueryRequestSchema,
  ModelCredentialSetSchema,
  SkillListRequestSchema,
  SkillProposalDecisionSchema,
  WorkspaceReadRequestSchema,
  WorkspaceWriteRequestSchema,
} from "../shared/ipc-protocol.js";
import type { DesktopAgent } from "./agent.js";
import type { ConfigService } from "./config-store.js";
import type { CredentialStore } from "./credential-store.js";

const DORMANT_NOT_ENABLED = {
  error: "dormant_not_enabled",
  hint: "Pass { enableDormant: true } to assembleDesktopAgent or set OPENINTJ_DORMANT=1",
};

const SKILLS_LEARNING_NOT_ENABLED = {
  error: "skills_learning_not_enabled",
  hint: "Pass { enableSkillLearning: true } to assembleDesktopAgent or set OPENINTJ_SKILLS_LEARN=1",
};

export interface IpcRegistration {
  unregister(): void;
}

/** 主进程注入的可选依赖（隔离 electron 专有能力，便于单测桩替换）。 */
export interface IpcDeps {
  /**
   * pickWorkspaceDir 的实现（通常注入 electron `dialog.showOpenDialog`）。
   * 不传 → WORKSPACE_PICK_DIR 总是返回 { canceled: true }。
   */
  pickDirectory?: () => Promise<string | null>;
  /** 应用配置服务；不传 → config.* channel 走内存空配置、不持久化。 */
  config?: ConfigService;
  /** Gracefully close state and relaunch the Electron process. */
  restart?: () => Promise<void>;
  /** Encrypted model API keys; never returned to renderer. */
  credentials?: CredentialStore;
  /** Environment-provided credentials count as configured but remain outside the store. */
  credentialEnv?: Partial<Record<"hunyuan" | "kimi" | "minimax" | "glm", boolean>>;
}

const workspaceError = (e: unknown): { error: "workspace_error"; message: string } => ({
  error: "workspace_error",
  message: e instanceof Error ? e.message : String(e),
});

/** 注册所有 IPC channel handler，并把 hook 事件转发为 webContents.send。 */
export const registerIpcHandlers = (
  agent: DesktopAgent,
  webContents?: WebContents,
  api: IpcMain = ipcMain,
  deps: IpcDeps = {},
): IpcRegistration => {
  const offs: Array<() => void> = [];

  api.handle(IPC.PING, async () => ({ ok: true, ts: Date.now() }));

  api.handle(IPC.STATUS, async () => {
    await agent.refreshModelRuntime();
    return agent.status();
  });

  api.handle(IPC.APP_RESTART, async () => {
    if (!deps.restart) return { ok: false, reason: "restart_unavailable" };
    await deps.restart();
    return { ok: true };
  });

  api.handle(IPC.MODEL_PROFILES, async () => {
    const custom = deps.config?.get().modelProfiles ?? [];
    const merged = new Map(
      [...DEFAULT_MODEL_PROFILES, ...custom].map((profile) => [profile.id, profile]),
    );
    return [...merged.values()].map((profile) => ({
      ...profile,
      hasCredential:
        profile.hasCredential === true ||
        deps.credentials?.has(profile.id) === true ||
        (profile.provider !== "auto" &&
          profile.provider !== "ollama" &&
          profile.provider !== "mock" &&
          deps.credentialEnv?.[profile.provider] === true),
    }));
  });

  api.handle(IPC.MODEL_CREDENTIAL_SET, async (_evt, raw: unknown) => {
    const parsed = ModelCredentialSetSchema.safeParse(raw);
    if (!parsed.success) return { ok: false, error: "invalid_request" };
    if (!deps.credentials) return { ok: false, error: "credential_store_unavailable" };
    deps.credentials.set(parsed.data.profileId, parsed.data.apiKey);
    return { ok: true };
  });

  api.handle(IPC.MODEL_CREDENTIAL_DELETE, async (_evt, profileId: unknown) => {
    if (typeof profileId !== "string") return { ok: false, error: "invalid_request" };
    if (!deps.credentials) return { ok: false, error: "credential_store_unavailable" };
    return { ok: true, deleted: deps.credentials.delete(profileId) };
  });

  api.handle(IPC.CHAT, async (_evt, raw: unknown) => {
    const parsed = ChatRequestSchema.safeParse(raw);
    if (!parsed.success) {
      return { error: "invalid_request", issues: parsed.error.issues };
    }
    const result = await withRootSpan("openintj.ipc.chat", () => agent.run(parsed.data.query), {
      attributes: { "ipc.channel": IPC.CHAT },
    });
    return {
      finalAnswer: result.finalAnswer,
      iterations: result.iterations,
      status: result.status,
      traceId: result.traceId,
    };
  });

  api.handle(IPC.MEMORY_QUERY, async (_evt, raw: unknown) => {
    const parsed = MemoryQueryRequestSchema.safeParse(raw);
    if (!parsed.success) {
      return { error: "invalid_request", issues: parsed.error.issues };
    }
    const { query, topK, mode: modeOverride, rrf } = parsed.data;
    const mode = modeOverride ?? agent.retrievalMode;
    if (typeof query === "string" && query.length > 0) {
      if (mode === "hybrid") {
        const hits = await agent.retrieveHybrid(query, {
          topK,
          ...(rrf !== undefined ? { config: { useRRF: rrf } } : {}),
        });
        return hits.map((h) => ({
          fragmentId: h.doc.id,
          content: h.doc.text,
          score: h.score,
          memoryType: h.doc.metadata.memoryType,
          taskTags: h.doc.metadata.taskTags,
        }));
      }
      const ranked = await agent.memory.retrieve(query, { topK });
      return ranked.map((r) => ({
        fragmentId: r.fragment.fragmentId,
        content: r.fragment.content,
        score: r.score,
        memoryType: r.fragment.memoryType,
        taskTags: r.fragment.taskTags,
      }));
    }
    const list = await agent.persistentStore.metadataStore.listFragmentMeta({
      limit: topK,
    });
    return list.map((m) => ({
      fragmentId: m.fragmentId,
      content: "",
      memoryType: m.memoryType,
      taskTags: m.taskTagsCsv ? m.taskTagsCsv.split(",") : [],
    }));
  });

  api.handle(IPC.AUDIT_RECENT, async (_evt, rawLimit: unknown) => {
    const limit =
      typeof rawLimit === "number" && Number.isFinite(rawLimit)
        ? Math.min(500, Math.max(1, Math.floor(rawLimit)))
        : 100;
    return {
      stats: agent.governance.auditTrail.getStats(),
      recent: agent.governance.auditTrail.query({ limit }),
    };
  });

  // ---------- RFC-004 §8 工作区系统能力面 ----------
  api.handle(IPC.WORKSPACE_INFO, async () => ({
    root: agent.workspace.config.root,
    enableCommands: agent.workspace.config.enableCommands,
    allowedCommands: agent.workspace.config.allowedCommands,
  }));

  // 与 agent 工具路径同一治理闸门：IPC 直连沙箱也要过策略 + 配额（RFC-004 §8）。
  const workspaceGate = createToolCallGate(agent.governance);

  api.handle(IPC.WORKSPACE_READ, async (_evt, raw: unknown) => {
    const parsed = WorkspaceReadRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      await workspaceGate({ tool: "read_file", params: { path: parsed.data.path } });
      // 复用 Agent read_file 工具的同一沙箱（路径越界 / 过大都在工具内拦截）。
      return await agent.workspace.tools.readFile({ path: parsed.data.path });
    } catch (e) {
      return workspaceError(e);
    }
  });

  api.handle(IPC.WORKSPACE_WRITE, async (_evt, raw: unknown) => {
    const parsed = WorkspaceWriteRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      await workspaceGate({
        tool: "write_file",
        params: { path: parsed.data.path, content: parsed.data.content },
      });
      return await agent.workspace.tools.writeFile({
        path: parsed.data.path,
        content: parsed.data.content,
      });
    } catch (e) {
      return workspaceError(e);
    }
  });

  api.handle(IPC.WORKSPACE_PICK_DIR, async () => {
    if (!deps.pickDirectory) return { canceled: true };
    try {
      const root = await deps.pickDirectory();
      return root ? { canceled: false, root } : { canceled: true };
    } catch (e) {
      return workspaceError(e);
    }
  });

  // ---------- 应用配置面 ----------
  api.handle(IPC.CONFIG_GET, async () => (deps.config ? deps.config.get() : {}));

  api.handle(IPC.CONFIG_UPDATE, async (_evt, raw: unknown) => {
    const parsed = AppConfigPatchSchema.safeParse(raw ?? {});
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    if (!deps.config) return parsed.data;
    try {
      return deps.config.update(parsed.data);
    } catch (e) {
      return { error: "invalid_request", message: e instanceof Error ? e.message : String(e) };
    }
  });

  // RFC-003 方向 3：Dormant Memory Learning IPC。
  api.handle(IPC.DORMANT_MINE, async () => {
    if (!agent.dormant) return DORMANT_NOT_ENABLED;
    const r = await agent.dormant.mine();
    return {
      scannedEvents: r.scannedEvents,
      patterns: r.patterns.map((p) => ({
        patternId: p.patternId,
        description: p.description,
        category: p.category,
        frequency: p.frequency,
        confidence: p.confidence,
      })),
      proposals: r.proposals.map((p) => ({
        proposalId: p.proposalId,
        targetField: p.targetField,
        value: p.value,
        status: p.status,
        patternDescription: p.pattern.description,
      })),
    };
  });

  api.handle(IPC.DORMANT_LIST, async (_evt, raw: unknown) => {
    if (!agent.dormant) return DORMANT_NOT_ENABLED;
    const parsed = DormantListRequestSchema.safeParse(raw ?? {});
    if (!parsed.success) {
      return { error: "invalid_request", issues: parsed.error.issues };
    }
    const list = agent.dormant.listProposals(parsed.data.status);
    return {
      total: list.length,
      proposals: list.map((p) => ({
        proposalId: p.proposalId,
        targetField: p.targetField,
        value: p.value,
        status: p.status,
        ts: p.ts,
        decidedAt: p.decidedAt,
        patternDescription: p.pattern.description,
        confidence: p.pattern.confidence,
        frequency: p.pattern.frequency,
      })),
    };
  });

  api.handle(IPC.DORMANT_APPROVE, async (_evt, raw: unknown) => {
    if (!agent.dormant) return DORMANT_NOT_ENABLED;
    const parsed = DormantProposalDecisionSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const out = agent.dormant.approve(parsed.data.proposalId);
    if (!out) return { error: "not_found_or_already_decided" };
    return { proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt };
  });

  api.handle(IPC.DORMANT_REJECT, async (_evt, raw: unknown) => {
    if (!agent.dormant) return DORMANT_NOT_ENABLED;
    const parsed = DormantProposalDecisionSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const out = agent.dormant.reject(parsed.data.proposalId);
    if (!out) return { error: "not_found_or_already_decided" };
    return { proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt };
  });

  api.handle(IPC.DORMANT_REVOKE, async (_evt, raw: unknown) => {
    if (!agent.dormant) return DORMANT_NOT_ENABLED;
    const parsed = DormantProposalDecisionSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const out = agent.dormant.revoke(parsed.data.proposalId);
    // 仅 applied 可撤销；找不到或非 applied → not_found_or_not_applied。
    if (!out) return { error: "not_found_or_not_applied" };
    return { proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt };
  });

  api.handle(IPC.DORMANT_PERSONA, async () => {
    if (!agent.dormant) return DORMANT_NOT_ENABLED;
    return agent.dormant.getPersona();
  });

  // 技能自学习 Phase 2：蒸馏 / 列表 / 审批。未启用统一返回 skills_learning_not_enabled。
  const skillProposalView = (p: {
    proposalId: string;
    candidate: { id: string; name: string; description: string };
    evidence: { queries: string[]; taskType?: string; count: number };
    status: string;
    ts: number;
    decidedAt?: number;
  }) => ({
    proposalId: p.proposalId,
    skillId: p.candidate.id,
    name: p.candidate.name,
    description: p.candidate.description,
    status: p.status,
    ts: p.ts,
    decidedAt: p.decidedAt,
    evidence: p.evidence,
  });

  api.handle(IPC.SKILLS_DISTILL, async () => {
    if (!agent.skillLearning) return SKILLS_LEARNING_NOT_ENABLED;
    const produced = await agent.skillLearning.distill();
    return { produced: produced.length, proposals: produced.map(skillProposalView) };
  });

  api.handle(IPC.SKILLS_LIST, async (_evt, raw: unknown) => {
    if (!agent.skillLearning) return SKILLS_LEARNING_NOT_ENABLED;
    const parsed = SkillListRequestSchema.safeParse(raw ?? {});
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const list = agent.skillLearning.listProposals(parsed.data.status);
    return { total: list.length, proposals: list.map(skillProposalView) };
  });

  api.handle(IPC.SKILLS_APPROVE, async (_evt, raw: unknown) => {
    if (!agent.skillLearning) return SKILLS_LEARNING_NOT_ENABLED;
    const parsed = SkillProposalDecisionSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const out = await agent.skillLearning.approve(parsed.data.proposalId);
    if (!out) return { error: "not_found_or_already_decided" };
    return { proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt };
  });

  api.handle(IPC.SKILLS_REJECT, async (_evt, raw: unknown) => {
    if (!agent.skillLearning) return SKILLS_LEARNING_NOT_ENABLED;
    const parsed = SkillProposalDecisionSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const out = agent.skillLearning.reject(parsed.data.proposalId);
    if (!out) return { error: "not_found_or_already_decided" };
    return { proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt };
  });

  api.handle(IPC.SKILLS_REVOKE, async (_evt, raw: unknown) => {
    if (!agent.skillLearning) return SKILLS_LEARNING_NOT_ENABLED;
    const parsed = SkillProposalDecisionSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const out = await agent.skillLearning.revoke(parsed.data.proposalId);
    if (!out) return { error: "not_found_or_not_approved" };
    return { proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt };
  });

  api.handle(IPC.SKILLS_ACTIVE, async () => {
    if (!agent.skillLearning) return SKILLS_LEARNING_NOT_ENABLED;
    const skills = agent.skillLearning.listApproved().map((s) => ({
      id: s.id,
      name: s.name,
      description: s.description,
      source: s.source,
      weight: agent.skillLearning!.weightFor(s.id),
    }));
    return { total: skills.length, skills };
  });

  // 把核心 hooks 推送给 renderer
  if (webContents) {
    const send = (channel: string, payload: unknown): void => {
      try {
        webContents.send(channel, payload);
      } catch {
        // renderer 已销毁；忽略
      }
    };
    offs.push(
      agent.hooks.on("tao.beforeThink", (ctx) =>
        send(IPC.EVT_TAO, { kind: "beforeThink", ...ctx.payload }),
      ),
    );
    offs.push(
      agent.hooks.on("tao.afterAct", (ctx) =>
        send(IPC.EVT_TAO, { kind: "afterAct", ...ctx.payload }),
      ),
    );
    offs.push(
      agent.hooks.on("react.afterThought", (ctx) =>
        send(IPC.EVT_REACT, { kind: "thought", ...ctx.payload }),
      ),
    );
    offs.push(
      agent.hooks.on("react.beforeAction", (ctx) =>
        send(IPC.EVT_REACT, { kind: "action", ...ctx.payload }),
      ),
    );
    offs.push(
      agent.hooks.on("react.afterAction", (ctx) =>
        send(IPC.EVT_REACT, { kind: "observation", ...ctx.payload }),
      ),
    );
    offs.push(agent.hooks.on("policy.afterCheck", (ctx) => send(IPC.EVT_AUDIT, ctx.payload)));

    // 工作区文件变更推送（onWorkspaceChange）。目录不可监听时静默跳过。
    let wsWatcher: FSWatcher | undefined;
    try {
      wsWatcher = watch(agent.workspace.config.root, { persistent: false }, (event, filename) => {
        send(IPC.EVT_WORKSPACE, {
          event: event === "rename" ? "rename" : "change",
          path: filename ? String(filename) : "",
        });
      });
    } catch {
      // 工作区目录不存在 / 平台不支持 watch → 不推送变更，但不影响其它能力。
    }
    if (wsWatcher) offs.push(() => wsWatcher?.close());
  }

  return {
    unregister(): void {
      api.removeHandler(IPC.PING);
      api.removeHandler(IPC.STATUS);
      api.removeHandler(IPC.APP_RESTART);
      api.removeHandler(IPC.MODEL_PROFILES);
      api.removeHandler(IPC.MODEL_CREDENTIAL_SET);
      api.removeHandler(IPC.MODEL_CREDENTIAL_DELETE);
      api.removeHandler(IPC.CHAT);
      api.removeHandler(IPC.MEMORY_QUERY);
      api.removeHandler(IPC.AUDIT_RECENT);
      api.removeHandler(IPC.WORKSPACE_INFO);
      api.removeHandler(IPC.WORKSPACE_READ);
      api.removeHandler(IPC.WORKSPACE_WRITE);
      api.removeHandler(IPC.WORKSPACE_PICK_DIR);
      api.removeHandler(IPC.CONFIG_GET);
      api.removeHandler(IPC.CONFIG_UPDATE);
      api.removeHandler(IPC.DORMANT_MINE);
      api.removeHandler(IPC.DORMANT_LIST);
      api.removeHandler(IPC.DORMANT_APPROVE);
      api.removeHandler(IPC.DORMANT_REJECT);
      api.removeHandler(IPC.DORMANT_REVOKE);
      api.removeHandler(IPC.DORMANT_PERSONA);
      for (const o of offs) o();
    },
  };
};
