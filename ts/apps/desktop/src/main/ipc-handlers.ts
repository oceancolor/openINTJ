import { randomUUID } from "node:crypto";
import { type FSWatcher, watch } from "node:fs";
import {
  type JournalTurnContext,
  type SlashCommandContext,
  deriveHistory,
} from "@openintj/core";
import { createToolCallGate } from "@openintj/plane-governance";
import { createTurnCoordinator, runCompactCommand, taoStatusToTurnEnd } from "@openintj/runtime";
import type { ScheduleTrigger } from "@openintj/scheduler";
import { withRootSpan } from "@openintj/telemetry-otel";
import { BrowserWindow, type IpcMain, type WebContents, app, ipcMain, shell } from "electron";
import {
  AGENT_RUNTIME_CONFIG_KEYS,
  AgentCancelRequestSchema,
  AgentSteerRequestSchema,
  AppConfigPatchSchema,
  ApprovalDecideRequestSchema,
  ChatRequestSchema,
  CommandsInvokeRequestSchema,
  DEFAULT_MODEL_PROFILES,
  DormantListRequestSchema,
  DormantProposalDecisionSchema,
  GrantCreateRequestSchema,
  IPC,
  IdentityAssertRequestSchema,
  IdentityConfirmRequestSchema,
  IdentityResolveRequestSchema,
  IdentityUnlinkRequestSchema,
  MemoryQueryRequestSchema,
  ModelCredentialSetSchema,
  ProfileUpsertRequestSchema,
  ScheduleAddRequestSchema,
  ScheduleUpdateRequestSchema,
  SkillListRequestSchema,
  SkillProposalDecisionSchema,
  SyncBundleViewSchema,
  VoiceTranscribeRequestSchema,
  WindowChromeActionSchema,
  WorkbenchCreateSchema,
  WorkbenchUpdateSchema,
  WorkspaceBytesRequestSchema,
  WorkspaceReadRequestSchema,
  WorkspaceRevealRequestSchema,
  WorkspaceWriteRequestSchema,
  isAgentRuntimeConfigPatch,
} from "../shared/ipc-protocol.js";
import {
  artifactsFromAnswer,
  extractWorkspaceArtifactPaths,
  preferredPreviewPath,
  requestsArtifactPreview,
} from "../shared/workspace-view.js";
import type { DesktopAgent } from "./agent.js";
import { requestsFileArtifact, resolveMissingArtifact } from "./artifact-fallback.js";
import type { ConfigService } from "./config-store.js";
import type { CredentialStore } from "./credential-store.js";
import type { ModelRegistry } from "./model-registry.js";
import type { WorkbenchStore } from "./workbench-store.js";
import { listWorkspaceEntries, readWorkspaceBytes, resolveWorkspaceItem } from "./workspace-fs.js";

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
  /**
   * Restart entry: unpackaged electron-vite keeps the window (reload agent +
   * renderer); packaged builds close state and relaunch the process.
   */
  restart?: () => Promise<void>;
  /**
   * Reassemble the Desktop agent from the latest saved config without quitting.
   * Used after runtime settings (workspace, models, feature flags) change.
   */
  reloadAgent?: () => Promise<void>;
  /** Encrypted model API keys; never returned to renderer. */
  credentials?: CredentialStore;
  /** Environment-provided credentials count as configured but remain outside the store. */
  credentialEnv?: Partial<Record<"hunyuan" | "kimi" | "minimax" | "glm", boolean>>;
  modelRegistry?: ModelRegistry;
  workbench?: WorkbenchStore;
  /**
   * Open a workspace path in the OS file manager.
   * Tests inject a stub; production uses Electron `shell`.
   */
  revealItem?: (absPath: string, kind: "file" | "dir") => Promise<{ ok: boolean; error?: string }>;
}

const workspaceError = (e: unknown): { error: "workspace_error"; message: string } => ({
  error: "workspace_error",
  message: e instanceof Error ? e.message : String(e),
});

const hasSuccessfulFileWrite = (trajectory: readonly unknown[]): boolean =>
  trajectory.some((entry) => {
    const state = (entry as { state?: { type?: string; toolResult?: unknown } })?.state;
    if (state?.type !== "observation") return false;
    const result = state.toolResult as
      | { toolName?: string; success?: boolean; output?: unknown }
      | undefined;
    return result?.toolName === "write_file" && result.success === true;
  });

/** Rebuild a zod-parsed trigger as an exact scheduler trigger (drops undefined optionals). */
const normalizeTrigger = (
  trigger:
    | { kind: "once"; at: number }
    | { kind: "interval"; everyMs: number; startAt?: number | undefined }
    | { kind: "daily"; time: string },
): ScheduleTrigger => {
  switch (trigger.kind) {
    case "once":
      return { kind: "once", at: trigger.at };
    case "daily":
      return { kind: "daily", time: trigger.time };
    case "interval":
      return {
        kind: "interval",
        everyMs: trigger.everyMs,
        ...(trigger.startAt !== undefined ? { startAt: trigger.startAt } : {}),
      };
  }
};

/** 注册所有 IPC channel handler，并把 hook 事件转发为 webContents.send。 */
export const registerIpcHandlers = (
  agent: DesktopAgent,
  webContents?: WebContents,
  api: IpcMain = ipcMain,
  deps: IpcDeps = {},
): IpcRegistration => {
  const offs: Array<() => void> = [];
  // RFC-021 F02 (T04): chat abort controllers are scoped per conversation;
  // a cancel for one conversation must never abort another's stream.
  // App-restart / config-reload remain legitimately global aborts.
  const activeChats = new Map<string, Set<AbortController>>();
  const trackChat = (key: string, controller: AbortController): void => {
    const set = activeChats.get(key) ?? new Set<AbortController>();
    set.add(controller);
    activeChats.set(key, set);
  };
  const untrackChat = (controller: AbortController): void => {
    for (const [key, set] of activeChats) {
      set.delete(controller);
      if (set.size === 0) activeChats.delete(key);
    }
  };
  const abortChatsFor = (key: string, reason: Error): void => {
    for (const controller of activeChats.get(key) ?? []) controller.abort(reason);
  };
  const abortAllChats = (reason: Error): void => {
    for (const set of activeChats.values()) {
      for (const controller of set) controller.abort(reason);
    }
  };
  // RFC-021 F01: IPC owns the journal final for CHAT turns; the loop runs
  // with completionOwner="host" and this coordinator commits exactly the
  // post-processed answer the user was shown.
  const turnCoordinator = createTurnCoordinator();
  let restarting = false;

  const commandCtx = (conversationId?: string): Omit<SlashCommandContext, "raw"> => {
    const journal = deps.workbench?.journal() ?? agent.handle.journal;
    const handle = conversationId ? agent.handleFor(conversationId, journal) : agent.handle;
    return conversationId
      ? {
          journal,
          handle,
          conversationId,
          // RFC-012: /compact runs against the workbench journal.
          compact: () => runCompactCommand(agent.compactor, journal, conversationId),
        }
      : { journal, handle };
  };

  if (deps.workbench) {
    agent.commands.register({
      name: "/clear",
      aliases: ["/clear history"],
      description: "清空当前会话（journal 追加 session.cleared）",
      async run(ctx) {
        if (ctx.conversationId) deps.workbench?.clearMessages(ctx.conversationId);
        ctx.handle.cancel("command");
        return { kind: "ok", message: "对话已清空。" };
      },
    });
  }

  api.handle(IPC.PING, async () => ({ ok: true, ts: Date.now() }));

  api.handle(IPC.STATUS, async () => {
    await agent.refreshModelRuntime();
    return agent.status();
  });

  api.handle(IPC.APP_RESTART, async () => {
    if (!deps.restart) return { ok: false, reason: "restart_unavailable" };
    restarting = true;
    abortAllChats(new Error("app_restarting"));
    await deps.restart();
    return { ok: true };
  });

  api.handle(IPC.WINDOW_CHROME, async (evt, raw: unknown) => {
    const parsed = WindowChromeActionSchema.safeParse(raw);
    if (!parsed.success) return { ok: false };
    const win = BrowserWindow.fromWebContents(evt.sender);
    if (!win || win.isDestroyed()) return { ok: false };
    switch (parsed.data) {
      case "quit":
        app.quit();
        return { ok: true };
      case "reload":
        win.webContents.reloadIgnoringCache();
        return { ok: true };
      case "toggleDevTools":
        win.webContents.toggleDevTools();
        return { ok: true };
      case "toggleFullScreen":
        win.setFullScreen(!win.isFullScreen());
        return { ok: true };
      case "minimize":
        win.minimize();
        return { ok: true };
      case "maximize":
        if (win.isMaximized()) win.unmaximize();
        else win.maximize();
        return { ok: true };
      case "close":
        win.close();
        return { ok: true };
    }
  });

  api.handle(IPC.MODEL_PROFILES, async () => {
    const custom = deps.config?.get().modelProfiles ?? [];
    const profiles = [...DEFAULT_MODEL_PROFILES, ...custom] as Array<
      (typeof DEFAULT_MODEL_PROFILES)[number]
    >;
    const merged = new Map(profiles.map((profile) => [profile.id, profile]));
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
    deps.modelRegistry?.clear(parsed.data.profileId);
    return { ok: true };
  });

  api.handle(IPC.MODEL_CREDENTIAL_DELETE, async (_evt, profileId: unknown) => {
    if (typeof profileId !== "string") return { ok: false, error: "invalid_request" };
    if (!deps.credentials) return { ok: false, error: "credential_store_unavailable" };
    const deleted = deps.credentials.delete(profileId);
    deps.modelRegistry?.clear(profileId);
    return { ok: true, deleted };
  });

  api.handle(IPC.MODEL_TEST, async (_evt, profileId: unknown) => {
    if (typeof profileId !== "string") return { ok: false, error: "invalid_request" };
    if (!deps.modelRegistry) return { ok: false, error: "model_registry_unavailable" };
    return deps.modelRegistry.test(profileId);
  });

  api.handle(IPC.WORKBENCH_BOOTSTRAP, async () => {
    if (!deps.workbench) return { workspaces: [], tasks: [], conversations: [] };
    return deps.workbench.bootstrap();
  });
  api.handle(IPC.WORKBENCH_WORKSPACE_CREATE, async (_evt, raw: unknown) => {
    const parsed = WorkbenchCreateSchema.safeParse(raw);
    if (!parsed.success || !parsed.data.name || !parsed.data.rootPath || !deps.workbench) {
      return { error: "invalid_request" };
    }
    return deps.workbench.createWorkspace({
      name: parsed.data.name,
      rootPath: parsed.data.rootPath,
    });
  });
  api.handle(IPC.WORKBENCH_WORKSPACE_CLOSE, async (_evt, raw: unknown) => {
    const parsed = WorkbenchUpdateSchema.safeParse(raw);
    if (!parsed.success || !deps.workbench) return { error: "invalid_request" };
    return deps.workbench.closeWorkspace(parsed.data.id);
  });
  api.handle(IPC.WORKBENCH_TASK_CREATE, async (_evt, raw: unknown) => {
    const parsed = WorkbenchCreateSchema.safeParse(raw);
    if (!parsed.success || !parsed.data.parentId || !parsed.data.title || !deps.workbench) {
      return { error: "invalid_request" };
    }
    return deps.workbench.createTask({
      workspaceId: parsed.data.parentId,
      title: parsed.data.title,
    });
  });
  api.handle(IPC.WORKBENCH_TASK_UPDATE, async (_evt, raw: unknown) => {
    const parsed = WorkbenchUpdateSchema.safeParse(raw);
    if (!parsed.success || !deps.workbench) return { error: "invalid_request" };
    return deps.workbench.updateTask(parsed.data.id, {
      ...(parsed.data.title !== undefined ? { title: parsed.data.title } : {}),
      ...(parsed.data.status !== undefined ? { status: parsed.data.status } : {}),
      ...(parsed.data.taskPoolRunId !== undefined
        ? { taskPoolRunId: parsed.data.taskPoolRunId }
        : {}),
    });
  });
  api.handle(IPC.WORKBENCH_CONVERSATION_CREATE, async (_evt, raw: unknown) => {
    const parsed = WorkbenchCreateSchema.safeParse(raw);
    if (!parsed.success || !parsed.data.parentId || !parsed.data.title || !deps.workbench) {
      return { error: "invalid_request" };
    }
    return deps.workbench.createConversation({
      taskId: parsed.data.parentId,
      title: parsed.data.title,
      modelProfileId: parsed.data.modelProfileId ?? "auto",
    });
  });
  api.handle(IPC.WORKBENCH_CONVERSATION_UPDATE, async (_evt, raw: unknown) => {
    const parsed = WorkbenchUpdateSchema.safeParse(raw);
    if (!parsed.success || !deps.workbench) return { error: "invalid_request" };
    return deps.workbench.updateConversation(parsed.data.id, {
      ...(parsed.data.title !== undefined ? { title: parsed.data.title } : {}),
      ...(parsed.data.modelProfileId !== undefined
        ? { modelProfileId: parsed.data.modelProfileId }
        : {}),
    });
  });
  api.handle(IPC.WORKBENCH_MESSAGES, async (_evt, conversationId: unknown) => {
    if (typeof conversationId !== "string" || !deps.workbench) return [];
    return deps.workbench.listMessages(conversationId);
  });
  api.handle(IPC.WORKBENCH_MESSAGES_CLEAR, async (_evt, conversationId: unknown) => {
    if (typeof conversationId !== "string" || !deps.workbench) return { error: "invalid_request" };
    const command = agent.commands.match("/clear");
    if (!command) return { error: "invalid_request" };
    await command.run({ ...commandCtx(conversationId), raw: "/clear" });
    return { ok: true };
  });

  api.handle(IPC.COMMANDS_INVOKE, async (_evt, raw: unknown) => {
    const parsed = CommandsInvokeRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const command = agent.commands.match(parsed.data.text);
    if (!command) return { error: "unknown_command" };
    const result = await command.run({
      ...commandCtx(parsed.data.conversationId),
      raw: parsed.data.text,
    });
    return { name: command.name, result };
  });

  api.handle(IPC.AGENT_STEER, async (_evt, raw: unknown) => {
    const parsed = AgentSteerRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    commandCtx(parsed.data.conversationId).handle.steer(parsed.data.message);
    return { ok: true };
  });

  api.handle(IPC.AGENT_INJECT, async (_evt, raw: unknown) => {
    const parsed = AgentSteerRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    commandCtx(parsed.data.conversationId).handle.inject(parsed.data.message);
    return { ok: true };
  });

  api.handle(IPC.AGENT_CANCEL, async (_evt, raw: unknown) => {
    const parsed = AgentCancelRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    commandCtx(parsed.data.conversationId).handle.cancel(parsed.data.cause);
    abortChatsFor(parsed.data.conversationId, new Error(parsed.data.cause));
    return { ok: true };
  });

  // RFC-013: interactive approval + capability grants.
  api.handle(IPC.APPROVAL_LIST, async () => agent.approvals?.listPending() ?? []);
  api.handle(IPC.APPROVAL_DECIDE, async (_evt, raw: unknown) => {
    const parsed = ApprovalDecideRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    if (!agent.approvals) return { error: "approval_not_enabled" };
    const ok = agent.approvals.decide(parsed.data.id, {
      decision: parsed.data.decision,
      ...(parsed.data.reason ? { reason: parsed.data.reason } : {}),
      ...(parsed.data.grant ? { grant: parsed.data.grant } : {}),
    } as Parameters<typeof agent.approvals.decide>[1]);
    return ok ? { ok: true } : { error: "not_pending" };
  });
  api.handle(IPC.GRANTS_LIST, async () => agent.grants.list());
  api.handle(IPC.GRANTS_GRANT, async (_evt, raw: unknown) => {
    const parsed = GrantCreateRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    return agent.grants.grant({
      tool: parsed.data.tool,
      risk: parsed.data.risk,
      ...(parsed.data.scope ? { scope: parsed.data.scope } : {}),
      ...(parsed.data.ttlMs ? { expiresAt: Date.now() + parsed.data.ttlMs } : {}),
      ...(parsed.data.label ? { label: parsed.data.label } : {}),
    });
  });
  api.handle(IPC.GRANTS_REVOKE, async (_evt, raw: unknown) => {
    if (typeof raw !== "string" || !raw) return { error: "invalid_request" };
    return agent.grants.revoke(raw) ? { ok: true } : { error: "not_found" };
  });

  // RFC-014: personal profile (memory control plane).
  api.handle(IPC.PROFILE_LIST, async () => agent.profile.list());
  api.handle(IPC.PROFILE_UPSERT, async (_evt, raw: unknown) => {
    const parsed = ProfileUpsertRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const entry = agent.profile.upsert({
      section: parsed.data.section,
      key: parsed.data.key,
      value: parsed.data.value,
      ...(parsed.data.validUntil ? { validUntil: parsed.data.validUntil } : {}),
    });
    return entry ?? { error: "conflict" };
  });
  api.handle(IPC.PROFILE_REMOVE, async (_evt, raw: unknown) => {
    if (typeof raw !== "string" || !raw) return { error: "invalid_request" };
    return agent.profile.remove(raw) ? { ok: true } : { error: "not_found" };
  });
  api.handle(IPC.PROFILE_EXPORT, async () => ({ profile: agent.profile.exportJSON() }));

  // RFC-020: identity graph. Resolvers only propose; confirm() is the only
  // way into the graph, and owned values can never be silently moved.
  api.handle(IPC.IDENTITY_PERSONS, async () => agent.identity.listPersons());
  api.handle(IPC.IDENTITY_RESOLVE, async (_evt, raw: unknown) => {
    const parsed = IdentityResolveRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      return await agent.identity.resolve(parsed.data);
    } catch (err) {
      return { error: err instanceof Error ? err.message : "resolve_failed" };
    }
  });
  api.handle(IPC.IDENTITY_CONFIRM, async (_evt, raw: unknown) => {
    const parsed = IdentityConfirmRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      return agent.identity.confirm(parsed.data.proposals);
    } catch (err) {
      return { error: err instanceof Error ? err.message : "confirm_failed" };
    }
  });
  api.handle(IPC.IDENTITY_ASSERT, async (_evt, raw: unknown) => {
    const parsed = IdentityAssertRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      return agent.identity.assertAccount(parsed.data);
    } catch (err) {
      return { error: err instanceof Error ? err.message : "assert_failed" };
    }
  });
  api.handle(IPC.IDENTITY_UNLINK, async (_evt, raw: unknown) => {
    const parsed = IdentityUnlinkRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const result = agent.identity.unlink(parsed.data.accountId);
    return result.removed ? result : { error: "not_found" };
  });

  // RFC-015: scheduler. The runner is (re)attached here so scheduled turns
  // land in the workbench journal and show up in the target conversation.
  agent.scheduler.attachRunner(async (job) => {
    const journal = deps.workbench?.journal() ?? agent.handle.journal;
    const handle = agent.handleFor(job.conversationId ?? "scheduler", journal);
    const result = await handle.run(job.prompt);
    if (result.status !== "completed") {
      throw new Error(`scheduled run ended with status ${result.status}`);
    }
  });
  api.handle(IPC.SCHEDULE_LIST, async () => agent.scheduler.list());
  api.handle(IPC.SCHEDULE_ADD, async (_evt, raw: unknown) => {
    const parsed = ScheduleAddRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      const d = parsed.data;
      return agent.scheduler.add({
        title: d.title,
        prompt: d.prompt,
        trigger: normalizeTrigger(d.trigger),
        ...(d.conversationId ? { conversationId: d.conversationId } : {}),
        ...(d.enabled !== undefined ? { enabled: d.enabled } : {}),
        ...(d.maxRuns ? { maxRuns: d.maxRuns } : {}),
      });
    } catch (e) {
      return { error: "invalid_request", message: (e as Error).message };
    }
  });
  api.handle(IPC.SCHEDULE_UPDATE, async (_evt, raw: unknown) => {
    const req = (raw ?? {}) as { id?: unknown; patch?: unknown };
    if (typeof req.id !== "string" || !req.id) return { error: "invalid_request" };
    const parsed = ScheduleUpdateRequestSchema.safeParse(req.patch);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      const d = parsed.data;
      const job = agent.scheduler.update(req.id, {
        ...(d.title !== undefined ? { title: d.title } : {}),
        ...(d.prompt !== undefined ? { prompt: d.prompt } : {}),
        ...(d.conversationId !== undefined ? { conversationId: d.conversationId } : {}),
        ...(d.trigger ? { trigger: normalizeTrigger(d.trigger) } : {}),
        ...(d.enabled !== undefined ? { enabled: d.enabled } : {}),
        ...(d.maxRuns !== undefined ? { maxRuns: d.maxRuns } : {}),
      });
      return job ?? { error: "not_found" };
    } catch (e) {
      return { error: "invalid_request", message: (e as Error).message };
    }
  });
  api.handle(IPC.SCHEDULE_REMOVE, async (_evt, raw: unknown) => {
    if (typeof raw !== "string" || !raw) return { error: "invalid_request" };
    return agent.scheduler.remove(raw) ? { ok: true } : { error: "not_found" };
  });

  // RFC-016: connector statuses (connector tools live in the agent ToolHub).
  api.handle(IPC.CONNECTORS_LIST, async () => agent.connectors.statuses());

  // RFC-012: TaskPool checkpoints (list / resume / cancel; no renderer panel yet).
  api.handle(IPC.TASKPOOL_CHECKPOINTS, async (_evt, raw: unknown) => {
    if (raw !== undefined && (typeof raw !== "number" || !Number.isFinite(raw))) {
      return { error: "invalid_request" };
    }
    return agent.taskPoolCheckpoints(raw);
  });
  api.handle(IPC.TASKPOOL_RESUME, async (_evt, raw: unknown) => {
    if (typeof raw !== "string" || !raw) return { error: "invalid_request" };
    const result = await agent.resumeTaskRun(raw);
    if (!result) return { error: "not_found_or_not_resumable" };
    return {
      runId: result.runId,
      planId: result.planId,
      status: result.status,
      results: Object.fromEntries(result.results),
      errors: Object.fromEntries(
        [...result.errors].map(([taskId, error]) => [
          taskId,
          error instanceof Error ? error.message : String(error),
        ]),
      ),
      states: Object.fromEntries(result.states),
      attempts: Object.fromEntries(result.attempts),
    };
  });
  api.handle(IPC.TASKPOOL_CANCEL, async (_evt, raw: unknown) => {
    if (typeof raw !== "string" || !raw) return { error: "invalid_request" };
    return (await agent.cancelTaskRun(raw))
      ? { ok: true }
      : { error: "not_found_or_not_cancellable" };
  });

  // RFC-012 §3: subagent records (spawning stays model-side via delegate_task).
  api.handle(IPC.SUBAGENTS_LIST, async () => agent.subagents.list());

  // RFC-018: outbound channels (status + test send).
  api.handle(IPC.CHANNELS_LIST, async () => agent.channels.list());
  api.handle(IPC.CHANNELS_TEST, async (_evt, raw: unknown) => {
    if (typeof raw !== "string" || !raw) return { error: "invalid_request" };
    const result = await agent.channels.test(raw);
    return result.ok ? result : { error: result.error ?? "channel_test_failed" };
  });

  // RFC-019: cross-device sync bundle (export/import; main validates shape).
  api.handle(IPC.SYNC_EXPORT, async () => agent.exportSyncBundle());
  api.handle(IPC.SYNC_IMPORT, async (_evt, raw: unknown) => {
    const parsed = SyncBundleViewSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_bundle", issues: parsed.error.issues };
    try {
      return { report: agent.importSyncBundle(parsed.data) };
    } catch (e) {
      return { error: "invalid_bundle", detail: String(e) };
    }
  });

  // RFC-019: speech-to-text (error when no provider configured via env).
  api.handle(IPC.VOICE_TRANSCRIBE, async (_evt, raw: unknown) => {
    if (!agent.stt) return { error: "stt_not_configured" };
    const parsed = VoiceTranscribeRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    const audio = Buffer.from(parsed.data.audioBase64, "base64");
    if (audio.byteLength > 26_000_000) return { error: "audio_too_large" };
    try {
      return await agent.stt.transcribe({
        audio: new Uint8Array(audio),
        mimeType: parsed.data.mimeType,
        ...(parsed.data.language ? { language: parsed.data.language } : {}),
      });
    } catch (e) {
      return { error: "transcribe_failed", detail: String(e) };
    }
  });

  api.handle(IPC.CHAT, async (_evt, raw: unknown) => {
    const parsed = ChatRequestSchema.safeParse(raw);
    if (!parsed.success) {
      return { error: "invalid_request", issues: parsed.error.issues };
    }
    if (restarting) return { error: "app_restarting" };
    const controller = new AbortController();
    // Always tracked (pending key until the conversation id is known) so
    // global aborts still reach brand-new conversations.
    trackChat(parsed.data.conversationId ?? `chat:${randomUUID()}`, controller);
    let persistedConversationId: string | undefined;
    let journalCtx: JournalTurnContext | undefined;
    try {
      const selected = parsed.data.conversationId
        ? deps.workbench?.getConversation(parsed.data.conversationId)
        : undefined;
      const listedMessages = selected
        ? deps.workbench?.listMessages(selected.conversation.id, 40)
        : undefined;
      const history = selected
        ? deriveHistory(deps.workbench?.listJournal(selected.conversation.id) ?? [])
        : undefined;
      const profileId = parsed.data.modelProfileId ?? selected?.conversation.modelProfileId;
      const selectedLlm =
        profileId && deps.modelRegistry ? await deps.modelRegistry.resolve(profileId) : undefined;
      const slash = agent.commands.match(parsed.data.query);
      if (slash) {
        const result = await slash.run({
          ...commandCtx(selected?.conversation.id),
          raw: parsed.data.query,
        });
        return {
          finalAnswer: result.message ?? "",
          iterations: 0,
          status: result.kind === "ok" ? "completed" : "failed",
          traceId: "command",
          command: { name: slash.name, kind: result.kind },
          context: agent.status().context,
        };
      }
      if (selected && deps.workbench) {
        persistedConversationId = selected.conversation.id;
        trackChat(selected.conversation.id, controller);
        journalCtx = {
          conversationId: selected.conversation.id,
          turnId: randomUUID(),
          sink: deps.workbench.journal(),
          originalInput: parsed.data.query,
        };
        deps.workbench.appendMessage({
          conversationId: selected.conversation.id,
          role: "user",
          content: parsed.data.query,
          messageKind: "message",
        });
      }
      if (requestsArtifactPreview(parsed.data.query)) {
        const historyPaths = (listedMessages ?? []).flatMap((entry) =>
          extractWorkspaceArtifactPaths(entry.content),
        );
        let listedPaths: string[] = [];
        try {
          listedPaths = (await listWorkspaceEntries(agent.workspace.config.root))
            .filter((entry) => entry.type === "file")
            .map((entry) => entry.path);
        } catch {
          listedPaths = [];
        }
        const path = preferredPreviewPath([...historyPaths, ...listedPaths]);
        const finalAnswer = path
          ? `已在预览区打开：\`${path}\``
          : "工作区里还没有可预览的产物。请先生成图片或文件。";
        const artifacts = path ? artifactsFromAnswer(`\`${path}\``) : [];
        if (selected && deps.workbench && journalCtx) {
          turnCoordinator.commitTurn(deps.workbench.journal(), journalCtx, {
            userContent: parsed.data.query,
            originalInput: parsed.data.query,
            assistantContent: finalAnswer,
            messageKind: "answer",
            status: "completed",
            endStatus: "completed",
            traceId: "preview",
          });
          deps.workbench.appendMessage({
            conversationId: selected.conversation.id,
            role: "assistant",
            content: finalAnswer,
            status: "completed",
            messageKind: "answer",
          });
        }
        return {
          finalAnswer,
          iterations: 0,
          status: "completed",
          traceId: "preview",
          context: agent.status().context,
          ...(artifacts.length > 0 ? { artifacts } : {}),
        };
      }
      const result = await withRootSpan(
        "openintj.ipc.chat",
        () =>
          agent.run(parsed.data.query, {
            signal: controller.signal,
            completionOwner: "host",
            ...(history ? { history } : {}),
            ...(journalCtx ? { journal: journalCtx } : {}),
            ...(selectedLlm ? { llm: selectedLlm } : {}),
            ...(selected
              ? {
                  contextTags: [
                    `workspace:${selected.workspace.id}`,
                    `task:${selected.task.id}`,
                    `conversation:${selected.conversation.id}`,
                  ],
                }
              : {}),
          }),
        { attributes: { "ipc.channel": IPC.CHAT } },
      );
      let finalAnswer = result.finalAnswer;
      if (
        selected &&
        result.status === "completed" &&
        result.inputStructure?.action !== "clarify" &&
        requestsFileArtifact(parsed.data.query) &&
        !hasSuccessfulFileWrite(result.trajectory) &&
        !result.metrics?.imageGenerated &&
        !result.metrics?.imageGenerationFallback
      ) {
        const artifact = resolveMissingArtifact(
          parsed.data.query,
          selected.task.id,
          result.finalAnswer,
        );
        // RFC-021 F03 (T06): host fallback goes through the governed tool
        // facade with an explicit origin ? never a silent direct write, and
        // a failed/cancelled run never earns a "written" claim.
        const writeResult = await agent.toolHub.call(
          "write_file",
          { path: artifact.path, content: artifact.content },
          {
            origin: "host_fallback",
            ...(journalCtx
              ? { runId: `run:${journalCtx.conversationId}:${journalCtx.turnId}` }
              : {}),
            signal: controller.signal,
          },
        );
        finalAnswer =
          writeResult.businessStatus !== "failure" && writeResult.success
            ? result.finalAnswer + "\n\n已写入工作区：`" + artifact.path + "`"
            : result.finalAnswer + "（自动补写文件失败：" + (writeResult.error ?? writeResult.businessStatus ?? "unknown") + "）";
      }
      if (selected && deps.workbench && journalCtx) {
        turnCoordinator.commitTurn(deps.workbench.journal(), journalCtx, {
          userContent: parsed.data.query,
          originalInput: parsed.data.query,
          ...(result.inputStructure ? { inputStructure: result.inputStructure } : {}),
          assistantContent: finalAnswer,
          messageKind: result.inputStructure?.action === "clarify" ? "clarification" : "answer",
          status: result.status,
          endStatus: taoStatusToTurnEnd(result.status, result.inputStructure?.action === "clarify"),
          traceId: result.traceId,
        });
        deps.workbench.appendMessage({
          conversationId: selected.conversation.id,
          role: "assistant",
          content: finalAnswer,
          traceId: result.traceId,
          tokens: result.totalTokensSpent,
          status: result.status,
          messageKind: result.inputStructure?.action === "clarify" ? "clarification" : "answer",
          ...(result.inputStructure ? { inputStructure: result.inputStructure } : {}),
        });
      }
      const modelStatus = selectedLlm?.getStatus();
      const context = agent.status().context;
      const artifacts = artifactsFromAnswer(finalAnswer);
      return {
        finalAnswer,
        iterations: result.iterations,
        status: result.status,
        traceId: result.traceId,
        context,
        ...(result.inputStructure ? { inputStructure: result.inputStructure } : {}),
        ...(modelStatus ? { provider: modelStatus.provider, model: modelStatus.model } : {}),
        ...(artifacts.length > 0 ? { artifacts } : {}),
      };
    } catch (error) {
      if (persistedConversationId && deps.workbench && journalCtx) {
        const failed = error instanceof Error ? error.message : String(error);
        turnCoordinator.commitTurn(deps.workbench.journal(), journalCtx, {
          userContent: parsed.data.query,
          originalInput: parsed.data.query,
          assistantContent: failed,
          messageKind: "answer",
          status: "failed",
          endStatus: "failed",
        });
        deps.workbench.appendMessage({
          conversationId: persistedConversationId,
          role: "assistant",
          content: failed,
          status: "failed",
          messageKind: "answer",
        });
      }
      throw error;
    } finally {
      untrackChat(controller);
    }
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

  // ---------- G5.2 记忆审核台（F14 状态面 + 影子候选审核） ----------
  api.handle(IPC.MEMORY_STATUS, async () => agent.memoryStatus());

  api.handle(IPC.MEMORY_CANDIDATES, async () => agent.memoryCandidates());

  const candidateDecision = async (
    raw: unknown,
    decide: (candidateId: string) => Promise<{ candidateId: string; status: string; fragmentId?: string } | { error: string }>,
  ): Promise<{ candidateId: string; status: string; fragmentId?: string } | { error: string }> => {
    const candidateId =
      typeof raw === "object" && raw !== null ? (raw as { candidateId?: unknown }).candidateId : undefined;
    if (typeof candidateId !== "string" || candidateId.length === 0) {
      return { error: "invalid_request" };
    }
    return decide(candidateId);
  };

  api.handle(IPC.MEMORY_CANDIDATE_APPROVE, async (_evt, raw: unknown) =>
    candidateDecision(raw, (id) => agent.approveMemoryCandidate(id)),
  );

  api.handle(IPC.MEMORY_CANDIDATE_REJECT, async (_evt, raw: unknown) =>
    candidateDecision(raw, (id) => agent.rejectMemoryCandidate(id)),
  );

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
      if (!root) return { canceled: true };
      if (deps.reloadAgent) {
        abortAllChats(new Error("config_reloading"));
        await deps.reloadAgent();
      }
      return { canceled: false, root };
    } catch (e) {
      return workspaceError(e);
    }
  });

  api.handle(IPC.WORKSPACE_LIST, async () => {
    try {
      const root = agent.workspace.config.root;
      return { root, entries: await listWorkspaceEntries(root) };
    } catch (e) {
      return workspaceError(e);
    }
  });

  api.handle(IPC.WORKSPACE_READ_BYTES, async (_evt, raw: unknown) => {
    const parsed = WorkspaceBytesRequestSchema.safeParse(raw);
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      await workspaceGate({ tool: "read_file", params: { path: parsed.data.path } });
      return await readWorkspaceBytes(agent.workspace.config.root, parsed.data.path);
    } catch (e) {
      return workspaceError(e);
    }
  });

  api.handle(IPC.WORKSPACE_REVEAL, async (_evt, raw: unknown) => {
    const parsed = WorkspaceRevealRequestSchema.safeParse(raw ?? {});
    if (!parsed.success) return { error: "invalid_request", issues: parsed.error.issues };
    try {
      const item = await resolveWorkspaceItem(agent.workspace.config.root, parsed.data.path);
      const reveal =
        deps.revealItem ??
        (async (absPath, kind) => {
          if (kind === "dir") {
            const error = await shell.openPath(absPath);
            return error ? { ok: false, error } : { ok: true };
          }
          shell.showItemInFolder(absPath);
          return { ok: true };
        });
      return await reveal(item.abs, item.kind);
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
      const previous = deps.config.get();
      const updated = deps.config.update(parsed.data);
      if (parsed.data.modelProfiles || parsed.data.activeModelProfileId) {
        deps.modelRegistry?.clear();
      }
      const runtimeChanged = AGENT_RUNTIME_CONFIG_KEYS.some(
        (key) => JSON.stringify(previous[key]) !== JSON.stringify(updated[key]),
      );
      if (deps.reloadAgent && runtimeChanged) {
        abortAllChats(new Error("config_reloading"));
        await deps.reloadAgent();
      }
      return updated;
    } catch (e) {
      return {
        error: isAgentRuntimeConfigPatch(parsed.data) ? "reload_failed" : "invalid_request",
        message: e instanceof Error ? e.message : String(e),
      };
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

  api.handle(IPC.SKILLS_FEEDBACK, async (_evt, raw: unknown) => {
    const body = (typeof raw === "object" && raw !== null ? raw : {}) as {
      conversationId?: unknown;
      sentiment?: unknown;
    };
    if (
      typeof body.conversationId !== "string" ||
      body.conversationId.length === 0 ||
      (body.sentiment !== "positive" && body.sentiment !== "negative")
    ) {
      return { error: "invalid_request" };
    }
    return agent.recordSkillFeedback(body.conversationId, body.sentiment);
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

  api.handle(IPC.SKILLS_CATALOG, async () => {
    const listed = agent.skillContext?.list() ?? [];
    const skills = listed.map((s) => ({
      id: s.id,
      name: s.name,
      description: s.description,
      source: s.source,
      version: s.version,
      tools: s.tools,
      triggers: s.triggers,
      taskTypes: s.taskTypes,
      origin: s.source.startsWith("learned:") ? ("learned" as const) : ("authored" as const),
    }));
    return {
      enabled: Boolean(agent.skillContext),
      learning: Boolean(agent.skillLearning),
      total: skills.length,
      skills,
    };
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
    offs.push(
      agent.hooks.on("event.SKILL_SELECTED", (ctx) =>
        send(IPC.EVT_SKILL, { kind: "selected", ...ctx.payload }),
      ),
    );
    offs.push(
      agent.hooks.on("event.ASSISTANT_DELTA", (ctx) => send(IPC.EVT_ASSISTANT_DELTA, ctx.payload)),
    );
    offs.push(
      agent.hooks.on("event.APPROVAL_REQUESTED", (ctx) =>
        send(IPC.EVT_APPROVAL, { kind: "requested", ...ctx.payload }),
      ),
    );
    offs.push(
      agent.hooks.on("event.APPROVAL_DECIDED", (ctx) =>
        send(IPC.EVT_APPROVAL, { kind: "decided", ...ctx.payload }),
      ),
    );
    offs.push(agent.scheduler.on((event) => send(IPC.EVT_SCHEDULE, event)));
    offs.push(
      agent.hooks.on("event.SKILL_PROPOSED", (ctx) =>
        send(IPC.EVT_SKILL, { kind: "proposed", ...ctx.payload }),
      ),
    );

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
      api.removeHandler(IPC.WINDOW_CHROME);
      api.removeHandler(IPC.MODEL_PROFILES);
      api.removeHandler(IPC.MODEL_CREDENTIAL_SET);
      api.removeHandler(IPC.MODEL_CREDENTIAL_DELETE);
      api.removeHandler(IPC.MODEL_TEST);
      api.removeHandler(IPC.WORKBENCH_BOOTSTRAP);
      api.removeHandler(IPC.WORKBENCH_WORKSPACE_CREATE);
      api.removeHandler(IPC.WORKBENCH_WORKSPACE_CLOSE);
      api.removeHandler(IPC.WORKBENCH_TASK_CREATE);
      api.removeHandler(IPC.WORKBENCH_TASK_UPDATE);
      api.removeHandler(IPC.WORKBENCH_CONVERSATION_CREATE);
      api.removeHandler(IPC.WORKBENCH_CONVERSATION_UPDATE);
      api.removeHandler(IPC.WORKBENCH_MESSAGES);
      api.removeHandler(IPC.WORKBENCH_MESSAGES_CLEAR);
      api.removeHandler(IPC.COMMANDS_INVOKE);
      api.removeHandler(IPC.AGENT_STEER);
      api.removeHandler(IPC.AGENT_INJECT);
      api.removeHandler(IPC.AGENT_CANCEL);
      api.removeHandler(IPC.APPROVAL_LIST);
      api.removeHandler(IPC.APPROVAL_DECIDE);
      api.removeHandler(IPC.GRANTS_LIST);
      api.removeHandler(IPC.GRANTS_GRANT);
      api.removeHandler(IPC.GRANTS_REVOKE);
      api.removeHandler(IPC.PROFILE_LIST);
      api.removeHandler(IPC.PROFILE_UPSERT);
      api.removeHandler(IPC.PROFILE_REMOVE);
      api.removeHandler(IPC.PROFILE_EXPORT);
      api.removeHandler(IPC.SCHEDULE_LIST);
      api.removeHandler(IPC.SCHEDULE_ADD);
      api.removeHandler(IPC.SCHEDULE_UPDATE);
      api.removeHandler(IPC.SCHEDULE_REMOVE);
      api.removeHandler(IPC.IDENTITY_PERSONS);
      api.removeHandler(IPC.IDENTITY_RESOLVE);
      api.removeHandler(IPC.IDENTITY_CONFIRM);
      api.removeHandler(IPC.IDENTITY_ASSERT);
      api.removeHandler(IPC.IDENTITY_UNLINK);
      api.removeHandler(IPC.CONNECTORS_LIST);
      api.removeHandler(IPC.CHAT);
      api.removeHandler(IPC.MEMORY_QUERY);
      api.removeHandler(IPC.MEMORY_STATUS);
      api.removeHandler(IPC.MEMORY_CANDIDATES);
      api.removeHandler(IPC.MEMORY_CANDIDATE_APPROVE);
      api.removeHandler(IPC.MEMORY_CANDIDATE_REJECT);
      api.removeHandler(IPC.AUDIT_RECENT);
      api.removeHandler(IPC.WORKSPACE_INFO);
      api.removeHandler(IPC.WORKSPACE_READ);
      api.removeHandler(IPC.WORKSPACE_WRITE);
      api.removeHandler(IPC.WORKSPACE_PICK_DIR);
      api.removeHandler(IPC.WORKSPACE_LIST);
      api.removeHandler(IPC.WORKSPACE_READ_BYTES);
      api.removeHandler(IPC.WORKSPACE_REVEAL);
      api.removeHandler(IPC.CONFIG_GET);
      api.removeHandler(IPC.CONFIG_UPDATE);
      api.removeHandler(IPC.DORMANT_MINE);
      api.removeHandler(IPC.DORMANT_LIST);
      api.removeHandler(IPC.DORMANT_APPROVE);
      api.removeHandler(IPC.DORMANT_REJECT);
      api.removeHandler(IPC.DORMANT_REVOKE);
      api.removeHandler(IPC.DORMANT_PERSONA);
      api.removeHandler(IPC.SKILLS_DISTILL);
      api.removeHandler(IPC.SKILLS_FEEDBACK);
      api.removeHandler(IPC.SKILLS_LIST);
      api.removeHandler(IPC.SKILLS_APPROVE);
      api.removeHandler(IPC.SKILLS_REJECT);
      api.removeHandler(IPC.SKILLS_REVOKE);
      api.removeHandler(IPC.SKILLS_ACTIVE);
      api.removeHandler(IPC.SKILLS_CATALOG);
      for (const o of offs) o();
    },
  };
};
