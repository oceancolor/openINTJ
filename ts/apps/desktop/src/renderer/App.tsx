import React from "react";
import type {
  ModelProfile,
  WorkbenchConversation,
  WorkbenchTask,
  WorkbenchWorkspace,
} from "../shared/ipc-protocol.js";
import { DEFAULT_DESKTOP_MODEL_PROFILE_ID } from "../shared/ipc-protocol.js";
import type { ApprovalRequestView } from "../shared/ipc-protocol.js";
import {
  artifactsFromAnswer,
  extractWorkspaceArtifactPaths,
  preferredPreviewPath,
} from "../shared/workspace-view.js";
import { ApprovalBanner } from "./components/ApprovalBanner.js";
import { type ChatMessage, ChatPanel } from "./components/ChatPanel.js";
import { DormantPanel } from "./components/DormantPanel.js";
import { IdentityPanel } from "./components/IdentityPanel.js";
import { MemoryPanel } from "./components/MemoryPanel.js";
import { ProfilePanel } from "./components/ProfilePanel.js";
import { SchedulePanel } from "./components/SchedulePanel.js";
import { SettingsPanel } from "./components/SettingsPanel.js";
import { SkillPanel } from "./components/SkillPanel.js";
import { StatusBar, type StatusSnapshot } from "./components/StatusBar.js";
import { SystemPanel } from "./components/SystemPanel.js";
import { TaskSidebar } from "./components/TaskSidebar.js";
import { TitleBar } from "./components/TitleBar.js";
import { type TrajectoryEntry, TrajectoryPanel } from "./components/TrajectoryPanel.js";
import { UpdateBanner } from "./components/UpdateBanner.js";
import { FilePreview, WorkspaceWorkbench, desktopWorkspaceClient } from "./workspace/index.js";
import "./types.js";

type RightTab =
  | "chat"
  | "tasks"
  | "trajectory"
  | "memory"
  | "profile"
  | "identity"
  | "schedule"
  | "dormant"
  | "skills"
  | "system"
  | "settings"
  | "preview";

const sameFsPath = (left: string, right: string): boolean =>
  left.replace(/\\/g, "/").toLowerCase() === right.replace(/\\/g, "/").toLowerCase();

export const App = (): JSX.Element => {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [trajectory, setTrajectory] = React.useState<TrajectoryEntry[]>([]);
  const [status, setStatus] = React.useState<StatusSnapshot | undefined>();
  const [pending, setPending] = React.useState(false);
  const [approvals, setApprovals] = React.useState<ApprovalRequestView[]>([]);
  const [rightTab, setRightTab] = React.useState<RightTab>("chat");
  const [profiles, setProfiles] = React.useState<ModelProfile[]>([]);
  const [workspaces, setWorkspaces] = React.useState<WorkbenchWorkspace[]>([]);
  const [tasks, setTasks] = React.useState<WorkbenchTask[]>([]);
  const [conversations, setConversations] = React.useState<WorkbenchConversation[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = React.useState<string>();
  const [activeTaskId, setActiveTaskId] = React.useState<string>();
  const [activeConversationId, setActiveConversationId] = React.useState<string>();
  const [selectedWorkspacePath, setSelectedWorkspacePath] = React.useState<string>();
  const [workspaceRoot, setWorkspaceRoot] = React.useState<string>();
  const workspaceClient = React.useMemo(() => desktopWorkspaceClient(), []);

  const refreshWorkbench = React.useCallback(async (): Promise<void> => {
    const [snapshot, modelProfiles, config] = await Promise.all([
      window.openintj.workbenchBootstrap(),
      window.openintj.modelProfiles(),
      window.openintj.getConfig(),
    ]);
    setWorkspaces(snapshot.workspaces);
    setTasks(snapshot.tasks);
    setConversations(snapshot.conversations);
    setProfiles(modelProfiles);
    const workspaceId =
      snapshot.workspaces.find((item) => item.id === config.activeWorkspaceId)?.id ??
      snapshot.workspaces[0]?.id;
    const taskId =
      snapshot.tasks.find(
        (item) =>
          item.id === config.activeTaskId &&
          item.workspaceId === workspaceId &&
          item.status !== "archived",
      )?.id ?? snapshot.tasks.find((item) => item.workspaceId === workspaceId)?.id;
    const conversationId =
      snapshot.conversations.find(
        (item) => item.id === config.activeConversationId && item.taskId === taskId,
      )?.id ?? snapshot.conversations.find((item) => item.taskId === taskId)?.id;
    setActiveWorkspaceId(workspaceId);
    setActiveTaskId(taskId);
    setActiveConversationId(conversationId);
  }, []);

  React.useEffect(() => {
    void refreshWorkbench();
  }, [refreshWorkbench]);

  React.useEffect(() => {
    if (!activeConversationId) {
      setMessages([]);
      return;
    }
    void window.openintj.workbenchMessages(activeConversationId).then((entries) => {
      setMessages(
        entries.map(({ role, content, messageKind, inputStructure }) => ({
          role,
          content,
          messageKind,
          ...(inputStructure ? { inputStructure } : {}),
        })),
      );
      setTrajectory([]);
      const lastArtifact = preferredPreviewPath(
        entries.flatMap((entry) => extractWorkspaceArtifactPaths(entry.content)),
      );
      if (lastArtifact) setSelectedWorkspacePath(lastArtifact);
    });
  }, [activeConversationId]);

  React.useEffect(() => {
    void window.openintj.workspaceInfo().then((info) => setWorkspaceRoot(info.root));
  }, []);

  React.useEffect(() => {
    const api = window.openintj;
    if (!api) return;
    const offs: Array<() => void> = [];
    offs.push(
      api.onTaoEvent((p) => {
        const payload = (p ?? {}) as Record<string, unknown> & {
          kind?: string;
        };
        const kind: TrajectoryEntry["kind"] =
          payload.kind === "afterAct" ? "tao.afterAct" : "tao.beforeThink";
        setTrajectory((t) => [...t, { kind, payload, ts: Date.now() }]);
      }),
    );
    offs.push(
      api.onReactEvent((p) => {
        const payload = (p ?? {}) as Record<string, unknown> & {
          kind?: string;
        };
        const map: Record<string, TrajectoryEntry["kind"]> = {
          thought: "react.thought",
          action: "react.action",
          observation: "react.observation",
        };
        const kind = map[payload.kind ?? "thought"] ?? "react.thought";
        setTrajectory((t) => [...t, { kind, payload, ts: Date.now() }]);
      }),
    );
    offs.push(
      api.onAuditEvent((p) =>
        setTrajectory((t) => [
          ...t,
          {
            kind: "audit",
            payload: (p ?? {}) as Record<string, unknown>,
            ts: Date.now(),
          },
        ]),
      ),
    );
    offs.push(
      api.onAssistantDelta((p) => {
        const payload = (p ?? {}) as { text?: string };
        const text = typeof payload.text === "string" ? payload.text : "";
        if (!text) return;
        setMessages((current) => {
          const last = current[current.length - 1];
          if (last?.role === "assistant" && last.streaming) {
            return [...current.slice(0, -1), { ...last, content: text }];
          }
          return current;
        });
      }),
    );
    return () => {
      for (const o of offs) o();
    };
  }, []);

  React.useEffect(() => {
    const refresh = async (): Promise<void> => {
      try {
        const s = await window.openintj?.status();
        setStatus(s);
      } catch {
        // ignore
      }
    };
    void refresh();
    const id = window.setInterval(refresh, 2_000);
    return () => window.clearInterval(id);
  }, []);

  React.useEffect(() => {
    const api = window.openintj;
    if (!api?.approvalList) return;
    let alive = true;
    const refresh = async (): Promise<void> => {
      try {
        const list = await api.approvalList();
        if (alive) setApprovals(list);
      } catch {
        // approval not enabled in this host build
      }
    };
    void refresh();
    const off = api.onApprovalEvent?.(() => void refresh());
    const id = window.setInterval(() => void refresh(), 2_000);
    return () => {
      alive = false;
      off?.();
      window.clearInterval(id);
    };
  }, []);

  const decideApproval = (
    req: ApprovalRequestView,
    decision: "approved" | "denied",
    withGrant: boolean,
  ): void => {
    const path = typeof req.params["path"] === "string" ? (req.params["path"] as string) : "";
    const scope = path.includes("/") ? path.slice(0, path.lastIndexOf("/") + 1) : path || undefined;
    void window.openintj
      .approvalDecide({
        id: req.id,
        decision,
        ...(decision === "approved" && withGrant
          ? { grant: { ...(scope ? { scope } : {}), ttlMs: 3_600_000 } }
          : {}),
      })
      .then(() => window.openintj.approvalList())
      .then(setApprovals)
      .catch(() => {});
  };

  const handleSend = async (text: string): Promise<void> => {
    setPending(true);
    const activeConversation = conversations.find(
      (conversation) => conversation.id === activeConversationId,
    );
    const skipOptimistic = text.trim().startsWith("/");
    if (!skipOptimistic) {
      setMessages((current) => [
        ...current,
        { role: "user", content: text },
        { role: "assistant", content: "", streaming: true },
      ]);
    }
    try {
      const res = await window.openintj.chat({
        query: text,
        ...(activeConversationId ? { conversationId: activeConversationId } : {}),
        ...(activeConversation ? { modelProfileId: activeConversation.modelProfileId } : {}),
      });
      if (res.command) {
        if (res.command.name === "/clear" && res.command.kind === "ok") {
          setMessages([{ role: "system", content: res.finalAnswer }]);
          setTrajectory([]);
          setSelectedWorkspacePath(undefined);
        } else {
          setMessages((current) => [
            ...current.filter((message) => !message.streaming),
            { role: "system", content: res.finalAnswer },
          ]);
        }
        return;
      }
      const assistant: ChatMessage = {
        role: "assistant",
        content: res.finalAnswer,
        messageKind: res.inputStructure?.action === "clarify" ? "clarification" : "answer",
        ...(res.inputStructure ? { inputStructure: res.inputStructure } : {}),
      };
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last?.streaming) return [...current.slice(0, -1), assistant];
        return [...current, { role: "user", content: text }, assistant];
      });
      if (res.context) {
        setStatus((current) => (current ? { ...current, context: res.context } : current));
      }
      const artifactPath =
        preferredPreviewPath([
          ...(res.artifacts ?? []).map((item) => item.path),
          ...extractWorkspaceArtifactPaths(res.finalAnswer),
        ]) ?? artifactsFromAnswer(res.finalAnswer)[0]?.path;
      if (artifactPath) {
        setSelectedWorkspacePath(artifactPath);
        setRightTab("preview");
      }
    } catch (e) {
      const errorLine: ChatMessage = { role: "system", content: `[错误] ${(e as Error).message}` };
      setMessages((current) => {
        const last = current[current.length - 1];
        if (last?.streaming) return [...current.slice(0, -1), errorLine];
        return [...current, { role: "user", content: text }, errorLine];
      });
    } finally {
      setPending(false);
    }
  };

  const selectTask = (taskId: string): void => {
    setActiveTaskId(taskId);
    const conversationId = conversations.find((item) => item.taskId === taskId)?.id;
    setActiveConversationId(conversationId);
    void window.openintj.updateConfig({
      activeTaskId: taskId,
      activeConversationId: conversationId,
    });
  };

  const selectConversation = (conversationId: string): void => {
    setActiveConversationId(conversationId);
    void window.openintj.updateConfig({ activeConversationId: conversationId });
  };

  const activateWorkspace = async (created: WorkbenchWorkspace): Promise<void> => {
    const snapshot = await window.openintj.workbenchBootstrap();
    let taskId = snapshot.tasks.find(
      (task) => task.workspaceId === created.id && task.status !== "archived",
    )?.id;
    let conversationId = snapshot.conversations.find((item) => item.taskId === taskId)?.id;
    if (!taskId) {
      const createdTask = await window.openintj.createWorkbenchTask({
        parentId: created.id,
        title: "Inbox",
      });
      const createdConversation = await window.openintj.createWorkbenchConversation({
        parentId: createdTask.id,
        title: "新对话",
        modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
      });
      taskId = createdTask.id;
      conversationId = createdConversation.id;
    }
    await window.openintj.updateConfig({
      workspaceDir: created.rootPath,
      activeWorkspaceId: created.id,
      activeTaskId: taskId,
      activeConversationId: conversationId,
    });
    await refreshWorkbench();
    setSelectedWorkspacePath(undefined);
  };

  const createWorkspace = async (): Promise<void> => {
    if (activeWorkspaceId) {
      const proceed = window.confirm(
        "同时只能打开一个工作区。打开新目录会关闭当前工作区（任务和对话会保留，磁盘文件不会删除）。继续？",
      );
      if (!proceed) return;
    }
    const picked = await window.openintj.workspacePickDir();
    if (picked.canceled || !picked.root) return;
    const current = workspaces.find((workspace) => workspace.id === activeWorkspaceId);
    if (current && sameFsPath(current.rootPath, picked.root)) return;
    const suggested = picked.root.split(/[\\/]/).filter(Boolean).at(-1) ?? "工作区";
    const name = window.prompt("工作区名称", suggested)?.trim();
    if (!name) return;
    const created = await window.openintj.createWorkbenchWorkspace({
      name,
      rootPath: picked.root,
    });
    if (!created.id) return;
    await activateWorkspace(created);
    if (window.confirm("工作区已切换。现在重启并加载该目录？")) {
      await window.openintj.restartApp();
    }
  };

  const closeWorkspace = async (): Promise<void> => {
    if (!activeWorkspaceId) return;
    const proceed = window.confirm(
      "关闭当前工作区？任务和对话会保留，磁盘文件不会删除。再次打开同一目录可恢复。",
    );
    if (!proceed) return;
    await window.openintj.closeWorkbenchWorkspace({ id: activeWorkspaceId });
    setWorkspaces([]);
    setTasks([]);
    setConversations([]);
    setActiveWorkspaceId(undefined);
    setActiveTaskId(undefined);
    setActiveConversationId(undefined);
    setSelectedWorkspacePath(undefined);
    setMessages([]);
    await refreshWorkbench();
  };

  const createTask = async (): Promise<void> => {
    if (!activeWorkspaceId) return;
    const created = await window.openintj.createWorkbenchTask({
      parentId: activeWorkspaceId,
      title: "新任务",
    });
    setTasks((current) => [created, ...current]);
    setActiveTaskId(created.id);
    const createdConversation = await window.openintj.createWorkbenchConversation({
      parentId: created.id,
      title: "新对话",
      modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
    });
    setConversations((current) => [createdConversation, ...current]);
    setActiveConversationId(createdConversation.id);
  };

  const createConversation = async (): Promise<void> => {
    if (!activeTaskId) return;
    const created = await window.openintj.createWorkbenchConversation({
      parentId: activeTaskId,
      title: "新对话",
      modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
    });
    setConversations((current) => [created, ...current]);
    setActiveConversationId(created.id);
  };

  const updateTask = async (
    selected: WorkbenchTask,
    patch: { title?: string; status?: "active" | "completed" | "archived" },
  ): Promise<void> => {
    const updated = await window.openintj.updateWorkbenchTask({ id: selected.id, ...patch });
    setTasks((current) => current.map((task) => (task.id === updated.id ? updated : task)));
    if (updated.status === "archived") {
      const next = tasks.find(
        (task) =>
          task.workspaceId === selected.workspaceId &&
          task.id !== selected.id &&
          task.status !== "archived",
      );
      if (next) selectTask(next.id);
    }
  };

  const changeConversationModel = async (modelProfileId: string): Promise<void> => {
    if (!activeConversationId) return;
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === activeConversationId
          ? { ...conversation, modelProfileId }
          : conversation,
      ),
    );
    const updated = await window.openintj.updateWorkbenchConversation({
      id: activeConversationId,
      modelProfileId,
    });
    setConversations((current) =>
      current.map((conversation) => (conversation.id === updated.id ? updated : conversation)),
    );
  };

  const dormantEnabled = status?.dormant?.enabled === true;
  const dormantPending = status?.dormant?.pendingProposals ?? 0;
  const skillsEnabled = status?.skills?.enabled === true;
  const learningEnabled = status?.skills?.learning === true;
  const skillsPending = status?.skills?.pendingProposals ?? 0;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-backdrop text-ink">
      <TitleBar />
      <UpdateBanner />
      <div className="flex-1 min-h-0 flex">
        <div className="flex-1 min-w-0 min-h-0">
          <WorkspaceWorkbench
            client={workspaceClient}
            rootLabel={
              workspaces.find((item) => item.id === activeWorkspaceId)?.rootPath ?? workspaceRoot
            }
            selectedPath={selectedWorkspacePath}
            onSelectPath={(path) => setSelectedWorkspacePath(path)}
            workspaceOpen={Boolean(activeWorkspaceId)}
          />
        </div>
        <div className="w-[380px] shrink-0 flex flex-col h-full min-h-0 bg-panel border-l border-line">
          <div className="shrink-0 px-4 py-3 border-b border-line text-xs bg-inspector/60">
            <div className="text-[10px] font-bold uppercase tracking-[0.13em] text-muted">
              工作区
            </div>
            <div
              className="truncate mt-1 text-[13px] font-semibold text-ink"
              title={workspaces.find((item) => item.id === activeWorkspaceId)?.rootPath}
            >
              {workspaces.find((item) => item.id === activeWorkspaceId)?.name ?? "未打开工作区"}
            </div>
            <div className="mt-2.5 flex gap-1.5">
              <button
                type="button"
                className="flex-1 px-2 py-1.5 rounded-lg bg-ink text-white hover:bg-inksoft"
                onClick={() => void createWorkspace()}
              >
                打开工作区
              </button>
              <button
                type="button"
                className="flex-1 px-2 py-1.5 rounded-lg border border-line bg-white hover:border-linedark text-ink disabled:opacity-40"
                disabled={!activeWorkspaceId}
                onClick={() => void closeWorkspace()}
              >
                关闭工作区
              </button>
            </div>
            {activeWorkspaceId ? (
              <button
                type="button"
                className="mt-1.5 w-full px-2 py-1.5 rounded-lg border border-line bg-white hover:border-linedark text-ink"
                onClick={() => void workspaceClient.reveal()}
              >
                打开工作区目录
              </button>
            ) : null}
          </div>
          <div className="shrink-0 flex items-center gap-1 border-b border-line bg-rail/50 px-2 py-1.5 text-xs overflow-x-auto whitespace-nowrap">
            {(
              [
                ["chat", "对话"],
                ["tasks", "任务"],
                ["trajectory", "推理轨迹"],
                ["memory", "记忆"],
                ["profile", "档案"],
                ["identity", "身份"],
                ["schedule", "定时"],
                ["dormant", "Dormant"],
                ["skills", "技能"],
                ["system", "系统"],
                ["settings", "设置"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setRightTab(id)}
                className={
                  rightTab === id
                    ? "shrink-0 rounded-lg bg-white px-3 py-1.5 font-semibold text-ink shadow-card"
                    : "shrink-0 rounded-lg px-3 py-1.5 text-muted hover:bg-white/70 hover:text-ink"
                }
              >
                {label}
                {id === "trajectory" && trajectory.length > 0 ? (
                  <span className="ml-1.5 font-mono text-[10px] text-faint">
                    {trajectory.length}
                  </span>
                ) : null}
                {id === "dormant" && dormantEnabled && dormantPending > 0 ? (
                  <span className="ml-1.5 px-1.5 py-0.5 rounded-full text-[10px] bg-ambertint text-amberdeep">
                    {dormantPending}
                  </span>
                ) : null}
                {id === "skills" && learningEnabled && skillsPending > 0 ? (
                  <span className="ml-1.5 px-1.5 py-0.5 rounded-full text-[10px] bg-ambertint text-amberdeep">
                    {skillsPending}
                  </span>
                ) : null}
              </button>
            ))}
            {selectedWorkspacePath ? (
              <button
                type="button"
                onClick={() => setRightTab("preview")}
                className={
                  rightTab === "preview"
                    ? "shrink-0 rounded-lg bg-white px-3 py-1.5 font-semibold text-ink shadow-card"
                    : "shrink-0 rounded-lg px-3 py-1.5 text-muted hover:bg-white/70 hover:text-ink"
                }
              >
                预览
              </button>
            ) : null}
          </div>
          <ApprovalBanner pending={approvals} onDecide={decideApproval} />
          <div className="flex-1 min-h-0">
            {rightTab === "chat" ? (
              <ChatPanel
                messages={messages}
                onSend={(t) => void handleSend(t)}
                pending={pending}
                profiles={profiles}
                modelProfileId={
                  conversations.find((conversation) => conversation.id === activeConversationId)
                    ?.modelProfileId
                }
                onModelChange={(id) => void changeConversationModel(id)}
                context={status?.context}
                onOpenWorkspacePath={(path) => {
                  setSelectedWorkspacePath(path);
                  setRightTab("preview");
                }}
                onSkillFeedback={
                  activeConversationId
                    ? (sentiment) => {
                        void window.openintj.skillsFeedback({
                          conversationId: activeConversationId,
                          sentiment,
                        });
                      }
                    : undefined
                }
              />
            ) : rightTab === "tasks" ? (
              <TaskSidebar
                showWorkspace={false}
                workspaces={workspaces}
                tasks={tasks}
                conversations={conversations}
                activeWorkspaceId={activeWorkspaceId}
                activeTaskId={activeTaskId}
                activeConversationId={activeConversationId}
                onCreateWorkspace={() => void createWorkspace()}
                onCloseWorkspace={() => void closeWorkspace()}
                onSelectTask={selectTask}
                onSelectConversation={selectConversation}
                onCreateTask={() => void createTask()}
                onCreateConversation={() => void createConversation()}
                onRenameTask={(task) => {
                  const title = window.prompt("任务名称", task.title)?.trim();
                  if (title) void updateTask(task, { title });
                }}
                onArchiveTask={(task) => void updateTask(task, { status: "archived" })}
              />
            ) : rightTab === "trajectory" ? (
              <TrajectoryPanel entries={trajectory} />
            ) : rightTab === "memory" ? (
              <MemoryPanel />
            ) : rightTab === "profile" ? (
              <ProfilePanel />
            ) : rightTab === "identity" ? (
              <IdentityPanel />
            ) : rightTab === "schedule" ? (
              <SchedulePanel />
            ) : rightTab === "dormant" ? (
              <DormantPanel enabled={dormantEnabled} />
            ) : rightTab === "skills" ? (
              <SkillPanel skillsEnabled={skillsEnabled} learningEnabled={learningEnabled} />
            ) : rightTab === "system" ? (
              <SystemPanel />
            ) : rightTab === "preview" && selectedWorkspacePath ? (
              <FilePreview
                client={workspaceClient}
                path={selectedWorkspacePath}
                onReveal={(path) => void workspaceClient.reveal(path)}
              />
            ) : (
              <SettingsPanel />
            )}
          </div>
        </div>
      </div>
      <StatusBar status={status} />
    </div>
  );
};
