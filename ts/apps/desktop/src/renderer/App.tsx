import React from "react";
import type {
  ModelProfile,
  WorkbenchConversation,
  WorkbenchTask,
  WorkbenchWorkspace,
} from "../shared/ipc-protocol.js";
import { DEFAULT_DESKTOP_MODEL_PROFILE_ID } from "../shared/ipc-protocol.js";
import {
  artifactsFromAnswer,
  extractWorkspaceArtifactPaths,
  preferredPreviewPath,
} from "../shared/workspace-view.js";
import { type ChatMessage, ChatPanel } from "./components/ChatPanel.js";
import { DormantPanel } from "./components/DormantPanel.js";
import { MemoryPanel } from "./components/MemoryPanel.js";
import { SettingsPanel } from "./components/SettingsPanel.js";
import { SkillPanel } from "./components/SkillPanel.js";
import { StatusBar, type StatusSnapshot } from "./components/StatusBar.js";
import { TaskSidebar } from "./components/TaskSidebar.js";
import { type TrajectoryEntry, TrajectoryPanel } from "./components/TrajectoryPanel.js";
import { UpdateBanner } from "./components/UpdateBanner.js";
import {
  EditorWorkbench,
  WorkspaceWorkbench,
  desktopWorkspaceClient,
} from "./workspace/index.js";
import "./types.js";

type RightTab = "workspace" | "trajectory" | "memory" | "dormant" | "skills" | "settings";

const sameFsPath = (left: string, right: string): boolean =>
  left.replace(/\\/g, "/").toLowerCase() === right.replace(/\\/g, "/").toLowerCase();

export const App = (): JSX.Element => {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [trajectory, setTrajectory] = React.useState<TrajectoryEntry[]>([]);
  const [status, setStatus] = React.useState<StatusSnapshot | undefined>();
  const [pending, setPending] = React.useState(false);
  const [rightTab, setRightTab] = React.useState<RightTab>("trajectory");
  const [profiles, setProfiles] = React.useState<ModelProfile[]>([]);
  const [workspaces, setWorkspaces] = React.useState<WorkbenchWorkspace[]>([]);
  const [tasks, setTasks] = React.useState<WorkbenchTask[]>([]);
  const [conversations, setConversations] = React.useState<WorkbenchConversation[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = React.useState<string>();
  const [activeTaskId, setActiveTaskId] = React.useState<string>();
  const [activeConversationId, setActiveConversationId] = React.useState<string>();
  const [selectedWorkspacePath, setSelectedWorkspacePath] = React.useState<string>();
  const [workspaceRoot, setWorkspaceRoot] = React.useState<string>();
  const [workspaceRefresh, setWorkspaceRefresh] = React.useState(0);
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
      api.onWorkspaceEvent(() => setWorkspaceRefresh((value) => value + 1)),
    );
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

  const handleSend = async (text: string): Promise<void> => {
    setMessages((m) => [...m, { role: "user", content: text }]);
    setPending(true);
    try {
      const activeConversation = conversations.find(
        (conversation) => conversation.id === activeConversationId,
      );
      const res = await window.openintj.chat({
        query: text,
        ...(activeConversationId ? { conversationId: activeConversationId } : {}),
        ...(activeConversation ? { modelProfileId: activeConversation.modelProfileId } : {}),
      });
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: res.finalAnswer,
          messageKind: res.inputStructure?.action === "clarify" ? "clarification" : "answer",
          ...(res.inputStructure ? { inputStructure: res.inputStructure } : {}),
        },
      ]);
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
        setRightTab("workspace");
        setWorkspaceRefresh((value) => value + 1);
      }
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "system",
          content: `[错误] ${(e as Error).message}`,
        },
      ]);
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
    <div className="flex flex-col h-screen overflow-hidden text-gray-200">
      <header className="shrink-0 px-4 py-2 border-b border-gray-800 bg-[#181825] flex items-center gap-2">
        <span className="text-base font-semibold">OpenINTJ</span>
        <span className="text-xs text-gray-500">v0.3.1</span>
        <button
          type="button"
          onClick={() => void window.openintj.restartApp()}
          className="ml-auto px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-xs"
        >
          重启
        </button>
      </header>
      <UpdateBanner />
      <div
        className="flex-1 grid min-h-0"
        style={{ gridTemplateColumns: "240px minmax(0, 1fr) 380px" }}
      >
        <TaskSidebar
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
          onOpenWorkspaceDir={() => void workspaceClient.reveal()}
        />
        <div className="flex flex-col min-h-0 min-w-0">
          <div className="flex-1 min-h-0">
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
              }}
            />
          </div>
          <div className="h-[46%] min-h-[260px] border-t border-gray-800">
            <EditorWorkbench
              key={activeWorkspaceId ?? "none"}
              client={workspaceClient}
              openPath={selectedWorkspacePath}
              rootLabel={workspaceRoot}
              refreshKey={workspaceRefresh}
              workspaceOpen={Boolean(activeWorkspaceId)}
              onActivePathChange={setSelectedWorkspacePath}
            />
          </div>
        </div>
        <div className="flex flex-col h-full min-h-0 bg-[#11111b] border-l border-gray-800">
          <div className="shrink-0 flex items-center border-b border-gray-800 text-xs overflow-x-auto whitespace-nowrap">
            <button
              type="button"
              onClick={() => setRightTab("workspace")}
              className={
                rightTab === "workspace"
                  ? "shrink-0 px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "shrink-0 px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              工作区
            </button>
            <button
              type="button"
              onClick={() => setRightTab("trajectory")}
              className={
                rightTab === "trajectory"
                  ? "shrink-0 px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "shrink-0 px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              推理轨迹
              {trajectory.length > 0 ? (
                <span className="ml-1.5 text-[10px] text-gray-500">{trajectory.length}</span>
              ) : null}
            </button>
            <button
              type="button"
              onClick={() => setRightTab("memory")}
              className={
                rightTab === "memory"
                  ? "shrink-0 px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "shrink-0 px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              记忆
            </button>
            <button
              type="button"
              onClick={() => setRightTab("dormant")}
              className={
                rightTab === "dormant"
                  ? "shrink-0 px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "shrink-0 px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              Dormant
              {dormantEnabled && dormantPending > 0 ? (
                <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] bg-yellow-700 text-yellow-100">
                  {dormantPending}
                </span>
              ) : null}
            </button>
            <button
              type="button"
              onClick={() => setRightTab("skills")}
              className={
                rightTab === "skills"
                  ? "shrink-0 px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "shrink-0 px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              技能
              {learningEnabled && skillsPending > 0 ? (
                <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] bg-yellow-700 text-yellow-100">
                  {skillsPending}
                </span>
              ) : null}
            </button>
            <button
              type="button"
              onClick={() => setRightTab("settings")}
              className={
                rightTab === "settings"
                  ? "shrink-0 px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "shrink-0 px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              设置
            </button>
          </div>
          <div className="flex-1 min-h-0">
            {rightTab === "workspace" ? (
              <WorkspaceWorkbench
                client={workspaceClient}
                rootLabel={workspaceRoot}
                selectedPath={selectedWorkspacePath}
                onSelectPath={(path) => setSelectedWorkspacePath(path)}
                showPreview={false}
                refreshKey={workspaceRefresh}
                workspaceOpen={Boolean(activeWorkspaceId)}
              />
            ) : rightTab === "trajectory" ? (
              <TrajectoryPanel entries={trajectory} />
            ) : rightTab === "memory" ? (
              <MemoryPanel />
            ) : rightTab === "dormant" ? (
              <DormantPanel enabled={dormantEnabled} />
            ) : rightTab === "skills" ? (
              <SkillPanel skillsEnabled={skillsEnabled} learningEnabled={learningEnabled} />
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
