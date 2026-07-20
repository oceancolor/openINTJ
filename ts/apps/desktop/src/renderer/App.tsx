import React from "react";
import type {
  ModelProfile,
  WorkbenchConversation,
  WorkbenchTask,
  WorkbenchWorkspace,
} from "../shared/ipc-protocol.js";
import { DEFAULT_DESKTOP_MODEL_PROFILE_ID } from "../shared/ipc-protocol.js";
import { type ChatMessage, ChatPanel } from "./components/ChatPanel.js";
import { DormantPanel } from "./components/DormantPanel.js";
import { MemoryPanel } from "./components/MemoryPanel.js";
import { SettingsPanel } from "./components/SettingsPanel.js";
import { SkillPanel } from "./components/SkillPanel.js";
import { StatusBar, type StatusSnapshot } from "./components/StatusBar.js";
import { TaskSidebar } from "./components/TaskSidebar.js";
import { type TrajectoryEntry, TrajectoryPanel } from "./components/TrajectoryPanel.js";
import { UpdateBanner } from "./components/UpdateBanner.js";
import "./types.js";

type RightTab = "trajectory" | "memory" | "dormant" | "skills" | "settings";

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
      setMessages(entries.map(({ role, content }) => ({ role, content })));
      setTrajectory([]);
    });
  }, [activeConversationId]);

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
      setMessages((m) => [...m, { role: "assistant", content: res.finalAnswer }]);
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

  const selectWorkspace = async (workspaceId: string): Promise<void> => {
    const selected = workspaces.find((workspace) => workspace.id === workspaceId);
    if (!selected) return;
    const taskId = tasks.find(
      (task) => task.workspaceId === workspaceId && task.status !== "archived",
    )?.id;
    const conversationId = conversations.find((item) => item.taskId === taskId)?.id;
    setActiveWorkspaceId(workspaceId);
    setActiveTaskId(taskId);
    setActiveConversationId(conversationId);
    await window.openintj.updateConfig({
      workspaceDir: selected.rootPath,
      activeWorkspaceId: workspaceId,
      activeTaskId: taskId,
      activeConversationId: conversationId,
    });
    if (window.confirm("切换工作区需要重装配 Agent。现在重启 OpenINTJ？")) {
      await window.openintj.restartApp();
    }
  };

  const createWorkspace = async (): Promise<void> => {
    const picked = await window.openintj.workspacePickDir();
    if (picked.canceled || !picked.root) return;
    const suggested = picked.root.split(/[\\/]/).filter(Boolean).at(-1) ?? "工作区";
    const name = window.prompt("工作区名称", suggested)?.trim();
    if (!name) return;
    const created = await window.openintj.createWorkbenchWorkspace({
      name,
      rootPath: picked.root,
    });
    const createdTask = await window.openintj.createWorkbenchTask({
      parentId: created.id,
      title: "Inbox",
    });
    const createdConversation = await window.openintj.createWorkbenchConversation({
      parentId: createdTask.id,
      title: "新对话",
      modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
    });
    setWorkspaces((current) => [created, ...current]);
    setTasks((current) => [createdTask, ...current]);
    setConversations((current) => [createdConversation, ...current]);
    setActiveWorkspaceId(created.id);
    setActiveTaskId(createdTask.id);
    setActiveConversationId(createdConversation.id);
    await window.openintj.updateConfig({
      workspaceDir: created.rootPath,
      activeWorkspaceId: created.id,
      activeTaskId: createdTask.id,
      activeConversationId: createdConversation.id,
    });
    if (window.confirm("新工作区已创建。现在重启并切换？")) {
      await window.openintj.restartApp();
    }
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
  const skillsPending = status?.skills?.pendingProposals ?? 0;

  return (
    <div className="flex flex-col h-screen text-gray-200">
      <header className="px-4 py-2 border-b border-gray-800 bg-[#181825] flex items-center gap-2">
        <span className="text-base font-semibold">OpenINTJ</span>
        <span className="text-xs text-gray-500">v3.0 Local Desktop</span>
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
          onSelectWorkspace={(id) => void selectWorkspace(id)}
          onCreateWorkspace={() => void createWorkspace()}
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
        />
        <div className="flex flex-col h-full bg-[#11111b] border-l border-gray-800">
          <div className="flex items-center border-b border-gray-800 text-xs">
            <button
              type="button"
              onClick={() => setRightTab("trajectory")}
              className={
                rightTab === "trajectory"
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
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
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              记忆
            </button>
            <button
              type="button"
              onClick={() => setRightTab("dormant")}
              className={
                rightTab === "dormant"
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
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
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              技能
              {skillsEnabled && skillsPending > 0 ? (
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
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              设置
            </button>
          </div>
          <div className="flex-1 min-h-0">
            {rightTab === "trajectory" ? (
              <TrajectoryPanel entries={trajectory} />
            ) : rightTab === "memory" ? (
              <MemoryPanel />
            ) : rightTab === "dormant" ? (
              <DormantPanel enabled={dormantEnabled} />
            ) : rightTab === "skills" ? (
              <SkillPanel enabled={skillsEnabled} />
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
