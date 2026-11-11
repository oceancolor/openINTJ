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
import { AgentContextBar } from "./components/AgentContextBar.js";
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
import { TimelinePanel } from "./components/TimelinePanel.js";
import { TitleBar } from "./components/TitleBar.js";
import { type TrajectoryEntry } from "./components/TrajectoryPanel.js";
import { UpdateBanner } from "./components/UpdateBanner.js";
import { FilePreview, WorkspaceWorkbench, desktopWorkspaceClient } from "./workspace/index.js";
import "./types.js";

type RightTab =
  | "tasks"
  | "timeline"
  | "memory"
  | "profile"
  | "identity"
  | "schedule"
  | "dormant"
  | "skills"
  | "system"
  | "settings";

const sameFsPath = (left: string, right: string): boolean =>
  left.replace(/\\/g, "/").toLowerCase() === right.replace(/\\/g, "/").toLowerCase();

/* ---- 三栏布局尺寸（可拖拽，localStorage 持久化） ---- */
const clampSize = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const loadLayoutNumber = (key: string, fallback: number): number => {
  try {
    const raw = Number(window.localStorage.getItem(key));
    return Number.isFinite(raw) && raw > 0 ? raw : fallback;
  } catch {
    return fallback;
  }
};

const saveLayoutNumber = (key: string, value: number): void => {
  try {
    window.localStorage.setItem(key, String(value));
  } catch {
    // 隐私模式等场景下静默放弃持久化，不影响拖拽本身。
  }
};

const LEFT_WIDTH = { key: "openintj.layout.leftWidth", def: 268, min: 200, max: 460 };
const RIGHT_WIDTH = { key: "openintj.layout.rightWidth", def: 360, min: 260, max: 580 };
const PREVIEW_RATIO = { key: "openintj.layout.previewRatio", def: 0.36, min: 0.14, max: 0.72 };

interface SplitterProps {
  orientation: "vertical" | "horizontal";
  testId: string;
  label: string;
  onPointerDown: (event: React.PointerEvent) => void;
  onDoubleClick: () => void;
  onNudge: (delta: number) => void;
}

/** 可拖拽分隔条：鼠标拖动 / 方向键微调 / 双击复位，悬停时呈浅紫高亮。 */
const Splitter: React.FC<SplitterProps> = ({
  orientation,
  testId,
  label,
  onPointerDown,
  onDoubleClick,
  onNudge,
}) => {
  const vertical = orientation === "vertical";
  return (
    <div
      role="separator"
      aria-orientation={orientation}
      aria-label={label}
      title={`${label}（拖动调整，双击复位）`}
      data-testid={testId}
      tabIndex={0}
      className={
        vertical
          ? "group relative z-10 w-[7px] shrink-0 cursor-col-resize touch-none outline-none"
          : "group relative z-10 h-[7px] shrink-0 cursor-row-resize touch-none outline-none"
      }
      onPointerDown={onPointerDown}
      onDoubleClick={onDoubleClick}
      onKeyDown={(event) => {
        const keys = vertical ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
        if (!keys.includes(event.key)) return;
        event.preventDefault();
        onNudge(event.key === keys[1] ? 10 : -10);
      }}
    >
      <span
        className={
          (vertical
            ? "absolute inset-y-0 left-1/2 w-px -translate-x-1/2 "
            : "absolute inset-x-0 top-1/2 h-px -translate-y-1/2 ") +
          "bg-line transition-all group-hover:bg-[#8f7fe0] group-focus-visible:bg-[#8f7fe0] " +
          (vertical ? "group-hover:w-[3px] group-focus-visible:w-[3px]" : "group-hover:h-[3px] group-focus-visible:h-[3px]")
        }
      />
    </div>
  );
};

export const App = (): JSX.Element => {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [trajectory, setTrajectory] = React.useState<TrajectoryEntry[]>([]);
  const [status, setStatus] = React.useState<StatusSnapshot | undefined>();
  const [pending, setPending] = React.useState(false);
  const [approvals, setApprovals] = React.useState<ApprovalRequestView[]>([]);
  const [rightTab, setRightTab] = React.useState<RightTab>("tasks");
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

  /* ---- 可拖拽布局尺寸 ---- */
  const [leftWidth, setLeftWidth] = React.useState(() =>
    clampSize(loadLayoutNumber(LEFT_WIDTH.key, LEFT_WIDTH.def), LEFT_WIDTH.min, LEFT_WIDTH.max),
  );
  const [rightWidth, setRightWidth] = React.useState(() =>
    clampSize(loadLayoutNumber(RIGHT_WIDTH.key, RIGHT_WIDTH.def), RIGHT_WIDTH.min, RIGHT_WIDTH.max),
  );
  const [previewRatio, setPreviewRatio] = React.useState(() =>
    clampSize(
      loadLayoutNumber(PREVIEW_RATIO.key, PREVIEW_RATIO.def),
      PREVIEW_RATIO.min,
      PREVIEW_RATIO.max,
    ),
  );
  const centerColRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => saveLayoutNumber(LEFT_WIDTH.key, leftWidth), [leftWidth]);
  React.useEffect(() => saveLayoutNumber(RIGHT_WIDTH.key, rightWidth), [rightWidth]);
  React.useEffect(() => saveLayoutNumber(PREVIEW_RATIO.key, previewRatio), [previewRatio]);

  /** 通用拖拽会话：按下后跟随指针，松开清理；期间锁定光标与文本选区。 */
  const beginDrag = (
    event: React.PointerEvent,
    cursor: string,
    onMove: (dx: number, dy: number) => void,
  ): void => {
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const move = (e: PointerEvent): void => onMove(e.clientX - startX, e.clientY - startY);
    const up = (): void => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    document.body.style.cursor = cursor;
    document.body.style.userSelect = "none";
  };

  const dragLeftWidth = (event: React.PointerEvent): void => {
    const start = leftWidth;
    beginDrag(event, "col-resize", (dx) =>
      setLeftWidth(clampSize(start + dx, LEFT_WIDTH.min, LEFT_WIDTH.max)),
    );
  };
  const dragRightWidth = (event: React.PointerEvent): void => {
    const start = rightWidth;
    beginDrag(event, "col-resize", (dx) =>
      setRightWidth(clampSize(start - dx, RIGHT_WIDTH.min, RIGHT_WIDTH.max)),
    );
  };
  const dragPreviewRatio = (event: React.PointerEvent): void => {
    const start = previewRatio;
    const height = centerColRef.current?.getBoundingClientRect().height ?? 0;
    if (height <= 0) return;
    beginDrag(event, "row-resize", (_dx, dy) =>
      setPreviewRatio(clampSize(start + dy / height, PREVIEW_RATIO.min, PREVIEW_RATIO.max)),
    );
  };

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

  const activeWorkspace = workspaces.find((item) => item.id === activeWorkspaceId);
  // 本任务产物（轻联动）：从当前对话消息中提取已生成的文件路径。
  const taskArtifacts = [
    ...new Set(messages.flatMap((message) => extractWorkspaceArtifactPaths(message.content))),
  ].slice(0, 12);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-backdrop text-ink">
      <TitleBar />
      <UpdateBanner />
      <div className="flex-1 min-h-0 flex">
        {/* 左栏：工作区（切换 + 文件管理器） */}
        <div
          className="shrink-0 flex flex-col h-full min-h-0 bg-panel"
          style={{ width: leftWidth }}
        >
          <div className="shrink-0 px-3 py-2.5 border-b border-line text-xs">
            <div className="text-[10px] font-bold uppercase tracking-[0.13em] text-muted">
              工作区
            </div>
            <div
              className="truncate mt-1 text-[13px] font-semibold text-ink"
              title={activeWorkspace?.rootPath}
            >
              {activeWorkspace?.name ?? "未打开工作区"}
            </div>
            <div className="mt-2 flex gap-1.5">
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
          <div className="flex-1 min-h-0">
            <WorkspaceWorkbench
              client={workspaceClient}
              rootLabel={activeWorkspace?.rootPath ?? workspaceRoot}
              selectedPath={selectedWorkspacePath}
              onSelectPath={(path) => setSelectedWorkspacePath(path)}
              showPreview={false}
              workspaceOpen={Boolean(activeWorkspaceId)}
            />
          </div>
        </div>
        <Splitter
          orientation="vertical"
          testId="layout-splitter-left"
          label="调整工作区栏宽度"
          onPointerDown={dragLeftWidth}
          onDoubleClick={() => setLeftWidth(LEFT_WIDTH.def)}
          onNudge={(delta) =>
            setLeftWidth((w) => clampSize(w + delta, LEFT_WIDTH.min, LEFT_WIDTH.max))
          }
        />
        {/* 中栏：Agent 主视图（上下文条 + 可分屏预览 + 对话） */}
        <div ref={centerColRef} className="flex-1 min-w-0 flex flex-col h-full min-h-0">
          <AgentContextBar
            workspaceName={activeWorkspace?.name}
            tasks={tasks.filter(
              (task) => task.workspaceId === activeWorkspaceId && task.status !== "archived",
            )}
            conversations={conversations}
            activeTaskId={activeTaskId}
            activeConversationId={activeConversationId}
            onSelectTask={selectTask}
            onSelectConversation={selectConversation}
            context={status?.context}
            previewPath={selectedWorkspacePath}
            onClosePreview={() => setSelectedWorkspacePath(undefined)}
          />
          {selectedWorkspacePath ? (
            <>
              <div
                className="shrink-0 min-h-0 bg-panel"
                style={{ height: `${previewRatio * 100}%` }}
              >
                <FilePreview
                  client={workspaceClient}
                  path={selectedWorkspacePath}
                  onReveal={(path) => void workspaceClient.reveal(path)}
                />
              </div>
              <Splitter
                orientation="horizontal"
                testId="layout-splitter-preview"
                label="调整预览区高度"
                onPointerDown={dragPreviewRatio}
                onDoubleClick={() => setPreviewRatio(PREVIEW_RATIO.def)}
                onNudge={(delta) =>
                  setPreviewRatio((r) =>
                    clampSize(r + delta / 400, PREVIEW_RATIO.min, PREVIEW_RATIO.max),
                  )
                }
              />
            </>
          ) : null}
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
          </div>
        </div>
        <Splitter
          orientation="vertical"
          testId="layout-splitter-right"
          label="调整检查器栏宽度"
          onPointerDown={dragRightWidth}
          onDoubleClick={() => setRightWidth(RIGHT_WIDTH.def)}
          onNudge={(delta) =>
            setRightWidth((w) => clampSize(w - delta, RIGHT_WIDTH.min, RIGHT_WIDTH.max))
          }
        />
        {/* 右栏：检查器（任务 / 时间线 / 记忆 …） */}
        <div
          className="shrink-0 flex flex-col h-full min-h-0 bg-panel"
          style={{ width: rightWidth }}
        >
          <div className="shrink-0 flex items-center gap-1 border-b border-line bg-rail/50 px-2 py-1.5 text-xs overflow-x-auto whitespace-nowrap">
            {(
              [
                ["tasks", "任务"],
                ["timeline", "时间线"],
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
                {id === "timeline" && trajectory.length > 0 ? (
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
          </div>
          <ApprovalBanner pending={approvals} onDecide={decideApproval} />
          <div className="flex-1 min-h-0">
            {rightTab === "tasks" ? (
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
                taskArtifacts={taskArtifacts}
                onOpenArtifact={(path) => setSelectedWorkspacePath(path)}
              />
            ) : rightTab === "timeline" ? (
              <TimelinePanel
                entries={trajectory}
                taskPoolActive={status?.taskPool?.active ?? false}
                pendingApprovals={approvals.length}
                running={pending}
              />
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
