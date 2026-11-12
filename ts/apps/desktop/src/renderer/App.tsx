import React from "react";
import type {
  ModelProfile,
  WorkbenchConversation,
  WorkbenchTask,
  WorkbenchWorkspace,
  WorkspacePickResponse,
} from "../shared/ipc-protocol.js";
import {
  DEFAULT_DESKTOP_MODEL_PROFILE_ID,
  DEFAULT_NEW_TASK_TITLE,
} from "../shared/ipc-protocol.js";
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
import { ConfirmDialog } from "./components/ConfirmDialog.js";
import { MemoryPanel } from "./components/MemoryPanel.js";
import { NameDialog } from "./components/NameDialog.js";
import { OllamaBanner } from "./components/OllamaBanner.js";
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
  /** 已从页签行关闭的会话 id（仅隐藏页签，会话仍在任务列表里可重开/删除）。 */
  const [closedTabIds, setClosedTabIds] = React.useState<string[]>([]);
  const [profiles, setProfiles] = React.useState<ModelProfile[]>([]);
  const [workspaces, setWorkspaces] = React.useState<WorkbenchWorkspace[]>([]);
  const [tasks, setTasks] = React.useState<WorkbenchTask[]>([]);
  const [conversations, setConversations] = React.useState<WorkbenchConversation[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = React.useState<string>();
  const [activeTaskId, setActiveTaskId] = React.useState<string>();
  const [activeConversationId, setActiveConversationId] = React.useState<string>();
  const [selectedWorkspacePath, setSelectedWorkspacePath] = React.useState<string>();
  const [workspaceRoot, setWorkspaceRoot] = React.useState<string>();
  /** 文件树刷新代数：agent 文件根变化（工作区切换/热重载）时递增，强制左栏重拉目录。 */
  const [workspaceTreeKey, setWorkspaceTreeKey] = React.useState(0);
  React.useEffect(() => {
    if (workspaceRoot) setWorkspaceTreeKey((key) => key + 1);
  }, [workspaceRoot]);
  /** 首屏 bootstrap 是否完成（用于判断是否需要提示选择工作区）。 */
  const [workbenchLoaded, setWorkbenchLoaded] = React.useState(false);
  /* Ollama 降级提示：用户明确接受降级后持久化，不再每次启动打扰。 */
  const [ollamaBusy, setOllamaBusy] = React.useState(false);
  const [ollamaNote, setOllamaNote] = React.useState<string>();
  const [degradedAccepted, setDegradedAccepted] = React.useState<boolean>(() => {
    try {
      return window.localStorage.getItem("openintj.embed.degradedAccepted") === "1";
    } catch {
      return false;
    }
  });
  const workspaceClient = React.useMemo(() => desktopWorkspaceClient(), []);
  /** 工作台切换（新建任务/对话）后，等新会话消息加载完再补一条系统提示。 */
  const pendingNoticeRef = React.useRef<string | undefined>(undefined);

  /* 应用内命名对话框（替代 Electron 不支持的 window.prompt）。 */
  const [nameDialog, setNameDialog] = React.useState<{
    title: string;
    label?: string;
    initialValue: string;
    resolve: (value: string | null) => void;
  } | null>(null);
  const askName = (
    title: string,
    initialValue: string,
    label?: string,
  ): Promise<string | null> =>
    new Promise((resolve) => {
      setNameDialog({ title, initialValue, ...(label ? { label } : {}), resolve });
    });
  const closeNameDialog = (value: string | null): void => {
    nameDialog?.resolve(value);
    setNameDialog(null);
  };

  /* 应用内确认对话框（替代 window.confirm）。 */
  const [confirmDialog, setConfirmDialog] = React.useState<{
    title: string;
    message: string;
    confirmLabel?: string;
    danger?: boolean;
    resolve: (value: boolean) => void;
  } | null>(null);
  const askConfirm = (
    title: string,
    message: string,
    opts: { confirmLabel?: string; danger?: boolean } = {},
  ): Promise<boolean> =>
    new Promise((resolve) => {
      setConfirmDialog({ title, message, ...opts, resolve });
    });
  const closeConfirmDialog = (value: boolean): void => {
    confirmDialog?.resolve(value);
    setConfirmDialog(null);
  };

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
    setWorkbenchLoaded(true);
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
      const loaded: ChatMessage[] = entries.map(
        ({ role, content, messageKind, inputStructure }) => ({
          role,
          content,
          messageKind,
          ...(inputStructure ? { inputStructure } : {}),
        }),
      );
      const notice = pendingNoticeRef.current;
      if (notice) {
        pendingNoticeRef.current = undefined;
        loaded.push({ role: "system", content: notice });
      }
      setMessages(loaded);
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

  /* ---- Ollama 显式生命周期（自检 / 启动 / 接受降级） ---- */
  const embedStatus = status?.modelRuntime.embed;
  const embedDegraded = Boolean(
    embedStatus &&
      (embedStatus.fallbackFrom === "ollama" ||
        (embedStatus.provider === "simple-sha256" && embedStatus.mode === "mock")),
  );

  const launchOllama = async (): Promise<void> => {
    setOllamaBusy(true);
    setOllamaNote(undefined);
    try {
      const res = await window.openintj.ollamaLaunch();
      if (res.running && res.modelInstalled) {
        setOllamaNote(
          res.launched
            ? `自检通过：Ollama 已启动，模型 ${res.embedModel} 就绪，embedding 已切回真实向量。`
            : `自检通过：Ollama 在运行，模型 ${res.embedModel} 就绪，embedding 已重新绑定。`,
        );
        try {
          setStatus(await window.openintj.status());
        } catch {
          // 状态轮询会随后刷新，失败不影响结论。
        }
      } else if (res.running) {
        setOllamaNote(
          `Ollama 已启动，但 embedding 模型 ${res.embedModel} 未安装：在终端执行 ollama pull ${res.embedModel} 后再自检；在此之前仍是降级向量。`,
        );
      } else {
        setOllamaNote(
          res.launchError ??
            `自检未通过：${res.reason ?? "服务未响应"}（${res.endpoint}）。可手动启动 Ollama 后重新自检，或选择降级使用。`,
        );
      }
    } catch (error) {
      setOllamaNote(
        `启动失败：${error instanceof Error ? error.message : String(error)}。可手动启动 Ollama 后重新自检，或选择降级使用。`,
      );
    } finally {
      setOllamaBusy(false);
    }
  };

  const recheckOllama = async (): Promise<void> => {
    setOllamaBusy(true);
    setOllamaNote(undefined);
    try {
      const res = await window.openintj.ollamaCheck();
      setOllamaNote(
        res.running
          ? `服务已在运行（${res.endpoint}），模型 ${res.embedModel} 已安装。点「启动 Ollama 并自检」即可把 embedding 重新绑定为真实向量。`
          : `仍未连上 Ollama：${res.reason ?? "服务未响应"}（${res.endpoint}）。`,
      );
    } catch (error) {
      setOllamaNote(
        `自检失败：${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      setOllamaBusy(false);
    }
  };

  const acceptDegraded = (): void => {
    try {
      window.localStorage.setItem("openintj.embed.degradedAccepted", "1");
    } catch {
      // 存不下就只关本次会话的横幅。
    }
    setDegradedAccepted(true);
  };

  const handleSend = async (text: string): Promise<void> => {
    // 工作台意图拦截：「新任务/新对话/打开工作区」这类操作由客户端真的
    // 执行并切换视图，不交给模型假装完成；「清空会话」转成 /clear 命令
    // 走既有命令通道（服务端留痕、历史派生排除）。
    const trimmed = text.trim();
    if (/^(清空|清除)(当前)?(会话|对话)(记录)?[。.!！]?$/.test(trimmed)) {
      return handleSend("/clear");
    }
    const newTaskMatch = trimmed.match(
      /^(?:新建|创建|开启|开始|新开)(?:一个)?(?:新)?任务(?:[：:\s]+(.+?))?[。.!！]?$/,
    );
    if (newTaskMatch) {
      if (!activeWorkspaceId) {
        // 先让用户选工作区目录，选完在新工作区里建这个任务。
        const activated = await createWorkspace();
        if (!activated) return;
        const title = newTaskMatch[1]?.trim() || defaultTaskTitle();
        pendingNoticeRef.current = `已新建任务「${title}」并切换到新对话（工作区已建同名文件夹），可以直接描述这个任务的目标。`;
        await createTask(title, activated.workspaceId);
        return;
      }
      // 任务必须命名：话里没带名字就用默认名（日期时间戳+当日序号）。
      const title = newTaskMatch[1]?.trim() || defaultTaskTitle();
      pendingNoticeRef.current = `已新建任务「${title}」并切换到新对话（工作区已建同名文件夹），可以直接描述这个任务的目标。`;
      await createTask(title);
      return;
    }
    if (
      /^(新建|创建|开启|开始|新开)(一个)?(新)?(对话|会话)[。.!！]?$/.test(trimmed) ||
      /^新(对话|会话)[。.!！]?$/.test(trimmed)
    ) {
      if (!activeTaskId) {
        // 没有活动任务：交给 createConversation 分流（无工作区先选
        // 工作区，有工作区先建任务），不再回一句「请先新建任务」。
        await createConversation();
        return;
      }
      pendingNoticeRef.current = "已新建对话。";
      await createConversation();
      return;
    }
    if (/^(打开|切换|新建)(一个)?(新)?工作区[。.!！]?$/.test(trimmed)) {
      await createWorkspace();
      return;
    }
    // 无绑定会话（全新对话尚无任务/工作区）时，先请用户选工作区目录：
    // 选完即绑定该工作区的任务与新对话，本次消息直接发进这个新对话；
    // 取消选择则不发送，并说明原因——不再出现「聊了但无处落盘」。
    let boundConversationId = activeConversationId;
    let boundTask: WorkbenchTask | undefined = tasks.find((task) => task.id === activeTaskId);
    if (!boundConversationId) {
      let activated: Awaited<ReturnType<typeof createWorkspace>>;
      try {
        activated = await createWorkspace();
      } catch (error) {
        // 建区过程抛错（如 IPC 异常）时把原因摆到界面上，不再静默吞掉。
        setMessages((current) => [
          ...current,
          { role: "user", content: text },
          {
            role: "system",
            content: `工作区绑定失败，本次消息没有发送：${error instanceof Error ? error.message : String(error)}`,
          },
        ]);
        return;
      }
      if (!activated) {
        setMessages((current) => [
          ...current,
          { role: "user", content: text },
          {
            role: "system",
            content:
              "未选择工作区，本次消息没有发送。先选一个目录作为工作区，对话与制品才会保存进去（左栏「打开工作区」可随时选）。",
          },
        ]);
        return;
      }
      boundConversationId = activated.conversationId;
      boundTask = {
        id: activated.taskId,
        workspaceId: activated.workspaceId,
        title: activated.taskTitle,
        status: "active",
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
    }
    setPending(true);
    const activeConversation = conversations.find(
      (conversation) => conversation.id === boundConversationId,
    );
    boundTask ??= tasks.find((task) => task.id === activeConversation?.taskId);
    const isFirstUserMessage =
      !text.trim().startsWith("/") && !messages.some((message) => message.role === "user");
    // 首句命名：仍叫「新对话」且本会话尚无用户消息时，用这句话的摘要改名。
    if (activeConversation && activeConversation.title === "新对话" && isFirstUserMessage) {
      const title = summarizeConversationTitle(text);
      if (title !== "新对话") {
        void window.openintj
          .updateWorkbenchConversation({ id: activeConversation.id, title })
          .then((updated) =>
            setConversations((current) =>
              current.map((item) => (item.id === updated.id ? updated : item)),
            ),
          )
          .catch(() => {});
      }
    }
    // 默认任务首句命名：任务仍是自动创建时的占位名（用户未改名）时，
    // 用同一摘要改任务名并同步文件夹。只要任务还停留在占位名，就说明
    // 此前没有一次普通首句成功为它命名，因此新会话的首句仍可命名它；
    // 用户已手动改过名时，标题不再等于占位名，绝不覆盖。
    if (boundTask && boundTask.title === DEFAULT_NEW_TASK_TITLE && isFirstUserMessage) {
      const title = summarizeConversationTitle(text);
      if (title !== "新对话" && title !== boundTask.title) {
        void updateTask(boundTask, { title }).catch(() => {});
      }
    }
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
        ...(boundConversationId ? { conversationId: boundConversationId } : {}),
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
    setClosedTabIds((current) => current.filter((id) => id !== conversationId));
    void window.openintj.updateConfig({ activeConversationId: conversationId });
  };

  /** 关闭会话页签：仅隐藏；若关的是当前会话则切到同任务下一个可见会话。 */
  const closeConversationTab = (conversationId: string): void => {
    setClosedTabIds((current) =>
      current.includes(conversationId) ? current : [...current, conversationId],
    );
    if (conversationId !== activeConversationId) return;
    const siblings = conversations.filter(
      (item) => item.taskId === activeTaskId && item.id !== conversationId,
    );
    const next = siblings.find((item) => !closedTabIds.includes(item.id));
    if (next) selectConversation(next.id);
  };

  const activateWorkspace = async (
    created: WorkbenchWorkspace,
  ): Promise<
    { workspaceId: string; taskId: string; conversationId: string; taskTitle: string } | undefined
  > => {
    const snapshot = await window.openintj.workbenchBootstrap();
    let taskId = snapshot.tasks.find(
      (task) => task.workspaceId === created.id && task.status !== "archived",
    )?.id;
    let conversationId = snapshot.conversations.find((item) => item.taskId === taskId)?.id;
    let taskTitle = snapshot.tasks.find((task) => task.id === taskId)?.title;
    if (!taskId) {
      const createdTask = await window.openintj.createWorkbenchTask({
        parentId: created.id,
        title: DEFAULT_NEW_TASK_TITLE,
      });
      const createdConversation = await window.openintj.createWorkbenchConversation({
        parentId: createdTask.id,
        title: "新对话",
        modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
      });
      taskId = createdTask.id;
      conversationId = createdConversation.id;
      taskTitle = createdTask.title;
    }
    // 联动第一步：先把活动工作区/任务/会话写进渲染状态，三栏当场
    // 对齐；随后再持久化配置（可能触发 agent 热重载，期间 IPC 短暂
    // 不可用也不能让界面看起来毫无变化）。
    setActiveWorkspaceId(created.id);
    if (taskId) setActiveTaskId(taskId);
    if (conversationId) setActiveConversationId(conversationId);
    await window.openintj.updateConfig({
      workspaceDir: created.rootPath,
      activeWorkspaceId: created.id,
      activeTaskId: taskId,
      activeConversationId: conversationId,
    });
    try {
      const info = await window.openintj.workspaceInfo();
      setWorkspaceRoot(info.root);
    } catch {
      // agent 尚在热重载时取不到根，联动校验 effect 会随后对齐。
    }
    await refreshWorkbench();
    setSelectedWorkspacePath(undefined);
    if (!taskId || !conversationId) return undefined;
    return {
      workspaceId: created.id,
      taskId,
      conversationId,
      taskTitle: taskTitle ?? DEFAULT_NEW_TASK_TITLE,
    };
  };

  /**
   * 首次启动默认动作（显式版）：没有任何打开的工作区时，弹提示请
   * 用户自己选/新建一个本地目录作为工作区，选中后绑定「新创建的任务」
   * 与新对话，制品从第一句起落进该目录。不再拿 agent 文件根静默
   * 顶替——那个根可能是历史遗留路径，用户根本不知道它从哪来。
   * 用户取消则停留在无工作区状态，左栏「打开工作区」随时可再选。
   */
  const workspacePromptedRef = React.useRef(false);
  React.useEffect(() => {
    if (!workbenchLoaded || workspacePromptedRef.current) return;
    workspacePromptedRef.current = true;
    if (workspaces.length > 0) return;
    void (async () => {
      const proceed = await askConfirm(
        "先选择一个工作区",
        "当前没有打开的工作区。对话要先绑定工作区与任务，聊天产生的制品才会落盘到工作区目录。接下来请选一个本地目录（可在系统对话框里新建文件夹），选中后会立即建为工作区并绑定「新创建的任务」与新对话（工作区名默认取目录名，之后可改名）。",
        { confirmLabel: "选择/新建目录" },
      );
      if (proceed) await createWorkspace();
    })();
  }, [workbenchLoaded, workspaces.length]);

  const createWorkspace = async (): Promise<
    | { workspaceId: string; taskId: string; conversationId: string; taskTitle: string }
    | undefined
  > => {
    if (activeWorkspaceId) {
      const proceed = await askConfirm(
        "打开新工作区",
        "同时只能打开一个工作区。打开新目录会关闭当前工作区（任务和对话会保留，磁盘文件不会删除）。继续？",
      );
      if (!proceed) return undefined;
    }
    let picked: WorkspacePickResponse;
    try {
      picked = await window.openintj.workspacePickDir();
    } catch (error) {
      // IPC 异常（通道未注册/主进程重载中等）必须显形，不能让用户
      // 面对「点了没反应」：把原因写进聊天区并终止本次建区。
      setMessages((current) => [
        ...current,
        {
          role: "system",
          content: `打开目录选择失败：${error instanceof Error ? error.message : String(error)}。工作区尚未创建，请稍后重试；若反复出现请重启应用。`,
        },
      ]);
      return undefined;
    }
    if (picked.canceled || !picked.root) return undefined;
    const current = workspaces.find((workspace) => workspace.id === activeWorkspaceId);
    if (current && sameFsPath(current.rootPath, picked.root)) {
      // 选的就是当前工作区：直接重新激活并返回绑定信息，不静默退出。
      return activateWorkspace(current);
    }
    // 选目录即建工作区（原子化）：名称默认取目录名，之后可在右栏
    // 改名。此前「选完目录还要再填名字，取消就静默断链」会让用户
    // 以为已建好、实际工作台里根本没有这条工作区。
    const name = picked.root.split(/[\\/]/).filter(Boolean).at(-1) ?? "工作区";
    let created: WorkbenchWorkspace;
    try {
      created = await window.openintj.createWorkbenchWorkspace({
        name,
        rootPath: picked.root,
      });
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "system",
          content: `工作区创建失败：${error instanceof Error ? error.message : String(error)}。目录虽已选定，但尚未绑定为工作区，请重试。`,
        },
      ]);
      return undefined;
    }
    if (!created.id) {
      setMessages((current) => [
        ...current,
        {
          role: "system",
          content:
            "工作区创建未完成：目录虽已选定，但尚未绑定为工作区。请重新打开工作区再选一次。",
        },
      ]);
      return undefined;
    }
    // activateWorkspace 写回 workspaceDir 会触发 agent 热重载，
    // 文件树随即对齐新工作区，无需重启应用。
    let activated: Awaited<ReturnType<typeof activateWorkspace>>;
    try {
      activated = await activateWorkspace(created);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "system",
          content: `工作区「${created.name}」已创建但绑定失败：${error instanceof Error ? error.message : String(error)}。请稍后重试或重启应用。`,
        },
      ]);
      return undefined;
    }
    if (activated) {
      void ensureTaskFolder(activated.taskTitle);
      // 联动确认：新会话消息加载后补一条系统提示，明确当前对话
      // 绑定的工作区与任务，制品落盘位置不再靠猜。
      pendingNoticeRef.current = `已打开工作区「${created.name}」（${created.rootPath}），当前对话已绑定任务「${activated.taskTitle}」，制品会保存到该工作区。`;
    }
    return activated;
  };

  const closeWorkspace = async (): Promise<void> => {
    if (!activeWorkspaceId) return;
    const proceed = await askConfirm(
      "关闭当前工作区",
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

  /** 任务名 → 文件工作区目录名：去掉 Windows 非法字符，避免空名/尾点。 */
  const taskFolderName = (title: string): string =>
    title
      .replace(/[\\/:*?"<>|]/g, "-")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/[. ]+$/g, "")
      .slice(0, 60);

  /** 建任务同名文件夹（尽力而为，失败不阻断任务本身）。 */
  const ensureTaskFolder = async (title: string): Promise<void> => {
    const folder = taskFolderName(title);
    if (!folder) return;
    try {
      await workspaceClient.mkdir(folder);
    } catch {
      // 文件工作区未就绪（如尚未重启加载新工作区）时静默跳过。
    }
  };

  /** 改名同步文件夹：旧目录在则改名，不在则按新名新建。 */
  const renameTaskFolder = async (oldTitle: string, newTitle: string): Promise<void> => {
    const from = taskFolderName(oldTitle);
    const to = taskFolderName(newTitle);
    if (!to || from === to) return;
    try {
      if (from) {
        const renamed = await workspaceClient.renameDir(from, to);
        if (!("error" in renamed)) return;
      }
      await workspaceClient.mkdir(to);
    } catch {
      try {
        await workspaceClient.mkdir(to);
      } catch {
        // 同上，尽力而为。
      }
    }
  };

  const createTask = async (title: string, workspaceId?: string): Promise<void> => {
    const targetWorkspaceId = workspaceId ?? activeWorkspaceId;
    if (!targetWorkspaceId) return;
    const created = await window.openintj.createWorkbenchTask({
      parentId: targetWorkspaceId,
      title,
    });
    void ensureTaskFolder(created.title);
    setTasks((current) => [created, ...current]);
    setActiveTaskId(created.id);
    const createdConversation = await window.openintj.createWorkbenchConversation({
      parentId: created.id,
      title: "新对话",
      modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
    });
    setConversations((current) => [createdConversation, ...current]);
    setActiveConversationId(createdConversation.id);
    setTrajectory([]);
    setSelectedWorkspacePath(undefined);
    await window.openintj.updateConfig({
      activeTaskId: created.id,
      activeConversationId: createdConversation.id,
    });
  };

  /** 默认任务名：新建任务-日期-时间-当日序号（如 新建任务-20261111-153012-01）。 */
  const defaultTaskTitle = (): string => {
    const nowDate = new Date();
    const pad = (n: number): string => String(n).padStart(2, "0");
    const date = `${nowDate.getFullYear()}${pad(nowDate.getMonth() + 1)}${pad(nowDate.getDate())}`;
    const time = `${pad(nowDate.getHours())}${pad(nowDate.getMinutes())}${pad(nowDate.getSeconds())}`;
    const sameDay = tasks.filter((task) => {
      if (task.workspaceId !== activeWorkspaceId) return false;
      const created = new Date(task.createdAt);
      return (
        created.getFullYear() === nowDate.getFullYear() &&
        created.getMonth() === nowDate.getMonth() &&
        created.getDate() === nowDate.getDate()
      );
    }).length;
    return `新建任务-${date}-${time}-${pad(sameDay + 1)}`;
  };

  /** 会话自动命名：取第一句话的首个分句，截 24 字。 */
  const summarizeConversationTitle = (text: string): string => {
    const clean = text.replace(/\s+/g, " ").trim();
    const first = clean.split(/[。！？.!?；;]/)[0]?.trim() || clean;
    if (!first) return "新对话";
    return first.length > 24 ? `${first.slice(0, 24)}…` : first;
  };

  /** 新建任务必须命名：对话框预填默认名（日期时间戳+当日序号），可改；无工作区时先走打开工作区。 */
  const requestCreateTask = (): void => {
    if (!activeWorkspaceId) {
      void createWorkspace();
      return;
    }
    void (async () => {
      const title = (
        await askName("新建任务", defaultTaskTitle(), "任务名称（将同时在工作区建同名文件夹）")
      )?.trim();
      if (title) await createTask(title);
    })();
  };

  const renameTask = async (task: WorkbenchTask): Promise<void> => {
    const title = (
      await askName("重命名任务", task.title, "任务名称（工作区同名文件夹将同步改名）")
    )?.trim();
    if (title && title !== task.title) await updateTask(task, { title });
  };

  const renameWorkspace = async (): Promise<void> => {
    const current = workspaces.find((item) => item.id === activeWorkspaceId);
    if (!current) return;
    const name = (
      await askName("重命名工作区", current.name, "只改显示名，不影响目录路径与数据")
    )?.trim();
    if (!name || name === current.name) return;
    const updated = await window.openintj.updateWorkbenchWorkspace({ id: current.id, name });
    setWorkspaces((items) => items.map((item) => (item.id === updated.id ? updated : item)));
  };

  const renameConversation = async (conversation: WorkbenchConversation): Promise<void> => {
    const title = (await askName("重命名会话", conversation.title))?.trim();
    if (!title || title === conversation.title) return;
    const updated = await window.openintj.updateWorkbenchConversation({
      id: conversation.id,
      title,
    });
    setConversations((items) => items.map((item) => (item.id === updated.id ? updated : item)));
  };

  const createConversation = async (): Promise<void> => {
    if (!activeTaskId) {
      // 没有活动任务时不静默吞掉：无工作区先选工作区（自动绑定
      // 「新创建的任务」与新对话）；有工作区则走新建任务（命名后建对话）。
      if (!activeWorkspaceId) {
        await createWorkspace();
        return;
      }
      requestCreateTask();
      return;
    }
    const created = await window.openintj.createWorkbenchConversation({
      parentId: activeTaskId,
      title: "新对话",
      modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
    });
    setConversations((current) => [created, ...current]);
    setActiveConversationId(created.id);
    setTrajectory([]);
    setSelectedWorkspacePath(undefined);
    await window.openintj.updateConfig({ activeConversationId: created.id });
  };

  const updateTask = async (
    selected: WorkbenchTask,
    patch: { title?: string; status?: "active" | "completed" | "archived" },
  ): Promise<void> => {
    const updated = await window.openintj.updateWorkbenchTask({ id: selected.id, ...patch });
    setTasks((current) => current.map((task) => (task.id === updated.id ? updated : task)));
    if (patch.title && patch.title !== selected.title) {
      void renameTaskFolder(selected.title, patch.title);
    }
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

  /** 删除任务：会话记录一并删，工作区同名文件夹保留在硬盘（可再被加载）。 */
  const deleteTask = async (selected: WorkbenchTask): Promise<void> => {
    const proceed = await askConfirm(
      `删除任务「${selected.title}」`,
      "该任务的会话记录会一并删除；工作区文件夹会保留在本地硬盘，可再次加载。",
      { confirmLabel: "删除", danger: true },
    );
    if (!proceed) return;
    await window.openintj.deleteWorkbenchTask({ id: selected.id });
    const remainingTasks = tasks.filter((task) => task.id !== selected.id);
    const remainingConversations = conversations.filter(
      (conversation) => conversation.taskId !== selected.id,
    );
    setTasks(remainingTasks);
    setConversations(remainingConversations);
    if (selected.id !== activeTaskId) return;
    const next = remainingTasks.find(
      (task) => task.workspaceId === selected.workspaceId && task.status !== "archived",
    );
    if (next) {
      setActiveTaskId(next.id);
      const nextConversation = remainingConversations.find((item) => item.taskId === next.id);
      setActiveConversationId(nextConversation?.id);
      await window.openintj.updateConfig({
        activeTaskId: next.id,
        activeConversationId: nextConversation?.id,
      });
    } else {
      setActiveTaskId(undefined);
      setActiveConversationId(undefined);
      setMessages([]);
      await window.openintj.updateConfig({ activeTaskId: undefined, activeConversationId: undefined });
    }
  };

  /** 删除会话：消息/日志一并删；删的是当前会话则切到同任务下一个。 */
  const deleteConversation = async (selected: WorkbenchConversation): Promise<void> => {
    const proceed = await askConfirm(
      `删除会话「${selected.title}」`,
      "对话记录会一并删除。",
      { confirmLabel: "删除", danger: true },
    );
    if (!proceed) return;
    await window.openintj.deleteWorkbenchConversation({ id: selected.id });
    const remaining = conversations.filter((item) => item.id !== selected.id);
    setConversations(remaining);
    setClosedTabIds((current) => current.filter((id) => id !== selected.id));
    if (selected.id !== activeConversationId) return;
    const next = remaining.find((item) => item.taskId === selected.taskId);
    setActiveConversationId(next?.id);
    if (!next) setMessages([]);
    await window.openintj.updateConfig({ activeConversationId: next?.id });
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

  // 工作区联动校验：任务会话栏的当前工作区与文件树根不一致时，
  // 写回配置触发 agent 热重载（workspaceDir 是运行时配置键）。
  React.useEffect(() => {
    const rootPath = activeWorkspace?.rootPath;
    if (!rootPath || !workspaceRoot) return;
    if (sameFsPath(rootPath, workspaceRoot)) return;
    void window.openintj
      .updateConfig({ workspaceDir: rootPath })
      .then(() => window.openintj.workspaceInfo())
      .then((info) => setWorkspaceRoot(info.root))
      .catch(() => {});
  }, [activeWorkspaceId, activeWorkspace?.rootPath, workspaceRoot]);
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
          </div>
          <div className="flex-1 min-h-0">
            <WorkspaceWorkbench
              client={workspaceClient}
              rootLabel={activeWorkspace?.rootPath ?? workspaceRoot}
              selectedPath={selectedWorkspacePath}
              onSelectPath={(path) => setSelectedWorkspacePath(path)}
              showPreview={false}
              refreshKey={workspaceTreeKey}
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
            onCreateConversation={() => void createConversation()}
            onCreateTask={requestCreateTask}
            closedTabIds={closedTabIds}
            onCloseConversationTab={closeConversationTab}
          />
          {embedDegraded && !degradedAccepted && embedStatus ? (
            <OllamaBanner
              embedLabel={`${embedStatus.provider} (${embedStatus.model}, ${embedStatus.dimension}d)`}
              busy={ollamaBusy}
              {...(ollamaNote ? { note: ollamaNote } : {})}
              onLaunch={() => void launchOllama()}
              onRecheck={() => void recheckOllama()}
              onAcceptDegraded={acceptDegraded}
            />
          ) : null}
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
                onCreateConversation={() => void createConversation()}
                onRenameTask={(task) => void renameTask(task)}
                onDeleteTask={(task) => void deleteTask(task)}
                onDeleteConversation={(conversation) => void deleteConversation(conversation)}
                onRenameWorkspace={() => void renameWorkspace()}
                onRenameConversation={(conversation) => void renameConversation(conversation)}
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
      {nameDialog ? (
        <NameDialog
          title={nameDialog.title}
          label={nameDialog.label}
          initialValue={nameDialog.initialValue}
          onSubmit={(value) => closeNameDialog(value)}
          onCancel={() => closeNameDialog(null)}
        />
      ) : null}
      {confirmDialog ? (
        <ConfirmDialog
          title={confirmDialog.title}
          message={confirmDialog.message}
          confirmLabel={confirmDialog.confirmLabel}
          danger={confirmDialog.danger}
          onConfirm={() => closeConfirmDialog(true)}
          onCancel={() => closeConfirmDialog(false)}
        />
      ) : null}
    </div>
  );
};
