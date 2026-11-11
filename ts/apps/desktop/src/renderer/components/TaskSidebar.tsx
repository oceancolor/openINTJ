import type React from "react";
import type {
  WorkbenchConversation,
  WorkbenchTask,
  WorkbenchWorkspace,
} from "../../shared/ipc-protocol.js";

export const TaskSidebar: React.FC<{
  workspaces: WorkbenchWorkspace[];
  tasks: WorkbenchTask[];
  conversations: WorkbenchConversation[];
  activeWorkspaceId?: string | undefined;
  activeTaskId?: string | undefined;
  activeConversationId?: string | undefined;
  onCreateWorkspace: () => void;
  onCloseWorkspace: () => void;
  onSelectTask: (id: string) => void;
  onSelectConversation: (id: string) => void;
  onCreateTask: () => void;
  onCreateConversation: () => void;
  onRenameTask: (task: WorkbenchTask) => void;
  onArchiveTask: (task: WorkbenchTask) => void;
  onOpenWorkspaceDir?: (() => void) | undefined;
  showWorkspace?: boolean | undefined;
  /** 当前任务（当前对话）已生成的文件路径，点击在中栏预览。 */
  taskArtifacts?: string[] | undefined;
  onOpenArtifact?: ((path: string) => void) | undefined;
}> = (props) => {
  const current = props.workspaces.find((workspace) => workspace.id === props.activeWorkspaceId);
  const showWorkspace = props.showWorkspace !== false;
  return (
    <aside className="flex h-full min-h-0 flex-col bg-panel text-xs">
      {showWorkspace ? (
        <div className="shrink-0 border-b border-line p-3">
          <div className="mb-1 flex justify-between text-[10px] font-bold uppercase tracking-[0.13em] text-muted">
            <span>工作区</span>
          </div>
          <div className="mb-2 truncate text-ink" title={current?.rootPath}>
            {current?.name ?? "未打开工作区"}
          </div>
          <div className="flex gap-1">
            <button
              type="button"
              className="flex-1 rounded-lg bg-ink px-2 py-1.5 font-medium text-white hover:bg-inksoft"
              onClick={props.onCreateWorkspace}
            >
              打开工作区
            </button>
            <button
              type="button"
              className="flex-1 rounded-lg border border-line bg-white px-2 py-1.5 font-medium text-ink hover:border-linedark disabled:opacity-40"
              disabled={!props.activeWorkspaceId}
              onClick={props.onCloseWorkspace}
            >
              关闭工作区
            </button>
          </div>
          {props.onOpenWorkspaceDir && props.activeWorkspaceId ? (
            <button
              type="button"
              className="mt-2 w-full rounded-lg border border-line bg-white px-2 py-1.5 font-medium text-ink hover:border-linedark"
              onClick={props.onOpenWorkspaceDir}
            >
              打开工作区目录
            </button>
          ) : null}
        </div>
      ) : null}
      <div className="flex shrink-0 items-center justify-between p-2">
        <span className="text-[10px] font-bold uppercase tracking-[0.13em] text-muted">任务</span>
        <button
          type="button"
          className="rounded-lg bg-ink px-2 py-1 font-medium text-white hover:bg-inksoft disabled:opacity-40"
          disabled={!props.activeWorkspaceId}
          onClick={props.onCreateTask}
        >
          + 新任务
        </button>
      </div>
      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-2">
        {props.tasks
          .filter(
            (task) => task.workspaceId === props.activeWorkspaceId && task.status !== "archived",
          )
          .map((task) => {
            const isActive = props.activeTaskId === task.id;
            const dot =
              task.status === "active"
                ? { cls: "bg-amberdot", label: "进行中" }
                : task.status === "completed"
                  ? { cls: "bg-[#62a36d]", label: "已完成" }
                  : { cls: "bg-[#c4c7bf]", label: task.status };
            return (
              <div key={task.id}>
                <button
                  type="button"
                  className={`w-full rounded-xl px-2.5 py-2 text-left ${
                    isActive ? "bg-white shadow-card" : "hover:bg-white/60"
                  }`}
                  onClick={() => props.onSelectTask(task.id)}
                >
                  <span
                    className={`block truncate text-xs text-ink ${isActive ? "font-semibold" : ""}`}
                  >
                    {task.title}
                  </span>
                  <span className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted">
                    <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${dot.cls}`} />
                    {dot.label}
                  </span>
                </button>
                {isActive ? (
                  <div className="ml-3 mt-1 space-y-0.5 border-l border-line pl-2">
                    <div className="flex gap-2 text-[10px]">
                      <button
                        type="button"
                        className="text-muted hover:text-ink"
                        onClick={() => props.onRenameTask(task)}
                      >
                        重命名
                      </button>
                      <button
                        type="button"
                        className="text-muted hover:text-signal"
                        onClick={() => props.onArchiveTask(task)}
                      >
                        归档
                      </button>
                    </div>
                    {props.conversations
                      .filter((conversation) => conversation.taskId === task.id)
                      .map((conversation) => (
                        <button
                          type="button"
                          key={conversation.id}
                          className={`block w-full truncate rounded-lg px-1.5 py-1 text-left ${
                            props.activeConversationId === conversation.id
                              ? "bg-mosstint text-mossdeep"
                              : "text-muted hover:text-ink"
                          }`}
                          onClick={() => props.onSelectConversation(conversation.id)}
                        >
                          {conversation.title}
                        </button>
                      ))}
                    <button
                      type="button"
                      className="px-1.5 py-1 text-muted hover:text-ink"
                      onClick={props.onCreateConversation}
                    >
                      + 新对话
                    </button>
                    {props.taskArtifacts && props.taskArtifacts.length > 0 ? (
                      <div className="pt-1">
                        <div className="px-1.5 text-[10px] font-bold uppercase tracking-[0.13em] text-muted">
                          本任务产物
                        </div>
                        {props.taskArtifacts.map((path) => (
                          <button
                            type="button"
                            key={path}
                            className="block w-full truncate rounded-lg px-1.5 py-1 text-left text-muted hover:text-ink"
                            title={path}
                            onClick={() => props.onOpenArtifact?.(path)}
                          >
                            {path.split(/[\\/]/).pop() ?? path}
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            );
          })}
      </div>
    </aside>
  );
};
