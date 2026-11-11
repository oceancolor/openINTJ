import type React from "react";
import type { WorkbenchConversation, WorkbenchTask } from "../../shared/ipc-protocol.js";

/**
 * Agent 上下文条（中栏顶部）：当前工作区 > 任务 > 对话的面包屑切换，
 * 加上下文用量与当前预览文件——让"agent 现在站在哪里、脑子里装了
 * 多少"一眼可见。数据全部来自宿主已确认的状态，不新增 IPC。
 */
export const AgentContextBar: React.FC<{
  workspaceName?: string | undefined;
  tasks: WorkbenchTask[];
  conversations: WorkbenchConversation[];
  activeTaskId?: string | undefined;
  activeConversationId?: string | undefined;
  onSelectTask: (id: string) => void;
  onSelectConversation: (id: string) => void;
  context?: { used: number; max: number; ratio: number } | undefined;
  previewPath?: string | undefined;
  onClosePreview?: (() => void) | undefined;
}> = (props) => {
  const taskConversations = props.conversations.filter((c) => c.taskId === props.activeTaskId);
  const usedK = props.context ? Math.round(props.context.used / 100) / 10 : 0;
  const maxK = props.context ? Math.round(props.context.max / 1000) : 0;
  const pct = props.context ? Math.min(100, Math.round(props.context.ratio * 100)) : 0;
  return (
    <div className="shrink-0 flex items-center gap-2 border-b border-line bg-panel px-3 py-1.5 text-xs">
      <span className="shrink-0 font-semibold text-ink truncate max-w-[10rem]" title={props.workspaceName}>
        {props.workspaceName ?? "未打开工作区"}
      </span>
      <span className="text-faint">›</span>
      <select
        aria-label="当前任务"
        className="max-w-[11rem] truncate rounded-md border border-line bg-white px-1.5 py-1 text-ink disabled:opacity-40"
        value={props.activeTaskId ?? ""}
        disabled={props.tasks.length === 0}
        onChange={(e) => props.onSelectTask(e.target.value)}
      >
        {props.tasks.length === 0 ? <option value="">无任务</option> : null}
        {props.tasks.map((task) => (
          <option key={task.id} value={task.id}>
            {task.title}
          </option>
        ))}
      </select>
      <span className="text-faint">›</span>
      <select
        aria-label="当前对话"
        className="max-w-[11rem] truncate rounded-md border border-line bg-white px-1.5 py-1 text-ink disabled:opacity-40"
        value={props.activeConversationId ?? ""}
        disabled={taskConversations.length === 0}
        onChange={(e) => props.onSelectConversation(e.target.value)}
      >
        {taskConversations.length === 0 ? <option value="">无对话</option> : null}
        {taskConversations.map((conversation) => (
          <option key={conversation.id} value={conversation.id}>
            {conversation.title}
          </option>
        ))}
      </select>
      {props.context ? (
        <span className="ml-auto flex shrink-0 items-center gap-1.5 text-muted" title={`上下文 ${props.context.used}/${props.context.max}`}>
          <span className="hidden xl:inline">上下文</span>
          <span className="inline-block h-1.5 w-16 overflow-hidden rounded-full bg-line">
            <span
              className={`block h-full ${pct > 85 ? "bg-signal" : pct > 60 ? "bg-amberdot" : "bg-[#62a36d]"}`}
              style={{ width: `${pct}%` }}
            />
          </span>
          <span className="font-mono text-[10px]">
            {usedK}k/{maxK}k
          </span>
        </span>
      ) : null}
      {props.previewPath ? (
        <span className="flex shrink-0 items-center gap-1 rounded-md bg-mosstint px-1.5 py-0.5 text-mossdeep">
          <span className="max-w-[10rem] truncate" title={props.previewPath}>
            {props.previewPath.split(/[\\/]/).pop()}
          </span>
          <button
            type="button"
            aria-label="关闭预览"
            className="text-mossdeep/60 hover:text-mossdeep"
            onClick={props.onClosePreview}
          >
            ×
          </button>
        </span>
      ) : null}
    </div>
  );
};
