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
  onCreateConversation?: (() => void) | undefined;
  onCreateTask?: (() => void) | undefined;
  closedTabIds?: string[] | undefined;
  onCloseConversationTab?: ((id: string) => void) | undefined;
}> = (props) => {
  const taskConversations = props.conversations.filter((c) => c.taskId === props.activeTaskId);
  const usedK = props.context ? Math.round(props.context.used / 100) / 10 : 0;
  const maxK = props.context ? Math.round(props.context.max / 1000) : 0;
  const pct = props.context ? Math.min(100, Math.round(props.context.ratio * 100)) : 0;
  return (
    <div className="shrink-0 border-b border-line bg-panel">
    <div className="flex items-center gap-2 px-3 py-2 text-[13px]">
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
    {props.onCreateConversation ? (
      <div
        className="flex items-center gap-1.5 overflow-x-auto border-t border-line/60 px-3 py-2"
        data-testid="conversation-tabs"
      >
        {taskConversations
          .filter((conversation) => !props.closedTabIds?.includes(conversation.id))
          .map((conversation) => (
            <span
              key={conversation.id}
              className={
                conversation.id === props.activeConversationId
                  ? "flex max-w-[11rem] shrink-0 items-center gap-0.5 rounded-full bg-mosstint py-1 pl-3 pr-1.5 text-xs font-semibold text-mossdeep"
                  : "flex max-w-[11rem] shrink-0 items-center gap-0.5 rounded-full py-1 pl-3 pr-1.5 text-xs text-muted hover:bg-white hover:text-ink"
              }
            >
              <button
                type="button"
                title={conversation.title}
                className="truncate"
                onClick={() => props.onSelectConversation(conversation.id)}
              >
                {conversation.title}
              </button>
              {props.onCloseConversationTab ? (
                <button
                  type="button"
                  aria-label={`关闭页签 ${conversation.title}`}
                  title="关闭页签（会话保留在任务列表）"
                  className="shrink-0 rounded-full px-1 leading-none text-faint hover:text-signal"
                  onClick={() => props.onCloseConversationTab?.(conversation.id)}
                >
                  ×
                </button>
              ) : null}
            </span>
          ))}
        <button
          type="button"
          className="shrink-0 rounded-full border border-dashed border-linedark px-3 py-1 text-xs font-medium text-ink hover:border-moss disabled:opacity-40"
          disabled={!props.activeTaskId}
          onClick={props.onCreateConversation}
        >
          + 新对话
        </button>
        {props.onCreateTask ? (
          <button
            type="button"
            className="ml-auto shrink-0 rounded-full bg-ink px-3.5 py-1.5 text-xs font-semibold text-white shadow-card hover:bg-inksoft"
            onClick={props.onCreateTask}
          >
            + 新任务
          </button>
        ) : null}
      </div>
    ) : null}
    </div>
  );
};
