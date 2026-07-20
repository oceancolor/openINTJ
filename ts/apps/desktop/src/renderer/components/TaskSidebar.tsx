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
  onSelectWorkspace: (id: string) => void;
  onCreateWorkspace: () => void;
  onSelectTask: (id: string) => void;
  onSelectConversation: (id: string) => void;
  onCreateTask: () => void;
  onCreateConversation: () => void;
  onRenameTask: (task: WorkbenchTask) => void;
  onArchiveTask: (task: WorkbenchTask) => void;
}> = (props) => (
  <aside className="h-full bg-[#11111b] border-r border-gray-800 flex flex-col text-xs">
    <div className="p-3 border-b border-gray-800">
      <div className="flex justify-between text-gray-400 mb-1">
        <span>工作区</span>
        <button type="button" onClick={props.onCreateWorkspace} className="hover:text-gray-200">
          + 添加
        </button>
      </div>
      <select
        className="w-full bg-gray-800 text-gray-200 rounded px-2 py-1"
        value={props.activeWorkspaceId ?? ""}
        onChange={(event) => props.onSelectWorkspace(event.target.value)}
      >
        {props.workspaces.map((workspace) => (
          <option key={workspace.id} value={workspace.id}>
            {workspace.name}
          </option>
        ))}
      </select>
    </div>
    <div className="p-2 flex items-center justify-between">
      <span className="text-gray-400">任务</span>
      <button
        type="button"
        className="px-2 py-0.5 rounded bg-purple-800 hover:bg-purple-700"
        onClick={props.onCreateTask}
      >
        + 任务
      </button>
    </div>
    <div className="flex-1 overflow-y-auto px-2 space-y-1">
      {props.tasks
        .filter(
          (task) => task.workspaceId === props.activeWorkspaceId && task.status !== "archived",
        )
        .map((task) => (
          <div key={task.id}>
            <button
              type="button"
              className={`w-full text-left rounded px-2 py-1 ${
                props.activeTaskId === task.id
                  ? "bg-purple-900 text-purple-100"
                  : "text-gray-300 hover:bg-gray-800"
              }`}
              onClick={() => props.onSelectTask(task.id)}
            >
              <span className="block truncate">{task.title}</span>
              <span className="text-[10px] text-gray-500">{task.status}</span>
            </button>
            {props.activeTaskId === task.id ? (
              <div className="ml-3 mt-1 space-y-0.5 border-l border-gray-800 pl-2">
                <div className="flex gap-2 text-[10px]">
                  <button
                    type="button"
                    className="text-gray-500 hover:text-gray-300"
                    onClick={() => props.onRenameTask(task)}
                  >
                    重命名
                  </button>
                  <button
                    type="button"
                    className="text-gray-500 hover:text-red-300"
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
                      className={`block w-full text-left truncate rounded px-1.5 py-1 ${
                        props.activeConversationId === conversation.id
                          ? "text-cyan-300 bg-gray-800"
                          : "text-gray-500 hover:text-gray-300"
                      }`}
                      onClick={() => props.onSelectConversation(conversation.id)}
                    >
                      {conversation.title}
                    </button>
                  ))}
                <button
                  type="button"
                  className="text-gray-500 hover:text-gray-300 px-1.5 py-1"
                  onClick={props.onCreateConversation}
                >
                  + 新对话
                </button>
              </div>
            ) : null}
          </div>
        ))}
    </div>
  </aside>
);
