import React from "react";
import type {
  ChannelStatusView,
  ConnectorStatusView,
  SubagentRecordView,
  TaskCheckpointView,
} from "../../shared/ipc-protocol.js";

const fmtTime = (ms?: number): string => (ms === undefined ? "—" : new Date(ms).toLocaleString());

const shortId = (id: string): string => (id.length > 12 ? `${id.slice(0, 8)}…` : id);

const STATUS_BADGE: Record<string, string> = {
  running: "bg-mosstint text-mossdeep",
  completed: "bg-mossmist text-moss",
  failed: "bg-[#f8e9e4] text-signal",
  cancelled: "bg-inset text-muted",
};

const Badge: React.FC<{ text: string; tone?: string | undefined }> = ({ text, tone }) => (
  <span className={`px-1 rounded text-[10px] ${tone ?? "bg-inset text-muted"}`}>{text}</span>
);

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="space-y-1.5">
    <div className="text-muted font-medium">{title}</div>
    {children}
  </div>
);

const Empty: React.FC<{ text: string }> = ({ text }) => <div className="text-muted">{text}</div>;

/**
 * 运维聚合页：连接器状态、TaskPool 检查点（恢复/取消）、通知通道（测试发送）、
 * 子智能体记录。四块都走既有 IPC；检查点与子智能体在本页可见时 5 秒轮询
 * （切走 tab 即卸载，轮询自动停）。
 */
export const SystemPanel: React.FC = () => {
  const [connectors, setConnectors] = React.useState<ConnectorStatusView[]>([]);
  const [checkpoints, setCheckpoints] = React.useState<TaskCheckpointView[]>([]);
  const [channels, setChannels] = React.useState<ChannelStatusView[]>([]);
  const [subagents, setSubagents] = React.useState<SubagentRecordView[]>([]);
  const [error, setError] = React.useState<string | undefined>();
  const [notice, setNotice] = React.useState<string | undefined>();
  const [busy, setBusy] = React.useState(false);

  const loadStatic = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    try {
      const [c, ch] = await Promise.all([
        api.connectorsList ? api.connectorsList() : [],
        api.channelsList ? api.channelsList() : [],
      ]);
      setConnectors(c);
      setChannels(ch);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const loadLive = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    try {
      const [cp, sub] = await Promise.all([
        api.taskpoolCheckpoints ? api.taskpoolCheckpoints() : [],
        api.subagentsList ? api.subagentsList() : [],
      ]);
      if (Array.isArray(cp)) setCheckpoints(cp);
      else setError(cp.error);
      setSubagents(sub);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const refreshAll = React.useCallback(async (): Promise<void> => {
    await Promise.all([loadStatic(), loadLive()]);
  }, [loadStatic, loadLive]);

  React.useEffect(() => {
    void refreshAll();
    const timer = setInterval(() => void loadLive(), 5_000);
    return () => clearInterval(timer);
  }, [refreshAll, loadLive]);

  const resume = async (runId: string): Promise<void> => {
    const api = window.openintj;
    if (!api?.taskpoolResume) return;
    setBusy(true);
    setError(undefined);
    setNotice(undefined);
    try {
      const res = await api.taskpoolResume(runId);
      if ("error" in res) setError(res.error);
      else setNotice(`已恢复运行 ${shortId(runId)}（状态 ${res.status}）`);
      await loadLive();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (runId: string): Promise<void> => {
    const api = window.openintj;
    if (!api?.taskpoolCancel) return;
    setBusy(true);
    setError(undefined);
    setNotice(undefined);
    try {
      const res = await api.taskpoolCancel(runId);
      if ("error" in res) setError(res.error);
      else setNotice(`已取消运行 ${shortId(runId)}`);
      await loadLive();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const testChannel = async (id: string): Promise<void> => {
    const api = window.openintj;
    if (!api?.channelsTest) return;
    setError(undefined);
    setNotice(undefined);
    try {
      const res = await api.channelsTest(id);
      if ("error" in res) setError(`${id}：${res.error}`);
      else setNotice(`测试消息已发送到 ${id}`);
      await loadStatic();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0 text-xs">
      <div className="shrink-0 p-2 border-b border-line flex items-center gap-2">
        <span className="text-muted">连接器 · 检查点 · 通道 · 子智能体</span>
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => void refreshAll()}
          className="px-2 py-0.5 rounded bg-inset hover:bg-rail text-ink"
        >
          刷新
        </button>
      </div>
      {notice ? <div className="shrink-0 px-2 pt-1 text-moss">{notice}</div> : null}
      {error ? <div className="shrink-0 px-2 pt-1 text-signal">[错误] {error}</div> : null}
      <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-4">
        <Section title="连接器">
          {connectors.length === 0 ? (
            <Empty text="未配置连接器（经 env 启用笔记/日历/邮件/MCP 后显示）。" />
          ) : (
            connectors.map((c) => (
              <div key={c.id} className="border-l border-linedark pl-2">
                <div className="flex items-center gap-2">
                  <span className="text-ink">{c.displayName}</span>
                  <Badge text={c.id} />
                  <Badge
                    text={c.ok ? "正常" : "异常"}
                    tone={c.ok ? STATUS_BADGE.completed : STATUS_BADGE.failed}
                  />
                  <span className="text-muted">{c.toolCount} 个工具</span>
                </div>
                {c.detail ? (
                  <div className="text-muted mt-0.5 break-words">{c.detail}</div>
                ) : null}
              </div>
            ))
          )}
        </Section>

        <Section title="任务检查点">
          {checkpoints.length === 0 ? (
            <Empty text="暂无 TaskPool 运行快照。" />
          ) : (
            checkpoints.map((cp) => {
              const done = cp.nodes.filter((n) => n.state === "completed").length;
              return (
                <div key={cp.runId} className="border-l border-linedark pl-2">
                  <div className="flex items-center gap-2">
                    <span className="text-ink">{cp.graph.goalInput ?? cp.planId}</span>
                    <Badge text={cp.status} tone={STATUS_BADGE[cp.status]} />
                    <span className="text-muted">
                      节点 {done}/{cp.nodes.length}
                    </span>
                    <span className="flex-1" />
                    {cp.status === "running" ? (
                      <>
                        <button
                          type="button"
                          disabled={busy}
                          onClick={() => void resume(cp.runId)}
                          className="text-muted hover:text-ink disabled:opacity-50"
                        >
                          恢复
                        </button>
                        <button
                          type="button"
                          disabled={busy}
                          onClick={() => void cancel(cp.runId)}
                          className="text-muted hover:text-signal disabled:opacity-50"
                        >
                          取消
                        </button>
                      </>
                    ) : null}
                  </div>
                  <div className="text-[10px] text-muted mt-0.5">
                    run {shortId(cp.runId)} · 创建 {fmtTime(cp.createdAt)} · 更新{" "}
                    {fmtTime(cp.updatedAt)}
                  </div>
                </div>
              );
            })
          )}
        </Section>

        <Section title="通知通道">
          {channels.length === 0 ? (
            <Empty text="未配置通道（经 OPENINTJ_CHANNELS 启用 webhook 后显示）。" />
          ) : (
            channels.map((ch) => (
              <div key={ch.id} className="border-l border-linedark pl-2">
                <div className="flex items-center gap-2">
                  <span className="text-ink">{ch.id}</span>
                  <Badge text={ch.kind} />
                  <Badge
                    text={ch.ok ? "正常" : "异常"}
                    tone={ch.ok ? STATUS_BADGE.completed : STATUS_BADGE.failed}
                  />
                  <span className="text-muted">已发送 {ch.sentCount}</span>
                  <span className="flex-1" />
                  <button
                    type="button"
                    onClick={() => void testChannel(ch.id)}
                    className="text-muted hover:text-ink"
                  >
                    测试发送
                  </button>
                </div>
                <div className="text-[10px] text-muted mt-0.5">
                  上次 {fmtTime(ch.lastSentAt)}
                  {ch.detail ? ` · ${ch.detail}` : ""}
                  {ch.lastError ? ` · 错误：${ch.lastError}` : ""}
                </div>
              </div>
            ))
          )}
        </Section>

        <Section title="子智能体">
          {subagents.length === 0 ? (
            <Empty text="暂无子智能体记录（启用 OPENINTJ_SUBAGENTS 后，delegate_task 的运行会显示在这里）。" />
          ) : (
            subagents.map((s) => (
              <div key={s.id} className="border-l border-linedark pl-2">
                <div className="flex items-center gap-2">
                  <span className="text-ink">{s.label}</span>
                  <Badge text={s.status} tone={STATUS_BADGE[s.status]} />
                  <span className="text-muted">{fmtTime(s.startedAt)}</span>
                </div>
                <div className="text-muted break-words mt-0.5">{s.task}</div>
                {s.finalAnswer ? (
                  <div className="text-muted break-words mt-0.5 line-clamp-3">
                    {s.finalAnswer}
                  </div>
                ) : null}
                {s.error ? <div className="text-signal break-words mt-0.5">{s.error}</div> : null}
              </div>
            ))
          )}
        </Section>
      </div>
    </div>
  );
};
