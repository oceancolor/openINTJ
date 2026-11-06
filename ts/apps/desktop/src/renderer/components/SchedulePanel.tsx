import React from "react";
import type {
  ScheduleTriggerView,
  ScheduledJobView,
  SchedulerEventView,
} from "../../shared/ipc-protocol.js";

type TriggerKind = ScheduleTriggerView["kind"];

const fmtTime = (ms?: number): string => (ms === undefined ? "—" : new Date(ms).toLocaleString());

const describeTrigger = (trigger: ScheduleTriggerView): string => {
  switch (trigger.kind) {
    case "once":
      return `一次 · ${fmtTime(trigger.at)}`;
    case "interval":
      return `每 ${Math.round(trigger.everyMs / 60_000) || 1} 分钟`;
    case "daily":
      return `每天 ${trigger.time}`;
  }
};

const KIND_LABELS: Record<SchedulerEventView["kind"], string> = {
  fired: "已触发",
  completed: "已完成",
  failed: "失败",
  deferred: "安静时段顺延",
};

/**
 * RFC-015 定时任务：持久化 job 列表 + 新建/启停/删除。到点后 agent 会把
 * prompt 作为一轮对话投递到目标会话；触发/完成/失败事件实时刷新本页。
 */
export const SchedulePanel: React.FC = () => {
  const [jobs, setJobs] = React.useState<ScheduledJobView[]>([]);
  const [lastEvent, setLastEvent] = React.useState<string | undefined>();
  const [title, setTitle] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [kind, setKind] = React.useState<TriggerKind>("once");
  const [when, setWhen] = React.useState("");
  const [everyMinutes, setEveryMinutes] = React.useState("60");
  const [dailyTime, setDailyTime] = React.useState("09:00");
  const [error, setError] = React.useState<string | undefined>();
  const [busy, setBusy] = React.useState(false);

  const refresh = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.scheduleList) return;
    try {
      setJobs(await api.scheduleList());
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  React.useEffect(() => {
    void refresh();
    const api = window.openintj;
    if (!api?.onScheduleEvent) return;
    return api.onScheduleEvent((payload) => {
      const event = payload as SchedulerEventView;
      setLastEvent(`${KIND_LABELS[event.kind] ?? event.kind}：${event.job.title}`);
      void refresh();
    });
  }, [refresh]);

  const buildTrigger = (): ScheduleTriggerView | undefined => {
    if (kind === "once") {
      const at = when ? new Date(when).getTime() : Number.NaN;
      if (!Number.isFinite(at)) {
        setError("请选择触发时间");
        return undefined;
      }
      return { kind: "once", at };
    }
    if (kind === "interval") {
      const minutes = Number.parseInt(everyMinutes, 10);
      if (!Number.isFinite(minutes) || minutes < 1) {
        setError("间隔分钟数无效");
        return undefined;
      }
      return { kind: "interval", everyMs: minutes * 60_000 };
    }
    return { kind: "daily", time: dailyTime };
  };

  const add = async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.scheduleAdd || !title.trim() || !prompt.trim()) return;
    const trigger = buildTrigger();
    if (!trigger) return;
    setBusy(true);
    setError(undefined);
    try {
      const res = await api.scheduleAdd({ title: title.trim(), prompt: prompt.trim(), trigger });
      if ("error" in res) setError(res.error);
      else {
        setTitle("");
        setPrompt("");
        await refresh();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const toggle = async (job: ScheduledJobView): Promise<void> => {
    await window.openintj?.scheduleUpdate(job.id, { enabled: !job.enabled });
    await refresh();
  };

  const remove = async (id: string): Promise<void> => {
    await window.openintj?.scheduleRemove(id);
    await refresh();
  };

  return (
    <div className="flex flex-col h-full min-h-0 text-xs">
      <div className="shrink-0 p-2 border-b border-line space-y-2">
        <div className="flex gap-2">
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="标题，如 晨间汇总"
            className="w-32 bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
          />
          <input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="到点交给 agent 的指令"
            className="flex-1 bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
          />
        </div>
        <div className="flex gap-2 items-center">
          <select
            value={kind}
            onChange={(e) => setKind(e.target.value as TriggerKind)}
            className="bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
          >
            <option value="once">一次</option>
            <option value="interval">间隔</option>
            <option value="daily">每天</option>
          </select>
          {kind === "once" ? (
            <input
              type="datetime-local"
              value={when}
              onChange={(e) => setWhen(e.target.value)}
              className="bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
            />
          ) : kind === "interval" ? (
            <span className="flex items-center gap-1 text-muted">
              每
              <input
                value={everyMinutes}
                onChange={(e) => setEveryMinutes(e.target.value)}
                className="w-16 bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
              />
              分钟
            </span>
          ) : (
            <input
              type="time"
              value={dailyTime}
              onChange={(e) => setDailyTime(e.target.value)}
              className="bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
            />
          )}
          <button
            type="button"
            onClick={() => void add()}
            disabled={busy || !title.trim() || !prompt.trim()}
            className="px-3 py-1 rounded bg-ink hover:bg-inksoft disabled:opacity-50 text-white"
          >
            添加
          </button>
        </div>
        <div className="text-muted">
          任务会持久化保存；应用离线期间错过的触发在下次启动时只补跑一次。
        </div>
        {lastEvent ? <div className="text-moss">{lastEvent}</div> : null}
        {error ? <div className="text-signal">[错误] {error}</div> : null}
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2">
        {jobs.length === 0 ? (
          <div className="text-muted">暂无定时任务。在上方创建第一个。</div>
        ) : (
          jobs.map((job) => (
            <div key={job.id} className="border-l border-linedark pl-2">
              <div className="flex items-center gap-2">
                <span className="text-ink">{job.title}</span>
                <span className="px-1 rounded bg-inset text-[10px] text-muted">
                  {describeTrigger(job.trigger)}
                </span>
                {job.lastStatus === "failed" ? (
                  <span className="px-1 rounded bg-[#f8e9e4] text-[10px] text-signaldeep">失败</span>
                ) : job.lastStatus === "completed" ? (
                  <span className="px-1 rounded bg-mossmist text-[10px] text-moss">完成</span>
                ) : null}
                <span className="flex-1" />
                <button
                  type="button"
                  onClick={() => void toggle(job)}
                  className="text-muted hover:text-ink"
                >
                  {job.enabled ? "暂停" : "启用"}
                </button>
                <button
                  type="button"
                  onClick={() => void remove(job.id)}
                  className="text-muted hover:text-signal"
                >
                  删除
                </button>
              </div>
              <div className="text-muted break-words mt-0.5">{job.prompt}</div>
              <div className="text-[10px] text-muted mt-0.5">
                下次：{job.enabled ? fmtTime(job.nextRunAt) : "已停用"} · 已运行 {job.runCount} 次
                {job.lastRunAt ? ` · 上次 ${fmtTime(job.lastRunAt)}` : ""}
                {job.lastError ? ` · ${job.lastError}` : ""}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
