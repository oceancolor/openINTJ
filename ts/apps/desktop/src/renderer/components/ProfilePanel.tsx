import React from "react";
import type { ProfileEntryView, ProfileSectionView } from "../../shared/ipc-protocol.js";

const SECTIONS: Array<[ProfileSectionView, string]> = [
  ["identity", "身份"],
  ["preferences", "偏好"],
  ["goals", "目标"],
  ["projects", "项目"],
  ["relationships", "关系"],
  ["constraints", "约束"],
  ["habits", "习惯"],
  ["notes", "备注"],
];

const SOURCE_LABELS: Record<ProfileEntryView["source"], string> = {
  user: "手动",
  dormant: "推断",
  import: "导入",
};

/**
 * RFC-014 个人档案：记忆控制面。用户可查看 / 新增 / 删除关于自己的结构化
 * 事实；推断（dormant）条目删除后不会被重新导入。档案内容会注入每轮 system
 * prompt 的 user-persona 段，优先于推断画像。
 */
export const ProfilePanel: React.FC = () => {
  const [entries, setEntries] = React.useState<ProfileEntryView[]>([]);
  const [section, setSection] = React.useState<ProfileSectionView>("preferences");
  const [key, setKey] = React.useState("");
  const [value, setValue] = React.useState("");
  const [error, setError] = React.useState<string | undefined>();
  const [busy, setBusy] = React.useState(false);

  const refresh = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.profileList) return;
    try {
      setEntries(await api.profileList());
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  React.useEffect(() => {
    void refresh();
  }, [refresh]);

  const add = async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.profileUpsert || !key.trim() || !value.trim()) return;
    setBusy(true);
    setError(undefined);
    try {
      const res = await api.profileUpsert({ section, key: key.trim(), value: value.trim() });
      if ("error" in res) setError(res.error);
      else {
        setKey("");
        setValue("");
        await refresh();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string): Promise<void> => {
    const api = window.openintj;
    if (!api?.profileRemove) return;
    await api.profileRemove(id);
    await refresh();
  };

  const exportJson = async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.profileExport) return;
    const { profile } = await api.profileExport();
    const url = URL.createObjectURL(new Blob([profile], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "openintj-profile.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full min-h-0 text-xs">
      <div className="shrink-0 p-2 border-b border-line space-y-2">
        <div className="flex gap-2">
          <select
            value={section}
            onChange={(e) => setSection(e.target.value as ProfileSectionView)}
            className="bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
          >
            {SECTIONS.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
          <input
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="键，如 editor"
            className="w-28 bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
          />
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void add();
            }}
            placeholder="值，如 VS Code"
            className="flex-1 bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
          />
          <button
            type="button"
            onClick={() => void add()}
            disabled={busy || !key.trim() || !value.trim()}
            className="px-3 py-1 rounded bg-ink hover:bg-inksoft disabled:opacity-50 text-white"
          >
            添加
          </button>
          <button
            type="button"
            onClick={() => void exportJson()}
            className="px-3 py-1 rounded bg-inset hover:bg-rail text-ink"
          >
            导出
          </button>
        </div>
        <div className="text-muted">
          档案会注入每轮对话并优先于推断画像；删除推断条目后不会被重新导入。
        </div>
        {error ? <div className="text-signal">[错误] {error}</div> : null}
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-3">
        {entries.length === 0 ? (
          <div className="text-muted">档案为空。在上方添加第一条关于自己的事实。</div>
        ) : (
          SECTIONS.map(([id, label]) => {
            const group = entries.filter((e) => e.section === id);
            if (group.length === 0) return null;
            return (
              <div key={id}>
                <div className="text-muted mb-1">{label}</div>
                <div className="space-y-1">
                  {group.map((entry) => (
                    <div
                      key={entry.id}
                      className="flex items-start gap-2 border-l border-linedark pl-2"
                    >
                      <div className="flex-1 min-w-0">
                        <span className="text-ink">{entry.key}</span>
                        <span className="text-muted"> = </span>
                        <span className="text-muted break-words">{entry.value}</span>
                        <span className="ml-2 px-1 rounded bg-inset text-[10px] text-muted">
                          {SOURCE_LABELS[entry.source]}
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => void remove(entry.id)}
                        className="shrink-0 text-muted hover:text-signal"
                        title="删除该条目"
                      >
                        删除
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
