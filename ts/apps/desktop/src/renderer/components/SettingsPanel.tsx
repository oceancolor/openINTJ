/**
 * 设置面板（RFC-004 §4 config 面 + §8 workspace 能力面）。
 *
 * 两段：
 *  1. 工作区：显示当前沙箱根 / 命令开关 / 白名单（workspaceInfo）；「选择目录」弹系统对话框
 *     （workspacePickDir，持久化到 config，下次启动生效）；底部实时显示 fs.watch 变更（onWorkspaceEvent）。
 *  2. 应用配置：读/写 AppConfig（getConfig/updateConfig）。运行时项保存后会热重装 agent；
 *     进程级项仍可用「重启 OpenINTJ」。
 *
 * 说明：desktop renderer 无 jsdom 单测；本面板逻辑经 IPC 契约测试（ipc-handlers.spec）间接覆盖，
 * 交互留给手动 / Playwright e2e。
 */
import React from "react";
import type {
  AppConfig,
  AppConfigPatch,
  ModelProfile,
  ModelProvider,
  WorkspaceInfo,
} from "../../shared/ipc-protocol.js";
import { DEFAULT_MODEL_PROFILES } from "../../shared/ipc-protocol.js";

const isError = (r: unknown): r is { error: string } =>
  typeof r === "object" && r !== null && "error" in r;

type ChangeEvent = { event: string; path: string; ts: number };

const PROVIDERS = ["auto", "mock", "ollama", "hunyuan", "kimi", "minimax", "glm"] as const;
const EMBED_PROVIDERS = ["auto", "simple", "ollama", "xenova", "mock"] as const;
const RETRIEVAL = ["vector", "hybrid"] as const;

interface ToggleRow {
  key: keyof AppConfig;
  label: string;
  hint: string;
}

const TOGGLES: ToggleRow[] = [
  {
    key: "enableProductBehavior",
    label: "产品行为契约",
    hint: "RFC-006 treatment/control A/B（默认开启）",
  },
  { key: "enableDormant", label: "钝化记忆", hint: "被动学习 → 审批 → persona 注入" },
  { key: "enablePersona", label: "注入 persona", hint: "已批准人格注入 system prompt（A/B）" },
  { key: "enableSkills", label: "技能系统", hint: "命中的能力包全文注入" },
  { key: "enableSkillLearning", label: "技能自学习", hint: "轨迹蒸馏 → 审批（隐含开技能系统）" },
  {
    key: "enableClassifier",
    label: "前端分类器",
    hint: "预分类降 token + 强化；TaskPool 开启时为必需项",
  },
  {
    key: "enableTaskPool",
    label: "任务池编排",
    hint: "RFC-007 planning/analysis 有界 DAG；会自动启用必需的前端分类器",
  },
  { key: "enableCommands", label: "允许执行命令", hint: "execute_command 沙箱（高危）" },
  { key: "autoUpdate", label: "自动更新", hint: "打包后检查并下载新版本" },
];

export const SettingsPanel: React.FC = () => {
  const [config, setConfig] = React.useState<AppConfig | undefined>();
  const [wsInfo, setWsInfo] = React.useState<WorkspaceInfo | undefined>();
  const [changes, setChanges] = React.useState<ChangeEvent[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | undefined>();
  const [saved, setSaved] = React.useState(false);
  const [profiles, setProfiles] = React.useState<ModelProfile[]>([]);
  const [credentialDrafts, setCredentialDrafts] = React.useState<Record<string, string>>({});
  const [profileTests, setProfileTests] = React.useState<Record<string, string>>({});
  const [newProfile, setNewProfile] = React.useState({
    name: "",
    provider: "kimi" as ModelProvider,
    model: "",
    baseUrl: "",
  });

  const refresh = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    try {
      const [c, w, p] = await Promise.all([
        api.getConfig(),
        api.workspaceInfo(),
        api.modelProfiles(),
      ]);
      setConfig(c);
      setWsInfo(w);
      setProfiles(p);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  React.useEffect(() => {
    void refresh();
  }, [refresh]);

  React.useEffect(() => {
    const api = window.openintj;
    if (!api?.onWorkspaceEvent) return;
    return api.onWorkspaceEvent((p) => {
      const ev = (p ?? {}) as { event?: string; path?: string };
      setChanges((c) =>
        [{ event: ev.event ?? "change", path: ev.path ?? "", ts: Date.now() }, ...c].slice(0, 20),
      );
    });
  }, []);

  const patch = async (p: AppConfigPatch): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    setBusy(true);
    setError(undefined);
    try {
      const r = await api.updateConfig(p);
      if (isError(r)) {
        const detail =
          "message" in r && typeof (r as { message?: unknown }).message === "string"
            ? `: ${(r as { message: string }).message}`
            : "";
        setError(`${r.error}${detail}`);
        return;
      }
      setConfig(r);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 1500);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const pickDir = async (): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    setBusy(true);
    try {
      const r = await api.workspacePickDir();
      if (!r.canceled && r.root) {
        setConfig((c) => ({ ...(c ?? {}), workspaceDir: r.root }));
        setSaved(true);
        window.setTimeout(() => setSaved(false), 1500);
      }
    } finally {
      setBusy(false);
    }
  };

  const restart = async (): Promise<void> => {
    const api = window.openintj;
    if (!api || !window.confirm("确认重启 OpenINTJ？当前请求会被终止。")) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await api.restartApp();
      if (!result.ok) setError(result.reason ?? "restart_failed");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };

  const saveCredential = async (profileId: string): Promise<void> => {
    const apiKey = credentialDrafts[profileId]?.trim();
    if (!apiKey) return;
    setBusy(true);
    setError(undefined);
    try {
      const result = await window.openintj.setModelCredential({ profileId, apiKey });
      if (!result.ok) throw new Error(result.error ?? "credential_save_failed");
      setCredentialDrafts((current) => ({ ...current, [profileId]: "" }));
      setProfiles(await window.openintj.modelProfiles());
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const testProfile = async (profileId: string): Promise<void> => {
    setProfileTests((current) => ({ ...current, [profileId]: "测试中…" }));
    const result = await window.openintj.testModelProfile(profileId);
    setProfileTests((current) => ({
      ...current,
      [profileId]: result.ok ? "连接成功" : `失败：${result.error ?? "unknown"}`,
    }));
  };

  const persistProfile = async (profile: ModelProfile): Promise<void> => {
    const next = [
      ...(config?.modelProfiles ?? []).filter((item) => item.id !== profile.id),
      {
        id: profile.id,
        name: profile.name,
        provider: profile.provider,
        model: profile.model,
        ...(profile.baseUrl ? { baseUrl: profile.baseUrl } : {}),
      },
    ];
    await patch({ modelProfiles: next });
    setProfiles(await window.openintj.modelProfiles());
  };

  const addProfile = async (): Promise<void> => {
    if (!newProfile.name.trim() || !newProfile.model.trim() || !newProfile.provider) return;
    const profile: ModelProfile = {
      id: `custom-${Date.now().toString(36)}`,
      name: newProfile.name.trim(),
      provider: newProfile.provider,
      model: newProfile.model.trim(),
      ...(newProfile.baseUrl.trim() ? { baseUrl: newProfile.baseUrl.trim() } : {}),
    };
    await persistProfile(profile);
    setNewProfile({ name: "", provider: "kimi", model: "", baseUrl: "" });
  };

  const deleteProfile = async (profile: ModelProfile): Promise<void> => {
    await patch({
      modelProfiles: (config?.modelProfiles ?? []).filter((item) => item.id !== profile.id),
    });
    await window.openintj.deleteModelCredential(profile.id);
    setProfiles(await window.openintj.modelProfiles());
  };

  const cfg = config ?? {};

  return (
    <div className="flex flex-col h-full overflow-y-auto text-xs">
      {error ? (
        <div className="mx-3 mt-2 px-2 py-1 bg-red-900/50 text-red-200 rounded">{error}</div>
      ) : null}
      {saved ? (
        <div className="mx-3 mt-2 px-2 py-1 bg-green-900/40 text-green-200 rounded">
          已保存并立即生效
        </div>
      ) : null}

      {/* 工作区 */}
      <section className="p-3 space-y-2 border-b border-gray-800">
        <div className="text-gray-300 font-medium">工作区</div>
        <div className="text-gray-400 break-all">
          根目录：<code className="text-cyan-400">{wsInfo?.root ?? cfg.workspaceDir ?? "—"}</code>
        </div>
        <div className="text-gray-500">
          命令：{wsInfo?.enableCommands ? "允许" : "禁用"}
          {wsInfo?.allowedCommands?.length ? ` · 白名单 ${wsInfo.allowedCommands.join(", ")}` : ""}
        </div>
        <button
          type="button"
          onClick={() => void pickDir()}
          disabled={busy}
          className="px-2 py-1 rounded bg-purple-700 hover:bg-purple-600 disabled:bg-gray-700 text-white"
        >
          选择目录…
        </button>
      </section>

      <section className="p-3 space-y-2 border-b border-gray-800">
        <div className="text-gray-300 font-medium">模型 Profiles</div>
        <div className="text-gray-500">API Key 使用系统安全存储加密，界面不会读取明文。</div>
        {profiles.map((profile) => {
          const needsKey = !["auto", "ollama", "mock"].includes(profile.provider);
          return (
            <div key={profile.id} className="rounded border border-gray-800 p-2 space-y-1">
              <div className="flex items-center justify-between gap-2">
                <button
                  type="button"
                  className={`text-left ${
                    cfg.activeModelProfileId === profile.id ? "text-cyan-300" : "text-gray-200"
                  }`}
                  onClick={() =>
                    void patch({
                      activeModelProfileId: profile.id,
                      llmProvider: profile.provider,
                    })
                  }
                >
                  {profile.name}
                </button>
                <span className={profile.hasCredential ? "text-green-400" : "text-amber-400"}>
                  {profile.hasCredential ? "可用" : "需密钥"}
                </span>
              </div>
              <div className="text-gray-500">
                {profile.provider} / {profile.model}
              </div>
              <div className="grid grid-cols-2 gap-1">
                <input
                  aria-label={`${profile.name} model`}
                  value={profile.model}
                  onChange={(event) =>
                    setProfiles((current) =>
                      current.map((item) =>
                        item.id === profile.id ? { ...item, model: event.target.value } : item,
                      ),
                    )
                  }
                  className="bg-gray-800 rounded px-2 py-1"
                />
                <input
                  aria-label={`${profile.name} base URL`}
                  value={profile.baseUrl ?? ""}
                  placeholder="Base URL（默认可留空）"
                  onChange={(event) =>
                    setProfiles((current) =>
                      current.map((item) =>
                        item.id === profile.id
                          ? { ...item, baseUrl: event.target.value || undefined }
                          : item,
                      ),
                    )
                  }
                  className="bg-gray-800 rounded px-2 py-1"
                />
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void persistProfile(profile)}
                  className="px-2 py-0.5 rounded bg-cyan-900 hover:bg-cyan-800"
                >
                  保存 Profile
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void testProfile(profile.id)}
                  className="px-2 py-0.5 rounded bg-gray-700 hover:bg-gray-600"
                >
                  测试连接
                </button>
                <span className="text-gray-500">{profileTests[profile.id]}</span>
                {!DEFAULT_MODEL_PROFILES.some((item) => item.id === profile.id) ? (
                  <button
                    type="button"
                    onClick={() => void deleteProfile(profile)}
                    className="ml-auto text-red-400"
                  >
                    删除
                  </button>
                ) : null}
              </div>
              {needsKey ? (
                <div className="flex gap-1">
                  <input
                    type="password"
                    autoComplete="new-password"
                    placeholder={profile.hasCredential ? "替换 API Key" : "输入 API Key"}
                    value={credentialDrafts[profile.id] ?? ""}
                    onChange={(event) =>
                      setCredentialDrafts((current) => ({
                        ...current,
                        [profile.id]: event.target.value,
                      }))
                    }
                    className="min-w-0 flex-1 bg-gray-800 text-gray-200 rounded px-2 py-1"
                  />
                  <button
                    type="button"
                    disabled={busy || !credentialDrafts[profile.id]?.trim()}
                    onClick={() => void saveCredential(profile.id)}
                    className="px-2 py-1 rounded bg-cyan-800 disabled:bg-gray-700"
                  >
                    保存
                  </button>
                </div>
              ) : null}
            </div>
          );
        })}
        <div className="rounded border border-dashed border-gray-700 p-2 space-y-1">
          <div className="text-gray-400">添加自定义 Profile</div>
          <div className="grid grid-cols-2 gap-1">
            <input
              placeholder="显示名称"
              value={newProfile.name}
              onChange={(event) =>
                setNewProfile((current) => ({ ...current, name: event.target.value }))
              }
              className="bg-gray-800 rounded px-2 py-1"
            />
            <select
              value={newProfile.provider}
              onChange={(event) =>
                setNewProfile((current) => ({
                  ...current,
                  provider: event.target.value as ModelProvider,
                }))
              }
              className="bg-gray-800 rounded px-2 py-1"
            >
              {PROVIDERS.filter((provider) => !["auto", "mock"].includes(provider)).map(
                (provider) => (
                  <option key={provider} value={provider}>
                    {provider}
                  </option>
                ),
              )}
            </select>
            <input
              placeholder="模型 ID"
              value={newProfile.model}
              onChange={(event) =>
                setNewProfile((current) => ({ ...current, model: event.target.value }))
              }
              className="bg-gray-800 rounded px-2 py-1"
            />
            <input
              placeholder="Base URL"
              value={newProfile.baseUrl}
              onChange={(event) =>
                setNewProfile((current) => ({ ...current, baseUrl: event.target.value }))
              }
              className="bg-gray-800 rounded px-2 py-1"
            />
          </div>
          <button
            type="button"
            disabled={busy || !newProfile.name.trim() || !newProfile.model.trim()}
            onClick={() => void addProfile()}
            className="px-2 py-1 rounded bg-purple-800 disabled:bg-gray-700"
          >
            添加 Profile
          </button>
        </div>
      </section>

      {/* 应用配置 */}
      <section className="p-3 space-y-3 border-b border-gray-800">
        <div className="text-gray-300 font-medium">应用配置</div>
        <button
          type="button"
          onClick={() => void restart()}
          disabled={busy}
          className="px-2 py-1 rounded bg-orange-700 hover:bg-orange-600 disabled:bg-gray-700 text-white"
        >
          重启 OpenINTJ
        </button>

        <label className="flex items-center gap-2">
          <span className="w-24 text-gray-400">LLM 提供方</span>
          <select
            value={cfg.llmProvider ?? "auto"}
            disabled={busy}
            onChange={(e) =>
              void patch({ llmProvider: e.target.value as AppConfig["llmProvider"] })
            }
            className="bg-gray-800 text-gray-200 rounded px-1 py-0.5"
          >
            {PROVIDERS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2">
          <span className="w-24 text-gray-400">Embed 提供方</span>
          <select
            value={cfg.embedProvider ?? "auto"}
            disabled={busy}
            onChange={(e) =>
              void patch({ embedProvider: e.target.value as AppConfig["embedProvider"] })
            }
            className="bg-gray-800 text-gray-200 rounded px-1 py-0.5"
          >
            {EMBED_PROVIDERS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-gray-400">Ollama</span>
          <input
            type="text"
            placeholder="OLLAMA_BASE_URL"
            value={cfg.ollamaBaseUrl ?? ""}
            disabled={busy}
            onChange={(e) =>
              setConfig((c) => ({ ...(c ?? {}), ollamaBaseUrl: e.target.value || undefined }))
            }
            onBlur={(e) => void patch({ ollamaBaseUrl: e.target.value.trim() || undefined })}
            className="bg-gray-800 text-gray-200 rounded px-2 py-1"
          />
          <input
            type="text"
            placeholder="OLLAMA_MODEL"
            value={cfg.ollamaModel ?? ""}
            disabled={busy}
            onChange={(e) =>
              setConfig((c) => ({ ...(c ?? {}), ollamaModel: e.target.value || undefined }))
            }
            onBlur={(e) => void patch({ ollamaModel: e.target.value.trim() || undefined })}
            className="bg-gray-800 text-gray-200 rounded px-2 py-1"
          />
          <input
            type="text"
            placeholder="OLLAMA_EMBED_MODEL"
            value={cfg.ollamaEmbedModel ?? ""}
            disabled={busy}
            onChange={(e) =>
              setConfig((c) => ({ ...(c ?? {}), ollamaEmbedModel: e.target.value || undefined }))
            }
            onBlur={(e) => void patch({ ollamaEmbedModel: e.target.value.trim() || undefined })}
            className="bg-gray-800 text-gray-200 rounded px-2 py-1"
          />
        </label>

        <label className="flex items-center gap-2">
          <span className="w-24 text-gray-400">检索模式</span>
          <select
            value={cfg.retrievalMode ?? "vector"}
            disabled={busy}
            onChange={(e) =>
              void patch({ retrievalMode: e.target.value as AppConfig["retrievalMode"] })
            }
            className="bg-gray-800 text-gray-200 rounded px-1 py-0.5"
          >
            {RETRIEVAL.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        {TOGGLES.map((t) => (
          <label key={t.key} className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={
                t.key === "enableProductBehavior"
                  ? cfg.enableProductBehavior !== false
                  : t.key === "enableClassifier" && cfg.enableTaskPool === true
                    ? true
                    : cfg[t.key] === true
              }
              disabled={busy || (t.key === "enableClassifier" && cfg.enableTaskPool === true)}
              onChange={(e) =>
                void patch(
                  t.key === "enableTaskPool" && e.target.checked
                    ? { enableTaskPool: true, enableClassifier: true }
                    : ({ [t.key]: e.target.checked } as AppConfigPatch),
                )
              }
              className="mt-0.5"
            />
            <span>
              <span className="text-gray-300">{t.label}</span>
              <span className="ml-1 text-gray-500">— {t.hint}</span>
            </span>
          </label>
        ))}
      </section>

      {/* 工作区实时变更 */}
      <section className="p-3 space-y-1 flex-1 min-h-0">
        <div className="text-gray-300 font-medium">工作区变更（实时）</div>
        {changes.length === 0 ? (
          <div className="text-gray-500">暂无变更 — 在工作区目录里增删改文件试试</div>
        ) : (
          <div className="space-y-0.5">
            {changes.map((c, i) => (
              <div key={`${c.ts}-${i}`} className="text-gray-400 flex gap-2">
                <span className="text-gray-600">{c.event}</span>
                <code className="text-cyan-400 truncate">{c.path || "(未知)"}</code>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
