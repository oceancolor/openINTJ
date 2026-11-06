import React from "react";
import type {
  IdentityPersonView,
  IdentityProposalView,
  IdentitySeedKindView,
} from "../../shared/ipc-protocol.js";

const KIND_LABELS: Record<IdentitySeedKindView, string> = {
  email: "邮箱",
  phone: "手机号",
  wechat: "微信号",
  username: "用户名",
  url: "链接",
};

const SOURCE_LABELS: Record<IdentityProposalView["account"]["source"], string> = {
  gravatar: "Gravatar 公开资料",
  mail: "本机邮件",
  calendar: "本机日历",
  notes: "本机笔记",
  manual: "手动登记",
};

const RESOLVER_LABELS: Record<string, string> = {
  gravatar: "Gravatar 公开资料",
  "local-corpus": "本机语料（邮件/日历/笔记）",
};

const INPUT =
  "bg-white border border-linedark rounded px-2 py-1.5 text-ink placeholder:text-faint focus:outline-none focus:border-moss min-w-0";
const BTN =
  "shrink-0 px-3 py-1.5 rounded bg-ink hover:bg-inksoft disabled:opacity-50 text-white";

type ResolveResult = {
  seed: { kind: IdentitySeedKindView; value: string };
  proposals: IdentityProposalView[];
  reports: { resolverId: string; ok: boolean; detail?: string | undefined; count: number }[];
};

/**
 * RFC-020 身份打通：输入一个邮箱 / 手机号 / 微信号作为种子，拉起本人分散在
 * 各应用、社区、工具里的账号候选；勾选确认后才写入身份图谱。
 *
 * 边界（诚实标注）：邮箱可查 Gravatar 公开资料；手机号 / 微信号没有合法的
 * 公开反查，只能命中本机邮件 / 日历 / 笔记语料 + 手动登记。
 */
export const IdentityPanel: React.FC = () => {
  const [persons, setPersons] = React.useState<IdentityPersonView[]>([]);
  const [seed, setSeed] = React.useState("");
  const [kind, setKind] = React.useState<IdentitySeedKindView | "auto">("auto");
  const [result, setResult] = React.useState<ResolveResult | undefined>();
  const [checked, setChecked] = React.useState<ReadonlySet<string>>(new Set());
  const [error, setError] = React.useState<string | undefined>();
  const [notice, setNotice] = React.useState<string | undefined>();
  const [busy, setBusy] = React.useState(false);

  // Manual assertion form.
  const [mPlatform, setMPlatform] = React.useState("github");
  const [mKind, setMKind] = React.useState<IdentitySeedKindView>("url");
  const [mValue, setMValue] = React.useState("");
  const [mPerson, setMPerson] = React.useState<string>("");

  const refresh = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.identityPersons) return;
    try {
      setPersons(await api.identityPersons());
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  React.useEffect(() => {
    void refresh();
  }, [refresh]);

  const resolve = async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.identityResolve || !seed.trim()) return;
    setBusy(true);
    setError(undefined);
    setNotice(undefined);
    try {
      const res = await api.identityResolve({
        ...(kind === "auto" ? {} : { kind }),
        value: seed.trim(),
      });
      if ("error" in res) {
        setError(res.error);
        setResult(undefined);
        return;
      }
      setResult(res);
      setChecked(new Set(res.proposals.map((p) => p.id)));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const confirm = async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.identityConfirm || !result) return;
    const selected = result.proposals.filter((p) => checked.has(p.id));
    if (selected.length === 0) return;
    setBusy(true);
    setError(undefined);
    try {
      const res = await api.identityConfirm({ proposals: selected });
      if ("error" in res) {
        setError(res.error);
        return;
      }
      setNotice(
        `已关联 ${res.linked.length} 个账号${res.skipped.length ? `，跳过 ${res.skipped.length} 个已关联` : ""}。`,
      );
      setResult(undefined);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  /** One-click claim: register the resolved seed itself as my account. */
  const claimSeed = async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.identityAssert || !result) return;
    const platformByKind: Record<IdentitySeedKindView, string> = {
      email: "mail",
      phone: "phone",
      wechat: "wechat",
      username: "handle",
      url: "web",
    };
    setBusy(true);
    setError(undefined);
    try {
      const res = await api.identityAssert({
        ...(persons[0] ? { personId: persons[0].id } : {}),
        platform: platformByKind[result.seed.kind],
        kind: result.seed.kind,
        value: result.seed.value,
      });
      if ("error" in res) {
        setError(res.error);
        return;
      }
      setNotice(`已认领 ${result.seed.value}（手动登记，未做公开验证）。`);
      setResult(undefined);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const assertManual = async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.identityAssert || !mValue.trim() || !mPlatform.trim()) return;
    setBusy(true);
    setError(undefined);
    try {
      const res = await api.identityAssert({
        ...(mPerson ? { personId: mPerson } : {}),
        platform: mPlatform.trim(),
        kind: mKind,
        value: mValue.trim(),
      });
      if ("error" in res) {
        setError(res.error);
        return;
      }
      setMValue("");
      setNotice("已手动登记（声明归属，未做公开验证）。");
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const unlink = async (accountId: string): Promise<void> => {
    const api = window.openintj;
    if (!api?.identityUnlink) return;
    const res = await api.identityUnlink(accountId);
    if ("error" in res) setError(res.error);
    else await refresh();
  };

  return (
    <div className="p-3 text-xs space-y-4 overflow-y-auto" data-testid="identity-panel">
      <p className="text-muted leading-relaxed break-words">
        输入你的邮箱 / 手机号 / 微信号，拉起你在各应用、社区、工具上的账号。邮箱可查 Gravatar
        公开资料；手机号与微信号没有合法的公开反查，只能命中本机邮件 / 日历 /
        笔记里已有的记录，或手动登记。候选经你确认后才入图谱。
      </p>

      <div className="space-y-2">
        <div className="flex gap-2">
          <select
            aria-label="种子类型"
            className={`${INPUT} w-[104px] shrink-0`}
            value={kind}
            onChange={(e) => setKind(e.target.value as typeof kind)}
          >
            <option value="auto">自动识别</option>
            {(Object.keys(KIND_LABELS) as IdentitySeedKindView[]).map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k]}
              </option>
            ))}
          </select>
          <button
            type="button"
            className={BTN}
            disabled={busy || !seed.trim()}
            onClick={() => void resolve()}
          >
            {busy ? "解析中…" : "解析身份"}
          </button>
        </div>
        <input
          aria-label="身份种子"
          className={`${INPUT} w-full`}
          placeholder="邮箱 / 手机号 / 微信号 / @用户名"
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void resolve();
          }}
        />
        {result ? (
          <div className="space-y-0.5 text-muted break-words">
            {result.reports.map((r) => (
              <div key={r.resolverId}>
                {RESOLVER_LABELS[r.resolverId] ?? r.resolverId}：
                {r.ok
                  ? r.count > 0
                    ? `找到 ${r.count} 个候选`
                    : r.resolverId === "gravatar"
                      ? "无公开资料（不代表没有注册过账号）"
                      : "0 个候选"
                  : `查询失败（${r.detail ?? "未知原因"}）`}
              </div>
            ))}
          </div>
        ) : null}
      </div>

      {error ? <div className="text-signal break-words">{error}</div> : null}
      {notice ? <div className="text-moss break-words">{notice}</div> : null}

      {result && result.proposals.length === 0 ? (
        <div className="space-y-2 rounded border border-linedark bg-white p-2">
          <div className="text-muted leading-relaxed break-words">
            没有解析到现成候选。可以先把这个{KIND_LABELS[result.seed.kind]}认领为你的账号，
            之后再把其他平台的账号逐个登记到同一个身份下。
          </div>
          <button type="button" className={BTN} disabled={busy} onClick={() => void claimSeed()}>
            认领 {result.seed.value}
          </button>
        </div>
      ) : null}

      {result && result.proposals.length > 0 ? (
        <div className="space-y-2">
          <div className="text-muted font-medium">候选账号（勾选确认）</div>
          {result.proposals.map((p) => (
            <label
              key={p.id}
              className="flex gap-2 items-start rounded border border-linedark bg-white p-2 cursor-pointer"
            >
              <input
                type="checkbox"
                className="mt-0.5 shrink-0"
                checked={checked.has(p.id)}
                onChange={(e) => {
                  setChecked((prev) => {
                    const next = new Set(prev);
                    if (e.target.checked) next.add(p.id);
                    else next.delete(p.id);
                    return next;
                  });
                }}
              />
              <span className="min-w-0 break-words">
                <span className="text-ink">
                  {p.account.platform} · {KIND_LABELS[p.account.kind]} · {p.account.value}
                </span>
                <span className="block text-muted">
                  {SOURCE_LABELS[p.account.source]} · 置信 {Math.round(p.account.confidence * 100)}%
                  {p.account.verifiedAt ? " · 已公开验证" : ""} · {p.account.evidence}
                </span>
              </span>
            </label>
          ))}
          <button
            type="button"
            className="px-3 py-1.5 rounded bg-moss hover:bg-mossdeep disabled:opacity-50 text-white"
            disabled={busy || checked.size === 0}
            onClick={() => void confirm()}
          >
            确认关联选中项
          </button>
        </div>
      ) : null}

      <div className="space-y-2">
        <div className="text-muted font-medium">手动登记账号</div>
        <div className="flex gap-2">
          <input
            aria-label="平台"
            className={`${INPUT} flex-1`}
            placeholder="平台，如 github"
            value={mPlatform}
            onChange={(e) => setMPlatform(e.target.value)}
          />
          <select
            aria-label="账号类型"
            className={`${INPUT} w-[104px] shrink-0`}
            value={mKind}
            onChange={(e) => setMKind(e.target.value as IdentitySeedKindView)}
          >
            {(Object.keys(KIND_LABELS) as IdentitySeedKindView[]).map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k]}
              </option>
            ))}
          </select>
        </div>
        <select
          aria-label="归属"
          className={`${INPUT} w-full`}
          value={mPerson}
          onChange={(e) => setMPerson(e.target.value)}
        >
          <option value="">归属：新建身份</option>
          {persons.map((p) => (
            <option key={p.id} value={p.id}>
              归属：{p.displayName}
            </option>
          ))}
        </select>
        <div className="flex gap-2">
          <input
            aria-label="账号值"
            className={`${INPUT} flex-1`}
            placeholder="账号 / 手机号 / 微信号 / 主页链接"
            value={mValue}
            onChange={(e) => setMValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void assertManual();
            }}
          />
          <button
            type="button"
            className={BTN}
            disabled={busy || !mValue.trim()}
            onClick={() => void assertManual()}
          >
            登记
          </button>
        </div>
      </div>

      <div className="space-y-3">
        <div className="text-muted font-medium">我的身份图谱</div>
        {persons.length === 0 ? (
          <div className="text-muted leading-relaxed">
            还没有关联任何身份。先解析一个种子，或手动登记。
          </div>
        ) : (
          persons.map((person) => (
            <div key={person.id} className="rounded border border-linedark bg-white p-2">
              <div className="text-ink mb-1 break-words">{person.displayName}</div>
              {person.accounts.map((a) => (
                <div key={a.id} className="flex items-center gap-2 py-0.5">
                  <span className="text-ink min-w-0 break-words">
                    {a.platform} · {KIND_LABELS[a.kind]} · {a.value}
                  </span>
                  <span className="text-muted shrink-0">
                    {SOURCE_LABELS[a.source]}
                    {a.verifiedAt ? " · 已验证" : ""}
                  </span>
                  <button
                    type="button"
                    className="ml-auto shrink-0 text-muted hover:text-signal"
                    onClick={() => void unlink(a.id)}
                  >
                    解绑
                  </button>
                </div>
              ))}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
