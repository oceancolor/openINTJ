import React from "react";
import { type ChatMessage, ChatPanel } from "./components/ChatPanel.js";
import { StatusBar, type StatusSnapshot } from "./components/StatusBar.js";
import { type TrajectoryEntry, TrajectoryPanel } from "./components/TrajectoryPanel.js";
import "./types.js";

export const App = (): JSX.Element => {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [trajectory, setTrajectory] = React.useState<TrajectoryEntry[]>([]);
  const [status, setStatus] = React.useState<StatusSnapshot | undefined>();
  const [pending, setPending] = React.useState(false);

  // 订阅 IPC 事件
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
    return () => {
      for (const o of offs) o();
    };
  }, []);

  // 周期刷新 status
  React.useEffect(() => {
    const refresh = async (): Promise<void> => {
      try {
        const s = (await window.openintj?.status()) as StatusSnapshot;
        setStatus(s);
      } catch {
        // ignore
      }
    };
    void refresh();
    const id = window.setInterval(refresh, 2_000);
    return () => window.clearInterval(id);
  }, []);

  const handleSend = async (text: string): Promise<void> => {
    setMessages((m) => [...m, { role: "user", content: text }]);
    setPending(true);
    try {
      const res = await window.openintj.chat({ query: text });
      setMessages((m) => [...m, { role: "assistant", content: res.finalAnswer }]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "system",
          content: `[错误] ${(e as Error).message}`,
        },
      ]);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex flex-col h-screen text-gray-200">
      <header className="px-4 py-2 border-b border-gray-800 bg-[#181825] flex items-center gap-2">
        <span className="text-base font-semibold">OpenINTJ</span>
        <span className="text-xs text-gray-500">v3.0 Local Desktop</span>
      </header>
      <div className="flex-1 grid grid-cols-[1fr_360px] min-h-0">
        <ChatPanel messages={messages} onSend={(t) => void handleSend(t)} pending={pending} />
        <TrajectoryPanel entries={trajectory} />
      </div>
      <StatusBar status={status} />
    </div>
  );
};
