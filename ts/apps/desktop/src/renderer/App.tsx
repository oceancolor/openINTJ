import React from "react";
import { type ChatMessage, ChatPanel } from "./components/ChatPanel.js";
import { DormantPanel } from "./components/DormantPanel.js";
import { MemoryPanel } from "./components/MemoryPanel.js";
import { SkillPanel } from "./components/SkillPanel.js";
import { StatusBar, type StatusSnapshot } from "./components/StatusBar.js";
import { type TrajectoryEntry, TrajectoryPanel } from "./components/TrajectoryPanel.js";
import { UpdateBanner } from "./components/UpdateBanner.js";
import "./types.js";

type RightTab = "trajectory" | "memory" | "dormant" | "skills";

export const App = (): JSX.Element => {
  const [messages, setMessages] = React.useState<ChatMessage[]>([]);
  const [trajectory, setTrajectory] = React.useState<TrajectoryEntry[]>([]);
  const [status, setStatus] = React.useState<StatusSnapshot | undefined>();
  const [pending, setPending] = React.useState(false);
  const [rightTab, setRightTab] = React.useState<RightTab>("trajectory");

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

  React.useEffect(() => {
    const refresh = async (): Promise<void> => {
      try {
        const s = await window.openintj?.status();
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

  const dormantEnabled = status?.dormant?.enabled === true;
  const dormantPending = status?.dormant?.pendingProposals ?? 0;
  const skillsEnabled = status?.skills?.enabled === true;
  const skillsPending = status?.skills?.pendingProposals ?? 0;

  return (
    <div className="flex flex-col h-screen text-gray-200">
      <header className="px-4 py-2 border-b border-gray-800 bg-[#181825] flex items-center gap-2">
        <span className="text-base font-semibold">OpenINTJ</span>
        <span className="text-xs text-gray-500">v3.0 Local Desktop</span>
      </header>
      <UpdateBanner />
      <div className="flex-1 grid grid-cols-[1fr_380px] min-h-0">
        <ChatPanel messages={messages} onSend={(t) => void handleSend(t)} pending={pending} />
        <div className="flex flex-col h-full bg-[#11111b] border-l border-gray-800">
          <div className="flex items-center border-b border-gray-800 text-xs">
            <button
              type="button"
              onClick={() => setRightTab("trajectory")}
              className={
                rightTab === "trajectory"
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              推理轨迹
              {trajectory.length > 0 ? (
                <span className="ml-1.5 text-[10px] text-gray-500">{trajectory.length}</span>
              ) : null}
            </button>
            <button
              type="button"
              onClick={() => setRightTab("memory")}
              className={
                rightTab === "memory"
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              记忆
            </button>
            <button
              type="button"
              onClick={() => setRightTab("dormant")}
              className={
                rightTab === "dormant"
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              Dormant
              {dormantEnabled && dormantPending > 0 ? (
                <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] bg-yellow-700 text-yellow-100">
                  {dormantPending}
                </span>
              ) : null}
            </button>
            <button
              type="button"
              onClick={() => setRightTab("skills")}
              className={
                rightTab === "skills"
                  ? "px-3 py-2 text-gray-100 border-b-2 border-purple-500"
                  : "px-3 py-2 text-gray-500 hover:text-gray-300"
              }
            >
              技能
              {skillsEnabled && skillsPending > 0 ? (
                <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] bg-yellow-700 text-yellow-100">
                  {skillsPending}
                </span>
              ) : null}
            </button>
          </div>
          <div className="flex-1 min-h-0">
            {rightTab === "trajectory" ? (
              <TrajectoryPanel entries={trajectory} />
            ) : rightTab === "memory" ? (
              <MemoryPanel />
            ) : rightTab === "dormant" ? (
              <DormantPanel enabled={dormantEnabled} />
            ) : (
              <SkillPanel enabled={skillsEnabled} />
            )}
          </div>
        </div>
      </div>
      <StatusBar status={status} />
    </div>
  );
};
