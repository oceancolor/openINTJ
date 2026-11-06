import { PROFILE_SECTIONS } from "@openintj/profile";
import type { ScheduledJobInput } from "@openintj/scheduler";
import { withRootSpan } from "@openintj/telemetry-otel";
import { type Context, Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";
import type { ServerAgent } from "./agent.js";
import { retrieveHybrid } from "./hybrid-retrieve.js";

const ChatBodySchema = z.object({
  query: z.string().min(1),
  stream: z.boolean().default(false),
});

const MemoryQuerySchema = z.object({
  q: z.string().optional(),
  topK: z.coerce.number().int().positive().max(50).default(10),
  /** 检索模式覆盖。不传则按 agent.retrievalMode 默认值。 */
  mode: z.enum(["vector", "hybrid"]).optional(),
  /** hybrid 模式启用 RRF 融合（默认按 alpha/beta 加权）。 */
  rrf: z.coerce.boolean().optional(),
});

const AuditQuerySchema = z.object({
  limit: z.coerce.number().int().positive().max(500).default(100),
});

const ProfileUpsertSchema = z.object({
  section: z.enum(PROFILE_SECTIONS),
  key: z.string().min(1),
  value: z.string(),
  validUntil: z.number().int().positive().optional(),
});

const ScheduleTriggerSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("once"), at: z.number() }),
  z.object({
    kind: z.literal("interval"),
    everyMs: z.number().int().min(1_000),
    startAt: z.number().optional(),
  }),
  z.object({ kind: z.literal("daily"), time: z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/) }),
]);

const ScheduleAddSchema = z.object({
  title: z.string().min(1),
  prompt: z.string().min(1),
  conversationId: z.string().optional(),
  trigger: ScheduleTriggerSchema,
  enabled: z.boolean().optional(),
  maxRuns: z.number().int().positive().optional(),
});

const SchedulePatchSchema = z.object({
  title: z.string().min(1).optional(),
  prompt: z.string().min(1).optional(),
  conversationId: z.string().optional(),
  trigger: ScheduleTriggerSchema.optional(),
  enabled: z.boolean().optional(),
  maxRuns: z.number().int().positive().optional(),
});

const IdentityResolveSchema = z.object({
  kind: z.enum(["email", "phone", "wechat", "username", "url"]).optional(),
  value: z.string().min(1),
});

const IdentityProposalSchema = z.object({
  id: z.string(),
  personId: z.string().nullable(),
  groupKey: z.string().optional(),
  displayName: z.string().optional(),
  account: z.object({
    platform: z.string().min(1),
    kind: z.enum(["email", "phone", "wechat", "username", "url"]),
    value: z.string().min(1),
    url: z.string().optional(),
    source: z.enum(["gravatar", "mail", "calendar", "notes", "manual"]),
    evidence: z.string(),
    confidence: z.number().min(0).max(1),
    verifiedAt: z.number().nullable(),
  }),
});

const IdentityConfirmSchema = z.object({
  proposals: z.array(IdentityProposalSchema).min(1),
});

const IdentityAssertSchema = z.object({
  personId: z.string().nullable().optional(),
  displayName: z.string().optional(),
  platform: z.string().min(1),
  kind: z.enum(["email", "phone", "wechat", "username", "url"]),
  value: z.string().min(1),
  url: z.string().optional(),
  evidence: z.string().optional(),
});

export const buildApp = (agent: ServerAgent): Hono => {
  const app = new Hono();

  app.get("/healthz", (c) => c.json({ ok: true }));

  app.get("/api/status", async (c) => {
    const s = await agent.status();
    return c.json(s);
  });

  app.post("/api/chat", async (c) => {
    const body = await c.req.json().catch(() => ({}));
    const parsed = ChatBodySchema.safeParse(body);
    if (!parsed.success) {
      return c.json({ error: "invalid_body", issues: parsed.error.issues }, 400 as const);
    }
    const { query, stream } = parsed.data;

    if (!stream) {
      const result = await withRootSpan("openintj.http.chat", () => agent.run(query), {
        attributes: { "http.route": "/api/chat", "http.stream": false },
      });
      return c.json({
        finalAnswer: result.finalAnswer,
        iterations: result.iterations,
        status: result.status,
      });
    }

    return streamSSE(c, async (s) => {
      // 订阅 hooks 把 react/tao 事件实时推给前端
      const send = (event: string, data: unknown): Promise<void> =>
        s.writeSSE({ event, data: JSON.stringify(data) });
      const offs: Array<() => void> = [];
      offs.push(
        agent.hooks.on("tao.beforeThink", async (ctx) => {
          await send("tao.beforeThink", ctx.payload);
        }),
      );
      offs.push(
        agent.hooks.on("tao.afterAct", async (ctx) => {
          await send("tao.afterAct", ctx.payload);
        }),
      );
      offs.push(
        agent.hooks.on("react.afterThought", async (ctx) => {
          await send("react.thought", ctx.payload);
        }),
      );
      offs.push(
        agent.hooks.on("react.beforeAction", async (ctx) => {
          await send("react.action", ctx.payload);
        }),
      );
      offs.push(
        agent.hooks.on("react.afterAction", async (ctx) => {
          await send("react.observation", ctx.payload);
        }),
      );
      try {
        const result = await withRootSpan("openintj.http.chat", () => agent.run(query), {
          attributes: { "http.route": "/api/chat", "http.stream": true },
        });
        await send("done", {
          finalAnswer: result.finalAnswer,
          iterations: result.iterations,
          status: result.status,
        });
      } catch (e) {
        await send("error", { message: (e as Error).message });
      } finally {
        for (const o of offs) o();
      }
    });
  });

  // RFC-013: interactive approval + capability grants.
  app.get("/api/approvals", (c) => c.json({ pending: agent.approvals?.listPending() ?? [] }));
  app.post("/api/approvals/:id/decide", async (c) => {
    if (!agent.approvals) return c.json({ error: "approval_not_enabled" }, 400);
    const body = (await c.req.json().catch(() => ({}))) as {
      decision?: string;
      reason?: string;
      grant?: { scope?: string; ttlMs?: number };
    };
    if (body.decision !== "approved" && body.decision !== "denied") {
      return c.json({ error: "invalid_body" }, 400);
    }
    const ok = agent.approvals.decide(
      c.req.param("id"),
      body.decision === "approved"
        ? { decision: "approved", ...(body.grant ? { grant: body.grant } : {}) }
        : { decision: "denied", ...(body.reason ? { reason: body.reason } : {}) },
    );
    return ok ? c.json({ ok: true }) : c.json({ error: "not_pending" }, 404);
  });
  app.get("/api/grants", (c) => c.json({ grants: agent.grants.list() }));
  app.post("/api/grants", async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as {
      tool?: string;
      risk?: "read" | "write" | "execute" | "external";
      scope?: string;
      ttlMs?: number;
      label?: string;
    };
    if (!body.tool || !body.risk) return c.json({ error: "invalid_body" }, 400);
    const grant = agent.grants.grant({
      tool: body.tool,
      risk: body.risk,
      ...(body.scope ? { scope: body.scope } : {}),
      ...(body.ttlMs ? { expiresAt: Date.now() + body.ttlMs } : {}),
      ...(body.label ? { label: body.label } : {}),
    });
    return c.json(grant);
  });
  app.delete("/api/grants/:id", (c) =>
    agent.grants.revoke(c.req.param("id"))
      ? c.json({ ok: true })
      : c.json({ error: "not_found" }, 404),
  );

  // RFC-014: personal profile (memory control plane).
  app.get("/api/profile", (c) => {
    const section = c.req.query("section");
    const parsed = z.enum(PROFILE_SECTIONS).safeParse(section);
    const entries =
      section && parsed.success
        ? agent.profile.list({ section: parsed.data })
        : agent.profile.list();
    return c.json({ entries });
  });
  app.get("/api/profile/export", (c) => c.json({ profile: agent.profile.exportJSON() }));
  app.post("/api/profile", async (c) => {
    const parsed = ProfileUpsertSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: "invalid_body", issues: parsed.error.issues }, 400);
    }
    const { validUntil, ...rest } = parsed.data;
    const entry = agent.profile.upsert({
      ...rest,
      ...(validUntil ? { validUntil } : {}),
    });
    return entry ? c.json(entry) : c.json({ error: "conflict" }, 409);
  });
  app.delete("/api/profile/:id", (c) =>
    agent.profile.remove(c.req.param("id"))
      ? c.json({ ok: true })
      : c.json({ error: "not_found" }, 404),
  );

  // RFC-020: identity graph. Resolvers only propose; confirm() is the only
  // way into the graph, and owned values can never be silently moved (409).
  app.get("/api/identity/persons", (c) => c.json({ persons: agent.identity.listPersons() }));
  app.post("/api/identity/resolve", async (c) => {
    const parsed = IdentityResolveSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: "invalid_body", issues: parsed.error.issues }, 400);
    }
    try {
      return c.json(await agent.identity.resolve(parsed.data));
    } catch (err) {
      return c.json({ error: err instanceof Error ? err.message : "resolve_failed" }, 400);
    }
  });
  app.post("/api/identity/confirm", async (c) => {
    const parsed = IdentityConfirmSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: "invalid_body", issues: parsed.error.issues }, 400);
    }
    try {
      return c.json(agent.identity.confirm(parsed.data.proposals));
    } catch (err) {
      const message = err instanceof Error ? err.message : "confirm_failed";
      return c.json({ error: message }, message.startsWith("identity_conflict") ? 409 : 400);
    }
  });
  app.post("/api/identity/assert", async (c) => {
    const parsed = IdentityAssertSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: "invalid_body", issues: parsed.error.issues }, 400);
    }
    try {
      return c.json(agent.identity.assertAccount(parsed.data));
    } catch (err) {
      const message = err instanceof Error ? err.message : "assert_failed";
      return c.json({ error: message }, message.startsWith("identity_conflict") ? 409 : 400);
    }
  });
  app.post("/api/identity/unlink", async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as { accountId?: unknown };
    if (typeof body.accountId !== "string" || !body.accountId) {
      return c.json({ error: "invalid_body" }, 400);
    }
    const result = agent.identity.unlink(body.accountId);
    return result.removed ? c.json(result) : c.json({ error: "not_found" }, 404);
  });

  // RFC-016: connector statuses (tools themselves are agent-internal).
  app.get("/api/connectors", (c) => c.json({ connectors: agent.connectors.statuses() }));

  // RFC-012: durable TaskPool checkpoints. Results are serialized explicitly
  // because TaskRunResult carries Maps.
  app.get("/api/taskpool/checkpoints", async (c) => {
    const limit = Number(c.req.query("limit") ?? 50);
    return c.json({ checkpoints: await agent.taskPoolCheckpoints(limit) });
  });
  app.post("/api/taskpool/runs/:id/resume", async (c) => {
    const result = await agent.resumeTaskRun(c.req.param("id"));
    if (!result) return c.json({ error: "not_found_or_not_resumable" }, 404);
    return c.json({
      result: {
        runId: result.runId,
        planId: result.planId,
        status: result.status,
        results: Object.fromEntries(result.results),
        errors: Object.fromEntries(
          [...result.errors].map(([taskId, error]) => [
            taskId,
            error instanceof Error ? error.message : String(error),
          ]),
        ),
        states: Object.fromEntries(result.states),
        attempts: Object.fromEntries(result.attempts),
      },
    });
  });
  app.post("/api/taskpool/runs/:id/cancel", async (c) =>
    (await agent.cancelTaskRun(c.req.param("id")))
      ? c.json({ ok: true })
      : c.json({ error: "not_found_or_not_cancellable" }, 404),
  );

  // RFC-012 §3: subagent run records (spawn happens via the delegate tool).
  app.get("/api/subagents", (c) => c.json({ subagents: agent.subagents.list() }));

  // RFC-018: outbound notification channels.
  app.get("/api/channels", (c) => c.json({ channels: agent.channels.list() }));
  app.post("/api/channels/:id/test", async (c) => {
    const result = await agent.channels.test(c.req.param("id"));
    return result.ok ? c.json(result) : c.json(result, 404);
  });

  // RFC-019: cross-device sync bundle (pure JSON; transport chosen by the user).
  app.get("/api/sync/export", (c) => c.json(agent.exportSyncBundle()));
  app.post("/api/sync/import", async (c) => {
    const body = await c.req.json().catch(() => null);
    try {
      return c.json({ report: agent.importSyncBundle(body) });
    } catch (e) {
      return c.json({ error: "invalid_bundle", detail: String(e) }, 400);
    }
  });

  // RFC-019: speech-to-text. 503 when no provider is configured via env.
  app.post("/api/voice/transcribe", async (c) => {
    if (!agent.stt) return c.json({ error: "stt_not_configured" }, 503);
    const body = (await c.req.json().catch(() => ({}))) as {
      audioBase64?: unknown;
      mimeType?: unknown;
      language?: unknown;
    };
    if (typeof body.audioBase64 !== "string" || typeof body.mimeType !== "string") {
      return c.json({ error: "invalid_body" }, 400);
    }
    const audio = Buffer.from(body.audioBase64, "base64");
    if (audio.byteLength > 26_000_000) return c.json({ error: "audio_too_large" }, 413);
    try {
      const out = await agent.stt.transcribe({
        audio: new Uint8Array(audio),
        mimeType: body.mimeType,
        ...(typeof body.language === "string" ? { language: body.language } : {}),
      });
      return c.json(out);
    } catch (e) {
      return c.json({ error: "transcribe_failed", detail: String(e) }, 502);
    }
  });

  // RFC-015: scheduler (persistent jobs). Firing state is observable via job fields.
  app.get("/api/schedules", (c) => c.json({ jobs: agent.scheduler.list() }));
  app.post("/api/schedules", async (c) => {
    const parsed = ScheduleAddSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: "invalid_body", issues: parsed.error.issues }, 400);
    }
    try {
      const d = parsed.data;
      return c.json(
        agent.scheduler.add({
          title: d.title,
          prompt: d.prompt,
          trigger: d.trigger as ScheduledJobInput["trigger"],
          ...(d.conversationId ? { conversationId: d.conversationId } : {}),
          ...(d.enabled !== undefined ? { enabled: d.enabled } : {}),
          ...(d.maxRuns ? { maxRuns: d.maxRuns } : {}),
        }),
      );
    } catch (e) {
      return c.json({ error: "invalid_body", message: (e as Error).message }, 400);
    }
  });
  app.patch("/api/schedules/:id", async (c) => {
    const parsed = SchedulePatchSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: "invalid_body", issues: parsed.error.issues }, 400);
    }
    try {
      const d = parsed.data;
      const job = agent.scheduler.update(c.req.param("id"), {
        ...(d.title !== undefined ? { title: d.title } : {}),
        ...(d.prompt !== undefined ? { prompt: d.prompt } : {}),
        ...(d.conversationId !== undefined ? { conversationId: d.conversationId } : {}),
        ...(d.trigger ? { trigger: d.trigger as ScheduledJobInput["trigger"] } : {}),
        ...(d.enabled !== undefined ? { enabled: d.enabled } : {}),
        ...(d.maxRuns !== undefined ? { maxRuns: d.maxRuns } : {}),
      });
      return job ? c.json(job) : c.json({ error: "not_found" }, 404);
    } catch (e) {
      return c.json({ error: "invalid_body", message: (e as Error).message }, 400);
    }
  });
  app.delete("/api/schedules/:id", (c) =>
    agent.scheduler.remove(c.req.param("id"))
      ? c.json({ ok: true })
      : c.json({ error: "not_found" }, 404),
  );

  app.get("/api/memory", async (c) => {
    const query = MemoryQuerySchema.safeParse(c.req.query());
    if (!query.success) {
      return c.json({ error: "invalid_query", issues: query.error.issues }, 400);
    }
    const { q, topK, mode: modeOverride, rrf } = query.data;
    const mode = modeOverride ?? agent.retrievalMode;

    if (typeof q === "string" && q.length > 0) {
      if (mode === "hybrid") {
        const hits = await retrieveHybrid(agent, q, {
          topK,
          ...(rrf !== undefined ? { config: { useRRF: rrf } } : {}),
        });
        return c.json({
          query: q,
          mode: "hybrid" as const,
          results: hits.map((h) => ({
            fragmentId: h.doc.id,
            content: h.doc.text,
            score: h.score,
            components: h.components,
            memoryType: h.doc.metadata.memoryType,
            taskTags: h.doc.metadata.taskTags,
          })),
        });
      }
      const ranked = await agent.memory.retrieve(q, { topK });
      return c.json({
        query: q,
        mode: "vector" as const,
        results: ranked.map((r) => ({
          fragmentId: r.fragment.fragmentId,
          content: r.fragment.content,
          score: r.score,
          components: r.components,
          memoryType: r.fragment.memoryType,
          taskTags: r.fragment.taskTags,
        })),
      });
    }
    // 不传 q：返回最近 topK fragments
    const list = await agent.persistentStore.metadataStore.listFragmentMeta({
      limit: topK,
    });
    return c.json({ recent: list, mode });
  });

  app.get("/api/audit", async (c) => {
    const parsed = AuditQuerySchema.safeParse(c.req.query());
    if (!parsed.success) {
      return c.json({ error: "invalid_query", issues: parsed.error.issues }, 400);
    }
    const { limit } = parsed.data;
    const stats = agent.governance.auditTrail.getStats();
    const recent = agent.governance.auditTrail.query({ limit });
    return c.json({ stats, recent });
  });

  // RFC-003 方向 3：Dormant Memory Learning。
  // 仅 enableDormant=true 时下面这组路由才能正常工作；未启用时统一 503。
  const requireDormant = (c: Context): Response | undefined => {
    if (!agent.dormant) {
      return c.json(
        {
          error: "dormant_not_enabled",
          hint: "Pass { enableDormant: true } to assembleServerAgent or set OPENINTJ_DORMANT=1",
        },
        503,
      );
    }
    return undefined;
  };

  app.post("/api/dormant/mine", async (c) => {
    const guard = requireDormant(c);
    if (guard) return guard;
    const r = await agent.dormant!.mine();
    return c.json({
      scannedEvents: r.scannedEvents,
      patterns: r.patterns.map((p) => ({
        patternId: p.patternId,
        description: p.description,
        category: p.category,
        frequency: p.frequency,
        confidence: p.confidence,
      })),
      proposals: r.proposals.map((p) => ({
        proposalId: p.proposalId,
        targetField: p.targetField,
        value: p.value,
        status: p.status,
        patternDescription: p.pattern.description,
      })),
    });
  });

  app.get("/api/dormant/proposals", async (c) => {
    const guard = requireDormant(c);
    if (guard) return guard;
    const statusParam = c.req.query("status");
    const validStatus =
      statusParam === "pending" ||
      statusParam === "approved" ||
      statusParam === "rejected" ||
      statusParam === "applied" ||
      statusParam === "revoked"
        ? statusParam
        : undefined;
    const list = agent.dormant!.listProposals(validStatus);
    return c.json({
      total: list.length,
      proposals: list.map((p) => ({
        proposalId: p.proposalId,
        targetField: p.targetField,
        value: p.value,
        status: p.status,
        ts: p.ts,
        decidedAt: p.decidedAt,
        patternDescription: p.pattern.description,
        confidence: p.pattern.confidence,
        frequency: p.pattern.frequency,
      })),
    });
  });

  app.post("/api/dormant/proposals/:id/approve", async (c) => {
    const guard = requireDormant(c);
    if (guard) return guard;
    const id = c.req.param("id");
    const out = agent.dormant!.approve(id);
    if (!out) return c.json({ error: "not_found_or_already_decided" }, 404);
    return c.json({
      proposalId: out.proposalId,
      status: out.status,
      decidedAt: out.decidedAt,
    });
  });

  app.post("/api/dormant/proposals/:id/reject", async (c) => {
    const guard = requireDormant(c);
    if (guard) return guard;
    const id = c.req.param("id");
    const out = agent.dormant!.reject(id);
    if (!out) return c.json({ error: "not_found_or_already_decided" }, 404);
    return c.json({
      proposalId: out.proposalId,
      status: out.status,
      decidedAt: out.decidedAt,
    });
  });

  app.post("/api/dormant/proposals/:id/revoke", async (c) => {
    const guard = requireDormant(c);
    if (guard) return guard;
    const id = c.req.param("id");
    // 仅 applied 可撤销：从 PersonaConfig 删字段 + 持久化新快照（RFC-003 §3.6 #4 可回退）。
    const out = agent.dormant!.revoke(id);
    if (!out) return c.json({ error: "not_found_or_not_applied" }, 404);
    return c.json({
      proposalId: out.proposalId,
      status: out.status,
      decidedAt: out.decidedAt,
    });
  });

  app.get("/api/dormant/persona", async (c) => {
    const guard = requireDormant(c);
    if (guard) return guard;
    return c.json(agent.dormant!.getPersona());
  });

  // 技能自学习（Phase 2）：仅 enableSkillLearning=true 时可用；未启用统一 503。
  const requireSkillLearning = (c: Context): Response | undefined => {
    if (!agent.skillLearning) {
      return c.json(
        {
          error: "skills_learning_not_enabled",
          hint: "Pass { enableSkillLearning: true } to assembleServerAgent or set OPENINTJ_SKILLS_LEARN=1",
        },
        503,
      );
    }
    return undefined;
  };

  const proposalView = (p: {
    proposalId: string;
    candidate: { id: string; name: string; description: string };
    evidence: { queries: string[]; taskType?: string; count: number };
    status: string;
    ts: number;
    decidedAt?: number;
  }) => ({
    proposalId: p.proposalId,
    skillId: p.candidate.id,
    name: p.candidate.name,
    description: p.candidate.description,
    status: p.status,
    ts: p.ts,
    decidedAt: p.decidedAt,
    evidence: p.evidence,
  });

  app.post("/api/skills/distill", async (c) => {
    const guard = requireSkillLearning(c);
    if (guard) return guard;
    const produced = await agent.skillLearning!.distill();
    return c.json({ produced: produced.length, proposals: produced.map(proposalView) });
  });

  app.get("/api/skills/proposals", async (c) => {
    const guard = requireSkillLearning(c);
    if (guard) return guard;
    const statusParam = c.req.query("status");
    const validStatus =
      statusParam === "pending" ||
      statusParam === "approved" ||
      statusParam === "rejected" ||
      statusParam === "revoked"
        ? statusParam
        : undefined;
    const list = agent.skillLearning!.listProposals(validStatus);
    return c.json({ total: list.length, proposals: list.map(proposalView) });
  });

  app.post("/api/skills/proposals/:id/approve", async (c) => {
    const guard = requireSkillLearning(c);
    if (guard) return guard;
    const out = await agent.skillLearning!.approve(c.req.param("id"));
    if (!out) return c.json({ error: "not_found_or_already_decided" }, 404);
    return c.json({ proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt });
  });

  app.post("/api/skills/proposals/:id/reject", async (c) => {
    const guard = requireSkillLearning(c);
    if (guard) return guard;
    const out = agent.skillLearning!.reject(c.req.param("id"));
    if (!out) return c.json({ error: "not_found_or_already_decided" }, 404);
    return c.json({ proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt });
  });

  app.post("/api/skills/proposals/:id/revoke", async (c) => {
    const guard = requireSkillLearning(c);
    if (guard) return guard;
    const out = await agent.skillLearning!.revoke(c.req.param("id"));
    if (!out) return c.json({ error: "not_found_or_not_approved" }, 404);
    return c.json({ proposalId: out.proposalId, status: out.status, decidedAt: out.decidedAt });
  });

  app.get("/api/skills", async (c) => {
    const guard = requireSkillLearning(c);
    if (guard) return guard;
    const active = agent.skillLearning!.listApproved().map((s) => ({
      id: s.id,
      name: s.name,
      description: s.description,
      source: s.source,
      weight: agent.skillLearning!.weightFor(s.id),
    }));
    return c.json({ total: active.length, skills: active });
  });

  return app;
};
