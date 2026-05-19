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
      const result = await agent.run(query);
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
        const result = await agent.run(query);
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
      statusParam === "applied"
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

  app.get("/api/dormant/persona", async (c) => {
    const guard = requireDormant(c);
    if (guard) return guard;
    return c.json(agent.dormant!.snapshot());
  });

  return app;
};
