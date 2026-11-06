import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { dirname } from "node:path";
import type { Connector, ConnectorTool } from "./types.js";

export interface MailConnectorOpts {
  id?: string;
  /** Path to an mbox file. Exactly one of mboxPath / mboxText. */
  mboxPath?: string;
  /** Inline mbox text (tests / pre-fetched content). */
  mboxText?: string;
  maxReadChars?: number;
  /**
   * RFC-017: when set, expose `mail_compose_draft` appending to this separate
   * drafts mbox. Drafts are never sent; the read mbox is never written.
   */
  draftMboxPath?: string;
  /** Default From header for drafts (default openintj@localhost). */
  draftFrom?: string;
}

export interface MailMessage {
  id: string;
  from: string;
  to?: string;
  subject: string;
  date?: number;
  body: string;
}

/** Decode RFC 2047 encoded-words (=?charset?B/Q?...?=) for headers. */
export const decodeEncodedWord = (value: string): string =>
  value.replace(/=\?([^?]+)\?([BbQq])\?([^?]*)\?=/g, (_m, _charset, enc, text) => {
    try {
      if ((enc as string).toUpperCase() === "B") {
        return Buffer.from(text as string, "base64").toString("utf8");
      }
      const qp = (text as string)
        .replace(/_/g, " ")
        .replace(/=([0-9A-Fa-f]{2})/g, (_h, hex) =>
          String.fromCharCode(Number.parseInt(hex as string, 16)),
        );
      return Buffer.from(qp, "latin1").toString("utf8");
    } catch {
      return text as string;
    }
  });

const parseHeaders = (block: string): Map<string, string> => {
  const headers = new Map<string, string>();
  let lastKey: string | undefined;
  for (const line of block.split(/\r?\n/)) {
    if ((line.startsWith(" ") || line.startsWith("\t")) && lastKey) {
      headers.set(lastKey, `${headers.get(lastKey)} ${line.trim()}`);
      continue;
    }
    const colon = line.indexOf(":");
    if (colon < 0) continue;
    lastKey = line.slice(0, colon).trim().toLowerCase();
    headers.set(lastKey, line.slice(colon + 1).trim());
  }
  return headers;
};

/** Extract a readable text body: first text/plain MIME part, else the raw body. */
const extractBody = (headers: Map<string, string>, body: string): string => {
  const contentType = headers.get("content-type") ?? "";
  const boundaryMatch = /boundary="?([^";]+)"?/.exec(contentType);
  if (contentType.includes("multipart/") && boundaryMatch) {
    const parts = body.split(`--${boundaryMatch[1]}`);
    for (const part of parts) {
      const splitAt = part.search(/\r?\n\r?\n/);
      if (splitAt < 0) continue;
      const partHeaders = parseHeaders(part.slice(0, splitAt));
      if ((partHeaders.get("content-type") ?? "text/plain").includes("text/plain")) {
        return part.slice(splitAt).trim();
      }
    }
    return body.trim();
  }
  return body.trim();
};

export const parseMbox = (text: string): MailMessage[] => {
  const chunks = text.split(/^From .+$/m).filter((chunk) => chunk.trim().length > 0);
  return chunks.map((chunk, index) => {
    const splitAt = chunk.search(/\r?\n\r?\n/);
    const headerBlock = splitAt >= 0 ? chunk.slice(0, splitAt) : "";
    const rawBody = splitAt >= 0 ? chunk.slice(splitAt).replace(/^\r?\n\r?\n/, "") : chunk;
    const headers = parseHeaders(headerBlock);
    const dateRaw = headers.get("date");
    const dateMs = dateRaw ? Date.parse(dateRaw) : Number.NaN;
    const message: MailMessage = {
      id: String(index),
      from: decodeEncodedWord(headers.get("from") ?? ""),
      subject: decodeEncodedWord(headers.get("subject") ?? "(无主题)"),
      body: extractBody(headers, rawBody),
    };
    const to = headers.get("to");
    if (to) message.to = decodeEncodedWord(to);
    if (Number.isFinite(dateMs)) message.date = dateMs;
    return message;
  });
};

const baseDescriptor = {
  permissions: ["filesystem.read"],
  idempotent: true,
  errorSemantics: "fail_fast" as const,
  timeoutS: 30,
};

const writeDescriptor = {
  permissions: ["filesystem.write"],
  idempotent: false,
  errorSemantics: "fail_fast" as const,
  timeoutS: 30,
};

const requireHeader = (value: unknown, field: string): string => {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`missing or invalid param: ${field}`);
  }
  if (/[\r\n]/.test(value)) throw new Error(`invalid header value: ${field}`);
  return value.trim();
};

/** RFC 2047 B-encode a header value when it carries non-ASCII characters. */
const encodeHeader = (value: string): string =>
  // biome-ignore lint/suspicious/noControlCharactersInRegex: ASCII range check is intentional
  /^[\x00-\x7F]*$/.test(value)
    ? value
    : `=?UTF-8?B?${Buffer.from(value, "utf8").toString("base64")}?=`;

/** Read-only connector over a local mbox archive. */
export const createMailConnector = (opts: MailConnectorOpts): Connector => {
  const id = opts.id ?? "mail";
  const maxReadChars = opts.maxReadChars ?? 20_000;
  let cache: { stamp: string; messages: MailMessage[] } | undefined;

  const load = (): MailMessage[] => {
    if (opts.mboxText !== undefined) {
      if (!cache) cache = { stamp: "inline", messages: parseMbox(opts.mboxText) };
      return cache.messages;
    }
    if (!opts.mboxPath) throw new Error("mail connector requires mboxPath or mboxText");
    const stat = statSync(opts.mboxPath);
    const stamp = `${stat.mtimeMs}:${stat.size}`;
    if (cache?.stamp !== stamp) {
      cache = { stamp, messages: parseMbox(readFileSync(opts.mboxPath, "utf8")) };
    }
    return cache.messages;
  };

  const summary = (m: MailMessage): Omit<MailMessage, "body"> => {
    const { body: _body, ...rest } = m;
    return rest;
  };

  const tools: ConnectorTool[] = [
    {
      descriptor: {
        name: "mail_list",
        description: "列出最近的邮件（发件人/主题/日期，按时间倒序，不含正文）。",
        inputSchema: { limit: "number?" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const limit =
          typeof params["limit"] === "number" ? Math.min(Math.max(params["limit"], 1), 100) : 20;
        const messages = [...load()]
          .sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
          .slice(0, limit)
          .map(summary);
        return { messages };
      },
    },
    {
      descriptor: {
        name: "mail_read",
        description: "按 mail_list 返回的 id 读取一封邮件的正文。",
        inputSchema: { id: "string" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const wanted = typeof params["id"] === "string" ? params["id"] : "";
        const message = load().find((m) => m.id === wanted);
        if (!message) throw new Error(`mail not found: ${wanted}`);
        const body =
          message.body.length > maxReadChars
            ? `${message.body.slice(0, maxReadChars)}\n…[truncated]`
            : message.body;
        return { ...summary(message), body };
      },
    },
    {
      descriptor: {
        name: "mail_search",
        description: "按关键词搜索邮件主题/发件人/正文（不区分大小写子串）。",
        inputSchema: { query: "string", limit: "number?" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const query = typeof params["query"] === "string" ? params["query"].toLowerCase() : "";
        if (!query) throw new Error("missing or invalid param: query");
        const limit =
          typeof params["limit"] === "number" ? Math.min(Math.max(params["limit"], 1), 100) : 20;
        const messages = load()
          .filter((m) =>
            [m.subject, m.from, m.body].some((field) => field.toLowerCase().includes(query)),
          )
          .sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
          .slice(0, limit)
          .map(summary);
        return { query: params["query"], messages };
      },
    },
  ];

  // RFC-017: compose drafts into a separate drafts mbox (never sends).
  if (opts.draftMboxPath) {
    const draftMboxPath = opts.draftMboxPath;
    tools.push({
      descriptor: {
        name: "mail_compose_draft",
        description: "把一封邮件写入草稿箱（本地 drafts mbox，只保存不发送）。写操作，需审批。",
        inputSchema: { to: "string", subject: "string", body: "string", from: "string?" },
        outputSchema: {},
        ...writeDescriptor,
      },
      handler: (params) => {
        const to = requireHeader(params["to"], "to");
        const subject = requireHeader(params["subject"], "subject");
        const from =
          params["from"] === undefined
            ? (opts.draftFrom ?? "openintj@localhost")
            : requireHeader(params["from"], "from");
        if (typeof params["body"] !== "string" || !params["body"]) {
          throw new Error("missing or invalid param: body");
        }
        const now = new Date();
        const body = params["body"].replace(/^From /gm, ">From ");
        const message = [
          `From ${from} ${now.toUTCString()}`,
          `From: ${encodeHeader(from)}`,
          `To: ${encodeHeader(to)}`,
          `Subject: ${encodeHeader(subject)}`,
          `Date: ${now.toUTCString()}`,
          "MIME-Version: 1.0",
          "Content-Type: text/plain; charset=utf-8",
          "",
          body,
          "",
        ].join("\n");
        mkdirSync(dirname(draftMboxPath), { recursive: true });
        const needsSeparator = existsSync(draftMboxPath) && statSync(draftMboxPath).size > 0;
        appendFileSync(draftMboxPath, `${needsSeparator ? "\n" : ""}${message}`, "utf8");
        return { draftMboxPath, to, subject };
      },
    });
  }

  return {
    id,
    displayName: "邮件（mbox 归档）",
    tools: () => tools,
    status: () => {
      try {
        const messages = load();
        return {
          id,
          displayName: "邮件（mbox 归档）",
          ok: true,
          detail: `${messages.length} messages`,
          toolCount: tools.length,
        };
      } catch (e) {
        return {
          id,
          displayName: "邮件（mbox 归档）",
          ok: false,
          detail: (e as Error).message,
          toolCount: tools.length,
        };
      }
    },
  };
};
