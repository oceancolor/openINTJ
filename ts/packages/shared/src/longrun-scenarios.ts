/**
 * 长跑评测场景脚本（fixtures）——有先后依赖的 query 序列。
 *
 * 设计原则：前几轮「告知信息」，后几轮「应当回忆」。memory-on 的 Agent 应在后半轮
 * 命中 expectRecall 金片段，memory-off 则命中率低 → 体现「越用越好」。
 */

import type { LongRunScript } from "./longrun-eval.js";
import { judgeNonEmpty } from "./task-eval.js";

/** 场景一：用户偏好——先告诉偏好，后续问答应记住。 */
export const SCENARIO_PREFERENCES: LongRunScript = {
  id: "user-preferences",
  description: "用户先陈述偏好与事实，后续轮次应回忆起来",
  turns: [
    { query: "记住：我最喜欢的编程语言是 Rust，我的代号是 Falcon。", judge: judgeNonEmpty },
    { query: "另外，我所在的城市是杭州，习惯用公制单位。", judge: judgeNonEmpty },
    { query: "顺便一提，我的项目叫 openINTJ，主仓库语言是 TypeScript。", judge: judgeNonEmpty },
    {
      query: "我最喜欢的编程语言是什么？",
      expectRecall: "Rust",
      judge: judgeNonEmpty,
    },
    {
      query: "我的代号叫什么？",
      expectRecall: "Falcon",
      judge: judgeNonEmpty,
    },
    {
      query: "我在哪个城市？",
      expectRecall: "杭州",
      judge: judgeNonEmpty,
    },
    {
      query: "我的项目名字是？",
      expectRecall: "openINTJ",
      judge: judgeNonEmpty,
    },
  ],
};

/** 场景二：技术决策——逐步给出约束，最后要求据此给方案，应回忆约束。 */
export const SCENARIO_DECISIONS: LongRunScript = {
  id: "tech-decisions",
  description: "逐步累积技术约束，末轮综合应回忆早先约束",
  turns: [
    { query: "约束 A：数据库必须用 SQLite，不能引入外部服务。", judge: judgeNonEmpty },
    { query: "约束 B：向量检索用 LanceDB 本地嵌入。", judge: judgeNonEmpty },
    { query: "约束 C：所有跨进程通信走 IPC，禁止开放网络端口。", judge: judgeNonEmpty },
    {
      query: "数据库选型的约束是什么？",
      expectRecall: "SQLite",
      judge: judgeNonEmpty,
    },
    {
      query: "向量检索用什么？",
      expectRecall: "LanceDB",
      judge: judgeNonEmpty,
    },
    {
      query: "进程间通信的约束是什么？",
      expectRecall: "IPC",
      judge: judgeNonEmpty,
    },
  ],
};

export const LONGRUN_SCENARIOS: readonly LongRunScript[] = [
  SCENARIO_PREFERENCES,
  SCENARIO_DECISIONS,
];
