/**
 * 冷启动种子 exemplar：每个 TaskType 几条中/英文样例。
 *
 * 仅用于让分类器开箱即用、不至于一开始全走兜底；真正的能力来自后续 reinforce 累积。
 * 不依赖 LLM 打标 → 零 token。
 */

import { TaskType } from "@openintj/core";
import type { SeedExample } from "./reinforcing-classifier.js";

export const DEFAULT_SEEDS: readonly SeedExample[] = [
  // 代码生成
  { text: "写一个快速排序函数", label: TaskType.CODE_GENERATION },
  { text: "实现一个 LRU 缓存类", label: TaskType.CODE_GENERATION },
  { text: "帮我修复这个 bug", label: TaskType.CODE_GENERATION },
  { text: "write a function to parse json", label: TaskType.CODE_GENERATION },
  { text: "refactor this class to use async", label: TaskType.CODE_GENERATION },

  // 技术写作
  { text: "给这个模块写一份 README 文档", label: TaskType.TECHNICAL_WRITING },
  { text: "写一篇关于事件循环的教程", label: TaskType.TECHNICAL_WRITING },
  { text: "draft documentation for the API", label: TaskType.TECHNICAL_WRITING },

  // 分析
  { text: "分析这两种方案的优劣", label: TaskType.ANALYSIS },
  { text: "评估当前架构的瓶颈", label: TaskType.ANALYSIS },
  { text: "compare redis and memcached for caching", label: TaskType.ANALYSIS },

  // 规划
  { text: "制定一个迁移到 TypeScript 的计划", label: TaskType.PLANNING },
  { text: "给出重构数据库层的方案和步骤", label: TaskType.PLANNING },
  { text: "plan the rollout for the new feature", label: TaskType.PLANNING },

  // 快速响应
  { text: "现在几点", label: TaskType.QUICK_RESPONSE },
  { text: "1+1 等于几", label: TaskType.QUICK_RESPONSE },
  { text: "yes or no", label: TaskType.QUICK_RESPONSE },

  // 一般对话
  { text: "你今天过得怎么样", label: TaskType.GENERAL_CHAT },
  { text: "我们聊聊周末的安排吧", label: TaskType.GENERAL_CHAT },
  { text: "tell me something interesting", label: TaskType.GENERAL_CHAT },
];
