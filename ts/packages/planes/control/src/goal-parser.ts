import { TaskType, type TaskTypeType } from "@openintj/core";
import { type Intent, type ParsedGoal, ParsedGoalSchema } from "./types.js";

const DEFAULT_INTENT_KEYWORDS: ReadonlyArray<readonly [string, Intent]> = [
  ["创建", "create"],
  ["生成", "create"],
  ["编写", "create"],
  ["写", "create"],
  ["修改", "modify"],
  ["更新", "modify"],
  ["修复", "modify"],
  ["改", "modify"],
  ["删除", "delete"],
  ["移除", "delete"],
  ["查询", "query"],
  ["搜索", "query"],
  ["查找", "query"],
  ["分析", "query"],
  ["执行", "execute"],
  ["运行", "execute"],
  ["部署", "execute"],
  ["规划", "plan"],
  ["设计", "plan"],
  ["架构", "plan"],
];

const URGENT_WORDS = new Set(["紧急", "立即", "马上", "urgent", "asap", "critical"]);

const QUOTE_CHARS = new Set(['"', "'", "\u201C", "\u201D", "\u2018", "\u2019"]);

export interface GoalParserConfig {
  intentKeywords?: ReadonlyArray<readonly [string, Intent]>;
  urgentWords?: ReadonlySet<string>;
}

export class GoalParser {
  private readonly intentKeywords: ReadonlyArray<readonly [string, Intent]>;
  private readonly urgentWords: ReadonlySet<string>;

  constructor(cfg: GoalParserConfig = {}) {
    this.intentKeywords = cfg.intentKeywords ?? DEFAULT_INTENT_KEYWORDS;
    this.urgentWords = cfg.urgentWords ?? URGENT_WORDS;
  }

  parse(rawInput: string, taskType: TaskTypeType = TaskType.GENERAL_CHAT): ParsedGoal {
    return ParsedGoalSchema.parse({
      rawInput,
      taskType,
      intent: this.extractIntent(rawInput),
      entities: this.extractEntities(rawInput),
      priority: this.estimatePriority(rawInput, taskType),
    });
  }

  private extractIntent(text: string): Intent {
    for (const [keyword, intent] of this.intentKeywords) {
      if (text.includes(keyword)) return intent;
    }
    return "general";
  }

  /** 提取引号内的实体（中英文引号皆可）。 */
  private extractEntities(text: string): string[] {
    const entities: string[] = [];
    let inQuote = false;
    let current = "";
    for (const char of text) {
      if (QUOTE_CHARS.has(char)) {
        if (inQuote) {
          const trimmed = current.trim();
          if (trimmed.length > 0) entities.push(trimmed);
          current = "";
        }
        inQuote = !inQuote;
      } else if (inQuote) {
        current += char;
      }
    }
    return entities;
  }

  private estimatePriority(text: string, taskType: TaskTypeType): number {
    let priority = 5;
    const lower = text.toLowerCase();
    for (const word of this.urgentWords) {
      if (lower.includes(word)) {
        priority = 9;
        break;
      }
    }
    if (taskType === TaskType.CODE_GENERATION) priority = Math.max(priority, 7);
    if (taskType === TaskType.QUICK_RESPONSE) priority = Math.max(priority, 8);
    return priority;
  }
}
