/**
 * ClassifierStore —— 分类器状态的持久化抽象（让「持续强化」跨重启）。
 *
 * 默认 InMemoryClassifierStore（CI / 无盘场景）；SQLite 适配见 @openintj/storage-sqlite。
 * 仿 dormant 的持久化分层：接口在本包，重适配（better-sqlite3）在 storage 包。
 */

import type { ClassifierState } from "./reinforcing-classifier.js";

export interface ClassifierStore {
  readonly name: string;
  /** 载入已保存状态；无则返回 undefined。 */
  load(): Promise<ClassifierState | undefined>;
  /** 保存当前状态（覆盖式）。热路径，实现不应抛错。 */
  save(state: ClassifierState): void | Promise<void>;
  /** 清空。 */
  clear(): void | Promise<void>;
}

/** 进程内默认实现：状态存在内存里，重启即丢。 */
export class InMemoryClassifierStore implements ClassifierStore {
  readonly name = "memory-classifier";
  private state: ClassifierState | undefined;

  async load(): Promise<ClassifierState | undefined> {
    return this.state ? { exemplars: this.state.exemplars.map((e) => ({ ...e })) } : undefined;
  }

  save(state: ClassifierState): void {
    this.state = { exemplars: state.exemplars.map((e) => ({ ...e })) };
  }

  clear(): void {
    this.state = undefined;
  }
}
