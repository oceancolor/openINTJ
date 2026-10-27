# src/vendor/openintj —— 从主干搬过来的 @openintj/* 包

## 从哪来、怎么来的

主干（`origin/main`，提交 `d225b31`）是一个 pnpm workspace monorepo，`ts/packages/` 下有 12 个包。
本分支是单包 Electron 应用（没有 workspace、没有 pnpm），所以这些包**无法按原样安装**，只能搬源码。

搬进来的三个包：

| 目录 | 上游位置 | 为什么需要 |
|---|---|---|
| `core/` | `ts/packages/core/src` | `AgentError` / `ErrorCode`（生图依赖）+ `input-structuring.ts`（输入结构化） |
| `shared/` | `ts/packages/shared/src` | core 的唯一内部依赖 |
| `llm-hunyuan/` | `ts/packages/llm/hunyuan/src` | `generateImage`：混元 Hy-Image-3.0 的 HTTP 实现 |

**没有搬**其余 9 个包（`planes/*`、`storage/*`、`embed/*`、`dormant`、`skills`、`taskpool`、`telemetry` 等）：它们服务于向量检索、技能系统、OpenTelemetry，与本项目当前的四项能力无关，且会引入 `apache-arrow`、`@opentelemetry/api` 两个重依赖。等真需要时再搬。

## 三条约束

1. **不要改这些文件里的 import**。跨包引用靠 `tsconfig.node.json` 的 `paths` 与 `electron.vite.config.ts` 的 `alias` 解析。改了源码里的 `@openintj/xxx`，下次同步上游就变成逐文件 diff。
2. **两边都要配**。tsconfig 的 paths 只让 tsc 认得；真正打包是 vite 在解析。只配一边的结果是：类型检查全绿，一 build 就报 "Failed to resolve import"。
3. **写入不许绕过裁决链**。上游 `artifact-fallback.ts` 那套「模型没写文件就自己 `fs.writeFile` 兜底」的做法在本项目里**不能照搬** —— 工作区内写入一律走 tool-host 的裁决与审计。搬过来的代码里凡是涉及写文件的，都要改走既有通道。

## 同步上游

上游再有更新时，按同样方式覆盖导出即可（保留本文件与 tsconfig/vite 的映射）：

```powershell
git ls-tree -r --name-only <commit> -- ts/packages/core/src | ForEach-Object { ... }
```

覆盖后跑 `npm run typecheck:node`，并对 `core/index.ts` 里被本项目用到的导出做一次确认。
