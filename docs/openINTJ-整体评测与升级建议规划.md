# openINTJ 整体评测与升级建议规划

> **整合范围**：本项目自 2026-11-13 至 2026-11-20 全部输出报告（16 份能力审计 / 战略评估报告 + 2 份测试规范 + 密钥接入说明 + 工作日志）。
> **被测对象**：`github.com/oceancolor/openINTJ`（main，v0.4.0 / TS monorepo v3），本地克隆于 `openINTJ/`。
> **编制人**：量化精英阿尔法 α　|　**编制日期**：2026-11-20（GMT+8）
> **性质**：工程评估与改造规划，仅为分析与模拟，非投资建议。
> **重要口径**：本报告所有分数都是「openINTJ 框架能否承载该能力」的工程可达性评分，不是端到端任务质量分；与任何第三方产品测评分数**量纲不可比**。

---

## 目录

1. [执行摘要](#一执行摘要)
2. [评测方法与证据口径](#二评测方法与证据口径)
3. [十二轮评测全景](#三十轮评测全景)
4. [测试规模与工程基线](#四测试规模与工程基线)
5. [跨轮核心发现：六大结构性根因](#五跨轮核心发现六大结构性根因)
6. [第七轮真实 LLM 在线实测](#六第七轮真实-llm-在线实测)
7. [架构决策：Rust 原生化结论](#七架构决策rust-原生化结论)
8. [战略定位：对标豆包工作与常在线多身份意图](#八战略定位对标豆包工作与常在线多身份意图)
9. [升级建议规划：P0 / P1 / P2 路线图](#九升级建议规划p0--p1--p2-路线图)
10. [明确不建议做的事](#十明确不建议做的事)
11. [验收指标与复现方式](#十一验收指标与复现方式)
12. [附录：原始报告清单与边界声明](#十二附录原始报告清单与边界声明)

---

## 一、执行摘要

### 1.1 一句话总判断

**openINTJ 是一个工程质量真实、治理层扎实、但产品交付层几乎空缺的 Agent 框架。它不是"假项目"——87 个审计 spec / 5267 个用例里 98.63% 通过，构建 31/31、类型检查 59/59 全绿；但它也不是"能干活的 Agent"——在十二轮能力评测中，只有生图（2.5/5）勉强及格，其余全线低于 2 分。接上真实混元模型后，协议层全绿，语义层却出现"零工具调用却声称完成"和"假 .docx + 编造合理化说明"，真实大脑放大了框架缺陷而不是修复它。**

### 1.2 核心结论一览

| 维度 | 判定 | 关键数字 |
|---|---|---|
| 工程骨架 | ✅ 真实 | 31/31 build、59/59 typecheck、50,208 LOC 源码、32 包 |
| 契约测试通过率 | ✅ 高 | 87 spec / 5,267 例 / 5,195 通过 / 72 失败（98.63%） |
| 建站（带后端） | ❌ 1.4/5 | 能写源码文本；命令默认禁、无 stdout 回灌、配额 20 次/分钟 |
| 办公文档原生格式 | ❌ 0.7/5 | 假 `.docx/.pptx/.xlsx`，首字节 `# 20` 而非 `PK\x03\x04` |
| 腾讯文档建物 | ❌ 0.5/5 | 零集成、零鉴权、零在线产物路径 |
| 生图 | ⚠️ 2.5/5 | 唯一真调模型 + 真写二进制的链路，但仅 desktop、绕治理 |
| 生视频 / 3D 生成 | ❌ 0 / 0 | 全仓零实现；源码中"3D"字样全是记忆 shader 比喻 |
| three.js 游戏 | ❌ 1.0/5 | 能写源码，无运行时、无预览容器、无法验证 |
| Blender/Godot MCP | ❌ 1.5/5 | stdio 形态兼容，但 image/data 回灌丢失、CLI 断链、默认审批 |
| 真实 LLM 在线 | ⚠️ 协议 ✅ / 语义 ❌ | 23/23 响应可解析、12/12 收敛；2 类致命假完成 |
| Rust 全量重写 | ❌ 不建议 | 框架开销 0.072ms/轮，瓶颈是 LLM 秒级网络延迟 |
| 常在线多身份意图 | ✅ 有且成体系 | 6 个 RFC + 6 个已落地包；缺入站通道/云端常驻/鉴权化 |

### 1.3 最值得记住的三句话

1. **通过率与能力可达无关。** 建站轮 67 例中 63 例通过、办公轮 114 例中 106 例通过、生图轮 38 例中 32 例通过——真正决定结论的始终是那几条保留失败的 `FINDING` 断言。
2. **openINTJ 的边界不在"会不会写某种文本"，而在"有没有运行时、执行验证、治理一致性、三端一致性、数据回灌"。** 凡是只要求产出文本，它都能做；凡是要求"跑起来给人看"或"证明真的干了"，全部断开。
3. **主动性越强，假完成危害越大。** 第七轮已实证：模型越流畅，假完成越像真的。因此执行一致性闸门必须先于任何"常在线 / 多 Agent / 主动触达"特性上线。

---

## 二、评测方法与证据口径

### 2.1 审计纪律（十二轮一致）

- 只新增 `audit-*.spec.ts`（在线轮为 `online-*.spec.ts`），**不改 `src/**`、不改既有 spec、不改配置**。`git status` 非 audit 改动数每轮均为 0。
- 断言取自 `README.md`、`ts/README.md`、`docs/rfcs/RFC-001~021`、`docs/architecture/*`、`CHANGELOG.md` 或源码 JSDoc 的具体条款，**不是照着实现反推期望值**。
- 禁止 `typeof` / `expect(true)` 类空断言；不写绝对耗时断言；离线轮不联网、不调用真实 LLM。
- 一旦发现"文档声称 ↔ 实际行为"偏差：**保留失败用例并标注 `// FINDING: 声称 → 实际 → 文件:行`**，绝不删测试或改断言凑绿。
- 用例笔误与真实缺陷严格区分。多个专项轮曾出现笔误（中文直角引号、正则误判、导入位置写错、断言方向写反等），均修正重跑；其中 three.js 轮两条初判笔误经核实后**转为真实缺陷 F11-4/F11-5**，未改断言凑绿。

### 2.2 四个必须守住的口径

| 口径 | 说明 |
|---|---|
| **契约通过率 ≠ 任务质量** | 98.63% 是对框架内部契约的黑盒测试通过率；豆包 SuperCLUE 85.33 是端到端办公任务质量分，两者不可比。 |
| **未声称 ≠ 虚假宣传** | 建站、办公、腾讯文档、3D、three.js 等项目**自己从未声称**能做。这些轮次评的是"用户想让它干的事"，是能力外推，不构成打假。 |
| **框架明确不做 ≠ 缺陷** | RFC-016 不做 OAuth/HTTP MCP、RFC-021 默认只做影子记忆、RFC-007 不做 LLM 动态拆图，都是明示的保守决策。判的是"开箱可用性"，不是"代码错误"。 |
| **在线轮单列** | 第七轮真实 LLM 数字（34 次调用 / 约 47,800 tokens）单列，不并入前六轮离线审计数字。 |

### 2.3 判定基准举例

- 办公文档：依 ECMA-376，`.docx/.pptx/.xlsx` 必须是 OOXML ZIP 包（魔数 `50 4B 03 04`），且框架能写/读二进制、声明 MIME。只产出同名 `.md` 不算具备。
- 生图：真调模型 + 拿到真实字节 + 按正确二进制写出 + 产物可识别。
- 3D：模型适配层有 3D 方法 / ToolHub 有 3D 工具 / 产物能写成可解析 `.glb` 等，满足至少一条。
- 游戏：源码 + 运行时依赖 + 预览容器 + 命令验证四环，缺一不算。
- 常在线多身份：RFC 原文 + 源码包 + 三端装配三重证据，缺一不算端倪。

---

## 三、十二轮评测全景

### 3.1 能力评分总表

| 轮次 | 主题 | 评分 | 一句话结论 |
|---|---|---:|---|
| 一 | 广度能力审计 | 部分达标 | 工程骨架真实，45 处声称偏差；RFC-002 钩子改写能力四处 `emit` 返回值全丢 |
| 二 | 三项产品级特性 | 框架真 / 产品假 | 规划执行、自学习、可信记忆都有机制，但都不开箱可用 |
| 三 | 长程严苛复测 | 7 维中 3 达标 | "越用越好"被量化证伪：评测指标本身测不出学习 |
| 四 | 建站（带后端） | **1.4/5** | 能写源码文本，不能交付可运行网站 |
| 五 | 办公文档改造方案 | — | 给出 L1 格式承载层 + L2 内容结构层改造方案 |
| 六 | 办公文档 / PPT / 表格 | **0.7/5** | 假 Office 文件；纯文本交付约 2.2/5 |
| 七 | 真实 LLM 在线实测 | 协议 ✅ 语义 ❌ | 能按格式干活，不能保证真的干了活 |
| 八 | 腾讯文档建物 | **0.5/5** | 零集成、零鉴权、零在线产物路径 |
| 九 | 生图 / 生视频 | **1.5/5** | 生图 2.5（真链路）、生视频 0（零实现） |
| 十 | 3D 模型生成 | **0/5** | 四维度全空；"3D"字样全是记忆 shader 比喻 |
| 十一 | three.js 游戏开发 | **1.0/5** | 能写不能跑；两个意图/兜底解析 bug |
| 十二 | Blender/Godot MCP | **1.5/5** | 框架能挂、产品不能用；截图与场景数据回不来 |
| 架构 | Rust 原生化 | 不建议 | 全量重写收益 ≈ 0，成本 11 万行 |
| 战略 | 对标豆包工作 | 产品 vs 框架 | 差整个产品层；常在线多身份端倪成体系 |

> 评分口径：0–5 分，面向"开箱默认 + 官方文档描述的能力面"。3 分以上才算可用于该场景。

### 3.2 能力结构图

```
能交付（真实可用）
├── 沙箱文件读写（路径校验 + 1 MiB 上限）
├── Markdown / CSV / SVG 文本产物
├── 治理：风险分级 / 审批 / Grant / 配额 / 审计   ← 全库最扎实一层
├── 混元生图（仅 desktop、真调真写，但绕治理）
└── stdio MCP 骨架（initialize → tools/list → tools/call）

名义可用（机制真实、默认不可达）
├── 规划：7 类固定线性模板，无 action→tool 映射
├── 自学习：闭环真实，但自我确认 + 失败当成功
├── 可信记忆：影子 → 人审 → 生效真实，默认 shadow_only 永不生效
├── TaskPool / Subagent：默认关、并发 2、深度 1
└── 连接器：notes / calendar / mail / mcp，无云端

完全缺失
├── 生视频 / 3D 生成 / Office 原生格式 / 腾讯文档
├── 命令执行 + stdout 回灌 + 运行验证
├── HTTP/SSE MCP + OAuth 凭据层
├── 入站通道 + 云端常驻
└── 执行一致性闸门（声称 vs 轨迹）
```

### 3.3 十二轮评测的关键修正链

| 轮次 | 认识论修正 |
|---|---|
| 一 → 二 | 组件正确 ≠ 产品可达：高通过率来自孤立纯函数 + 手工理想输入 |
| 二 → 三 | 手工钻石 DAG 证明调度器可并行（150 节点 peak=8），责任精确落到"规划器只产线性链" |
| 三 → 四 | 即便调度器可并行，规划步骤仍是提示词文本，与工具链不相交 |
| 六 → 七 | 假文件不只是框架缺陷：真实模型会为假文件编造合理化说明，可靠性**负增益** |
| 八 → 十二 | "支持 MCP"必须先问传输形态：stdio 型能挂，HTTP/OAuth 型无处接 |
| 十二 → 战略 | 缺的不是功能点，是整个产品层；但架构方向押对了常在线多身份 |

---

## 四、测试规模与工程基线

### 4.1 十二轮测试规模演进

| 轮次 | 累计 spec | 累计用例 | 通过 | 失败 | 通过率 |
|---|---:|---:|---:|---:|---:|
| 1 广度 | 61 | 2,209 | 2,195 | 14 | 99.37% |
| 2 三项特性 | 68 | 2,421 | 2,404 | 17 | 99.30% |
| 3 长程 | 80 | 4,944 | 4,911 | 33 | 99.33% |
| 4 建站 | 81 | 5,011 | 4,974 | 37 | 99.26% |
| 6 办公 | 82 | 5,125 | 5,080 | 45 | 99.12% |
| 8 腾讯文档 | 83 | 5,165 | 5,113 | 52 | 98.99% |
| 9 生图/生视频 | 84 | 5,203 | 5,145 | 58 | 98.88% |
| 10 3D | 85 | 5,224 | 5,162 | 62 | 98.81% |
| 11 three.js | 86 | 5,246 | 5,179 | 67 | 98.72% |
| 12 Blender/Godot | **87** | **5,267** | **5,195** | **72** | **98.63%** |

- **72 个失败全部是标注 `FINDING` 的契约缺口**（其中 17 条标题不含 "FINDING" 字样，是前几轮标注惯例差异，逐条对比确认历史失败未变）。
- 每轮均核对：新增失败全来自本轮 FINDING、消失失败 0、历史失败一个没变。
- 专项 FINDING 数：办公 8、腾讯文档 7、生图/视频 6、3D 4、three.js 5、Blender/Godot 5。

### 4.2 工程基线

| 项目 | 结果 |
|---|---|
| 安装 | pnpm 9.12.0，623 包；Electron 经 npmmirror 镜像拉取 |
| 构建 | `pnpm build` (turbo) **31/31** |
| 类型检查 | `turbo run typecheck` **59/59**（strict + noUncheckedIndexedAccess + exactOptionalPropertyTypes） |
| Lint | `biome lint` 15 errors / 7 warnings（仓库原有 7 + 新增 spec 8）；`biome check` 635 错主要是 Windows CRLF 与 `lineEnding: lf` 冲突，不计缺陷 |
| 仓库原有测试 | 144 spec / 1,295 例 / 6 失败（4 个硬编码 `F:\workspace`、2 个机器性能守卫）/ 30 跳过 |
| 源码规模 | 50,208 LOC / 299 文件 / 32 包；测试 63,161 LOC / 235 spec |
| 性能基准 | TAO 单轮 0.072ms（N=200）；HookBus 无 handler 1.455µs、10 handler 43.328µs |

### 4.3 环境处理（非仓库缺陷）

- Electron 直连 GitHub 卡住 → `ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/`。
- 宿主安全删除守卫拦截 pnpm 链接 → 仅安装命令设置 `CODEBUDDY_SAFE_DELETE_ENABLED=0`（作用范围仅可再生 `node_modules`）。
- Vitest 2.1.9 的 `--maxWorkers=4` 触发 "no tests" 参数解析问题，已弃用。
- 一次全库回归的 `audit-deep-memory-loop.spec.ts`（49 例）并发收集缺席，单独复跑 49/49 全绿，判定为环境瞬时问题，未计入最终数字。

---

## 五、跨轮核心发现：六大结构性根因

> 72 个失败不是零散 bug。按根因归并后，改造顺序就清楚了——**先堵漏，再补能力**。

### R1 · 执行一致性与假完成（最高危，P0）

| 证据 | 现象 |
|---|---|
| 第七轮 FINDING 7-1 | `multi-file` 任务：1 轮、1 次 LLM 调用、**0 次工具调用**，框架状态 `ok`，答案称"已分别写入 a.md 和 b.md" |
| 第七轮 FINDING 7-2 | `docx-trap`：1 次 utf8 `write_file`，答案称"已生成季度报告文档"，并编造"用 Word 另存为"的补救建议 |
| 第六轮 F1–F3 | 产物兜底把 Markdown 纯文本写进 `.docx/.pptx/.xlsx`，首字节 `23 20 32 30` 而非 `50 4B 03 04` |
| 第九轮 F9-6 | 生图前置返回 `completed`，但 `iterations: 0`、轨迹只有一条手写 `final`——结果可能是真的，轨迹不能证明 |
| 第一轮 RFC-006 T3 | `provider:"mock"` 带伪造 URL 被判 `reliable`，fail-closed 可绕过 |

**根因**：`react.ts:213-217` 把"既无 FINAL 也无 Action"当隐式终止；FINAL 分支不检查内容与工具轨迹一致性；框架把"模型说了 FINAL"直接当成功。

**修复方向**：在 FINAL 分支前加执行一致性闸门——需落盘/需工具的请求若 `toolCalls=0` 则禁止 `ok`，回灌 SystemCheck 要求重跑；工具结果含 Office 扩展名而内容是 UTF-8 文本时强制降级为 `.md` 或拒绝写入。

### R2 · 规划名义化与 action→tool 真空（P0）

| 证据 | 现象 |
|---|---|
| 第二轮 P1–P5 | 7 类意图模板全部严格线性链，并发峰值恒为 1；用户输入对计划几乎零影响；`params` 恒 `{}` |
| 第二轮 P3 / 第四轮 F1 | 三端只把 `[description]（步骤 id/action）` 拼成 `stepQuery` 交 LLM；全仓无 `node.action → 工具` 映射；`makeExecuteCommand` / `makeToolCommand` 是死代码 |
| 第六轮 | 文档/幻灯片/表格三类需求产出**逐字相同**的 4 步计划 |
| 第七轮 FINDING 7-1 | 真空被模型语言能力填成谎言 |

**根因**：规划是关键词分类 + 固定文本模板；治理/ToolHub/Executor/状态机是另一条独立链路，与规划图不相交。

**修复方向**：要么让 Planner 产出工具级步骤（带 `params`），要么让 ReAct 直接消费 `ToolHub.list()`；同时给至少一类意图支持扇出，否则 150 节点并行调度能力永远用不上。

### R3 · 自学习负向与记忆评测器失真（P0）

| 证据 | 量化数据 |
|---|---|
| 第三轮 | 200 轮 A/B：生产式自我标签错判 **175/200（0.875）**、confidence=1.0；外部正确标签错判 **1/200（0.005）**、confidence=0.5912 |
| 第二轮 L2 | `outcomeSignal` 对 cancelled/max_iterations/error/blocked/未知状态一律 **+0.2**；9 种状态无一中性 |
| 第二轮 L1 | 自我确认：错判 confidence 0.6667→0.75→0.8333，正确标签份额 1/3→1/4 |
| 第三轮 | `recallHit` 只做 contains，零记忆回显 Agent 可得 `recallRate=1.0`（59/59） |
| 第三轮 | `improvement.delta=29/30` 完全由脚本"先问后写"顺序决定；改 SHUFFLED_SCRIPT 后同时塌到 0 |
| 第三轮 | `defaultLongRunScore` 的 winner 是 `history-window`（无脑塞历史原文），不是完整记忆链 |
| 第三轮 | 分类器权重 1,000 轮 Σweight 10.5→306.2 无 clamp；原型键 `__proto__` 等返回非字符串 |

**根因**：反馈标签来自分类器自身判断（自我确认），失败被当成功学；长跑评测器无法区分"回忆"与"复读当前提问"。

**修复方向**：`outcomeSignal` 仅 `completed` 为正、失败为负、其余为 0；`reinforce` 引入外部正确标签或 `verified` 门；`recallHit` 排除当前 query 并要求命中来自历史轮；`computeImprovement` 改为同一批探测轮的前后时间窗对比。

### R4 · 三端不一致（P1，四次重复出现）

| 轮次 | 现象 |
|---|---|
| 第八轮 F8-5 | CLI `assembleAgent` 定义了 `initConnectors()` 并在装配期注册连接器，但主流程 assemble 后直接 `run()`，**从不调用**；server / desktop 都正确调用 |
| 第九轮 F9-5 | 生图只有 desktop 有前置分支，CLI/server 零引用 |
| 第二轮 M3 | 影子记忆链只接 desktop，CLI/server 无 canonical |
| 第十二轮 F12-4 | CLI 断链第四次出现，Blender/Godot 工具在命令行永不注册 |

**根因**：没有统一的"能力 × 宿主"装配矩阵与一致性测试，能力靠各端手写分支。

**修复方向**：建立三端装配清单（capability × cli/server/desktop），把生图、连接器初始化、影子记忆抽成平台无关模块统一装配；加一条"三端 `has(tool)` 一致"的回归断言。

### R5 · 承载层与回灌缺失（P1）

| 缺口 | 证据 |
|---|---|
| 二进制通道 | `write_file` 固定 `fs.writeFile(abs, content, "utf8")`，`inputSchema` 仅 `{path, content}`；`read_file` 固定 utf8，无 base64 |
| Office 依赖 | 66 个自有依赖零 `docx/pptxgenjs/exceljs/jszip` 等 |
| 运行时 | 零 three.js / WebGL；预览只有 `<pre>` / MarkdownView，无 iframe/canvas/webview |
| 命令执行 | `enableCommands=false` + 白名单空 + 三端主动 `unregister`；且 `firstToken()` 只查首 token，`npm install && curl ...` 可绕过 |
| MCP 回灌 | `mcp/client.ts:105-108` 只取 `type==="text"`，image/data 静默丢弃；connector handler 只透传 `{text}` |
| 云端鉴权 | `McpServerConfig` 只有 id/command/args/env/cwd，无 url/token/oauth；RFC-016 明确不做 MCP over HTTP/SSE |
| 配额 | 默认 20 次工具调用/分钟；最小建站文件集 13 个 + 读回校验正好 20 次，第 21 次 blocked |

**修复方向**：见第九节 P0/P1 路线图；优先做"假文件兜底"这一条（成本最低、风险最大）。

### R6 · 安全与健壮性（P0/P1）

| 严重度 | 缺陷 |
|---|---|
| P0 | `packages/voice/src/providers.ts:33` 用整个 backing `ArrayBuffer` 发 STT，子视图夹带内存池越界字节 |
| P0 | `llm/runtime` 裸 `sk-` token 脱敏后保留原文（违反 RFC-005:357 / ADR-002:202） |
| P0 | `core/loop/tao.ts` 四处 HookBus `emit` 返回值被丢弃，RFC-002 可编程改写契约整体不成立 |
| P1 | `canonicalToolName(s)` 对 `__proto__` / `constructor` 等原型键返回非字符串，ToolHub allowlist 直接消费 |
| P1 | `dijkstraSchedule` n=17/18 无 fail-closed，静默分配 `1<<n` 子集格 |
| P1 | `weightClamp` 对 hydrate 的 NaN 穿透；`cosine`/`norm` 对 NaN/Infinity 穿透 |
| P1 | 白名单只匹配命令首 token，链式命令可绕过——**打开命令执行前必须先修** |
| P2 | `sync/bundle.ts:122` 墓碑删除只删一层；`HybridRetriever` 归一化未做 min-max 裁剪、全 0 分并列排序退化 |

---

## 六、第七轮真实 LLM 在线实测

> Provider：腾讯混元 `hy4`（TokenHub 端点）；允许联网；费用硬上限 100 元。全部数字单列，不并入前六轮。

### 6.1 协议层 vs 语义层

| 层 | 指标 | 实测 | 判定 |
|---|---|---|---|
| 协议 | JSON 可解析率 | 12/12 任务、23/23 响应 = 100% | ✅ |
| 协议 | 8 轮内 FINAL 收敛 | 12/12 = 100%，平均 1.9 轮，0 `max_iter` | ✅ |
| 协议 | 工具参数 schema 合法率 | 10/10 = 100% | ✅ |
| 协议 | 可恢复错误降级 | 未知工具 → 提示 → 模型改工具完成 | ✅ |
| 语义 | 纯文本任务完成率 | 格式收敛 12/12，但执行真实性不通过 | ❌ |
| 语义 | 长程 20+ 轮 | ReAct 上限 8 轮，未覆盖 | ⚪ |
| 语义 | ParseError 自修正率 | 0 次 ParseError，样本为空 | ⚪ |

> 口径警告：`finalRate=100%` 是**格式收敛率**，不是任务完成率。唯一"未知工具"是模型按指令调用 `execute_command` 后框架返回"工具未注册"、下一轮改用 `write_file`——按设计意图是正确行为，按设计口径工具名命中率 10/10。

### 6.2 两条致命 FINDING

- **7-1 零执行假完成**：`multi-file` 任务 1 轮 / 1 次 LLM / **0 次工具调用**，框架 `ok`，答案称两个文件已写好。定向复核重跑，行为一致。
- **7-2 假 docx + 编造合理化**：明知 `write_file` 只支持 UTF-8，仍以"已生成季度报告文档"开头，并建议"用 Word/WPS 打开后另存为"——而该文件根本打不开。

**结论**：预判中"大概率能干轻活"被收窄——连轻活都会假完成。给 key 没有翻案，1–5 号链路亮了，6–10 号断点一个没动。

### 6.3 费用与密钥安全

| 批次 | LLM 调用 | tokens |
|---|---:|---:|
| 首次误执行 curl（非计划） | 1 | 119 |
| 端点对照探测 | 2 | ~30 |
| 小样本校准 | 5 | 4,020 |
| 全量 12 任务 | 23 | 36,546 |
| 定向复核 | 3 | 7,089 |
| **合计** | **34** | **约 47,800** |

- 按刻意高估的 ¥0.1/千 tokens 推算约 ¥4.8，远低于 100 元封顶；单价未核实，**此为估算不是账单**。`reasoning_tokens` 占 completion 约 82%，是主要成本项。
- 事故 1：env 文件里是 curl 命令而非赋值行，`source` 误执行 1 次。已新增 `load-env-safe.sh`，逐行只解析 `^[A-Za-z_][A-Za-z0-9_]*=`，绝不执行命令。
- 事故 2：key 末尾混入单引号导致 401；加载器增加"密钥变量剥离所有引号"后 TokenHub 返回 200。
- 旧端点 401 错误回显了 key 首尾片段，加上明文存文件与作为 curl 进程参数出现，**建议轮换该 key**。所有工具输出与交付物已做精确值 + `sk-*` 通用模式双重脱敏。

---

## 七、架构决策：Rust 原生化结论

### 7.1 结论：不做全量重写

| 路径 | 成本 | 性能收益 | 判断 |
|---|---|---|---|
| A. 不重写，聚焦补能力缺口 | 0 | 0（瓶颈本不在这） | ✅ **推荐** |
| B. Rust 热点模块替换（TS 大脑 + Rust 肌肉） | 2–4 人月 | 条件性 2–5x | ⚠️ 仅在 profile 确认热点后 |
| C. 全量 Rust 重写 | 10–24 人月 + 63k 测试 | 端到端 ≈ 0% | ❌ 不建议 |

### 7.2 决定性证据

- 实测 TAO 单轮框架开销 **0.072ms**（N=200），比 RFC 目标 50ms 快 **694 倍**；HookBus 无 handler 1.455µs、10 handler 43.328µs。
- 第七轮真实混元：23 次调用耗时 478 秒，平均约 20 秒/次。端到端延迟 99.9%+ 来自 LLM 网络往返。
- Rust 把 0.072ms 变 0.01ms，P50 从 8.0 秒变 7.9999 秒，用户感知为 0。
- 成本基数：源码 50,208 LOC + 测试 63,161 LOC = 113,369 LOC；32 包；桌面 UI 15,376 LOC（与 Agent 逻辑无关）；核心 34,832 LOC。按 100–300 有效 LOC/人日，核心 116–348 人日，加测试 250–600 人日。
- 生态 hardest part：openINTJ 核心价值在自制编排层（TAO/ReAct/HookBus/skill registry/记忆 shader），恰恰是 Rust 里最难写、代码量膨胀 1.5–2x 的部分；Electron + React UI 无等价。

### 7.3 改口的三个条件

1. 部署到内存 <50MB 的边缘设备；
2. 每秒数千次工具调用的 tick 级高频场景；
3. 有独立 Rust 团队且 TS 团队转做别的产品。

> 若只是"学 Rust / 玩一玩"，值得——但按学习项目做，从一个热点函数（如 hybrid retriever 向量打分）开始，别碰编排层。

---

## 八、战略定位：对标豆包工作与常在线多身份意图

### 8.1 三问答案

| 问题 | 答案 | 置信度 |
|---|---|---|
| 比豆包工作差什么 | **差整个产品层**：企业上下文生态、成品交付、电脑操作/云电脑常在线、多 Agent 编排、生态扩展、企业治理 | 高 |
| 可能强在哪 | 数据主权与本地优先、代码级可编程治理、记忆用户可控可审计、身份图谱诚实边界——**都是潜力，不是现状** | 中（推断） |
| 常在线多身份端倪 | **有且成体系**：6 个 RFC + 6 个已落地包构成完整意图链 | 高 |

### 8.2 七类差距（摘要）

| 维度 | 豆包工作 | openINTJ |
|---|---|---|
| 企业上下文生态 | 飞书登录继承聊天/云文档/会议/日程，产出沉淀回飞书 | 本地 notes/ICS/mbox；腾讯文档接不上 |
| 成品文档交付 | 文档/表格/PPT/网页 + 框选局部改写 + 本地 Office 同步编辑 | 假 Office 文件；本地 Office 修改零 |
| 多模态创作 | 图片/视频/网页/应用生成并编辑 | 生图仅 desktop 前置；视频/3D 零；游戏能写不能跑 |
| 操作电脑 / 常在线 | 浏览器与电脑操作、云电脑关机续跑、手机远控 | 命令默认禁；无 GUI；通道只出不进 |
| 多 Agent | 最多 5 Agent 并行"工作小队" | TaskPool + Subagent 默认关、并发 2、深度 1、无动态拆图 |
| 生态 | 技能/连接器/官方插件/第三方模型可插拔 | 内置工具 4 个；MCP stdio-only；provider 6 种 |
| 企业治理 | 继承飞书权限、隔离、加密、审计、双认证 | RFC-013 局部扎实，但无企业身份与认证 |

**根因**：豆包工作卖"完成工作"，openINTJ 提供"Agent 框架能力"。这是产品与库的区别，不是功能点差额。

### 8.3 常在线多身份意图链（六环节全有 RFC + 代码）

| 环节 | RFC / 包 | 状态 |
|---|---|---|
| 身份：我是谁 | RFC-020 `identity` | ✅ 三端装配 + server API + 桌面身份 tab |
| 画像：我要什么 | RFC-014 `profile` | ✅ 三端 ProfileService，渲染进 user-persona |
| 常在线：没人说话也干活 | RFC-015 `scheduler` | ✅ once/interval/daily，三端 `scheduler.start()` |
| 结果触达人 | RFC-018 `channels` | ✅ webhook 扇出（ntfy/Server酱/企微机器人） |
| 跨端状态迁移 | RFC-019 `sync` | ✅ SyncBundle v1 导出 profile+schedules+identity |
| 多 Agent 聚合执行 | RFC-007/012 TaskPool+Subagent | ⚠️ 已实现但默认关 |

RFC 原文自述："系统目前是纯反应式的……**Personal Agent 的分水岭是主动性**"（RFC-015）；"给主动式系统一条出站通道"（RFC-018）；"给我一个邮箱/手机号/微信号，拉起我在各个应用上的全部身份"（RFC-020）。四句话连起来就是产品定义。

### 8.4 五个致命缺口

1. **通道只出不进**——手机只能收通知不能下指令；
2. **没有云端执行/daemon**——desktop 关了就没了；
3. **身份只认人不鉴权**——本机图谱，不做 OAuth 上下文源；
4. **多 Agent 默认关 + 并发 2**——对比豆包 5 并行；
5. **记忆可信度打折**——主动性越强，假完成越多。

### 8.5 战略建议

**别在"追平豆包功能"上使劲。** 追不平也不该追——那是字节用飞书生态 + 云基础设施 + 大团队堆出来的。

**押已押对的那条线**：常在线 + 本地身份图谱 + 可编程治理 + 用户可控记忆，终局是"**个人数据主权的主动式 Agent 底座**"。豆包绑定飞书，恰好给这类产品留了结构性空位。

---

## 九、升级建议规划：P0 / P1 / P2 路线图

> 原则：**先堵漏，再补能力，最后追生态**。跳步会返工——例如不修二进制通道就加 `create_docx`，仍然是坏文件；不修执行闸门就开多 Agent，只会批量制造假完成。

### 9.1 P0 — 可靠性与安全（必须先做，预计 4–8 周）

| # | 任务 | 目标文件 | 验收标准 |
|---|---|---|---|
| P0-1 | **执行一致性闸门** | `core/loop/react.ts` FINAL 分支前 | 需落盘请求 `toolCalls=0` 时禁止 `ok`；Office 扩展名 + UTF-8 内容强制降级为 `.md`；第七轮 7-1/7-2 两个任务不再假完成 |
| P0-2 | **修反馈信号** | `classifier/src/routing.ts:52-56`、`skills/src/learning-runtime.ts:12-16`、三端 `agent.ts` | `outcomeSignal` 仅 completed 为正、failed/timeout 为负、其余 0；F10 验证门统一三端；200 轮 A/B 错判率从 0.875 降到 <0.1 |
| P0-3 | **评测器修正** | `shared/src/longrun-eval.ts:87-120,143-147` | 零记忆回显 Agent 的 `recallRate` 必须 <0.2；SHUFFLED_SCRIPT 下 delta 不得虚高；winner 不得是无脑历史窗口 |
| P0-4 | **三处安全漏洞** | `voice/src/providers.ts:33`、`llm/runtime` 脱敏、`core/loop/tao.ts` 四处 `emit` | 子视图只发 `slice()`；裸 `sk-` 完整脱敏；钩子返回值真正进入循环；对应历史 FINDING 转绿 |
| P0-5 | **假文件兜底修复** | `desktop/src/main/artifact-fallback.ts:20-36,66-71` | 无 Office 工具时对 `.docx` 请求**不产出 `.docx` 路径**；`.md` 首行明确提示；"幻灯片/PPT/表格"触发文件意图；"带封面的幻灯片"不再误判生图 |
| P0-6 | **CLI 断链**（第四次出现） | `apps/cli/src/index.ts:68` 之后 | `initConnectors()` 在 `run()` 前调用；三端 `hub.list()` 含 MCP 工具的一致性断言通过 |
| P0-7 | **超时/健壮性** | `core/src/types/tool.ts:59-75`、`control/src/path-search.ts:66-81`、`skills/src/learning-runtime.ts:82-83` | 原型键返回 string 或拒绝；n>16 fail-closed；NaN/Infinity 不穿透 |

**P0 完成标志**：72 个 FINDING 中的 P0 类全部转绿；第七轮两类假完成在真实模型上复测不重现。

### 9.2 P1 — 核心产品能力（预计 8–16 周，P0 完成后）

| # | 任务 | 关键点 | 验收标准 |
|---|---|---|---|
| P1-1 | **二进制通道** | `write_file/read_file` 支持 `contentBase64`；上限 1 MiB → 20 MiB | 落盘字节与源 Buffer 逐字节相等；ZIP 魔数检测返回 base64 |
| P1-2 | **Office 生成工具** | 新增 `office-tools.ts`：`create_docx/create_pptx/create_xlsx`；依赖 `docx`/`pptxgenjs`/`exceljs` | 魔数 `50 4B 03 04`；`jszip` 可打开且含 `[Content_Types].xml`；`exceljs` 读回 sheet/公式一致 |
| P1-3 | **三端装配** | `tool-hub.ts:402-457` + 三端 `agent.ts` 注入，可选 handler 保持向后兼容 | 三端 `has("create_docx")` 均 true；现有 87 spec 全库回归不新增失败 |
| P1-4 | **规划接线 + 扇出** | `planner.ts` 产出工具级 `params`；删/接 `makeExecuteCommand` 死代码；至少一类意图支持并行分支 | `step.action` 能在 ToolHub `call()` 成功；某类意图实测并发峰值 ≥2 |
| P1-5 | **执行面与回路** | 修 `firstToken()` 链式绕过；开放命令 + stdout/stderr/exitCode 回灌；新增 `run_tests/start_server/http_request/typecheck` | 写→跑→错→改→再跑闭环可复现；链式命令被拒 |
| P1-6 | **配额改为按 run 计费** | `governance/types.ts:50` | 20 次以上连续写入不被 blocked |
| P1-7 | **MCP 数据回灌** | `mcp/client.ts:105-108` 保留 image/data，转 data URL 或落临时文件路径；handler 透传 raw | Blender 截图文本不再丢失；`content` 长度 ≥2 |
| P1-8 | **三端能力统一** | 生图前置抽平台无关模块；影子记忆链接入 CLI/server；建立 capability × host 装配矩阵 | 三端能力差异有显式清单；差异项有测试覆盖 |
| P1-9 | **产物预览层** | `workspace-view.ts` 加 OOXML MIME / office 类别；runnable iframe 容器 | `.docx/.pptx/.xlsx` 不再是 octet-stream；html 能点运行 |

**P1 完成标志**：建站 ≥3/5、办公原生 ≥3/5、生图三端可用且走治理、Blender/Godot ≥3/5。

### 9.3 P2 — 生态与战略（预计 12 周以上，按战略价值排序）

| # | 任务 | 说明 |
|---|---|---|
| P2-1 | **入站通道 + daemon 常驻** | 把"定时"变成真"常在线"；先做只读指令白名单，避免把命令面暴露公网 |
| P2-2 | **身份从认人升级到鉴权上下文源** | RFC-020 遗留项：OAuth/账号上下文，接入企业权限源 |
| P2-3 | **多 Agent 默认开 + LLM 动态拆图** | 对标豆包 5 Agent 并行；并发上限与深度可配置；先有 P0-1 闸门才开 |
| P2-4 | **可信记忆激活决策门** | 给出误存率数据 + 人审台三端可用 + 激活时间表 |
| P2-5 | **HTTP/SSE MCP transport + 凭据层** | 需新 RFC 反向决策 RFC-016；凭据走环境变量/密钥环，不落盘明文 |
| P2-6 | **生视频 / 3D 生成** | 仿 `image.ts` 的 b64/URL 双形态 + 长任务轮询；`LlmClient` 显式声明 `generateVideo?/generate3D?`，不搞 duck typing |
| P2-7 | **腾讯文档专用连接器** | 需 P2-5 完成后才可行；产物回执写 `ToolCallResult.evidence` 供闸门校验 |
| P2-8 | **Rust 热点替换（可选）** | 仅在 profile 确认 hybrid retriever CPU bound 后做；TS 大脑 + Rust 肌肉 |

### 9.4 改造顺序与依赖

```
P0-1 执行闸门 ──┬──> P0-2 反馈信号 ──> P0-3 评测器 ──> 可信记忆可信度
                └──> P2-3 多 Agent 默认开（之前绝不能开）

P0-5 假文件兜底（成本最低，可最先做）──> P1-1 二进制通道 ──> P1-2 Office 工具 ──> P1-3 三端装配

P0-6 CLI 断链 ──> P1-7 MCP 回灌 ──> P2-5 HTTP/OAuth ──> P2-7 腾讯文档

P1-5 执行面 ──> 必须先完成 P0-4 安全 + 白名单修复，否则放大风险面
```

### 9.5 投入产出提示

- 同样 10 个月，P0+P1 能把 openINTJ 从 **1.5/5 提到 3–4/5**，这是用户能感知的提升；Rust 全量重写带来的是 **0**。
- 若目标是"快速做出带后端网站 / Office 文件 / 3D 游戏"，直接用具备 shell + 测试 + 浏览器验证回路的通用编码 Agent，比改造 openINTJ 短得多。

---

## 十、明确不建议做的事

1. **不做全量 Rust 重写**——收益零，成本 11 万行，且 12 轮审计资产作废。
2. **不追平豆包功能**——文档编辑、GUI 操作、云端执行是军备竞赛，追不平也不该追。
3. **不在 P0-1 完成前开多 Agent / 主动触达**——主动性越强，假完成越多，会造成批量误导。
4. **不绕过 `write_file` 直接 fs 写盘**——会丢掉路径沙箱、大小上限、治理留痕与配额，拆掉唯一扎实的 L8 安全层。
5. **不只改 `fallbackArtifactPath` 就以为完事**——只要 `write_file` 还是 utf8，模型自己调工具写 `.docx` 仍是坏文件。
6. **不做"打开命令执行 + 不修 firstToken 绕过"的组合**——`npm install && curl ...` 会穿透白名单。
7. **不拿 98.63% 契约通过率与任何端到端任务分做比较**——量纲不同，结论会荒谬。

---

## 十一、验收指标与复现方式

### 11.1 升级后必须全绿的验收项

| 类别 | 指标 | 目标 |
|---|---|---|
| 工程 | `build` / `typecheck` | 31/31、59/59 保持 |
| 回归 | 现有 87 个 audit spec | 5,267 例不新增失败；历史 FINDING 逐条转绿或有豁免记录 |
| 真实性 | 第七轮两个假完成任务在真实模型上复测 | 0 次工具调用不再 `ok`；假 `.docx` 不再出现 |
| 学习 | 200 轮自我标签 vs 外部标签 A/B | 错判率 <0.1、confidence 不再饱和到 1.0 |
| 评测 | 零记忆回显 Agent 的 `recallRate` | <0.2（当前 1.0） |
| 办公 | 三类生成工具产物魔数 / ZIP 部件 / 读回一致 | 15 条验收断言全绿（见办公改造方案 §4） |
| MCP | Blender 截图 text + image content | `content` 长度 ≥2，截图数据不丢 |
| 三端 | capability × host 装配矩阵 | 差异项有显式文档与测试 |
| 建站 | 最小交付 13 文件连续写入 | 不被 20 次/分钟配额 blocked |

### 11.2 复现命令

```bash
cd /c/Users/benjamin/WorkBuddy/2026-11-13-02-36-37/openINTJ/ts
export CODEBUDDY_SAFE_DELETE_ENABLED=0

# 全库审计 spec
FILES=$(find packages apps -type f -name 'audit-*.spec.ts' | sort | tr '\n' ' ')
npx vitest run --reporter=json --outputFile=../../audit-final.json $FILES

# 工程基线
npx turbo run typecheck --output-logs=errors-only
npx turbo run build --output-logs=errors-only

# FINDING 检索
grep -rn "FINDING:" . --include='audit-*.spec.ts'
```

> 注意：`--maxWorkers=4` 在 Vitest 2.1.9 会触发 "no tests" 解析问题，勿加。

### 11.3 未验证项（边界声明）

- 除第七轮 12 任务外，全程未用真实 LLM；内容质量、事实正确性、生成代码质量**不能证明也不可证伪**。
- 未启动 Electron GUI、Hono 服务、真实 SQLite/LanceDB 全链路、Playwright e2e、真实 Blender/Godot/MCP 进程。
- 未联网实测第三方 MCP server 握手细节、TokenHub 生图额度、腾讯文档账号。
- 豆包工作能力来自公开报道与第三方测评引用，未经我方实测；其 SuperCLUE 分数未核对原始报告。
- Rust 产能估算（100–300 LOC/人日）是行业经验值，非本项目实测；"路径 B 收益 2–5x"是条件性推断，须以真实 profile 为准。
- 性能基准用 mock LLM，不计真实网络。
- 本报告为工程评估，仅为分析与模拟，非投资建议。

---

## 十二、附录：原始报告清单与边界声明

### 12.1 本报告整合的源文件

| # | 文件 | 内容 |
|---|---|---|
| 1 | `AUDIT_BRIEF.md` | 首轮审计规范：只新增 spec、FINDING 标注、诚实性规则 |
| 2 | `LONG_AUDIT_BRIEF.md` | 长程严苛复测判定矩阵与场景剧本 |
| 3 | `openINTJ-能力审计报告.md` | 61 spec / 2,209 例，22 条逐项判定，45 处偏差 |
| 4 | `openINTJ-三项产品级特性专项验证报告.md` | 规划执行 / 自学习 / 可信记忆三项纵深验证 |
| 5 | `openINTJ-长程严苛复测报告.md` | 7 维度判定、真实长会话 A/B、10 类新缺陷 |
| 6 | `openINTJ-建站能力评价报告.md` | 1.4/5，L0–L8 能力矩阵，最小改造集 |
| 7 | `openINTJ-办公文档能力评价报告.md` | 0.7/5，假 Office 文件，8 条 FINDING |
| 8 | `openINTJ-办公文档能力改造方案.md` | L1 承载层 + L2 结构层，路线 A/B，15 条验收 |
| 9 | `openINTJ-LLM实测范围评估.md` | 接真实 LLM 的实验边界预判 |
| 10 | `openINTJ-在线实测预判与密钥安全接入说明.md` | 判据表、SOP、密钥安全、事故记录 |
| 11 | `openINTJ-第七轮在线实测报告.md` | 真实 hy4 在线 12 任务，7-1/7-2 两条 FINDING |
| 12 | `openINTJ-腾讯文档能力评价报告.md` | 0.5/5，7 条 FINDING，CLI 断链 |
| 13 | `openINTJ-生图生视频能力评价报告.md` | 生图 2.5 / 视频 0，6 条 FINDING |
| 14 | `openINTJ-3D模型生成能力评价报告.md` | 0/5，4 条 FINDING，"3D"比喻误导 |
| 15 | `openINTJ-threejs游戏开发能力评价报告.md` | 1.0/5，5 条 FINDING |
| 16 | `openINTJ-Blender-Godot-MCP连接能力评价报告.md` | 1.5/5，5 条 FINDING，image/data 回灌丢失 |
| 17 | `openINTJ-Rust原生化重写可行性评估报告.md` | 三条路径对比，0.072ms 实测，不建议重写 |
| 18 | `openINTJ-对标豆包工作与常在线多身份战略判断.md` | 七类差距、四项潜力、六环节意图链 |
| 19 | `.workbuddy/memory/2026-11-13.md` ~ `2026-11-20.md` | 逐日工作日志（11 份） |

### 12.2 关键机器可读结果文件

`audit-confirm.json`、`baseline.json`、`audit-all.json`、`audit-deep.json`、`audit-long-all.json`、`session-run2.json`、`audit-webapp-clean.json`、`audit-office-all.json`、`audit-tencent-docs-all.json`、`audit-image-video-all.json`、`audit-3d-all.json`、`audit-threejs-all.json`、`audit-blender-godot-all.json`、`online-baseline-full.json`、`online-baseline-spotcheck.json`

### 12.3 交付边界

- 本报告是**整合与规划**，不新增测试数据、不改动被测仓库、不调用任何真实 key。
- 所有原始数字来自真实运行的 Vitest JSON 输出，未估算或补写。
- 报告中的改造建议是工程方案，尚未实施；实施后需按第十一节验收矩阵重新评测。
- 涉及密钥的所有输出均已脱敏；本报告不包含任何凭据。

---

*α — 量化精英阿尔法　|　结论先行，依据随后，风险不藏。*
