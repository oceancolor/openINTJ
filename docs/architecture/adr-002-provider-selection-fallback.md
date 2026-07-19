# ADR-002：模型 Provider 选择、回退与持久化身份

- 状态：**已采纳（Accepted）** — 2026-07-14
- 关联：[RFC-005](../rfcs/RFC-005-local-model-runtime.md)、[ADR-001](./adr-001-react-tool-protocol.md)
- 决策者：核心架构

## 背景

OpenINTJ 已有 Hunyuan、Ollama LLM adapter，Ollama embedding adapter，以及用于测试/演示的 mock/simple 实现，但 provider 选择散落在三个入口：

- CLI 有 `auto`，现状优先 Hunyuan key，再选 Ollama；
- Server 和 Desktop 各自装配，现状默认 mock；
- Ollama/Hunyuan adapter 在部分失败路径会直接返回 mock 内容；
- LLM 与 embedding 没有统一但独立的选择契约；
- 持久化工厂只接收向量维度。LanceDB 固定了列维度，却不能识别“同维度、不同模型”的不兼容向量空间。

这会造成三类问题：

1. “本地优先”在不同入口含义不同；
2. 用户显式选择真实 provider 后，失败可能被 mock 响应掩盖；
3. 切换 embedding provider/model 后，旧向量可能被错误复用。

同时，当前依赖方向是 `core -> shared`，具体 LLM/embed adapter 依赖 core 接口。provider 编排若放入 shared，会引入反向依赖和环。

## 决策

### 1. 建立独立模型运行时包

新建：

```text
ts/packages/llm/runtime
@openintj/model-runtime
```

runtime 依赖 core 接口和具体 LLM/embed adapter，三端依赖 runtime。它不放入 `@openintj/shared`，core、shared 和具体 adapter 不反向依赖 runtime。

### 2. LLM 与 embedding 独立选择

LLM provider 集合为：

```text
auto | ollama | hunyuan | mock
```

embedding provider 集合为：

```text
auto | ollama | mock
```

LLM 选择 Hunyuan 不会自动改变 embedding；首期没有 Hunyuan embedding provider。

### 3. `auto` 采用本地优先顺序

LLM：

```text
Ollama 健康且模型存在
  -> 否则有 Hunyuan 凭据时使用 Hunyuan
  -> 否则使用明确可见的 mock
```

embedding：

```text
Ollama 健康且模型可用
  -> 否则使用明确可见的 simple-sha256 mock
```

mock fallback 必须显示在状态、日志和 UI 中；不能把 actual provider 标成 Ollama/Hunyuan。

### 4. 显式选择 fail closed

用户显式选择 `ollama` 时，服务不可达、模型未安装、超时或请求失败必须报错，禁止静默 mock。

相同原则适用于显式 Hunyuan：缺凭据或鉴权失败必须报错。显式 `mock` 是唯一主动请求 mock 的方式；`auto` 落到 mock 是选择状态机的可见结果。

### 5. 状态报告实际结果

runtime 状态必须分别报告 LLM 和 embedding，至少包含：

```text
requestedProvider
provider          # actual provider
model
mode
fallbackFrom
lastError
```

同时保留候选 attempts，以解释完整选择链。CLI、Server、Desktop 从同一状态对象投影，不能用 requested provider 推断实际 provider。

### 6. 持久化 embedding 指纹

持久化向量数据绑定：

```text
provider + model + dimension
```

打开 LanceDB/hydrate 前必须读取并比较指纹。任一字段不匹配就拒绝复用；即使 dimension 相同也不能混用。已有 fragment 但没有指纹的旧数据同样拒绝自动复用。

首期不在线迁移、不自动重算、不自动删除。用户可恢复原配置、换新数据目录，或显式清空后重建。

### 7. 保留 ADR-001 文本工具协议

本决策只改变 provider 解析、错误语义、状态和 embedding 身份，不改变 ReAct 的 `Thought/Action/Action-Input/FINAL` 文本协议。token streaming 和 OpenAI-compatible provider 留待后续。

## 备选方案

### A. 把选择逻辑放入 `@openintj/shared`

**未采用。**

优点是三端已经依赖 shared，表面接入成本低。缺点是 runtime 必须实例化依赖 core 的 adapter，而 core 本身依赖 shared，形成：

```text
shared -> runtime/provider -> core -> shared
```

这破坏 shared 的底层无 provider 依赖定位，也会给构建、测试和初始化顺序带来环依赖。

### B. 保持三端各自选择 provider

**未采用。**

短期改动少，但已经出现 CLI、Server、Desktop 默认值和 auto 语义不一致。后续 embedding、状态、health probe 和 fallback 测试会继续复制，难以保证契约一致。

### C. LLM 与 embedding 使用同一个 provider 开关

**未采用。**

LLM 能力与 embedding 能力并不对称：Hunyuan 当前只有 LLM adapter，Ollama 可同时提供两者，mock 的模型与维度也不同。绑定选择会导致无对应能力时出现隐式行为，并让 LLM 切换意外改变持久化向量空间。

### D. 云端优先：有 Hunyuan key 就先用 Hunyuan

**未采用。**

这延续当前 CLI 行为，但与产品的 local-first 定位冲突，增加数据外发、网络依赖和成本。凭据存在只表示可用候选，不应覆盖健康的本地模型。

### E. 真实 provider 失败时始终静默 mock

**未采用。**

优点是演示不易中断；代价是调用方无法区分真实推理与模板响应，配置错误、服务故障和鉴权失败会被隐藏。对显式选择尤其违反用户意图。

auto 仍可选择 mock，但它是 actual provider 的公开状态，而不是 adapter 内部伪装成功。

### F. 只按 embedding dimension 判断兼容

**未采用。**

相同维度不表示相同向量空间。两个 768 维模型的坐标语义可能完全不同，混用后相似度结果没有意义，却不一定触发运行时错误。

### G. 不持久化指纹，切换时自动重算

**未采用。**

没有指纹就无法可靠判断何时需要重算；自动重算还涉及原文完整性、长任务恢复、双写、磁盘空间、回滚和并发一致性。首期选择 fail closed，把迁移作为独立设计。

### H. 首期同时加入 streaming 与 OpenAI-compatible

**未采用。**

二者会扩大 LlmClient、IPC、取消/背压和 adapter 测试面，不是统一 provider 选择的前置条件。当前 `llm/openai-compat` 仍是 placeholder；先冻结 runtime 契约更容易验证。

## 后果

### 正面后果

- 三端获得一致的 local-first 行为和状态语义。
- 健康 Ollama 默认优先，减少云端依赖和数据外发。
- 显式选择失败不再被假响应掩盖，故障更容易定位。
- LLM 可以切换而不扰动 embedding 持久化；embedding 变更则被安全拦截。
- provider/model/dimension 指纹阻止“能打开但检索错误”的静默数据损坏。
- runtime 成为后续 provider、熔断和健康恢复的单一扩展点。

### 负面后果与成本

- provider resolution 变为异步，CLI 当前同步 `assembleAgent()` 需要调整。
- Ollama/Hunyuan adapter 的内部 mock 行为需要拆分或增加 strict 模式。
- Desktop IPC、config schema、SettingsPanel 和 StatusBar 都要升级。
- persistence 初始化顺序要改为“先 metadata 指纹、后 LanceDB/hydrate”。
- legacy 真盘数据没有指纹时会被拒绝，用户必须显式选择恢复或重建路径。
- auto 落到 Hunyuan 后，首个真实请求仍可能因鉴权/网络失败；首期不会在该请求中继续 mock。
- 维护两套独立 provider 配置增加了一些 UI 和测试复杂度。

## 实施约束

1. runtime 包不得被 core/shared/provider adapter 反向依赖。
2. 真实 adapter 的 strict 路径不得生成 mock 内容。
3. mock provider 必须有独立、可识别的 provider/model/status。
4. embedding resolution 和指纹校验必须早于向量 hydrate。
5. 指纹 mismatch 不得触发自动删除、覆盖或迁移。
6. 状态与 telemetry 必须脱敏，不记录 key 或 prompt。
7. ADR-001 parity fixtures 必须保持通过。

## 重新评估触发条件

满足以下任一条件时，应修订 RFC-005 或新增 ADR，而不是在入口中加入例外：

1. Hunyuan 或其他云服务提供正式 embedding adapter；
2. 需要运行期熔断并在真实 provider 间自动切换；
3. embedding 在线迁移成为产品要求；
4. Ollama tag 的可变性要求把 model digest/revision 纳入指纹；
5. token streaming 或 OpenAI-compatible 成为主路径；
6. mock 不再允许作为生产默认的最终候选。

