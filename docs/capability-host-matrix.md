# Capability × Host 装配矩阵（P1-8）

> 源数据：`ts/packages/shared/src/capability-matrix.ts`（测试同步断言）。

| Capability | CLI | Server | Desktop |
|---|---|---|---|
| office_tools | available | available | available |
| dev_tools | available | available | available |
| image_generation | partial | partial | available |
| shadow_memory | absent | opt_in | opt_in |
| mcp_connectors | opt_in | opt_in | opt_in |
| web_search | opt_in | opt_in | opt_in |
| workspace_preview | absent | absent | available |

**说明**
- `available`：默认装配
- `opt_in`：需环境变量/配置（如 `OPENINTJ_CONNECTORS`、搜索 key、`OPENINTJ_MEMORY_WRITER`）
- `partial`：有平台无关生图模块（`@openintj/plane-execution` `fulfillImageGenerationTask`），三端可调用；CLI/server 无 GUI 预览
- `absent`：该端未装配
