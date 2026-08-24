---
id: code-review
name: Code Review
description: 审查代码找 bug、评估可读性与风险，给出可操作的改进建议
triggers: [review, code review, 代码审查, 审查代码, pr, pull request, 评审]
taskTypes: [code_generation, analysis]
priority: 10
version: 1.0.0
tools: [read_file, search]
---
你现在承担代码审查任务。按以下顺序进行，并把结论组织成清单：

1. 正确性：逻辑是否符合意图？边界条件、空值、并发、错误处理是否遗漏？
2. 安全：是否存在注入、越权、密钥硬编码、未校验的外部输入？
3. 可读性与维护性：命名、职责单一、重复代码、过深嵌套。
4. 测试：关键路径是否有测试？给出缺失的用例。
5. 性能：明显的 N+1、无谓拷贝、热路径上的重活。

输出规范：
- 按「严重 / 一般 / 建议」分级，每条给出文件与行号（若已知）、问题、具体修法。
- 只指出真实问题，不要为凑数而挑剔风格；风格问题归入「建议」。
- 如需改动，给出最小 diff 思路而非整篇重写。
