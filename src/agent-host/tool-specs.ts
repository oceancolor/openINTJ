/**
 * Agent 循环可用的工具清单（双协议共用同一份声明）
 *
 * 设计约束：
 *   ① 工具只有两个 —— 少而稳。工具越多，模型的误选面越大，审批噪音越高。
 *   ② `run_command` **不声明**风险等级、不声明是否需要批准。
 *      它只能「请求执行」，分级与裁决全在 P-04 tool-host 内部完成（D10）。
 *      如果让模型自带 `risk: 'L0'` 这种字段，闸门就变成了摆设。
 *   ③ `finish` 是唯一的正常终止信号。没有它，模型只能因为「说到没话说了」而停，
 *      那种停法无法区分「做完了」和「放弃了」。
 */

import type { ToolSpec } from '../shared/llm'

/** 一次模型请求里最多允许声明的命令数 —— 防止一条消息塞 50 条命令绕过逐步审批的节奏 */
export const MAX_COMMANDS_PER_TURN = 6

export const AGENT_TOOLS: ToolSpec[] = [
  {
    name: 'run_command',
    description: [
      '在工作区根目录下执行一条非交互式 shell 命令，并把 stdout/stderr/exitCode 返回给你。',
      '',
      '重要：',
      '- 一次只跑一条命令。不要用 && / ; / | 把多条命令串起来 —— 串起来之后审批卡片上只有一个字符串，',
      '  用户看不出你到底要干什么，审计也无法对应到具体动作。',
      '- 命令必须能自己跑完并退出。你看不到终端，不能有交互式输入（不要用 vim、less、read、需要密码的 sudo）。',
      '- 你必须给出 purpose：这条命令是为了满足契约里的哪一条。说不清目的的命令不要发。',
      '- 安全分级与审批不在你这里做。你只管请求执行；需要用户点头时系统会自动弹卡片，',
      '  并且可能被拒绝。被拒不是错误，是用户的决定。'
    ].join('\n'),
    inputSchema: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: '完整命令，例如 "npm test" 或 "git status --porcelain"'
        },
        purpose: {
          type: 'string',
          description: '这条命令要达成的目的，以及它服务于契约里的哪一条成功判据'
        }
      },
      required: ['command', 'purpose'],
      additionalProperties: false
    }
  },
  {
    name: 'generate_image',
    description: [
      '按一句描述生成一张图，产物落在工作区的 artifacts/ 下，并把路径返回给你。',
      '',
      '重要：',
      '- prompt 用中文或英文描述画面本身（主体、风格、构图、光线），不要写「生成一张图」这种指令。',
      '- 它和 run_command 走**同一条审批链**：写文件那一步可能被拒绝。被拒不是错误，照实说明即可。',
      '- 生图凭据没配好时，它会落一张**占位图**并告诉你原因 —— 那不是成功，如实转述给用户。',
      '- 一次只生成一张。要出一组图就分多次调用。'
    ].join('\n'),
    inputSchema: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description: '画面描述，例如「清晨的江边步道，水彩风格，横向构图」'
        },
        purpose: {
          type: 'string',
          description: '这张图服务于契约里的哪一条成功判据'
        }
      },
      required: ['prompt', 'purpose'],
      additionalProperties: false
    }
  },
  {
    name: 'finish',
    description: [
      '声明本次任务结束。这是唯一的正常终止方式。',
      '',
      '必须在 satisfied 里逐条说明每条成功判据是如何被满足的，并附上证据（来自命令输出的事实，不是你的推测）。',
      '如果有判据没能满足，放进 unmet 并说明卡在哪里（被拒绝 / 缺依赖 / 需要用户提供信息），',
      '**不要**把它塞进 satisfied 里凑数。'
    ].join('\n'),
    inputSchema: {
      type: 'object',
      properties: {
        summary: {
          type: 'string',
          description: '一段话总结实际做了什么'
        },
        satisfied: {
          type: 'array',
          description: '已满足的成功判据',
          items: {
            type: 'object',
            properties: {
              criterion: { type: 'string', description: '契约里的那条判据原文' },
              evidence: { type: 'string', description: '证明它成立的具体事实（命令输出、退出码等）' }
            },
            required: ['criterion', 'evidence'],
            additionalProperties: false
          }
        },
        unmet: {
          type: 'array',
          description: '未满足的成功判据及原因；全部满足时给空数组',
          items: {
            type: 'object',
            properties: {
              criterion: { type: 'string' },
              reason: { type: 'string' }
            },
            required: ['criterion', 'reason'],
            additionalProperties: false
          }
        }
      },
      required: ['summary', 'satisfied', 'unmet'],
      additionalProperties: false
    }
  }
]

/** 执行代理的系统提示 */
export const AGENT_SYSTEM_PROMPT = `你是 openINTJ 的执行代理，在一个受控工作区里替用户完成一个已确认的目标。

你手上有一份**用户已确认**的目标契约（JSON）。契约是你唯一的授权来源：
- 只做 successCriteria 覆盖的事；nonGoals 里列的事一概不做。
- assumptions 只是假设，一旦发现它不成立，用 finish 的 unmet 说明，不要硬往下做。
- needsClarification 如果还有未回答的项，先用只读命令把能查清的部分查清；确实需要用户决定时，用 finish 提出。

关于执行：
1. 先看，再动。第一次行动一般是只读命令（列目录、读文件、看配置），不要上来就改东西。
2. 一次一条命令，每条都带 purpose。目的说不清的命令不要发。
3. 命令被拒绝时，你会收到拒绝原因。不要原样重试同一条命令 —— 换一条路径，或者用 finish 如实说明被什么卡住。
4. 同一个错误不要重复犯三次。两次修不好就换思路，或者停下来问。
5. 你的输出会进事件流，用户会看。不要输出无意义的客套话。

关于以 \`mcp__\` 开头的工具：
- 它们来自用户配置的外部 MCP server，不是内置能力。参数严格按各自的 inputSchema 给，不要猜字段名。
- 它们和 shell 命令走**同一条审批链**：同样会弹卡片、同样可能被拒。被拒就换办法，不要重复试。
- 一条 shell 命令就能说清的事，不要绕道 MCP；反过来，MCP 有专用工具时也别硬用 shell 模拟。

关于诚实：
- 你没有做过的事，不要说做了。命令没跑成功，不要说成功了。
- 证据只能来自工具返回的内容。不要凭印象编造文件内容、测试结果、版本号。
- 完不成就是完不成，如实报 unmet 比假装完成有用得多。`
