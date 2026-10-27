/**
 * 目标契约 —— 数据形状
 *
 * IC-07 裁决为**硬闸门**：契约未确认，不得执行任何工具。
 * 所以契约必须是结构化的、能被界面逐条展示、能被用户逐条否掉的东西，
 * 而不是一段「我理解你想……」的自然语言。
 */

export interface GoalContract {
  /** 一句话目标 —— 与用户原话的差异必须能被看出来 */
  objective: string
  /** 可验证的成功判据，每条都应是「能判定真假」的陈述 */
  successCriteria: string[]
  /** 明确不做的事 —— 防止范围自行膨胀 */
  nonGoals: string[]
  /** 依赖的假设；假设不成立时判据可能失效，用户看得到才能纠正 */
  assumptions: string[]
  /** 已知风险 */
  risks: string[]
  /** 需要用户澄清的问题；非空表示契约不完整，不应放行执行 */
  needsClarification: string[]
}

/**
 * 输入结构化的结论 —— 起草契约时一并产出，随契约一起给用户看。
 *
 * 为什么它要进契约而不是只留在日志里：澄清问题只有被用户看见才算问过。
 * 记在日志里的「我们认为这句话有 0.8 的歧义」对用户毫无意义 ——
 * 他需要的是界面上那三行问题，以及「为什么问这个」。
 */
export interface InputStructureHint {
  /** 0~1，越大越含糊。≥0.62 且有问题时判为需澄清（主干阈值） */
  ambiguityScore: number
  /** 最多 3 条 */
  questions: string[]
  /** pass-through（未触发）/ structured / clarification / fallback（失败回退） */
  mode: string
  /** 结构化后的执行输入；需澄清时为空 */
  executionInput: string
  tokensSpent: number
  durationMs: number
}

/** 契约草稿的生成结果 */
export type ContractDraft =
  | { status: 'drafted'; contract: GoalContract }
  /** 模型回复无法解析成契约 —— 如实报告，不硬凑一份 */
  | { status: 'unparseable'; reason: string; rawText: string }

export function isGoalContract(v: unknown): v is GoalContract {
  if (typeof v !== 'object' || v === null) return false
  const o = v as Record<string, unknown>
  const strArr = (x: unknown): x is string[] => Array.isArray(x) && x.every((i) => typeof i === 'string')
  return (
    typeof o['objective'] === 'string' &&
    o['objective'].trim().length > 0 &&
    strArr(o['successCriteria']) &&
    strArr(o['nonGoals']) &&
    strArr(o['assumptions']) &&
    strArr(o['risks']) &&
    strArr(o['needsClarification'])
  )
}

/** 从模型回复里抠 JSON：容忍 ```json 围栏与前后废话，但抠不出来就如实失败 */
export function extractContract(text: string): ContractDraft {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(text)
  const candidate = fenced ? fenced[1]! : text
  const start = candidate.indexOf('{')
  const end = candidate.lastIndexOf('}')
  if (start === -1 || end <= start) {
    return { status: 'unparseable', reason: '回复中找不到 JSON 对象', rawText: text.slice(0, 800) }
  }
  const slice = candidate.slice(start, end + 1)
  let parsed: unknown
  try {
    parsed = JSON.parse(slice)
  } catch (err) {
    return {
      status: 'unparseable',
      reason: `JSON 解析失败：${(err as Error).message}`,
      rawText: text.slice(0, 800)
    }
  }
  if (!isGoalContract(parsed)) {
    return { status: 'unparseable', reason: 'JSON 结构不符合契约字段定义', rawText: slice.slice(0, 800) }
  }
  return { status: 'drafted', contract: parsed }
}
