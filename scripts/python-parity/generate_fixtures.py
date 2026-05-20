"""
OpenINTJ Python v2 ↔ TS 行为对齐测试 fixture 生成器
====================================================

读取仓库根目录冻结的 Python v2.0 实现，在固定输入上跑核心组件，
输出 JSON fixture 给 TS 端 vitest 加载并断言等价。

约束（详见 docs/architecture/python-reference.md）：
- Python v2 已冻结（tag v2.0-python-reference）；本脚本只读，不修改 Python 代码。
- TS 端基于这些 fixture 写 parity spec；偏差在 docs/architecture/phase3-6-parity-tests.md 中显式记录。

用法：
    python scripts/python-parity/generate_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# 让 Python 能 import 仓库根目录下的 framework_core / memory_plane / ...
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from control_plane import GoalParser, Planner  # noqa: E402
from execution_plane import (  # noqa: E402
    Executor,
    ExecutionMode,
    Step,
    StepState,
    StepStateMachine,
)
from framework_core import (  # noqa: E402
    AgentError,
    ContextBudget,
    LODLevel,
    MemoryFragment,
    ShaderConfig,
    ShaderMode,
    TaskType,
)
from memory_plane import (  # noqa: E402
    MemoryRetriever,
    MemoryStore,
    cosine_similarity,
    simple_embedding,
)

OUT = {
    "core": REPO_ROOT / "ts/packages/core/__tests__/parity/fixtures/python-v2.json",
    "control": REPO_ROOT
    / "ts/packages/planes/control/__tests__/parity/fixtures/python-v2.json",
    "execution": REPO_ROOT
    / "ts/packages/planes/execution/__tests__/parity/fixtures/python-v2.json",
    "memory": REPO_ROOT
    / "ts/packages/planes/memory/__tests__/parity/fixtures/python-v2.json",
}


# ============================================================
# 1. core slice —— SimpleEmbedder / cosine / decayImportance
# ============================================================

EMBEDDING_INPUTS = [
    ("", 8),
    ("", 64),
    ("a", 8),
    ("a", 64),
    ("hello world", 8),
    ("hello world", 64),
    ("hello world", 128),
    ("The quick brown fox jumps over the lazy dog", 64),
    # CJK / 中英混杂
    ("你好，世界", 64),
    ("写一个 user.py 文件", 64),
    # 长字符串
    ("openintj " * 32, 64),
]

COSINE_CASES = [
    {"a": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]},  # self → 1
    {"a": [1.0, 0.0, 0.0], "b": [-1.0, 0.0, 0.0]},  # opposite → -1
    {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]},  # orthogonal → 0
    {"a": [3.0, 4.0], "b": [4.0, 3.0]},  # 24/25
    {"a": [0.0, 0.0, 0.0], "b": [1.0, 1.0, 1.0]},  # zero → 0
    {"a": [], "b": []},  # empty → 0
    {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]},  # diff length → 0
]

# decayImportance(importance, halfLifeHours, ageSeconds) =
#   importance * exp(-ln2 * (ageSeconds/3600) / halfLifeHours)
# Python 的 MemoryFragment.decay_importance 用 time.time() 当 now；
# 我们模拟一个 fragment 把 timestamp = now - ageSeconds 这种方式做测试。
DECAY_CASES = [
    {"importance": 1.0, "halfLifeHours": 24.0, "ageSeconds": 0.0},
    {"importance": 1.0, "halfLifeHours": 24.0, "ageSeconds": 3600.0 * 24.0},  # 1 半衰期 → 0.5
    {"importance": 0.5, "halfLifeHours": 1.0, "ageSeconds": 3600.0 * 1.0},  # 0.5 → 0.25
    {"importance": 0.7, "halfLifeHours": 24.0, "ageSeconds": 3600.0 * 12.0},  # 半半衰期 → 0.7*sqrt(0.5)
    {"importance": 0.3, "halfLifeHours": 12.0, "ageSeconds": 3600.0 * 48.0},  # 4 半衰期 → 0.3/16
]


def gen_core() -> dict[str, Any]:
    embeddings = []
    for text, dim in EMBEDDING_INPUTS:
        embeddings.append(
            {
                "input": text,
                "dim": dim,
                "vector": simple_embedding(text, dim),
            }
        )

    cosines = []
    for case in COSINE_CASES:
        cosines.append({**case, "expected": cosine_similarity(case["a"], case["b"])})

    decays = []
    import math

    for case in DECAY_CASES:
        # Python: decay = importance * exp(-0.693 * age_hours / half_life)
        age_h = case["ageSeconds"] / 3600.0
        expected = case["importance"] * math.exp(
            -0.693 * age_h / case["halfLifeHours"]
        )
        decays.append({**case, "expected": expected})

    return {
        "schemaVersion": 1,
        "generatedFrom": "memory_plane.simple_embedding / cosine_similarity, "
        "framework_core.MemoryFragment.decay_importance",
        "embeddings": embeddings,
        "cosineSimilarities": cosines,
        "decayImportance": decays,
    }


# ============================================================
# 2. control slice —— GoalParser / Planner
# ============================================================

GOAL_INPUTS: list[dict[str, Any]] = [
    {"input": "你好", "taskType": TaskType.GENERAL_CHAT.value},
    {"input": "创建一个用户登录页面", "taskType": TaskType.CODE_GENERATION.value},
    {"input": "生成测试数据", "taskType": TaskType.GENERAL_CHAT.value},
    {"input": "修改这个 bug", "taskType": TaskType.GENERAL_CHAT.value},
    {"input": "修复 login 路由", "taskType": TaskType.CODE_GENERATION.value},
    {"input": "删除用户 alice", "taskType": TaskType.GENERAL_CHAT.value},
    {"input": "查询数据库状态", "taskType": TaskType.ANALYSIS.value},
    {"input": "搜索关键字", "taskType": TaskType.GENERAL_CHAT.value},
    {"input": "执行部署脚本", "taskType": TaskType.GENERAL_CHAT.value},
    {"input": "规划下个季度任务", "taskType": TaskType.PLANNING.value},
    {"input": "设计新架构", "taskType": TaskType.PLANNING.value},
    # entity 抓取
    {
        "input": '创建一个 "user.py" 文件并实现 "main()" 函数',
        "taskType": TaskType.CODE_GENERATION.value,
    },
    {
        "input": "Please 'fix' the bug urgently",
        "taskType": TaskType.QUICK_RESPONSE.value,
    },
    # priority 升压
    {"input": "asap deploy this", "taskType": TaskType.GENERAL_CHAT.value},
    {"input": "立即停止服务", "taskType": TaskType.GENERAL_CHAT.value},
    # 默认 general
    {"input": "天气怎么样", "taskType": TaskType.GENERAL_CHAT.value},
]


# Python Planner 只覆盖 create / modify / query / plan / general；
# delete / execute 在 Python 端落回 general 分支。
# TS 端在 delete / execute 上做了扩展（specialized templates），
# parity 测试只跑公共 intent。
PARITY_PLAN_INTENTS = ["create", "modify", "query", "plan", "general"]


def _plan_steps_for_intent(intent: str) -> list[dict[str, Any]]:
    """造一个最小 ParsedGoal 让 Planner 生成对应 intent 的 PlanGraph，返回纯 steps。"""
    from control_plane import ParsedGoal

    goal = ParsedGoal(raw_input="_parity_", intent=intent)
    plan = Planner().create_plan(goal)
    return [
        {
            "stepId": s.step_id,
            "action": s.action,
            "description": s.description,
            "dependencies": list(s.dependencies),
            "status": s.status,
        }
        for s in plan.steps
    ]


def gen_control() -> dict[str, Any]:
    parser = GoalParser()
    parsed = []
    for case in GOAL_INPUTS:
        g = parser.parse(case["input"], TaskType(case["taskType"]))
        parsed.append(
            {
                "input": case["input"],
                "taskType": case["taskType"],
                "expected": {
                    "intent": g.intent,
                    "entities": list(g.entities),
                    "priority": g.priority,
                },
            }
        )

    plans = []
    for intent in PARITY_PLAN_INTENTS:
        plans.append({"intent": intent, "expected": _plan_steps_for_intent(intent)})

    return {
        "schemaVersion": 1,
        "generatedFrom": "control_plane.GoalParser / Planner (v2 frozen)",
        "notes": {
            "plannerSharedIntents": PARITY_PLAN_INTENTS,
            "plannerDivergence": [
                "Python Planner 把 delete/execute 都落回 general 分支；",
                "TS Planner 给 delete/execute 提供了专门的 3-step 模板，",
                "因此 parity 测试只对齐公共 intent；TS 扩展行为在自家单测里验证。",
            ],
        },
        "parsedGoals": parsed,
        "plans": plans,
    }


# ============================================================
# 3. execution slice —— StepStateMachine / Executor 简单序列
# ============================================================

TRANSITION_CASES = [
    {"from": "pending", "to": "ready"},
    {"from": "pending", "to": "skipped"},
    {"from": "pending", "to": "running"},  # illegal
    {"from": "ready", "to": "running"},
    {"from": "ready", "to": "skipped"},
    {"from": "ready", "to": "completed"},  # illegal
    {"from": "running", "to": "completed"},
    {"from": "running", "to": "failed"},
    {"from": "running", "to": "waiting_approval"},
    {"from": "running", "to": "skipped"},  # illegal
    {"from": "failed", "to": "ready"},  # 重试
    {"from": "failed", "to": "completed"},  # illegal
    {"from": "waiting_approval", "to": "running"},
    {"from": "waiting_approval", "to": "skipped"},
    {"from": "waiting_approval", "to": "completed"},  # illegal
]


def _check_transition(src: str, tgt: str) -> dict[str, Any]:
    sm = StepStateMachine()
    step = Step(step_id=f"t-{src}-{tgt}", action="noop", state=StepState(src))
    try:
        event = sm.transition(step, StepState(tgt))
        return {
            "from": src,
            "to": tgt,
            "allowed": True,
            "eventType": event.event_type.value,
            "eventSource": event.source,
            "eventPayload": {
                "step_id": event.payload["step_id"],
                "from": event.payload["from"],
                "to": event.payload["to"],
            },
        }
    except AgentError as e:
        return {
            "from": src,
            "to": tgt,
            "allowed": False,
            "errorCode": e.code.value,
        }


def gen_execution() -> dict[str, Any]:
    transitions = [_check_transition(c["from"], c["to"]) for c in TRANSITION_CASES]

    # 顺序执行三步骤，全部使用未注册 action（走默认分支 → 全部 completed）
    seq_steps = [
        Step(step_id="s1", action="think"),
        Step(step_id="s2", action="act"),
        Step(step_id="s3", action="respond"),
    ]
    exec1 = Executor()
    result1 = exec1.execute(seq_steps, mode=ExecutionMode.SEQUENTIAL)

    sequential = {
        "mode": "sequential",
        "steps": [{"stepId": s.step_id, "action": s.action} for s in seq_steps],
        "expected": {
            "success": result1.success,
            "finishedSteps": list(result1.finished_steps),
            "failedSteps": list(result1.failed_steps),
            "eventTrace": [
                {
                    "type": ev.event_type.value,
                    "stepId": ev.payload.get("step_id"),
                    "from": ev.payload.get("from"),
                    "to": ev.payload.get("to"),
                }
                for ev in result1.events
            ],
        },
    }

    # 并行：三步全 completed
    par_steps = [
        Step(step_id="p1", action="lookup"),
        Step(step_id="p2", action="lookup"),
        Step(step_id="p3", action="lookup"),
    ]
    exec2 = Executor()
    result2 = exec2.execute(par_steps, mode=ExecutionMode.PARALLEL)
    parallel = {
        "mode": "parallel",
        "steps": [{"stepId": s.step_id, "action": s.action} for s in par_steps],
        "expected": {
            "success": result2.success,
            "finishedSteps": list(result2.finished_steps),
            "failedSteps": list(result2.failed_steps),
            "eventTrace": [
                {
                    "type": ev.event_type.value,
                    "stepId": ev.payload.get("step_id"),
                    "from": ev.payload.get("from"),
                    "to": ev.payload.get("to"),
                }
                for ev in result2.events
            ],
        },
    }

    return {
        "schemaVersion": 1,
        "generatedFrom": "execution_plane.StepStateMachine / Executor (v2 frozen)",
        "notes": {
            "knownDivergence": [
                "Python Executor 在失败分支会触发 FAILED→FAILED（非法转换会抛错，",
                "Python 旧实现的 if/transition 双跳被记录为 v2 已知 bug；TS 已修复）。",
                "本 fixture 只跑全成功路径，避免触发该已知差异。",
            ]
        },
        "transitions": transitions,
        "executions": [sequential, parallel],
    }


# ============================================================
# 4. memory slice —— MemoryStore / MemoryRetriever 评分组件
# ============================================================

# 对齐口径：
# Python MemoryRetriever 的 recency_score 用 max_summary_length / 10 当半衰期（已知 bug）。
# TS MemoryRetriever 用 ShaderConfig.recencyHalfLifeHours（默认 24）作为半衰期。
# 为了让分数严格相等，本 fixture 把 Python max_summary_length 设为 240
# （→ Python 半衰期 = 24h），与 TS 默认对齐。
# 所有 fragment timestamp 都被显式设为 (test_now - ageSeconds)；test_now 取
# 调用 retrieve() 当时的 time.time()，并在 fixture 中导出 ageSeconds 让 TS 端复算。

STORE_OVERFLOW_CASES = [
    {
        "maxShortTerm": 3,
        "adds": ["alpha", "beta", "gamma", "delta", "epsilon"],
    },
    {
        "maxShortTerm": 5,
        "adds": ["a", "b", "c"],
    },
]


def gen_memory() -> dict[str, Any]:
    # 4.1 store overflow
    store_overflows = []
    for case in STORE_OVERFLOW_CASES:
        store = MemoryStore(max_short_term=case["maxShortTerm"])
        for content in case["adds"]:
            store.add_short_term(content)
        store_overflows.append(
            {
                "maxShortTerm": case["maxShortTerm"],
                "adds": list(case["adds"]),
                "expected": {
                    "shortTerm": [f.content for f in store.short_term],
                    "longTerm": [f.content for f in store.long_term],
                },
            }
        )

    # 4.2 retrieval score 组件（vec / keyword / decay）
    # 用固定 query + 固定 fragment 集合 + 固定 ageSeconds 数组；
    # 关键决定：Python max_summary_length=240 让 recency 半衰期 = 24h，与 TS 默认对齐。

    query = "hello world deploy"
    frags_spec = [
        # (content, importance, ageSeconds, taskTags)
        ("hello world", 0.8, 0.0, ["general_chat"]),
        ("openintj deploy script", 0.6, 3600.0 * 12, []),  # 12h ago
        ("unrelated noise", 0.2, 0.0, []),
        ("the quick brown fox", 0.5, 3600.0 * 24, []),  # 24h → decay 0.5
        ("hello deploy world", 0.9, 3600.0 * 1, ["code_generation"]),
    ]

    # 关键：MemoryRetriever.retrieve 用 time.time() 做"现在"。
    # 我们在调用前固定 test_now，然后用 test_now - age 设置 fragment.timestamp，
    # 调用过程中 time.time() 与 test_now 的偏差 < 几 ms → 在 1e-3 容差内对齐。
    shader = ShaderConfig(
        max_summary_length=240,  # → recency 半衰期 = 24h
        recency_weight=0.4,
        relevance_weight=0.4,
        importance_weight=0.2,
        importance_threshold=0.0,
        max_fragments_per_query=10,
    )
    store = MemoryStore()
    import time as _time

    test_now = _time.time()

    for content, imp, age, tags in frags_spec:
        f = store.add_short_term(content, importance=imp, task_tags=list(tags))
        f.timestamp = test_now - age

    retriever = MemoryRetriever(store=store, shader_config=shader)
    ranked = retriever.retrieve(query, top_k=10)

    # 把每个 ranked 结果的"score 组件"分解出来，方便 TS 端逐项断言
    # Python 评分公式（v2 frozen）：
    #   score = relevance_weight * vec
    #         + importance_weight * recency
    #         + recency_weight * keyword
    #   recency = fragment.decay_importance(max_summary_length / 10)  ← 半衰期 24h
    q_emb = simple_embedding(query, store.embedding_dim)
    q_kw = set(query.lower().split())
    results = []
    for frag, score in ranked:
        vec = cosine_similarity(q_emb, frag.embedding)
        content_words = set(frag.content.lower().split())
        overlap = len(q_kw & content_words)
        keyword = overlap / max(1, len(q_kw))
        recency = frag.decay_importance(shader.max_summary_length / 10)
        results.append(
            {
                "content": frag.content,
                "score": score,
                "components": {
                    "relevance": vec,
                    "keyword": keyword,
                    "recency": recency,
                },
                "importance": frag.importance,
                "taskTags": list(frag.task_tags),
            }
        )

    retrieval = {
        "query": query,
        "testNow": test_now,
        "shader": {
            "recencyHalfLifeHours": shader.max_summary_length / 10,
            "relevanceWeight": shader.relevance_weight,
            "recencyWeight": shader.recency_weight,
            "importanceWeight": shader.importance_weight,
        },
        "fragments": [
            {
                "content": c,
                "importance": imp,
                "ageSeconds": age,
                "taskTags": list(tags),
            }
            for c, imp, age, tags in frags_spec
        ],
        "expected": results,
    }

    return {
        "schemaVersion": 1,
        "generatedFrom": "memory_plane.MemoryStore / MemoryRetriever (v2 frozen)",
        "notes": {
            "halfLifeAlignment": (
                "Python MemoryRetriever 用 max_summary_length/10 当半衰期 (已知 bug)。"
                "本 fixture 把 max_summary_length 设为 240 → 半衰期 24h，"
                "与 TS ShaderConfig.recencyHalfLifeHours 默认值对齐。"
            ),
            "scoringFormula": (
                "score = relevance_weight*vec + importance_weight*recency + "
                "recency_weight*keyword（注意 Python 把 'recency_weight' 当 keyword 权重，"
                "把 'importance_weight' 当 decayed-importance 权重；TS 一致复制。）"
            ),
        },
        "storeOverflows": store_overflows,
        "retrieval": retrieval,
    }


# ============================================================
# 主入口
# ============================================================


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main() -> None:
    payloads = {
        "core": gen_core(),
        "control": gen_control(),
        "execution": gen_execution(),
        "memory": gen_memory(),
    }
    for slice_name, path in OUT.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_jsonable(payloads[slice_name]), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        rel = os.path.relpath(path, REPO_ROOT)
        print(f"[write] {rel}")


if __name__ == "__main__":
    main()
