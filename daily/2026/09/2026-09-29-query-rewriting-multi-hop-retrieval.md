# 2026-09-29：查询改写与多跳检索——意图保留、证据依赖和停止条件

## 今日目标

上一篇 [检索排序](2026-09-28-retrieval-ranking-bm25-rrf-evaluation.md)比较了候选融合与排序指标。今天讨论更靠前的问题：用户的问题应该怎样转成检索请求？如果第二次检索依赖第一次发现的实体，又如何避免把猜测传播成事实？

完成后应能回答：

1. 查询改写、扩展与问题分解有什么区别？
2. 哪些原始约束不能在改写时丢失？
3. 多跳检索如何记录中间实体的来源？
4. 证据缺失、冲突与预算耗尽应如何区分？
5. 如何公平比较单次检索与多次检索？

本篇使用人工构造的结构化事实表模拟两跳查询。没有运行改写模型、自然语言事实抽取、向量检索或真实外部知识库。

---

## 1. 三种操作，不是同一件事

- 改写：把代词、口语表达或上下文依赖转成可独立检索的问题。
- 扩展：保留原查询，同时加入同义词、别名或其他检索表达。
- 分解：将复合问题拆成多个具有依赖关系的子问题。

例如“它的负责人在哪个团队？”需要先明确“它”对应哪个项目。若上下文存在两个候选项目，应先澄清，而不是选择一个后假装问题本来如此。

[Query Rewriting in Retrieval-Augmented Large Language Models](https://aclanthology.org/2023.emnlp-main.322/)研究了 Rewrite-Retrieve-Read 流程。本文借此理解查询阶段可以独立优化，不复现论文训练方法。

## 2. 建立不能丢失的约束清单

改写前保留：

- 实体及其消歧信息。
- 时间点、版本、地区和适用对象。
- 否定、排除条件与比较方向。
- 输出目标，例如询问人、地点、原因或数值。
- 允许的数据源和权限范围。

“2024 年不支持离线模式的产品”不能改成“支持离线模式的产品”；“当时的负责人”不能偷换成“现任负责人”。

字符串里保留关键词只是必要性很弱的检查，不足以证明语义等价。需要抽样审查与成对测试，并保留原始查询用于诊断。

## 3. 不要把可能的答案写成已知事实

模型可以提出待验证假设，但检索查询中加入未经证实的人名、年份或结论，会把搜索引向支持该猜测的材料。

可以分别记录：

- 用户明确提供的条件。
- 已检索并核验的事实。
- 仍待验证的假设。

后续查询应说明绑定实体来自哪条证据。生成的摘要或解释不是新的独立来源，多轮重复也不会提高其真实性。

## 4. 多跳检索的依赖

教学问题：“Project Cedar 的负责人所在团队是什么？”

第一跳：查询项目的负责人，得到实体 Person Lin。

第二跳：以该实体查询所属团队，得到 Team Amber。

最终结论依赖两条事实。即使第二条真实，第一跳把负责人识别错了，整体答案仍然错误。

[IRCoT](https://aclanthology.org/2023.acl-long.557/)研究交错进行检索与多步问题处理。本文只演示显式事实依赖，不实现论文的生成流程。

## 5. 实体连接比字符串拼接更严格

同名人物不能自动视为同一人。生产系统应结合稳定 ID、组织、时间和来源信息消歧。

不同时间的两条事实也不能随意连接。例如“去年项目负责人是 Lin”和“今年 Lin 属于 Amber”不直接证明去年项目负责人所属团队是 Amber。

今天的数据统一采用同一合成快照，并使用稳定实体 ID。真实系统还需要时间有效区间和版本一致性检查。

## 6. 先定义停止原因

| 状态 | 含义 | 不应声称 |
| --- | --- | --- |
| complete | 所需关系均由当前证据支持 | 自动证明来源真实或仍有效 |
| missing | 当前查询没有发现所需关系 | 世界上不存在该关系 |
| conflict | 找到不同对象，无法唯一绑定 | 随意挑一个就能继续 |
| budget_exhausted | 检索次数达到上限 | 已经确认没有答案 |
| cycle | 重复进入同一子问题 | 多循环几次就增加了证据 |

查询去重应与索引快照、权限范围等条件关联。同一句文本在新版本语料上重查可能合理，不能永久禁止。

## 7. 最小实现：带预算的关系路径

事实记录包含 fact_id、subject、predicate、object。每一步先检查预算，再在允许范围内检索；发现多个不同 object 时停止。

代码中“关系已确认”只是相对于预先构造的事实表，不包含自然语言蕴含或来源可信度判断。

```python
from copy import deepcopy


def follow_path(start, relations, facts, allowed_fact_ids, max_calls):
    if not isinstance(start, str) or not start:
        raise ValueError("invalid start entity")
    if type(max_calls) is not int or max_calls < 0:
        raise ValueError("invalid budget")
    if not relations or not all(isinstance(x, str) and x for x in relations):
        raise ValueError("need a nonempty relation path")
    ids = [fact["fact_id"] for fact in facts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fact id")
    for fact in facts:
        if not all(isinstance(fact[key], str) and fact[key]
                   for key in ("fact_id", "subject", "predicate", "object")):
            raise ValueError("invalid fact")

    entity, calls, trace, seen = start, 0, [], set()

    def finish(status):
        return {
            "status": status,
            "answer": entity if status == "complete" else None,
            "calls": calls, "trace": deepcopy(trace),
        }

    for relation in relations:
        query = (entity, relation)
        if query in seen:
            return finish("cycle")
        if calls >= max_calls:
            return finish("budget_exhausted")
        seen.add(query)
        calls += 1
        matches = [
            fact for fact in facts
            if fact["fact_id"] in allowed_fact_ids
            and (fact["subject"], fact["predicate"]) == query
        ]
        matches.sort(key=lambda fact: fact["fact_id"])
        objects = sorted({fact["object"] for fact in matches})
        trace.append({
            "subject": entity, "predicate": relation,
            "evidence": [fact["fact_id"] for fact in matches],
            "objects": objects,
        })
        if not objects:
            return finish("missing")
        if len(objects) > 1:
            return finish("conflict")
        entity = objects[0]
    return finish("complete")
```

多个来源给出相同对象时，本例保留全部证据 ID，但不会把来源数量当作置信度。多个页面可能复制同一条错误信息。

这里的预算只计函数内检索次数。真实系统还应独立限制总时间、模型 token、候选量和费用；网络重试也不能变成不计费的“隐藏调用”。

## 8. 测试：完整路径与显式失败

以下代码接在上一段后运行。所有项目、人名与团队均为合成标识。

```python
facts = [
    dict(fact_id="f1", subject="project:cedar", predicate="owner",
         object="person:lin"),
    dict(fact_id="f2", subject="person:lin", predicate="team",
         object="team:amber"),
    dict(fact_id="f3", subject="person:other", predicate="team",
         object="team:blue"),
]
allowed = {"f1", "f2", "f3"}
path = ["owner", "team"]

complete = follow_path("project:cedar", path, facts, allowed, 2)
assert complete["status"] == "complete"
assert complete["answer"] == "team:amber"
assert complete["calls"] == 2
assert [step["evidence"] for step in complete["trace"]] == [["f1"], ["f2"]]
assert follow_path("project:cedar", path, facts[::-1], allowed, 2) == complete

limited = follow_path("project:cedar", path, facts, allowed, 1)
assert limited["status"] == "budget_exhausted"
assert limited["answer"] is None and limited["calls"] == 1
assert follow_path("project:cedar", path, facts, allowed, 0)["calls"] == 0

hidden = follow_path("project:cedar", path, facts, {"f1"}, 2)
assert hidden["status"] == "missing" and hidden["answer"] is None
assert hidden["trace"][-1]["evidence"] == []

conflicting = facts + [
    dict(fact_id="f4", subject="project:cedar", predicate="owner",
         object="person:other")
]
conflict = follow_path("project:cedar", path, conflicting, allowed | {"f4"}, 2)
assert conflict["status"] == "conflict" and conflict["calls"] == 1
assert conflict["answer"] is None

loop = [dict(fact_id="loop", subject="x", predicate="next", object="x")]
assert follow_path("x", ["next", "next"], loop, {"loop"}, 5)["status"] == "cycle"

# Repetition of an identical object is not an object conflict.
same = facts + [dict(facts[0], fact_id="f5")]
assert follow_path("project:cedar", path, same, allowed | {"f5"}, 2)["answer"] == "team:amber"

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: follow_path("project:cedar", path, facts + [facts[0]], allowed, 2))
reject(lambda: follow_path("project:cedar", path, facts, allowed, -1))
reject(lambda: follow_path("project:cedar", [], facts, allowed, 2))

print(f"status={complete['status']}, answer={complete['answer']}, calls=2")
print("evidence_chain=f1 -> f2")
print("missing, conflict, budget and cycle handled without guessed answers")
print("All multi-hop retrieval checks passed.")
```

这不是自动发现关系路径的算法。relations 由人预先给定；现实中提出子问题、抽取事实与确认连接都可能出错，需要分别评估。

## 9. 为什么不能无止境扩展查询

多生成几个查询可能增加覆盖，也可能重复召回同一批片段或引入噪声。

建议记录每轮新增的有效证据、重复比例和未解决子问题。没有新增 chunk 是一种启发式停止信号，不证明已经穷尽语料；可以在明确预算内尝试不同表达，再如实报告不足。

外部片段只能提供证据，不能通过“请继续搜索私有目录”等文本扩大权限。查询计划应保持原任务边界。

## 10. 如何公平比较单跳与多跳

固定语料快照、权限、问题集和最终上下文预算，比较：

1. 原始查询的一次检索。
2. 保留原查询并加入改写。
3. 多查询融合。
4. 带证据依赖的多跳检索。

除了最终正确率，还要报告调用次数、总延迟、模型 token、证据覆盖和拒答情况。给多跳更多资源时，应明确这是质量与成本的权衡，不是同预算胜出。

把桥接实体已知的对照与完整流程分开，可帮助判断问题出在第一跳实体发现，还是第二跳检索。不要将人工提供的中间答案混入真实评估。

## 11. 失败分析和日志

建议保存原始问题、改写候选、保留的约束、结构化子问题、检索参数、证据 ID 与版本，以及停止原因。

日志只需足够说明执行了哪些检索和证据如何连接，不必保存模型内部的私有推理过程。面向用户可提供简洁证据链和结论边界。

错误分析关注：

- 改写改变了实体、否定或时间条件。
- 第一跳答案仅是假设却被当作事实。
- 不同实体或时间的事实被错误连接。
- 预算结束却报告“没有答案”。
- 仅根据引用数量判断可信度。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
status=complete, answer=team:amber, calls=2
evidence_chain=f1 -> f2
missing, conflict, budget and cycle handled without guessed answers
All multi-hop retrieval checks passed.
```

已检查 UTF-8、代码围栏和本地链接；未进行 GitHub 页面视觉渲染验收。

示例验证两跳证据链、顺序稳定性、权限范围过滤、重复事实 ID 拒绝，以及缺失、冲突、预算耗尽与循环状态。

没有实现查询语义保持检查、事实抽取或自动规划。结构化事实中的正确连接也不自动证明真实世界答案正确。

## 参考资料

- [Ma 等：Query Rewriting in Retrieval-Augmented Large Language Models](https://aclanthology.org/2023.emnlp-main.322/)。
- [Trivedi 等：Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://aclanthology.org/2023.acl-long.557/)。
- [前篇：检索排序](2026-09-28-retrieval-ranking-bm25-rrf-evaluation.md)。

## 今日总结

1. 改写要保留原始意图，而不是提前填入猜测答案。
2. 多跳流程需要记录中间实体与来源证据的连接。
3. 权限、版本和时间约束应贯穿所有跳数。
4. 缺失、冲突和预算耗尽是不同状态。
5. 多次检索的质量收益应与成本、延迟一起评估。

## 下次衔接建议

继续学习 RAG 端到端评估与回归测试：结合检索覆盖、答案正确性、引用支持和拒答行为，建立小型固定测试集与错误分类。
