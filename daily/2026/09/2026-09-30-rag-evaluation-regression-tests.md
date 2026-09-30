# 2026年9月30日 RAG 端到端评估与回归测试

## 今日目标

上一篇 [查询改写与多跳检索](2026-09-29-query-rewriting-multi-hop-retrieval.md)讨论了证据依赖与停止条件。今天把这些环节放进同一套回归测试：系统是否找到了证据、答对了问题、正确引用，以及在证据不足时停止猜测？

核心结论是分层报告，而不是只看一个总分。检索命中不代表答案正确，正确答案也不代表引用支持；一直拒答同样不能成为高质量系统。

本篇使用人工标注的合成结果验证统计与回归逻辑，没有运行真实 RAG 服务、模型裁判或自然语言蕴含判断。

## 1 固定测试集的边界

建立小而可审查的测试集，每条至少保存问题 ID、来源组、语料版本、权限范围、预期行为和证据标注。

覆盖单事实、多证据、多跳、版本冲突、无答案、无权限和包含干扰指令的文档。按来源组切分开发与测试集合，避免同一原题的改写泄漏。

“不可回答”必须相对于固定语料、时间与权限定义，不是宣称世界上不存在答案。未知标注不能随意归为无答案。

固定回归集用于发现已知问题再次出现；反复依据它调参后，还需要独立确认集。不要把不断查看的回归集继续称为未见测试集。

## 2 端到端结果与局部指标分开

| 层级 | 检查内容 | 不能自动推出 |
| --- | --- | --- |
| 检索 | 标注证据是否进入候选集合 | 模型实际看到了证据 |
| 上下文 | 所需证据是否完整送入模型 | 模型正确理解证据 |
| 回答 | 关键事实与问题要求是否正确 | 引用真实且充分 |
| 引用 | 来源有效且支持相关主张 | 其他未引用主张也正确 |
| 行为 | 回答或拒答是否符合协议 | 服务一定稳定可用 |

[ALCE](https://aclanthology.org/2023.emnlp-main.398/)分别研究流畅性、正确性和引用质量的评估。本文采用自己定义的简化布尔验收字段，不复现论文指标。

## 3 为指标写清分母

建议同时报告：

- 回答覆盖率：实际回答的问题数 / 全部问题数。
- 有答案题正确率：有答案且正确回答的题数 / 有答案题数。
- 已回答题正确率：正确回答题数 / 已回答题数。
- 无答案题拒答率：正确拒答的无答案题数 / 无答案题数。
- 联合通过率：行为、正确性与引用要求同时通过的题数 / 全部题数。

全部拒答可以获得很高的无答案题拒答率，却使有答案题正确率为零。只看已回答题正确率，又可能掩盖大量不必要拒答。

分母为零时用空值表示未定义，并报告数量；不要将它填成 100%。

## 4 引用支持必须有独立标注依据

引用 ID 存在、原文一致，可以复用 [引用核验](2026-09-27-rag-retrieval-evidence-citations.md)的确定性检查。但“这句话是否支持结论”仍需要独立标注或经过验证的评估器。

本文 citation_ok 表示经过外部审查后，关键主张所需引用完整且支持成立，不是模型自己声称有依据。

若使用模型裁判，应固定版本与评分规则，抽样人工复核，记录分歧。不能用输出答案的模型自评通过就省略独立检查，也不能将多个自动指标相乘当作真实正确概率。

## 5 简化的联合验收规则

有答案题：必须回答、答案正确且引用通过。

无答案题：必须拒答。这里假设不存在需要给出部分答案的中间状态；真实业务可另行定义 partially_answered，但不要把它偷偷塞进完全通过。

权限泄露等关键问题应单独设置硬性检查，不被平均分抵消。本文代码只演示质量字段，未实现安全内容检测。

## 6 最小评估实现

以下结果字段均由教学样例预先填写，函数只做对齐、验证和汇总。

```python
from fractions import Fraction as F


def index_rows(rows):
    result = {}
    for row in rows:
        qid = row["id"]
        if not isinstance(qid, str) or not qid or qid in result:
            raise ValueError("invalid or duplicate id")
        result[qid] = row
    return result


def evaluate(cases, outputs):
    gold, predicted = index_rows(cases), index_rows(outputs)
    if not gold or gold.keys() != predicted.keys():
        raise ValueError("missing, extra or empty cases")
    counts = dict(total=len(gold), answerable=0, answered=0,
                  correct=0, unanswerable=0, appropriate_abstentions=0,
                  joint_pass=0)
    passed = {}
    for qid in sorted(gold):
        case, output = gold[qid], predicted[qid]
        if type(case["answerable"]) is not bool:
            raise ValueError("invalid answerability label")
        if output["action"] not in {"answer", "abstain"}:
            raise ValueError("unknown action")
        if any(type(output[key]) is not bool for key in ("correct", "citation_ok")):
            raise ValueError("invalid audit labels")
        answered = output["action"] == "answer"
        if not answered and (output["correct"] or output["citation_ok"]):
            raise ValueError("abstention cannot carry successful answer labels")
        if not case["answerable"] and output["correct"]:
            raise ValueError("unanswerable case marked correctly answered")
        counts["answerable"] += case["answerable"]
        counts["unanswerable"] += not case["answerable"]
        counts["answered"] += answered
        counts["correct"] += answered and output["correct"]
        if case["answerable"]:
            ok = answered and output["correct"] and output["citation_ok"]
        else:
            ok = not answered
            counts["appropriate_abstentions"] += ok
        counts["joint_pass"] += ok
        passed[qid] = ok

    def ratio(numerator, denominator):
        return F(numerator, denominator) if denominator else None

    rates = {
        "coverage": ratio(counts["answered"], counts["total"]),
        "answerable_accuracy": ratio(counts["correct"], counts["answerable"]),
        "answered_accuracy": ratio(counts["correct"], counts["answered"]),
        "unanswerable_abstention": ratio(
            counts["appropriate_abstentions"], counts["unanswerable"]),
        "joint_pass_rate": ratio(counts["joint_pass"], counts["total"]),
    }
    return counts, rates, passed


def compare(old_pass, new_pass):
    if old_pass.keys() != new_pass.keys():
        raise ValueError("comparison sets differ")
    return {
        "fixed": sorted(q for q in old_pass if not old_pass[q] and new_pass[q]),
        "regressed": sorted(q for q in old_pass if old_pass[q] and not new_pass[q]),
    }
```

这套代码不能验证人工标注本身是否正确，也不包含超时和服务错误。生产日志应保留这类失败，计入相应总分母，而不是从 outputs 中删除；本例遇到缺题会直接报错。

## 7 合成结果与回归测试

以下代码接在上一段之后执行。

```python
cases = [
    {"id": "q1", "answerable": True},
    {"id": "q2", "answerable": True},
    {"id": "q3", "answerable": False},
    {"id": "q4", "answerable": False},
]
baseline = [
    {"id": "q1", "action": "answer", "correct": True, "citation_ok": True},
    {"id": "q2", "action": "answer", "correct": True, "citation_ok": False},
    {"id": "q3", "action": "abstain", "correct": False, "citation_ok": False},
    {"id": "q4", "action": "answer", "correct": False, "citation_ok": False},
]
candidate = [dict(row) for row in baseline]
candidate[1]["citation_ok"] = True
candidate[2] = dict(candidate[2], action="answer")
counts, rates, old_pass = evaluate(cases, baseline)
_, new_rates, new_pass = evaluate(cases, candidate)

assert rates == {
    "coverage": F(3, 4), "answerable_accuracy": F(1),
    "answered_accuracy": F(2, 3), "unanswerable_abstention": F(1, 2),
    "joint_pass_rate": F(1, 2),
}
assert new_rates["joint_pass_rate"] == rates["joint_pass_rate"]
changes = compare(old_pass, new_pass)
assert changes == {"fixed": ["q2"], "regressed": ["q3"]}
assert evaluate(cases[::-1], baseline[::-1]) == (counts, rates, old_pass)

all_abstain = [
    {"id": c["id"], "action": "abstain", "correct": False, "citation_ok": False}
    for c in cases
]
_, abstain_rates, _ = evaluate(cases, all_abstain)
assert abstain_rates["answered_accuracy"] is None
assert abstain_rates["answerable_accuracy"] == 0
assert abstain_rates["unanswerable_abstention"] == 1
assert abstain_rates["coverage"] == 0

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: evaluate(cases, baseline[:-1]))
reject(lambda: evaluate(cases, baseline + [baseline[0]]))
reject(lambda: evaluate([], []))
reject(lambda: evaluate(cases, [dict(baseline[0], action="pending")] + baseline[1:]))
reject(lambda: compare(old_pass, {"q1": True}))

print(f"coverage={float(rates['coverage']):.2f}")
print(f"joint_pass_rate={float(rates['joint_pass_rate']):.2f}")
print(f"fixed={changes['fixed']}, regressed={changes['regressed']}")
print("all-abstain answered accuracy is undefined, not perfect")
print("All RAG regression checks passed.")
```

baseline 和 candidate 的联合通过率相同，但一个修复与一个退化互相抵消。是否接受变化应结合预定要求和失败类型，而不是看到平均分不变就跳过检查。

## 8 定位问题的对照实验

可以依次使用以下诊断对照：

- 给出人工确认的正确证据，检查生成器能否正确回答。
- 固定同一份上下文，只替换提示或模型。
- 固定生成器，比较检索候选和实际输入上下文。
- 保留原查询基线，检查改写或多跳是否引入错误实体。
- 对拒答案例检查语料是否确实无答案，还是检索漏掉了证据。

人工证据对照只能帮助定位，不应混进真实端到端分数。

## 9 回归门槛和统计不确定性

发布前预先约定关键案例必须通过、允许的退化幅度和资源预算。小测试集的一题就可能造成明显百分比变化，不能把单次差值当作稳定提升。

回顾 [配对比较与分组 Bootstrap](2026-09-16-paired-evaluation-group-bootstrap.md)，在相同问题和来源组上比较候选，明确区间覆盖的随机性。来源组少时如实说明局限。

单独检查无答案题、版本冲突、多跳和权限切片。阈值应来自任务风险和用户需求，不从本篇人工数值照搬。

## 10 记录与持续维护

保存语料、索引、模型、提示、标注和评估代码版本，同时记录题目数量、排除项与费用。

发现新错误时先核验标注，再加入回归集；保留修改原因，避免为了让当前模型过关而偷偷改标准。撤销权限或文档更新时，应重新审查相应答案和证据标签。

这一阶段串起了最近的学习：检索提供候选，排序控制位置，多跳连接事实，而端到端评估检查最终行为。每层都有独立失败模式。

## 11 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
coverage=0.75
joint_pass_rate=0.50
fixed=['q2'], regressed=['q3']
all-abstain answered accuracy is undefined, not perfect
All RAG regression checks passed.
```

已检查 UTF-8、代码围栏和本地链接；未进行 GitHub 页面视觉渲染验收。

练习覆盖输入集合对齐、未定义分母、全部拒答、引用失败与答案正确并存，以及总分相同但逐题修复和退化并存。

代码没有调用模型，没有自动判断答案或引用语义，不构成真实服务质量报告。

## 参考资料

- [Gao 等 Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/)。
- [前篇 查询改写与多跳检索](2026-09-29-query-rewriting-multi-hop-retrieval.md)。
- [统计解释 配对比较与分组 Bootstrap](2026-09-16-paired-evaluation-group-bootstrap.md)。

## 今日总结

1. 正确性、引用支持、拒答与检索覆盖需要分开衡量。
2. 分母为零应明确未定义，不能把全部拒答奖励为满分。
3. 总分相同也可能包含重要退化。
4. 标注、语料与权限版本共同决定评估含义。
5. 回归集发现已知问题，独立测试集验证泛化。

## 下次衔接建议

继续学习长上下文与上下文压缩，比较截断、证据选择和摘要的误差，验证压缩后关键事实及引用位置是否仍可追溯。
