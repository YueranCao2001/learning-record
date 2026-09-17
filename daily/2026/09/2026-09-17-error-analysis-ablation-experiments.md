# 2026-09-17：错误分析与消融实验——从失败案例到可检验的改进假设

## 今日目标

上一篇 [评估结果的统计解释](2026-09-16-paired-evaluation-group-bootstrap.md)讨论了配对比较与不确定性。今天继续回答：发现模型有问题后，如何定位值得修改的环节，并验证修改是否真正有帮助？

完成后应能回答：

1. 失败现象、评分错误和根因有什么区别？
2. 如何同时报告修复和新增退化，而不是只展示成功案例？
3. 如何把错误分析转成单因素对照？
4. 两个改动各自有效，为什么合起来不一定更好？
5. 如何避免在发现问题的同一批题上循环证明自己？

本篇包含实验设计和纯 Python 合成数据练习。没有训练真实模型，没有生成真实模型评估结果，也不把示例数值当作能力提升证据。

---

## 1. 错误分析先检查测量工具

看到一条失败记录，先保留原始输出，再检查：

- 输入、参考答案、来源组与题目版本是否匹配。
- 是否发生截断、超时、空响应或服务错误。
- 答案抽取器是否漏掉实际存在的答案。
- 评分器是否使用正确的单位、容差和归一化规则。
- 人工复核是否同意自动评分。

例如模型输出正确的数值和单位，但解析器只接受纯数字，这首先是评分协议问题，不应立即推断“模型不会计算”。

若评分器修正，两个模型都必须按同一新版本重新评分。保留旧版结果与变更原因，不把测量规则变化包装成模型提升。

## 2. 分开记录现象、假设与证据

建议每条案例至少包含：

| 字段 | 作用 |
| --- | --- |
| prompt_id / source_group | 对齐题目、追溯来源及相关样本 |
| raw_output / score_version | 保留原始证据与评分版本 |
| observed_failure | 可观察的失败，例如格式不合法 |
| suspected_cause | 待验证原因，例如训练缺少格式约束 |
| review_status | 未复核、已确认、存在分歧 |
| proposed_test | 能区分原因的下一步实验 |

“答案错误”是观察；“推理能力不足”通常还只是解释。

失败可有多个标签，例如同时出现格式不合规与数值错误。若需要各类比例相加为 100%，必须另行定义互斥的主类别和优先级；不能把多标签计数直接当作互斥分布。

不确定时保留 unknown，不要强迫每个案例都有确定根因。

## 3. 抽样审查不能只挑最显眼的失败

建议同时查看四种配对结果：

- 两者都成功。
- baseline 失败、candidate 成功：修复。
- baseline 成功、candidate 失败：新增退化。
- 两者都失败：未解决问题。

成功案例也值得抽查，因为评分器可能误判成功。

在错误分析阶段，可以分层抽样，额外查看罕见但重要的失败；但这种审查样本不能未经加权就用于估计全量发生率。报告“审查了多少条”和“完整评估集有多少条”是两个不同信息。

模型身份可以在人工审查时隐藏，并随机安排呈现顺序，以减少对候选模型的期待影响判断。

## 4. 净提升会隐藏修复与退化

对固定的 N 道题、二元成功指标，令 F 是修复数量，R 是新增退化数量：

```math
\widehat{\Delta}_{\mathrm{micro}} = \frac{F-R}{N}
```

因此，两种候选都可能净提升 5 个百分点，但一个只修复少量问题，另一个同时修复和破坏了大量题目。上线风险与后续诊断重点可能不同。

这条恒等式只针对每题等权的二元指标。若沿用昨天的来源组等权 macro 指标，应先在每组内部计算，再对组平均；不能直接用全局 F、R 替代。

## 5. 切片需要固定分母和定义

可以按领域、长度、语言、来源或能力标签划分切片。但需要：

- 每个切片报告题目数与独立来源组数。
- baseline 与 candidate 使用完全相同的题目集合。
- 空切片标为无数据，不写成 0% 成功率。
- 多标签切片可能重叠，不能把它们的数量直接相加。
- 比较前固定的切片与看结果后发现的切片分开报告。

“baseline 答错的题”可以帮助描述修复率，但这是按结果筛选的集合，不能据此推断候选在一般题目上有同样幅度的提升。

发现某类退化之后，可以提出假设；要确认其普遍性，应在新的独立来源样本上验证。

## 6. 从错误到可证伪的假设

一个可操作的示例：

- 观察：需要严格 JSON 输出的开发题中，内容可能正确但格式频繁失败。
- 假设：训练中格式约束覆盖不足，是失败的一个原因。
- 改动 A：在固定训练 token 预算下，用经过审查的格式训练样本替换一部分原数据。
- 预测：未参与构造训练样本的格式测试集成功率提高。
- 反证信号：只在见过的模板上提高，或语义正确率明显下降。
- 保留能力：同时观察普通问答和不需要 JSON 的任务。

这项实验测试的是“数据替换方案”这个整体干预，不是严格证明内部的某个机制。替换数据还会改变内容分布，需要明确这一限制。

如果同时增加训练量、换模板、换评分器，就无法把分数变化归给格式数据。

## 7. 消融不是随意删除模块

常见的两类问题：

- 添加式：在固定 baseline 上加入 A，比较 A 与 baseline。
- 删除式：从完整系统去掉 A，比较完整系统与缺少 A 的系统。

两者回答的条件不同。完整系统中 A 的边际贡献，不一定等于 baseline 上 A 的贡献。

删除时还要注意：是否重新训练、是否保持相同搜索预算、是否让剩余模块合理配置。直接关闭一个模型依赖的组件造成接口损坏，不是有解释力的能力消融。

一次只改一个因素是容易审查的起点，但不能充分识别因素间的交互。

## 8. 两因素实验：单独有效，组合可能失效

令 A 是格式训练数据替换，B 是推理时增加格式提示。安排四个配置：

| 配置 | A | B | 用途 |
| --- | --- | --- | --- |
| 00 | 关闭 | 关闭 | baseline |
| 10 | 开启 | 关闭 | A 的单独影响 |
| 01 | 关闭 | 开启 | B 的单独影响 |
| 11 | 开启 | 开启 | 组合及交互 |

这里 B 只影响推理，所以可对同一训练产物分别运行两种提示；不能声称这必然需要四次独立训练。

在成功率差值尺度上定义差中之差：

```math
I = (s_{11}-s_{01})-(s_{10}-s_{00})
```

如果四个成功率分别为 0.50、0.60、0.55、0.52，则 A 在 B 关闭时提升 0.10，在 B 开启时反而下降 0.03，交互差为 -0.13。

这是描述性对比，不是显著性检验，也不证明机制。交互还依赖响应尺度；成功率尺度与 log-odds 尺度的交互不等价。

全因子设计覆盖因素水平的所有组合，可用于分析组合影响，参见 [NIST：Full factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm)。本文示例采用 0/1 条件下的差中之差，不应直接混同于其他编码下的回归交互系数。

## 9. 固定预算与记录资源

“公平预算”必须说明具体含义：

- 固定训练 token 数，不保证固定优化器步数。
- 固定步数，不保证不同长度数据有相同 token 数。
- 固定推理最大长度，不保证实际生成 token 数相同。
- 固定墙钟时间，还会受硬件、并发和实现影响。

根据研究问题预先选择主要约束，同时记录其他资源。比较训练方法时，还应考虑调参次数与失败试验，而不只是最终一次训练成本。

多个训练重复可按预先安排的 seed 配对；相同 seed 只是控制的一部分，不保证所有配置共享完全相同的随机轨迹。

## 10. 最小实现：配对转移与切片报告

下面对已经冻结评分规则的二元结果进行分析。元数据及评分版本的外层验证仍不可省略。

```python
from collections import Counter
from math import isclose


def index_rows(rows):
    out = {}
    for row in rows:
        pid, group = row["id"], row["group"]
        if not isinstance(pid, str) or not pid or pid in out:
            raise ValueError("invalid or duplicate id")
        if not isinstance(group, str) or not group:
            raise ValueError("invalid group")
        if type(row["ok"]) is not bool:
            raise ValueError("ok must be bool")
        tags = row["tags"]
        if not isinstance(tags, (tuple, list, set)):
            raise ValueError("tags must be a collection")
        if not all(isinstance(tag, str) and tag for tag in tags):
            raise ValueError("invalid tag")
        out[pid] = (group, row["ok"], frozenset(tags))
    if not out:
        raise ValueError("empty evaluation")
    return out


def pair_rows(baseline, candidate):
    b, c = index_rows(baseline), index_rows(candidate)
    if b.keys() != c.keys():
        raise ValueError("id sets differ")
    pairs = []
    for pid in sorted(b):
        gb, sb, tb = b[pid]
        gc, sc, tc = c[pid]
        if gb != gc or tb != tc:
            raise ValueError("group or slice metadata changed")
        pairs.append((pid, gb, sb, sc, tb))
    return pairs


def summarize(pairs, tag=None):
    selected = [r for r in pairs if tag is None or tag in r[4]]
    if not selected:
        return None
    transitions = Counter((r[2], r[3]) for r in selected)
    n = len(selected)
    fixed = transitions[(False, True)]
    regressed = transitions[(True, False)]
    baseline = sum(r[2] for r in selected) / n
    candidate = sum(r[3] for r in selected) / n
    delta = (fixed - regressed) / n
    assert isclose(candidate - baseline, delta, abs_tol=1e-12)
    return dict(
        n=n, groups=len({r[1] for r in selected}),
        baseline=baseline, candidate=candidate, delta=delta,
        fixed=fixed, regressed=regressed,
        both_ok=transitions[(True, True)],
        both_fail=transitions[(False, False)],
    )


def interaction(scores):
    if set(scores) != {"00", "10", "01", "11"}:
        raise ValueError("need all four cells")
    return (scores["11"] - scores["01"]) - (scores["10"] - scores["00"])
```

这里的切片报告采用 micro 平均，是为了展示修复与退化的恒等式。代码不计算置信区间；需要推断时，按昨天的原则保留来源组依赖，并在所有配置中共同重采样组索引。

## 11. 合成数据测试

以下代码接在上一段之后运行。

```python
def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


base = [True, False, True, False, True, False, False, True]
cand = [True, True, False, True, True, False, True, False]
b, c = [], []
for i, (left, right) in enumerate(zip(base, cand)):
    meta = dict(
        id=f"p{i}", group=f"g{i // 2}",
        tags=("json", "short") if i < 4 else ("plain",),
    )
    b.append(dict(meta, ok=left))
    c.append(dict(meta, ok=right))

pairs = pair_rows(b, c)
report = summarize(pairs)
assert report["n"] == 8 and report["groups"] == 4
assert report["fixed"] == 3 and report["regressed"] == 2
assert report["both_ok"] == 2 and report["both_fail"] == 1
assert isclose(report["delta"], 0.125)
assert summarize(pair_rows(b[::-1], c)) == report
assert summarize(pairs, "unseen") is None
assert summarize(pairs, "json")["n"] == 4
assert summarize(pairs, "short") == summarize(pairs, "json")
assert isclose(summarize(pair_rows(c, b))["delta"], -report["delta"])
assert summarize(pair_rows(b, b))["fixed"] == 0
assert summarize(pair_rows(b, b))["regressed"] == 0

reject(lambda: pair_rows(b + [b[0]], c))
reject(lambda: pair_rows(b, c[:-1]))
reject(lambda: pair_rows(b, [dict(c[0], group="wrong")] + c[1:]))
reject(lambda: pair_rows(b, [dict(c[0], tags=("new",))] + c[1:]))
reject(lambda: pair_rows(b, [dict(c[0], ok=1)] + c[1:]))
reject(lambda: pair_rows([], []))

scores = {"00": 0.50, "10": 0.60, "01": 0.55, "11": 0.52}
value = interaction(scores)
assert isclose(value, -0.13)
additive = {"00": 0.50, "10": 0.60, "01": 0.55, "11": 0.65}
assert isclose(interaction(additive), 0.0, abs_tol=1e-12)
reject(lambda: interaction({"00": 0.5}))
print(f"fixed={report['fixed']}, regressed={report['regressed']}, "
      f"micro_delta={report['delta']:.3f}")
print(f"interaction={value:.3f}")
print("All error-analysis and ablation checks passed.")
```

只有 8 道合成题、4 个来源组，不足以支持真实效果结论。四配置分数另为人工设定，不是从这 8 道题计算出来的，也不是实际消融结果。

## 12. 行为测试与独立确认

[CheckList（Ribeiro 等，2020）](https://aclanthology.org/2020.acl-main.442/)将语言能力与测试类型结合，强调总体准确率之外的行为测试。

对本文实验，可设计三类教学测试：

- 最小能力：简单问题也要求合法 JSON，先验证基本约束。
- 不变性：只改变无关空白或等价表达，语义答案应保持。
- 定向变化：把要求从输出一个字段变成两个字段，结构应相应改变。

这些是本文构造的测试想法，不是论文实验复现。变换后仍需检查题意和参考答案是否正确；不能假设任何改写都语义不变。

从一个原题扩写出的测试应保留共同来源组。用这些失败案例调优后，它们成为开发或回归测试，不能继续当作完全未见的确认集。

建议分开维护：

1. 探索集：发现问题和构造假设。
2. 回归集：确保已修复行为没有再次失败。
3. 独立确认集：检查冻结方案在新来源上的表现。

三者都重要，但用途不相同。

## 13. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
fixed=3, regressed=2, micro_delta=0.125
interaction=-0.130
All error-analysis and ablation checks passed.
```

已检查 UTF-8 编码、代码围栏、公式结构和本地链接；未进行 GitHub 页面视觉渲染验收。

练习检查了配对结果守恒、输入重排不变性、交换模型后的符号反转、切片重叠、空切片，以及重复 ID 和元数据冲突的拒绝逻辑。

实验报告应同时提供整体指标、预定切片、修复与退化、资源预算，以及是否进行多次试验选择。事后发现的切片标为探索性，不能替代整体主指标。

代码没有运行模型、判断自然语言根因、估计区间或执行真实训练消融；这些仍需后续独立实验。

## 参考资料

- [Ribeiro 等：Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/)。
- [NIST/SEMATECH：Full factorial designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri333.htm)。
- [前篇：配对比较、分组 Bootstrap 与选择偏差](2026-09-16-paired-evaluation-group-bootstrap.md)。

## 今日总结

1. 先确认评分正确，再解释模型错误。
2. 现象是证据，根因通常还需要干预来验证。
3. 同时报告修复和退化，保留切片分母与来源组。
4. 单因素对照回答局部问题，组合实验才能暴露交互。
5. 已用于改进的案例适合回归测试，最终确认仍需要独立来源。

## 下次衔接建议

继续学习实验追踪与可复现报告：把数据、配置、模型产物、评分结果和失败案例关联到同一 run manifest，保证实验结论可以追溯和重新计算。
