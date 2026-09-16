# 2026-09-16：评估结果的统计解释——配对比较、分组 Bootstrap 与选择偏差

## 今日目标

上一篇 [小模型后训练实验设计](2026-09-15-small-model-experiment-design.md)确定了 baseline、数据切分和训练验收。今天讨论下一步：即使候选模型分数更高，证据是否足以支持“有改善”？

完成后应能回答：

1. 为什么必须在相同题目上比较两个模型？
2. 同源改写题为什么不能直接当作独立样本？
3. 按来源组平均与按题目平均有什么不同？
4. Bootstrap 区间反映什么，又遗漏什么？
5. 为什么选出最高分 checkpoint 后再计算区间，仍然可能高估效果？

本篇只运行纯 Python 合成数据测试，没有运行真实模型训练或评估。示例输出不能作为任何模型提升的证据。

---

## 1. 先冻结比较协议

统计方法不能弥补不公平的比较。开始前固定：

- baseline 与 candidate 的模型版本、tokenizer、chat template。
- prompt、来源组、数据版本与切分 manifest。
- 解码配置、生成预算、失败处理规则。
- 评分器版本、答案抽取规则、主指标与指标方向。
- 模型选择规则、最小有意义提升，以及保留能力的退化容限。

如果候选模型使用更长的生成预算，应明确这是“模型加推理配置”的比较，不能将差异全部归因于训练。

生成失败不等于可以删掉题目。预先规定如何计分并同时报告失败率；基础设施缺失则先补齐或按冻结协议重跑。不要只留下双方都答得顺利的子集。

## 2. 配对比较：题目相同，差值才有对应关系

令同一题上的 baseline 分数为 b，candidate 分数为 c，约定越高越好：

```math
d_i = c_i - b_i
```

重点是差值的分布，而不是两组互不对应的平均数。

实现时按稳定的 `prompt_id` 对齐，不按文件行号对齐；拒绝重复 ID、缺失 ID 和来源组冲突。评分配置等元数据也需要在外层校验，后文代码只检查逐题数据。

“配对”并不要求两个模型生成相同答案，也不自动消除解码随机性。随机生成可以预先安排重复试验；相同数字 seed 也不保证不同模型经历相同的随机事件。

## 3. 先声明要估计哪一种平均提升

假设有 G 个来源组，第 g 组有 n_g 道题。

每题等权的 micro 平均：

```math
\widehat{\Delta}_{\mathrm{micro}}
= \frac{\sum_{g=1}^{G}\sum_{i=1}^{n_g} d_{gi}}
       {\sum_{g=1}^{G} n_g}
```

每组等权的 macro 平均：

```math
\bar d_g = \frac{1}{n_g}\sum_{i=1}^{n_g}d_{gi},
\qquad
\widehat{\Delta}_{\mathrm{macro}}
= \frac{1}{G}\sum_{g=1}^{G}\bar d_g
```

例如一组有 2 题，平均提升 0.2；另一组有 8 题，平均退化 0.1：

- macro：平均提升 0.05。
- micro：平均退化 0.04。

这不是计算矛盾，而是组权重不同。来源组是否应该等权，取决于实际研究问题和目标分布，不能看到哪个结果更好才选哪个。

今天的实现明确采用 **来源组等权的 macro 平均**。代码不适用于直接计算 corpus BLEU、F1 等非线性整体指标；那些指标需要在重采样数据上重新计算。

## 4. 为什么按来源组重采样

同一原题的改写、同一文档抽出的问答，可能共享难点和错误模式。把所有行独立重采样，可能低估这类相关性带来的不确定性。

今天采用以下假设：

- 来源组之间可近似视为独立、来自目标来源总体。
- 组内题目允许相关。
- 已观测的每组题目集合，在本次计算中作为一个整体保留。
- 两个模型与评分器均固定。

每次抽取 G 个来源组，允许重复。抽到某组时，保留该组两模型的配对结果。对于今天的 macro 指标，等价于对 G 个组均值进行有放回重采样。

如果两个“不同来源组”实际上来自同一模板家族，独立性假设仍可能失败。分组字段本身不是独立性的证明。

如果希望同时推断“组内尚未观测的新题”的不确定性，还需要结合采样设计考虑多层重采样；今天的例子没有覆盖这一层。

## 5. Percentile 区间如何构造

重复重采样 B 次，每次计算一个平均差值，得到 bootstrap 分布。95% percentile 区间取其 2.5% 和 97.5% 分位数：

```math
I_{0.95}
=
\left[
Q_{0.025}\left(\widehat{\Delta}^{*}\right),
Q_{0.975}\left(\widehat{\Delta}^{*}\right)
\right]
```

这是教学中容易检查的一种近似区间，不是所有数据条件下都可靠的方法。增加 B 主要减少重采样计算的 Monte Carlo 波动，不会增加独立来源组数量。

SciPy 的 `bootstrap` 支持 paired 重采样，其默认区间方法是 BCa，而不是本文的 percentile。若传入的是逐题数组，`paired=True` 只保证两模型使用相同索引，并不会自动识别来源组。参见 [SciPy bootstrap 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。

## 6. 最小实现：严格对齐，再对组均值重采样

下面只依赖 Python 标准库。输入分数限定为 [0, 1]，既可表示二元成功率，也可表示已经归一化的逐题评分。

```python
import math
import random
from statistics import mean


def index_rows(rows):
    indexed = {}
    for row in rows:
        pid = row["prompt_id"]
        group = row["source_group"]
        score = float(row["score"])
        if not isinstance(pid, str) or not pid:
            raise ValueError("invalid prompt_id")
        if not isinstance(group, str) or not group:
            raise ValueError("invalid source_group")
        if pid in indexed:
            raise ValueError("duplicate prompt_id")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score must be finite and in [0, 1]")
        indexed[pid] = (group, score)
    if not indexed:
        raise ValueError("empty evaluation")
    return indexed


def paired_group_deltas(baseline, candidate):
    left, right = index_rows(baseline), index_rows(candidate)
    if left.keys() != right.keys():
        raise ValueError("prompt_id sets differ")
    grouped = {}
    for pid in sorted(left):
        group_b, score_b = left[pid]
        group_c, score_c = right[pid]
        if group_b != group_c:
            raise ValueError("source_group mismatch")
        grouped.setdefault(group_b, []).append(score_c - score_b)
    return {g: mean(grouped[g]) for g in sorted(grouped)}


def quantile_sorted(values, p):
    # Linear interpolation between adjacent order statistics.
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    fraction = position - lo
    return values[lo] * (1 - fraction) + values[hi] * fraction


def group_bootstrap(deltas, *, repeats=10000, seed=20260916):
    values = [float(deltas[g]) for g in sorted(deltas)]
    if len(values) < 2:
        raise ValueError("need at least two groups; this is not a sufficiency rule")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("non-finite group delta")
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("invalid repeats")
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(repeats)
    )
    return {
        "estimate": mean(values),
        "low": quantile_sorted(samples, 0.025),
        "high": quantile_sorted(samples, 0.975),
        "groups": count,
        "repeats": repeats,
        "seed": seed,
        "method": "paired source-group percentile bootstrap, macro mean",
    }
```

这里“至少两组”只是拒绝明显无意义输入的代码检查，绝不表示两组就足够进行可信推断。实际来源组很少时，应把区间视为脆弱的探索性结果。

## 7. 合成数据与不变量测试

以下代码接在上一段后运行。测试重点是统计量定义和配对逻辑，而不是仅检查程序没有报错。

```python
def make_rows(group_deltas, sizes):
    baseline, candidate = [], []
    for g, (delta, size) in enumerate(zip(group_deltas, sizes)):
        for i in range(size):
            common = {"prompt_id": f"{g}:{i}", "source_group": f"g{g}"}
            baseline.append({**common, "score": 0.5})
            candidate.append({**common, "score": 0.5 + delta})
    return baseline, candidate


def must_reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


# Unequal group sizes make macro and micro different.
b, c = make_rows([0.2, -0.1], [2, 8])
d = paired_group_deltas(b, c)
macro = mean(d.values())
micro = mean(y["score"] - x["score"] for x, y in zip(b, c))
assert math.isclose(macro, 0.05, abs_tol=1e-12)
assert math.isclose(micro, -0.04, abs_tol=1e-12)

# Reordering files must not affect pairing or seeded bootstrap results.
reference = group_bootstrap(d)
assert paired_group_deltas(b[::-1], c[::-1]) == d
assert group_bootstrap(paired_group_deltas(b[::-1], c)) == reference
assert group_bootstrap(d) == reference

# Duplicating every row in one group preserves that group's mean.
b_copy = [dict(r, prompt_id=r["prompt_id"] + ":copy")
          for r in b if r["source_group"] == "g0"]
c_copy = [dict(r, prompt_id=r["prompt_id"] + ":copy")
          for r in c if r["source_group"] == "g0"]
assert group_bootstrap(paired_group_deltas(b + b_copy, c + c_copy)) == reference

# Identity yields a degenerate interval, not proof of universal equivalence.
identity = group_bootstrap(paired_group_deltas(b, b))
assert identity["estimate"] == identity["low"] == identity["high"] == 0

# Reversing model order reverses the estimate and interval endpoints.
reverse = group_bootstrap(paired_group_deltas(c, b))
assert math.isclose(reverse["estimate"], -reference["estimate"], abs_tol=1e-12)
assert math.isclose(reverse["low"], -reference["high"], abs_tol=1e-12)
assert math.isclose(reverse["high"], -reference["low"], abs_tol=1e-12)

must_reject(lambda: paired_group_deltas(b + [b[0]], c))
must_reject(lambda: paired_group_deltas(b, c[:-1]))
wrong_group = [dict(c[0], source_group="wrong")] + c[1:]
must_reject(lambda: paired_group_deltas(b, wrong_group))
bad_score = [dict(c[0], score=float("nan"))] + c[1:]
must_reject(lambda: paired_group_deltas(b, bad_score))
must_reject(lambda: paired_group_deltas([], []))
must_reject(lambda: group_bootstrap({"only_group": 0.1}))

# An illustrative eight-group dataset, not a real evaluation.
b8, c8 = make_rows(
    [-0.20, -0.10, 0.00, 0.00, 0.05, 0.10, 0.15, 0.20],
    [2, 3, 1, 4, 2, 5, 3, 2],
)
result = group_bootstrap(paired_group_deltas(b8, c8))
assert math.isclose(result["estimate"], 0.025, abs_tol=1e-12)
assert result["low"] < 0 < result["high"]
print(f"macro={macro:.3f}, micro={micro:.3f}")
print(
    f"synthetic groups={result['groups']}, "
    f"delta={result['estimate']:.3f}, "
    f"95% percentile CI=[{result['low']:.3f}, {result['high']:.3f}]"
)
print("All paired/group-bootstrap checks passed.")
```

预期的核心结论：第一组例子 macro 与 micro 的符号相反；第二组例子点估计为 +0.025，但区间跨过零。后者不能支持“已经确定提升”的结论，也不能证明“两模型没有差异”。

8 个来源组仍很少，合成例子只用于验证实现；没有开展区间覆盖率模拟，也没有证明该方法对真实数据有良好覆盖率。

## 8. 怎样读区间，怎样不读区间

95% 置信区间对应的是重复抽样程序的覆盖率目标，且本文方法只是近似实现。不能解释成“固定的真实提升有 95% 概率位于这个已经算出的区间里”。

常见误读：

- 区间跨零 → “模型完全一样”：不成立，可能只是信息不足。
- 区间下界大于零 → “值得部署”：不成立，还需要实际收益、成本和退化约束。
- bootstrap 差值大于零的比例 → “候选更好的后验概率”：不成立。
- 直接将 bootstrap 的负值比例称为 p-value：没有给出零假设下的检验构造，不应这样命名。
- 重采样十万次 → “相当于十万组新数据”：不成立。
- 所有已观测差值都相同 → “总体没有不确定性”：不成立，经验分布没有覆盖未知情况。

若要主张等价，应预先定义等价界限并使用相应设计和检验；“没有发现显著差异”不是等价证据。

## 9. 哪些随机性没有包括进来

本篇区间条件于固定的两套模型和已生成、已评分的输出，重点反映来源组抽样变化。

它没有自动覆盖：

- 重新训练时的初始化、数据顺序和优化随机性。
- 再次随机解码导致的输出变化。
- 人工或模型评分器的评分波动与系统偏差。
- 数据污染、错误答案标签、目标分布迁移。

如果问题是“训练方法 A 是否稳定优于方法 B”，需要安排多个训练重复并记录预算。不能把同一 checkpoint 在不同题目上的差异当作训练 seed 的独立重复。

多 seed、多来源组可以构成多层或交叉设计；必须说明推断对象和依赖关系，不能简单把所有 seed × prompt 行压平。

## 10. 多次试验之后，最高分可能包含运气

假设反复调整学习率、训练步数、数据过滤和评分提示，再选开发集最高分。这个分数既包含真实能力，也可能包含对该开发集噪声的适配。

对胜出模型的同一份开发集输出进行 bootstrap，不会撤销此前的选择过程。模型选择也会过拟合有限样本的评估准则，参见 [Cawley 与 Talbot（2010）](https://www.jmlr.org/papers/v11/cawley10a.html)。

实验安排：

1. 在训练集拟合，在开发集做探索和选择。
2. 记录试过多少配置，不只记录胜出的一次。
3. 在查看最终测试结果前冻结模型、指标和比较规则。
4. 使用未参与选择的测试集做最终评估。
5. 看过测试结果后再调参，这份集合便不再是新一轮的未见测试集。

若没有足够数据，应如实标为开发集探索结果；复杂情形可考虑嵌套评估。多个正式假设或指标还需要预先考虑多重比较，而不是挑一个“显著”的结果报告。

## 11. 给昨天的小模型实验增加统计验收项

在 [昨天的实验计划](2026-09-15-small-model-experiment-design.md)上增加以下记录：

| 项目 | 应记录的内容 |
| --- | --- |
| 比较对象 | 固定的两个模型，还是两种训练流程 |
| 样本单位 | 题目数、来源组数、来源组构造方式 |
| 统计量 | micro 或 macro、指标方向、权重 |
| 不确定性 | 重采样单位、方法、次数、seed、区间水平 |
| 选择过程 | 已试配置数、选择用的数据、是否看过测试集 |
| 验收条件 | 预定最小收益、保留能力容限、推理预算 |
| 局限 | 来源依赖、少数组、评分误差、未覆盖的随机性 |

对二元成功率，还可以报告配对的“候选赢 / 持平 / 候选输”数量，并保留来源组切片。这些计数帮助解释变化，但不能代替与目标指标一致的统计分析。

先确认数据和评分协议一致，再解释点估计和区间，最后才讨论是否满足实际验收条件。

## 12. 今日练习结果与边界

已在本地执行上述两个 Python 代码块，全部断言通过。实际输出：

```text
macro=0.050, micro=-0.040
synthetic groups=8, delta=0.025, 95% percentile CI=[-0.062, 0.106]
All paired/group-bootstrap checks passed.
```

已检查 UTF-8 解码、代码围栏、公式花括号和本地链接；未进行 GitHub 页面视觉渲染验收。

本篇可执行代码覆盖：

- macro 与 micro 的不同权重及相反结论。
- ID 对齐、文件重排不变性和固定 seed 可复现性。
- 组内整体重复后，macro 统计不应增加该组权重。
- 相同模型的零差值，以及交换模型后的符号对称性。
- 重复 ID、缺失 ID、来源冲突、非有限分数与空输入拒绝。
- 合成数据中“点估计为正但区间跨零”的情况。

这些检查验证的是教学代码的特定性质，不是统计方法的一般有效性，更不是实际训练成功的证明。

## 参考资料

- [SciPy：bootstrap API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)：paired 参数、区间方法及随机数配置。
- [Koehn（2004）：Statistical Significance Tests for Machine Translation Evaluation](https://aclanthology.org/W04-3250/)：语言任务评估中统计显著性检验的延伸阅读。
- [Cawley 与 Talbot（2010）：On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html)：模型选择过拟合与评估选择偏差。

本文的来源组等权实现、合成数据和断言是教学示例，不是上述论文实验的复现。

## 今日总结

1. 先定义想估计的量，再决定如何平均和重采样。
2. 配对要保留题目对应关系，分组要反映实际依赖关系。
3. 题目多不等于独立来源多，重采样次数也不是新增样本量。
4. 区间跨零表示当前证据不足，不等于证明无差异。
5. 选择最高分和报告最高分之间，需要独立评估的边界。

## 下次衔接建议

继续学习错误分析与消融实验：从逐题失败分类提出可检验的修改假设，固定预算做单因素对照，避免把事后挑出的“好看切片”当作整体提升。
