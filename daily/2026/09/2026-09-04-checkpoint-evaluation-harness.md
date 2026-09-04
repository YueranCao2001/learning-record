# 2026-09-04：Checkpoint 评估——Validation Loss、任务指标、Calibration 与 Evaluation Harness

## 今日目标

昨天在[训练异常诊断](2026-09-03-training-instability-loss-spike-nan-recovery.md)中讨论了如何保存和恢复一个数值健康、状态一致的 checkpoint。今天进一步回答：

> 一个 checkpoint 可以继续训练，不代表它更值得使用。怎样证明它在目标能力上真正进步，而不只是某个平均分更好看？

完成后应能回答：

1. validation loss、任务正确率、生成质量、calibration 和效率分别测量什么？
2. PPL 为什么必须先汇总 NLL 和有效 token 数，再取指数？
3. likelihood 选择题与自由生成有什么不同？
4. 如何控制 few-shot、chat template、解码预算和答案提取？
5. 如何比较两个 checkpoint，并解释配对置信区间？
6. 污染、test 选择偏差和 LLM judge 偏差怎样影响结论？
7. 怎样构建可复现、能追到每道题的 evaluation harness？

本篇代码为教学骨架和实验模板；本次更新没有加载模型、运行 benchmark 或产生真实模型成绩。

---

## 1. 评估对象不只是权重

一次评测结果可以写成：

\[
R=f(\theta,D,P,T,G,E,A)
\]

- `\theta`：checkpoint；
- `D`：数据版本、split 和样本集合；
- `P`：prompt、few-shot 与选项顺序；
- `T`：tokenizer、chat template、上下文截断；
- `G`：生成参数、seed、推理与工具预算；
- `E`：parser、执行器或 judge；
- `A`：聚合与统计方法。

任一项改变，分差就不能直接归因于模型权重。比较时先冻结协议，再解释差异。

---

## 2. 分层评估，而不是寻找万能分数

| 层次 | 主要问题 | 典型指标 |
|---|---|---|
| 语言建模 | 对真实后续文本分配的概率是否更好？ | NLL、PPL、按域 loss |
| 任务能力 | 是否能做对目标任务？ | accuracy、EM、F1、functional correctness |
| 生成行为 | 开放回答是否正确、完整、符合指令？ | rubric、人审、pairwise preference |
| 可信程度 | 自信是否与正确率匹配？ | NLL、Brier、ECE、risk-coverage |
| 稳健性 | 改写、换语言或上下文后是否仍有效？ | 分层结果、扰动差值 |
| 成本 | 是否满足部署要求？ | latency、tokens/s、memory、cost |
| 风险 | 是否存在污染、泄漏或不当行为？ | 污染审计、风险测试、失败案例 |

PPL 降低不自动意味着指令遵循更好；代码 pass@k 上升也不代表问答更可靠。HELM 是多维评估的参考，实际 scorecard 仍应由用途决定。[HELM](https://crfm.stanford.edu/helm/latest/)

---

## 3. 冻结可识别的 Checkpoint

评估应指向不可变 artifact，而不只是不断移动的 `latest`：

```text
run_id / checkpoint_id
committed_optimizer_step / training_tokens
model_config_hash / weight_manifest_hash
tokenizer_revision
adapter_revision / merge_policy
precision / quantization
training_data_version / code_commit
```

还要明确：

- base、SFT 还是 preference-tuned；
- 是否使用 EMA；
- adapter 是否加载或合并；
- 分片导出是否完整；
- vocabulary、special tokens 是否一致；
- checkpoint 是否已经 COMMITTED。

“load 成功”只通过 artifact 检查，不代表评测协议正确。

---

## 4. Train、Development、Calibration 与 Test

- **Train**：更新模型参数。
- **Development / Validation**：选择 checkpoint、prompt、超参数和数据配比。
- **Calibration**：拟合 confidence calibration，可从开发数据单独划分。
- **Final Test**：主要选择冻结后，报告泛化表现。

如果每天测 test，再挑最高分 checkpoint，test 实际承担了 validation 的角色。即使没有对 test 反向传播，也发生了选择层面的信息利用。

记录数据集的实际用途，而不是只根据目录名判断是否独立。

---

## 5. Validation NLL 与 Perplexity

对真正计分的 token 集合 `V`：

\[
S=-\sum_{t\in V}\log p_\theta(x_t\mid x_{<t})
\]

\[
L_{\mathrm{token}}=\frac{S}{|V|},
\qquad
\mathrm{PPL}=\exp(L_{\mathrm{token}})
\]

log 使用自然对数。只有纯 hard-label NLL 才能这样转换为 PPL；含 label smoothing、正则项或辅助项的总训练 loss 不能直接指数化后叫作 PPL。

`V` 必须固定：

- causal shift 后的位置；
- PAD、prompt、边界和滑窗重叠 masks；
- BOS / EOS 是否评分；
- 文档起点如何获得上下文。

---

## 6. 不能平均 Batch Loss 或 Batch PPL

假设：

```text
A: 10 个有效 token，mean NLL = 2
B: 90 个有效 token，mean NLL = 4
```

正确：

\[
L=\frac{10\times2+90\times4}{100}=3.8
\]

\[
\mathrm{PPL}=\exp(3.8)\approx44.70
\]

错误的等权 batch mean 是 `3`，对应 PPL 约 `20.09`。

跨 batch 和 ranks 的聚合单元应是：

```text
NLL sum + scored token count
```

不是 rank means，也不是 PPL 的算术平均。

---

## 7. 从 Shift 后的 Labels 计数

假设 logits 为 `[B,T,V]`，labels 为 `[B,T]`；labels 与 input positions 对齐，mask 通过 `-100` 编码，本函数执行唯一一次 shift：

```python
import torch
import torch.nn.functional as F

@torch.inference_mode()
def causal_nll_stats(logits, labels, ignore_index=-100):
    pred = logits[:, :-1, :].contiguous()
    target = labels[:, 1:].contiguous()
    count = target.ne(ignore_index).sum(dtype=torch.int64)

    if count.item() == 0:
        raise ValueError("no scored next-token targets")

    nll_sum = F.cross_entropy(
        pred.float().view(-1, pred.size(-1)),
        target.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    if not torch.isfinite(nll_sum):
        raise FloatingPointError("non-finite evaluation NLL")

    return nll_sum.double(), count
```

若模型在 `model(..., labels=labels)` 内部 shift，不要重复 shift。

不要无条件采用“有效 labels 数减 batch size”：每行首个 label 可能早已 mask，盲减会少计。直接统计实际 shifted targets 最稳妥。

这是口径示例；超大词表时完整 FP32 logits 成本高，生产中可用经过等价性测试的分块或融合 loss。

---

## 8. 长文本 PPL 的 Context 协议

- 不重叠切块：快，但每块开头上下文不足。
- 滑动窗口：保留更多历史；重叠 tokens 只作 context，不重复计分。

必须记录 context length、stride、文档边界、BOS/EOS、第一 token 处理、跨文档 attention 和评分覆盖。

改变 stride 就可能改变 PPL，不能把协议变化当成模型进步。[Hugging Face PPL 文档](https://huggingface.co/docs/transformers/main/perplexity)

最小测试：为 scored token 记录 `document_id + token_offset`，验证每个应评分目标恰好出现一次。

---

## 9. 不同 Tokenizer 的 PPL 不宜直接横比

PPL 的单位是每 token，token 粒度改变会改变尺度。

可以补充：

\[
\mathrm{BPB}
=
\frac{S}{\ln2\cdot N_{\mathrm{scored\ UTF8\ bytes}}}
\]

但仍要统一原始文本、规范化、计分区间、边界、special tokens 和 tokenizer 的有损行为。

BPB 也不会自动消除 context length 按 token 定义产生的差异。它是补充口径，不是万能公平修正。

---

## 10. Teacher Forcing 不等于自由生成

NLL 在真实前文下预测：

```text
真实 prefix → 下一个真实 token 的概率
```

自由生成则在自己的输出上继续：

```text
生成 prefix → 下一个生成 token
```

更低 NLL 不保证长回答更少跑题、推理更正确、JSON 更稳定或更愿意承认不知道。应同时包含 likelihood 和 generation 评估。

---

## 11. `eval()` 与禁用梯度不同

```python
previous_mode = model.training
model.eval()
try:
    with torch.inference_mode():
        outputs = model(**model_inputs)
finally:
    model.train(previous_mode)
```

`eval()` 改变 Dropout、BatchNorm 等行为；`inference_mode()` 控制 autograd，不自动设置 evaluation mode。

训练进程内评估还需保护训练 RNG、缓存和可变模块状态。随机生成可用隔离进程或显式保存恢复状态。

记录 dtype、quantization、attention backend 和 batch size，不承诺跨硬件完全一致。

---

## 12. 分布式评测不能补齐重复样本

训练用 `DistributedSampler(drop_last=False)` 可能为对齐长度补索引；评测直接复用会重复计分。

应：

- 使用不补齐分片或按 stable sample ID 去重；
- 不因 `drop_last=True` 丢弃尾部；
- 聚合 sums/counts；
- 核对实际样本集合等于预期集合。

若 ranks 批次数不等，而 forward 包装器每次执行 collective，仍可能 hang。分片方案和执行通信必须一起设计，不能只改 sampler。

---

## 13. 选择题：对 Continuation 评分

给定问题 `q` 与候选 `a_i`：

\[
s_i=\sum_{t=1}^{|a_i|}
\log p_\theta(a_{i,t}\mid q,a_{i,<t}),
\qquad
\hat i=\arg\max_i s_i
\]

通常只计 continuation，不把共享 prompt 的概率加入候选分数。

固定：

- 候选前的空格与分隔符；
- 候选是字母还是完整答案；
- context-continuation tokenization 边界；
- few-shot、选项顺序；
- EOS；
- chat template 与答案前缀。

生成字母再解析，与直接比较候选 likelihood 是不同协议，不能混报。

---

## 14. `acc_norm` 的分母要看实现

长度归一化可写为：

\[
s_i^{\mathrm{norm}}=\frac{s_i}{d(a_i)}
\]

`d(a_i)` 由任务实现决定，不应默认等于 token 数。

本次核对的官方 Harness `main` 实现中：

```text
acc       : argmax log P(choice | context)
acc_norm  : argmax log P(choice | context) / len(choice)
acc_bytes : argmax log P(choice | context) / len(choice.encode("utf-8"))
```

这里 `len(choice)` 是 Python 字符串长度，不是 tokenizer token 数。自定义 task 可以覆盖规则，所以仍应固定 commit，记录真实 denominator。[官方计分源码](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py)

归一化是先改变候选排序再计算 accuracy，不是把最终 accuracy 除以长度，也不等于 calibration。

---

## 15. 生成任务：Parser 属于评测协议

常见指标包括 EM、token F1、numerical accuracy、symbolic equivalence、format compliance 和 test pass rate。

每条结果保存：

```text
raw output
  → extractor/version
  → extracted answer
  → reference
  → score
  → failure reason
```

检查：

- 最终答案与中间推理如何区分；
- 多个数字、单位和多个答案如何处理；
- 大小写、空白、标点的规范化；
- 空输出、refusal、截断和 timeout；
- parser failure 的数量。

过宽正则可能从错误答案中抓出正确数字；过窄 parser 也可能误判正确输出。所有模型必须使用同样冻结的规则。

---

## 16. 代码生成与 `pass@k`

每题生成 `n` 个候选，其中 `c` 个通过测试，`n\ge k` 时：

\[
\widehat{\mathrm{pass@k}}
=
1-\frac{\binom{n-c}{k}}{\binom nk}
\]

最后对题目取均值。它衡量 k 次尝试至少一次通过，不是 top-k classification，也不表示系统能选出那条正确候选。

其常用无偏解释依赖固定协议下独立同分布采样等假设；beam search、去重或成功后自适应停止不能直接套用。也不要替换成有偏的 `1-(1-c/n)^k` 插件估计。[原始论文](https://arxiv.org/abs/2107.03374)

固定 sampling 参数、测试版本、runtime、timeout、工具/修复预算。通过有限 tests 不等于数学证明程序正确。

生成代码是不可信输入，只能在无凭据、受限网络和资源的隔离环境测试；本次未执行模型代码。

---

## 17. 解码预算也是实验条件

记录：

- `do_sample`、temperature、top-p、top-k；
- seed、采样次数；
- `max_new_tokens`、stop strings、EOS；
- context truncation；
- thinking/reasoning mode；
- 工具调用与修复次数；
- timeout；
- 实际 token、延迟和费用。

32 次采样与单次生成，或不同推理预算，不能只按裸分数比较权重优劣。

Greedy decoding 减少采样随机性，但不保证跨 backend、硬件和版本 bitwise 一致。

---

## 18. Calibration：先定义 Confidence 的事件

“自信 80%”的样本长期只有 50% 正确，说明该 confidence overconfident。但 confidence 可以指：

- token 概率；
- 候选答案概率；
- 整体回答正确概率；
- 语言化的“80% 把握”；
- 独立 correctness predictor 输出。

它们不是同一个事件。对候选分数做 softmax：

\[
p_i=\frac{\exp(s_i)}{\sum_j\exp(s_j)}
\]

只是指定候选集合和 scoring protocol 下的相对权重，不自动等于现实正确率。尤其长度归一化 scores 不能未经说明当成完整序列概率。

---

## 19. Brier 与 ECE

多分类 Brier 的一个约定：

\[
\mathrm{Brier}
=
\frac1N\sum_{i=1}^N\sum_{c=1}^C(p_{ic}-y_{ic})^2
\]

`y` 是 one-hot；这里不除类别数或 2，范围为 `[0,2]`。二元单事件版本 `N^{-1}\sum_i(q_i-z_i)^2` 的范围是 `[0,1]`，不要混用。

按 top-label confidence 分 bins `B_m`：

\[
\mathrm{ECE}
=
\sum_m\frac{|B_m|}{N}
\left|
\operatorname{acc}(B_m)-\operatorname{conf}(B_m)
\right|
\]

明确 bin 数、边界、等宽/等频、样本量，以及 top-label/classwise/binary-event 定义。ECE 对分箱和有限样本敏感。

同时报告 reliability diagram、bin counts、NLL/Brier 和分层结果。NLL/Brier 是概率预测质量指标，不是只测 calibration 的纯误差。[Calibration 研究](https://proceedings.mlr.press/v70/guo17a.html)

---

## 20. Calibration Temperature 不等于生成 Temperature

对分类 logits：

\[
p_c=\operatorname{softmax}(z/T)_c,\qquad T>0
\]

在独立 calibration set 上拟合 `T`，冻结后测 test。固定 logits 下正 temperature 不改变 argmax，但改变 confidence；效果仍需验证。

生成 temperature 改变逐 token sampling，会改变答案本身。把生成参数设为 0.7 不等于完成答案正确率校准。

---

## 21. 拒答系统的 Risk 与 Coverage

设 confidence 为 `c_i`：

\[
\mathrm{coverage}(\tau)
=
\frac{\#\{i:c_i\ge\tau\}}N
\]

\[
\mathrm{risk}(\tau)
=
\frac{\#\{i:c_i\ge\tau,\hat y_i\ne y_i\}}
{\#\{i:c_i\ge\tau\}}
\]

只回答少数容易题可能 risk 很低，但 coverage 也低。两者必须一起报告；阈值在 dev/calibration 上选择。

零接受样本时 risk 未定义，不能为了让曲线好看把它填成 0。

---

## 22. Macro、Micro 与能力分层

- Micro：按题数加权，大任务影响更大。
- Macro：先算每任务指标，再按任务权重聚合。

冻结 task weights、指标方向、缺失任务处理、language/domain 分层和关键风险门槛。

不要直接把 PPL、accuracy、latency、ECE 原值求平均。综合指数若确有用途，转换、权重和缺失策略都应透明。

还要检查平均分背后的 regression：一个重要领域下降，可能被多个小幅上升掩盖。

---

## 23. 不确定性有多个来源

1. 题目采样；
2. few-shot 和 prompt 选择；
3. generation seed；
4. training seed；
5. judge 随机性与系统偏差；
6. 硬件、数值 backend 与 timeout。

只 bootstrap test items 不覆盖其他全部来源。若只比较单个训练 run，应把结论限定为“这些 checkpoints 在此协议下的比较”，而不是证明训练方法普遍更优。

固定 benchmark 上的实测均值本身是确定结果；CI 想推广到什么任务分布、依赖哪些抽样假设，需要写清楚。

---

## 24. 两个 Checkpoint 应做 Paired 比较

相同题目上的分数为 `a_i,b_i`：

\[
\Delta=\frac1N\sum_i(b_i-a_i)
\]

先按 stable sample ID join，再让 A/B 共同重采样同一批索引，保留题目难度相关性。

分别独立 resample 或观察两条独立 CI 是否重叠，都不能替代配对差值推断。

同时记录：

- 两者都对；
- A 对 B 错；
- A 错 B 对；
- 两者都错；
- 缺失或未配对题目。

这些切片比单一 delta 更容易解释能力变化。

---

## 25. Paired Bootstrap 教学实现

以下适用于独立题目上的可加分数，例如 accuracy 的 `0/1`，不是 corpus-level metric 的通用实现：

```python
import math
import random

def percentile(sorted_values, p):
    position = (len(sorted_values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    weight = position - lo
    return (
        sorted_values[lo] * (1 - weight)
        + sorted_values[hi] * weight
    )

def paired_bootstrap(a, b, repeats=2000, seed=42):
    if len(a) != len(b) or not a:
        raise ValueError("aligned non-empty scores required")
    if repeats < 2:
        raise ValueError("too few replicates")
    if not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError("non-finite score")

    delta = [y - x for x, y in zip(a, b)]
    observed = sum(delta) / len(delta)
    rng = random.Random(seed)

    draws = []
    for _ in range(repeats):
        sampled = [
            delta[rng.randrange(len(delta))]
            for _ in range(len(delta))
        ]
        draws.append(sum(sampled) / len(sampled))

    draws.sort()
    return {
        "delta": observed,
        "ci95": (
            percentile(draws, 0.025),
            percentile(draws, 0.975),
        ),
        "n": len(delta),
        "repeats": repeats,
        "seed": seed,
    }
```

每次重采样必须计算目标指标；数据依赖结构与 estimand 不匹配时，再多 bootstrap iterations 也无法修复错误假设。[NLP 统计检验指南](https://aclanthology.org/P18-1128/)

---

## 26. Cluster Bootstrap 与 Ratio of Sums

同一文档、用户、仓库、对话或 problem template 可能派生多个相关题目。应以合理独立 cluster 抽样，保留簇内全部记录和 A/B pairing。

NLL/PPL 每次重算：

\[
L^*=\frac{\sum_{d\in D^*}S_d}{\sum_{d\in D^*}T_d},
\qquad
\mathrm{PPL}^*=\exp(L^*)
\]

不是逐文档 PPL 平均。同一道题的 100 次生成也不是 100 道独立题目。

Macro aggregate 可在任务内 resample，再按冻结任务权重重算。独立簇太少时，bootstrap 不能凭空制造信息。

---

## 27. 正确解释置信区间

例如：

```text
B - A = +0.8 percentage points
95% paired bootstrap CI = [-0.4, +2.0] points
```

意味着当前样本与方法下，差异方向仍有不确定性。

不能解释为：

- 固定区间有 95% 概率包含真值；
- CI 含 0，所以两个模型等价；
- CI 不含 0，所以业务收益一定重要。

95% 指方法在重复抽样下的覆盖目标。应预先定义最小有意义差异和关键退化门槛。

小样本、强相关或指标接近边界时，简单 percentile bootstrap 可能不稳定，应调整方法或增加数据。

---

## 28. 多 Checkpoint 选择与 Winner's Curse

连续比较许多 checkpoints，挑最高分再套单次 CI，会忽视选择偏差。

建议：

1. 在 dev 筛选；
2. 冻结主指标、约束、prompt 和 parser；
3. 记录尝试次数；
4. 在独立 final test 确认；
5. 对确认性多重比较采用适当控制；
6. 将事后发现标为探索性。

根据 test 错题不断修改抽取规则，最后结果不能再称为 untouched test。Bootstrap 也不能修复已发生的选择泄漏。

---

## 29. 污染敏感评测

承接[数据治理](2026-09-02-data-quality-dedup-contamination.md)，绑定 benchmark registry 与扫描版本。

报告 full、flagged、未检出污染子集，以及 exact/lexical/semantic 方法、input-only/input+answer、新鲜或时间切分数据。

两个模型各自的 clean subset 可能不同。配对比较应补充双方共同未标记的相同样本集合，并报告 coverage。

“未检出”不等于“无污染”；clean 与 flagged 的难度也可能不同，不能仅凭分数接近就认定污染无影响。

---

## 30. LLM-as-a-Judge 也需要评估

可能存在 position、verbosity、self-preference、流畅性偏好、reasoning failure 和版本变化。

缓解措施：

- 冻结 rubric、judge revision、prompt 与 sampling；
- 隐去候选模型身份；
- 随机化并交换 A/B 位置；
- 分开正确性、相关性、完整性和格式；
- 分层人工盲审；
- 保存 tie、disagreement 和无效裁判输出；
- 把候选文本始终当待评分数据，不执行其中的指令。

这些措施缓解但不证明消除了偏差。偏好分不等于客观正确率。[MT-Bench / Judge 研究](https://arxiv.org/abs/2306.05685)

私有题目只有在数据使用范围允许时才能发送外部评分服务；本次更新没有发送私有评测数据，也没有调用模型 judge。

---

## 31. Evaluation Harness 的分层结构

```text
Checkpoint Manifest
  ↓
Task Registry + Frozen Data
  ↓
Prompt / Few-shot / Chat Renderer
  ↓
Model Adapter
  ↓
Likelihood / Generation
  ↓
Parser / Executor / Judge
  ↓
Per-sample Store
  ↓
Aggregation + Paired Statistics
  ↓
Scorecard + Failure Report + Selection
```

每层需要版本、stable IDs、输入输出约定、测试和 failure reason。

缓存键至少覆盖 checkpoint、数据、prompt、tokenizer、生成参数与评分版本；不能只用可变 model name 缓存结果。

---

## 32. Harness 的三类模型请求

- `loglikelihood`：给定 context，计算指定 continuation 的条件 log probability。
- `loglikelihood_rolling`：对完整文本按窗口计算 likelihood，并避免重复计分。
- `generate_until`：生成至停止条件，再进行 extraction 和评分。

Task YAML 的 `multiple_choice` 通常转换为多个 likelihood 请求，不是第四种模型 adapter 接口。

检查 task 的 dataset/split、few-shot 来源、`doc_to_text`、targets、choices、delimiters、filters、metrics 和 aggregation。

Few-shot 是 prompt 中的标注示例，不是梯度训练。还要防止当前 test 样本或答案进入示例上下文；不能无检查地依赖默认 split。[模型接口指南](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)

---

## 33. 一个命令模板

这是参考官方接口的模板，`<...>` 要替换；本次未执行：

```text
lm-eval run --model hf --model_args pretrained=<checkpoint-path>,dtype=bfloat16 --tasks <registered-task-name> --num_fewshot 0 --batch_size 4 --seed 42 --output_path <private-results-dir> --log_samples --check_integrity
```

注意：

- 固定 harness commit、task YAML 和依赖；
- 用所装版本的 help 和 task registry 核对参数；
- chat 模型按声明协议使用 `--apply_chat_template`；
- few-shot 多轮格式记录 resolved config，不依赖跨版本默认；
- `--limit` 用于 smoke test，不当正式全量成绩；
- `--log_samples` 可能保存私有题目、答案和输出；
- 不默认启用 remote code 或不安全执行。

解码参数对 likelihood task 并不具有自由生成时的含义。[官方 CLI](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)

---

## 34. Manifest 与逐题日志

```json
{
  "evaluation_id": "eval-2026-09-04-example",
  "checkpoint_id": "immutable-checkpoint-id",
  "weight_hash": "...",
  "tokenizer_hash": "...",
  "harness_commit": "...",
  "task_config_hash": "...",
  "dataset_revision": "...",
  "split": "validation",
  "sample_set_hash": "...",
  "prompt_hash": "...",
  "fewshot_ids": [],
  "chat_template_hash": "...",
  "context_length": 4096,
  "generation_config_hash": "...",
  "dtype": "bfloat16",
  "backend": "...",
  "scoring_revision": "...",
  "judge_revision": null,
  "bootstrap_unit": "document",
  "seeds": {
    "fewshot": 42,
    "generation": 42,
    "statistics": 42
  },
  "status": "planned"
}
```

上述数值只是 schema 示例，不是完成的运行。

逐题记录：

```text
sample_id / cluster_id / task / domain
prompt hash / raw output 或受控引用
choice scores / normalization denominators
extracted answer / parser revision
score / latency / generated tokens
status / failure reason / contamination flags
checkpoint_id / evaluation_id
```

Aggregate report 应能从逐题数据重新生成。

---

## 35. Missing、Timeout 与失败分母

区分 success、model failure、format failure、timeout、infrastructure error、missing 和预声明 exclusion。

不能只平均“有结果的题”，静默删除难题、超时或格式失败。

事先规定：

- 模型失败如何计分；
- 系统故障如何重试；
- seed 与 sample ID 是否保持；
- timeout 是否属于固定预算失败；
- 什么时候报告 incomplete；
- exclusion 是否在看结果前确定。

报告 requested、completed、scored、excluded、failed counts 和集合校验。

---

## 36. Latest-Good、Best-Dev 与 Release-Candidate

- **Latest-Good**：数值健康、状态可恢复的训练 checkpoint。
- **Best-Dev**：冻结开发协议下满足约束的最佳候选。
- **Release-Candidate**：通过独立确认、风险与部署成本门槛的候选。

```text
artifact smoke test
  → held-out NLL
  → core tasks / regressions
  → generation / calibration
  → paired comparison + uncertainty
  → risk + cost gates
  → explicit decision
```

评测不能覆盖训练恢复指针。Aggregate 上升也不能抵消关键领域退化或超出成本预算。

---

## 37. 常见错误

### 把总训练 Loss 指数化就叫 PPL

含辅助项、正则或 label smoothing 时，它不再是标准 token NLL。

### 平均各 Batch 的 Loss 或 PPL

有效 token 数不同会改变权重，应聚合 NLL sum 和 count。

### 无条件从有效 Labels 中减去 Batch Size

首位已 mask 时会少计；直接检查 shift 后的 targets。

### `acc_norm` 一定按 Token 归一化

实际分母由 task/版本定义，不能靠指标名字猜测。

### PPL 更低等于自由生成一定更好

Teacher forcing 与模型自生成历史上的行为不同。

### `pass@k` 等于系统一定能挑中正确候选

它只是至少一条通过的概率估计，不提供 oracle 选择器。

### ECE 很低就说明答案正确率高

校准良好不代表判别能力强；总是给基础正确率也可能有低 ECE。

### CI 覆盖所有不确定性

题目 bootstrap 不自动覆盖训练 seed、解码或 judge 偏差。

### 两条 CI 重叠就表示两模型没有差异

比较同批题目应分析配对差值。

### 从许多 Test 结果挑最高，再称为独立测试

这是选择偏差，普通 bootstrap 不能修复。

### 只对成功返回的题目计分

Timeout、missing 和 parser failure 不能静默从分母消失。

### 一个 Aggregate 就决定替换模型

还要检查关键分层、风险、成本与统计证据。

---

## 38. 概念检查

**为什么 PPL 应最后取指数？**

因为指数非线性，目标是 token 平均 NLL 的指数，不是样本或 batch PPL 平均。

**不同 tokenizer 换成 BPB 就完全公平吗？**

不是，文本预处理、context、边界与覆盖仍须一致。

**Few-shot 是否会更新参数？**

这里指提示中的示例，不包含梯度训练，但示例选择会影响评测。

**Calibration temperature 和 decoding temperature 相同吗？**

形式相似，目的和作用不同：前者校准固定预测的 confidence，后者改变生成采样。

**CI 包含 0，能否说模型等价？**

不能，只能说当前证据不足以确定差异方向；等价性需另定义容忍范围与检验。

**同一题的 100 次生成可以当 100 个独立题吗？**

不可以，应保留题目层级依赖，按目标指标选择重采样单位。

**为什么要共同 Clean Subset？**

各自 clean 集可能不同，配对比较需要同样的题目。

**Latest-Good 和 Best-Dev 有什么区别？**

前者是恢复点，后者是开发评估选择；不要让评测选择破坏训练恢复协议。

---

## 39. 最小测试与今日练习

### 正确性测试清单

1. 10-token mean NLL=2、90-token mean NLL=4，整体 NLL 应为 3.8。
2. 增加 PAD 或重新分 batch，不改变总 NLL 与 count。
3. shift 前首个 label 已 mask 时，不重复扣除 count。
4. 滑窗的 `document_id + offset` 目标覆盖无重复。
5. 单卡/多 rank 样本集合、NLL sums 和 counts 一致。
6. A/B 分数完全相同，paired delta 和所有 bootstrap 差值均为 0。
7. B 每题比 A 固定增加 `d`，delta 与 CI 都为 `d`。
8. 交换 A/B，delta 与区间符号翻转。
9. 同时重排 A/B，observed delta 不变。
10. pass@k 边界：`c=0` 得 0，`c=n` 得 1，`k=1` 得 `c/n`，拒绝 `k>n`。
11. `n=10,c=2,k=3` 得 `8/15`。
12. 四题 confidence 均为 0.75、正确性为 `[1,1,1,0]`：ECE=0，binary Brier=0.1875。
13. Parser 覆盖空答、多答案、单位、截断和 refusal。
14. 修改 checkpoint、prompt、数据或 generation 参数，缓存正确失效。
15. 注入 missing/timeout，report 不会悄悄缩小分母。
16. Judge A/B 换位与无信息冗长测试能暴露不合理偏差。

### 本次整理的校验范围

已使用合成数据验证：

- Python 示例的语法；
- paired bootstrap 的相同分数、固定偏移、交换方向不变量；
- token 加权 NLL=3.8，PPL≈44.7012；
- pass@k 的上述组合公式与边界；
- ECE=0、binary Brier=0.1875 的数值示例。

未运行真实模型、PyTorch 前向、分布式评测、LLM judge 或正式 benchmark。PAD、滑窗、分布式与 parser 集成测试仍是后续练习，不应把示例数值记成模型实测成绩。

### 动手练习

1. 为两个候选 checkpoints 编写冻结 manifest，暂不运行。
2. 实现 NLL 的 shift/mask/PAD 单元测试。
3. 完整阅读一个 harness task 配置，写出 prompt、few-shot split 和 metric 分母。
4. 用 stable ID 对齐逐题结果，实现 paired/cluster bootstrap。
5. 比较 accuracy、Brier、不同分箱下的 ECE。
6. 建立 parser regression suite 与失败分类。
7. 设计不超过十项的主 scorecard，事先声明退化门槛。
8. 在隔离环境完成 smoke test 后，再跑完整正式评测。

---

## 40. 参考资料

- [Hugging Face：Perplexity of Fixed-Length Models](https://huggingface.co/docs/transformers/main/perplexity)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Harness 模型接口](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)
- [Harness CLI](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
- [Harness Task 计分源码](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py)
- [HELM](https://crfm.stanford.edu/helm/latest/)
- [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [NLP 统计显著性检验指南](https://aclanthology.org/P18-1128/)
- [LLM-as-a-Judge / MT-Bench](https://arxiv.org/abs/2306.05685)
- [PyTorch Inference Mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

---

## 今日总结

今天把“checkpoint 正常运行”推进到“checkpoint 的价值可被可靠比较”：

1. 分数属于模型、数据、prompt、tokenizer、解码、评分与聚合共同组成的协议；
2. NLL/PPL 必须按 shift 后有效 token 聚合；
3. likelihood、自由生成、代码执行测量不同能力；
4. calibration 要明确 confidence 事件，独立拟合；
5. checkpoint 比较保留逐题 pairing 与 cluster 结构；
6. CI 不覆盖所有不确定性，统计显著不等于实际有用；
7. 污染、test 选择和 judge 偏差可能制造虚假进步；
8. manifest 与逐题日志是复现和错误分析基础；
9. missing、timeout、parser failure 与分母必须透明；
10. Latest-Good、Best-Dev、Release-Candidate 是不同决策。

最核心的一句话是：

> 可靠评测不是给 checkpoint 打一个漂亮分数，而是在冻结协议下，用逐题证据说明它哪里更好、哪里退步，以及我们对判断有多大把握。

## 明日衔接建议

从“如何评估模型”继续进入“如何让基础模型学会按指令回答”：学习 Supervised Fine-Tuning 的数据组织、chat template、assistant-only loss mask、多轮对话边界、packing、全参数与 LoRA 微调，以及如何用今天的 evaluation harness 检查能力提升和遗忘。
