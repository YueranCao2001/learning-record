# 2026-09-08：Direct Preference Optimization——偏好对、Reference Model、DPO Loss 与长度偏差

## 今日目标

上一篇[SFT：Chat Template、Assistant-only Loss、Packing 与 LoRA](2026-09-07-supervised-fine-tuning-chat-template-lora.md)建立了监督微调的数据与训练协议。今天继续学习 DPO：当同一个问题有两个候选回答，并且知道哪个更好时，如何把这种相对偏好转化为模型更新。

完成后应能回答：

1. 偏好数据与 SFT 数据有什么区别？
2. Bradley–Terry 偏好概率怎样连接 KL 正则目标与 DPO？
3. 为什么每个偏好对需要 policy/reference 的四个回答 log probability？
4. beta、reference 与 completion mask 分别控制什么？
5. 为什么 preference accuracy 上升，不等于生成质量必然上升？
6. 怎样用合成测试发现正负样本颠倒、shift、padding 与缓存错误？

本篇包含数学推导、纯 Python 数值练习与待运行的框架骨架。练习结果来自合成输入；真实模型训练、生成评测和 GPU 性能实验留作后续动手任务。

---

## 1. 从“模仿回答”到“比较两个回答”

SFT 的样本通常告诉模型：给定问题，应该输出这份示范答案。偏好样本则提供同一问题下的两个回答，以及一个比较标签。

```text
prompt x
  ├─ chosen   y_w：在指定评价标准下更受偏好
  └─ rejected y_l：相对较差
```

chosen 不一定完美，rejected 也不一定全错。它们的标签只表示在某个任务、评审标准和上下文中存在相对偏好。若评审只是偏爱更长、更自信或格式更漂亮的回答，模型也可能学到这些捷径。

本文聚焦固定离线偏好数据、固定 reference、普通 sigmoid loss 的 DPO。数据收集阶段可以生成候选，但这个基础训练循环不需要每一步在线生成 rollout，也不需要单独训练显式 reward model。[DPO 原论文](https://arxiv.org/abs/2305.18290)

---

## 2. 一条偏好记录的最小结构

推荐使用显式 prompt，避免从两个完整回答中猜共同前缀：

```json
{
  "id": "preference-000042",
  "group_id": "source-question-17",
  "prompt": [
    {"role": "user", "content": "Why must padding be excluded from loss?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "Padding is batching structure, so it should not contribute training targets."}
  ],
  "rejected": [
    {"role": "assistant", "content": "Padding contains the most important semantic information."}
  ],
  "rubric_version": "accuracy-v1",
  "label_source": "synthetic-teaching-example"
}
```

这里的 prompt 只保存一次，chosen/rejected 都是后续 completion。生产数据还应保存候选生成模型、采样配置、评审来源、原始打分、tie/abstain、去重组和数据许可。示例中的偏好标签是教学构造，并非真实评审实验。

TRL 支持标准文本和 conversational preference 格式；对 conversational 格式会应用 chat template。[TRL Dataset Formats](https://huggingface.co/docs/trl/v1.12.0/dataset_formats)

---

## 3. 比较标签需要先过数据检查

同一偏好对必须共享相同问题、历史消息、工具定义和可见上下文。否则“选中哪个回答”可能只是比较了不同问题。

进入训练前检查：

- ID 唯一；同一 prompt、来源组及其所有候选对留在同一个 split。
- prompt 和两侧 completion 非空，角色顺序符合项目 schema。
- chosen/rejected 不能完全相同；渲染、分词和截断后还要再检查一次。
- 去除近重复泄漏，并扫描 benchmark prompt/answer 污染。
- 记录 tie、低置信度和标注冲突；普通硬标签 DPO 不应强行把平局变成 chosen。
- 区分准确性、指令遵循、风格等 rubric，避免把相互矛盾的标准混成一个无条件标签。

一个 prompt 产生很多偏好对时，它会在 pair mean 中获得更多权重。按 prompt 抽样或重新加权会改变优化分布，应显式记录。

---

## 4. Bradley–Terry：把奖励差转成偏好概率

假设存在标量 reward，给定 prompt 和 completion 后输出偏好分数。Bradley–Terry 模型把比较概率写为：

```math
P(y_w \succ y_l\mid x)
=
\frac{\exp(r(x,y_w))}
{\exp(r(x,y_w))+\exp(r(x,y_l))}
=
\sigma\!\left(r(x,y_w)-r(x,y_l)\right)
```

其中 sigmoid 把实数映射到 0 到 1。reward 的绝对零点不重要：对同一个 prompt，两侧 reward 同时加一个常数，比较概率不变。

这是一种建模假设。真实偏好可能依赖评审者、展示顺序或多个无法压缩到单一标量的维度；它不能保证所有偏好都具有稳定的全局排序。

---

## 5. KL 正则为何引入 Reference

先看一个理想化的 reward 最大化目标：

```math
\max_{\pi}\;
\mathbb{E}_{x}
\left[
\mathbb{E}_{y\sim\pi(\cdot\mid x)}r(x,y)
-
\beta D_{\mathrm{KL}}
\!\left(\pi(\cdot\mid x)\,\Vert\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)
\right],
\qquad \beta \gt 0
```

reference 是固定的行为基准，通常选择进入 DPO 前的 SFT checkpoint。KL 项约束新策略相对这个基准的变化；这里的方向是 policy 到 reference。

对给定 reward、正 beta、合适的支持集及可归一化条件，该目标的最优策略为：

```math
\pi^*(y\mid x)
=
\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}
{Z(x)},
\qquad
Z(x)=
\sum_y \pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)
```

这是理论上的最优分布表达式。有限数据、神经网络参数约束和优化误差都会使实际训练偏离理想情形。

---

## 6. Reward 可以用 Policy 的 Log-ratio 表示

对上一式取对数并移项：

```math
r(x,y)
=
\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
+
\beta\log Z(x)
```

将最优策略替换成可学习的 policy，即得到隐式 reward 参数化。对同一 prompt 的两个回答求差时，归一化常数对应的项抵消：

```math
r_\theta(x,y_w)-r_\theta(x,y_l)
=
\beta
\left[
\log\frac{\pi_\theta(y_w\mid x)}
{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}
{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]
```

这样就能把偏好比较直接写成 policy/reference 的 log probability，不需要显式计算覆盖所有回答的配分函数。

---

## 7. 四个 Log Probability 与一个 Margin

对每条偏好对，分别计算：

| 量 | 含义 | 需要梯度 |
|---|---|---|
| `logp_w` | policy 对 chosen 的 completion log probability | 是 |
| `logp_l` | policy 对 rejected 的 completion log probability | 是 |
| `ref_w` | reference 对 chosen 的 completion log probability | 否 |
| `ref_l` | reference 对 rejected 的 completion log probability | 否 |

定义未缩放 margin：

```math
m_\theta
=
(\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x))
-
(\log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x))
```

DPO 比较的是 policy 相对于 reference 的 chosen/rejected 对数优势变化。因此 policy 对 chosen 的绝对概率较低，并不自动意味着 DPO margin 为负。

---

## 8. 普通 DPO 的 Loss

对偏好数据集取负对数似然：

```math
\mathcal{L}_{\mathrm{DPO}}(\theta)
=
-\mathbb{E}_{(x,y_w,y_l)\sim D}
\log \sigma(\beta m_\theta)
```

单对损失也可以稳定地写成：

```math
\ell=\mathrm{softplus}(-\beta m_\theta)
```

实际实现使用 `logsigmoid` 或稳定的 softplus；直接算 `-log(sigmoid(z))` 在极端负 margin 时可能下溢。

这里对偏好对做平均。每个回答内部的 log probability 则是 token log probability 之和；这两个归约层次应分清。

---

## 9. 梯度方向与 Beta 的两个层面

对未缩放 margin 求导：

```math
\frac{\partial\ell}{\partial m_\theta}
=
-\beta\,\sigma(-\beta m_\theta)
```

在把两个 policy log probability 暂视作独立坐标时，梯度下降推动 chosen 上升、rejected 下降。真实神经网络共享参数，两侧并不独立；因此不能承诺每一步 chosen 的绝对概率都会上升。

beta 要结合两个层面理解：

- 在固定 reward 的 KL 正则问题中，beta 越大，最优分布通常越接近 reference。
- 在 DPO 的有限步训练中，beta 同时缩放 margin 和梯度；例如初始 margin 为 0 时，导数为 `-beta/2`。

所以 beta 不是“越大更新越小”的学习率旋钮。它与学习率、训练步数、数据噪声及 reference 一起决定实际效果，需要通过开发集选择。

---

## 10. Policy 等于 Reference 时应发生什么

若两者从相同权重开始，使用相同 token、mask、dtype 与确定性前向，四个 log probability 两两相等：

```math
m_\theta=0,\qquad
P(y_w\succ y_l\mid x)=\frac12,\qquad
\ell=\log 2\approx0.693147
```

这是很有价值的初始化检查。允许声明范围内的浮点误差；若初始 loss 差异很大，应排查模板、dropout、reference checkpoint、adapter 状态和缓存。

相同权重不等于同一个可变对象。若让 reference 与训练 policy 共享会被更新的参数存储，优化一步后基准也会漂移。

---

## 11. 一个可以手算的例子

取四个 sequence log probability：

```text
policy chosen   = -8
policy rejected = -10
reference chosen   = -9
reference rejected = -9
beta = 0.1
```

未缩放 margin 为 `(-8 - (-10)) - (-9 - (-9)) = 2`，缩放后为 `0.2`。

对应偏好概率约为 `0.549834`，loss 约为 `0.598139`。若同时交换 policy 和 reference 的 chosen/rejected 两列，margin 变成 `-2`，loss 约为 `0.798139`。

这个例子验证的是公式符号与数值实现。它不代表真实回答质量或训练增益。

---

## 12. Completion Log Probability 必须正确求和

对 completion 的自回归概率：

```math
\log\pi_\theta(y\mid x)
=
\sum_{t=1}^{|y|}
\log\pi_\theta(y_t\mid x,y_{\lt t})
```

prompt 提供条件上下文，不计入回答 log probability。历史 assistant 消息若位于 prompt，也不参与这次 completion 求和。

这与昨天的 assistant-only SFT 有一个关键区别：SFT 可以监督历史多个 assistant 回合；今天的偏好对比较的是在共同 prompt 后分叉的 continuation，不能直接沿用“所有 assistant token 都打分”的 mask。

普通 sigmoid DPO 使用 sequence sum。把它改成 token mean 会改变目标，属于需要另行声明与评估的变体。

---

## 13. Causal Shift、EOS 与 Padding

令每个序列已经包含 prompt 和一侧 completion，mask 的 1 表示要打分的 completion token：

```text
input_ids:        p0 p1 y0 y1 EOS PAD
completion_mask:  0  0  1  1   1   0
logits used:         ↑  ↑  ↑
targets:            y0 y1 EOS
```

模型位置 `t-1` 的 logit 预测位置 `t` 的 token，因此第一个 completion token 由最后一个 prompt token 的 logit 预测。绝不能漏掉这个位置。

EOS/turn-end 是否算入 completion 由固定模板协议决定；两侧和两个模型必须一致。PAD 即使与 EOS 共用 ID，也只能根据位置级 attention mask 排除，不能按 token ID 全部删除。

对无前文的序列首位不能打分。左 padding 后第一个真实 token 的前驱若是 PAD，也应排除。

---

## 14. 纯 Python：稳定的 Sequence Scoring

下面的教学函数处理一条未 packing 的序列。`logits[i]` 表示在位置 `i` 预测下一 token 的词表 logits；它返回 completion 的 log probability sum 和实际计分数。

```python
import math

def logsumexp(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("finite, non-empty logits required")
    largest = max(values)
    return largest + math.log(sum(math.exp(v - largest) for v in values))

def completion_logp(logits, input_ids, completion_mask, attention_mask):
    n = len(input_ids)
    if n < 2 or any(
        len(v) != n for v in (logits, completion_mask, attention_mask)
    ):
        raise ValueError("invalid sequence lengths")
    for mask in (completion_mask, attention_mask):
        if any(v not in (0, 1) for v in mask):
            raise ValueError("mask must contain only 0/1")

    real = [i for i, value in enumerate(attention_mask) if value]
    if not real or real != list(range(real[0], real[-1] + 1)):
        raise ValueError("real tokens must form one contiguous span")

    score, count = 0.0, 0
    for target_pos in range(1, n):
        if not (
            completion_mask[target_pos]
            and attention_mask[target_pos]
            and attention_mask[target_pos - 1]
        ):
            continue
        row = logits[target_pos - 1]
        token = input_ids[target_pos]
        if not isinstance(token, int) or not 0 <= token < len(row):
            raise ValueError("target ID outside vocabulary")
        if not row or not all(math.isfinite(v) for v in row):
            raise ValueError("finite, non-empty logits required")
        largest = max(row)
        centered = [value - largest for value in row]
        score += centered[token] - logsumexp(centered)
        count += 1
    if count == 0:
        raise ValueError("no scoreable completion targets")
    return score, count

def softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))

def dpo_pair(logp_w, logp_l, ref_w, ref_l, beta=0.1):
    if not all(math.isfinite(v) for v in (logp_w, logp_l, ref_w, ref_l, beta)):
        raise ValueError("finite scores and beta required")
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = (logp_w - logp_l) - (ref_w - ref_l)
    z = beta * margin
    if not math.isfinite(z):
        raise ValueError("scaled margin overflow")
    loss = softplus(-z)
    return {"margin": margin, "scaled_margin": z, "loss": loss}
```

真实模型应采用 batch tensor 实现，并验证它与这个小规模 oracle 一致。该函数不处理 packed segment，不能把它直接套到跨样本拼接的输入。

---

## 15. 数值和边界测试

```python
def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

def test_dpo_math():
    same = dpo_pair(-8, -10, -8, -10)
    assert same["margin"] == 0
    assert math.isclose(same["loss"], math.log(2))

    good = dpo_pair(-8, -10, -9, -9)
    swapped = dpo_pair(-10, -8, -9, -9)
    assert good["margin"] == 2
    assert math.isclose(good["loss"], 0.5981388693815918)
    assert math.isclose(swapped["loss"], 0.7981388693815918)
    assert good["loss"] < same["loss"] < swapped["loss"]

    # 两侧 reference 同加常数，差值和 loss 不变。
    shifted_ref = dpo_pair(-8, -10, -4, -4)
    assert math.isclose(good["loss"], shifted_ref["loss"])
    assert math.isfinite(dpo_pair(-10000, 0, 0, 0, beta=1)["loss"])
    assert math.isfinite(dpo_pair(0, -10000, 0, 0, beta=1)["loss"])
    expect_value_error(lambda: dpo_pair(-1, -2, -1, -2, beta=0))

    # 在合成 log-prob 坐标上验证 loss 对 chosen 的导数。
    eps = 1e-5
    numerical = (
        dpo_pair(-8 + eps, -10, -9, -9)["loss"]
        - dpo_pair(-8 - eps, -10, -9, -9)["loss"]
    ) / (2 * eps)
    analytical = -0.1 / (1 + math.exp(0.2))
    assert math.isclose(numerical, analytical, rel_tol=1e-6)

def test_completion_scoring():
    logits = [
        [0.0, 2.0, -1.0],
        [0.0, -1.0, 2.0],
        [7.0, -7.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    # token 2 同时用作真实 EOS 与 PAD，按位置区分。
    ids = [0, 1, 2, 2]
    masks = [0, 1, 1, 1], [1, 1, 1, 0]
    actual, count = completion_logp(logits, ids, *masks)
    expected = (
        logits[0][1] - logsumexp(logits[0])
        + logits[1][2] - logsumexp(logits[1])
    )
    assert count == 2
    assert math.isclose(actual, expected)

    # log-softmax 对公共偏置不变，先居中避免大数相消。
    huge, huge_count = completion_logp(
        [[1e16, 1e16], [0.0, 0.0]],
        [0, 1], [0, 1], [1, 1],
    )
    assert huge_count == 1
    assert math.isclose(huge, -math.log(2))

    # 改变只会预测 PAD 的 logit，不应改变回答分数。
    modified = [row[:] for row in logits]
    modified[2] = [-99.0, 99.0, 0.0]
    assert completion_logp(modified, ids, *masks) == (actual, count)

    # 左侧补 PAD，原序列与 logits 一起平移，分数不变。
    padded = completion_logp(
        [[0.0, 0.0, 0.0]] + logits,
        [2] + ids,
        [0] + masks[0],
        [0] + masks[1],
    )
    assert padded == (actual, count)
    expect_value_error(
        lambda: completion_logp(logits, ids, [0, 0, 0, 0], masks[1])
    )
    expect_value_error(
        lambda: completion_logp(logits, ids, [1, 0, 0, 0], masks[1])
    )
    expect_value_error(
        lambda: completion_logp(logits, ids, [0, 2, 1, 0], masks[1])
    )

test_dpo_math()
test_completion_scoring()
```

这些测试覆盖初始化、正负方向、常数抵消、数值稳定性、有限差分梯度、首次回答 token、EOS/PAD、左 padding 与零有效目标。它们验证合成计算路径，真实 tokenizer 和模型集成仍需单独验收。

---

## 16. PyTorch 中保持正确的梯度边界

下面的 loss 接收已经完成 completion 求和的四个 `[batch]` tensor：

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_w, policy_l, reference_w, reference_l, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    if policy_w.ndim != 1 or policy_w.numel() == 0:
        raise ValueError("expected a non-empty [batch] tensor")
    if any(
        tensor.shape != policy_w.shape
        for tensor in (policy_l, reference_w, reference_l)
    ):
        raise ValueError("pair score shapes must match")

    policy_margin = policy_w.float() - policy_l.float()
    reference_margin = reference_w.detach().float() - reference_l.detach().float()
    scaled_margin = beta * (policy_margin - reference_margin)
    return -F.logsigmoid(scaled_margin).mean()
```

reference 前向应使用 `no_grad()` 并关闭 dropout；policy 的 chosen 和 rejected 两侧都保留计算图。对 policy 分数做 `detach()` 会直接破坏训练。

这里的 `float()` 在 pair loss 阶段使用 FP32。模型侧的 log-softmax、gather 和 sequence reduction 也应选择合适精度；丢失的信息无法靠最后一次转换恢复。

---

## 17. Chat Template 应先于 Log Probability 缓存

沿用昨天的 tokenizer 和模板 revision，分别形成：

```text
prompt + chosen   → input_ids_w + completion_mask_w
prompt + rejected → input_ids_l + completion_mask_l
                    ↓ 同一 tokenization / mask 规则
             policy scores + reference scores
```

不能假定“单独 tokenize(prompt)”必然是“tokenize(prompt + completion)”的逐 token 前缀。BPE、空格、角色标记和模板的 generation prefix 都可能影响边界。

保存 golden fixtures：完整 rendered text、token IDs、首个 completion token、实际 target 坐标和 end-of-turn。对两侧确认 prompt 前缀相同；需要模型特定边界处理时，把规则固定为预处理版本。

在 TRL v1.12.0 的 conversational 路径，EOS 由模板决定；模板前缀不匹配时实现可能只发出 warning，因此预检必须把它升级为明确失败，不能继续信任切片得到的 completion 边界。

不要先自行渲染一次，又把结构化消息交给 Trainer 重复套模板。显式声明由哪一层负责渲染，并检查最终输入。

---

## 18. 截断会改变偏好问题本身

如果 chosen 的关键结论在尾部，截断后标签可能失去依据。两条原本不同的回答也可能截成完全相同的 token 序列。

项目基线采用完整偏好对的长度预检：

1. 按最终模板 tokenize 两侧完整序列。
2. 任意一侧超过预算，按预先声明的规则整对过滤或重新构造。
3. 不允许某一侧只剩 prompt、零 completion target 或丢掉关键语义。
4. 记录两侧保留 token 数、过滤率和来源分布，检查是否系统性排除了某类任务。

TRL `v1.12.0` 的 `max_length` 对完整 tokenized sequence 生效；本篇显式使用 `truncation_mode="keep_start"`。不能沿用旧教程里的长度参数而不核对当前 API。[DPOConfig](https://huggingface.co/docs/trl/v1.12.0/dpo_trainer#trl.DPOConfig)

配置中的截断上限仍应存在，但数据预检需要保证正常样本不会在训练时被静默改写。开发集与训练集必须采用同一组预处理规则。

---

## 19. 长度偏差有两个来源

第一种来自数据标签：如果更长的回答更常被选中，模型可能学习冗长性。第二种来自打分目标：sequence log probability 是 token 项的累加，长度变化会影响数值和梯度。

reference subtraction 能提供相对基准，但不能保证自动消除长度偏差。即便两侧平均 token log-ratio 相同，较长一侧的 sequence log-ratio 仍可能更大或更小，取决于各项符号。

评估至少分层记录 chosen/rejected 长度差、长度相近子集、生成长度分布和质量指标。不要把“除以长度”当成无影响的修复；若使用 `sigmoid_norm` 或其他变体，应重新声明 loss，并与基础 sigmoid DPO 分开比较。

---

## 20. Reference 冻结、缓存与 Adapter 状态

固定 reference 可以逐 batch 前向，也可以预计算每条记录两侧的 reference log probability。缓存有效的前提是生成它的模型和预处理定义完全一致。

缓存键至少绑定：

```text
reference weights + adapter state
tokenizer + chat template revision
sample ID + chosen/rejected token hashes
completion mask + EOS + truncation policy
sequence-sum scoring version + compute dtype
```

改变这些字段时重新计算缓存。抽样用在线 reference 前向验证 cache，再按 stable ID join；不能仅依赖 dataset row index。

v1.12.0 的预计算需要 map-style Dataset，不能与 reference 同步或 Liger 一起使用。显式传入的 reference 仍可由 Trainer 持有，打开 precompute 不代表它会自动释放显存；后续新增评估数据也需要原始 reference 或对应的有效缓存。

LoRA 场景尤其要检查 reference 究竟包含哪个 adapter。关闭 DPO adapter 后露出的模型，未必就是期望的 SFT reference；若 SFT 本身由另一个 adapter 承载，需要确认其仍处于正确激活状态。

下面的框架骨架使用两个独立的完整 SFT 模型对象，并关闭 precompute，先建立容易检查的基线。

---

## 21. TRL v1.12.0 的最小配置骨架

假设 `policy` 和 `reference` 是从同一固定 SFT checkpoint 独立加载的 causal LM；`tokenizer` 有经过验证的 chat template；`train_pairs`、`dev_pairs` 是完成长度和数据预检的 map-style conversational preference datasets。

```python
from trl import DPOConfig, DPOTrainer

if policy is reference:
    raise ValueError("policy and reference must be independent objects")
reference.requires_grad_(False)
reference.eval()

tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is None:
        raise ValueError("a valid padding token must be configured")
    tokenizer.pad_token = tokenizer.eos_token

args = DPOConfig(
    output_dir="outputs/dpo-example",
    loss_type=["sigmoid"],
    beta=0.1,
    label_smoothing=0.0,
    f_divergence_type="reverse_kl",
    use_weighting=False,
    ld_alpha=None,
    max_length=2048,
    truncation_mode="keep_start",
    padding_free=False,
    disable_dropout=True,
    precompute_ref_log_probs=False,
    sync_ref_model=False,
    use_liger_kernel=False,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    report_to="none",
    push_to_hub=False,
    seed=42,
    data_seed=42,
)

trainer = DPOTrainer(
    model=policy,
    ref_model=reference,
    args=args,
    processing_class=tokenizer,
    train_dataset=train_pairs,
    eval_dataset=dev_pairs,
)
# 完成真实 batch 的 token/mask/初始化检查后，再调用 trainer.train()。
```

这些数值只是 full-model 教学实验起点。精度、batch、学习率和显存预算需要结合实际硬件与模型确定；本文未实例化 Trainer 或运行训练。

tokenizer 的 left padding 设置遵循生成接口约定；这个 tag 的默认文本训练 collator 实际使用 right padding，仍须检查真实 batch 的 attention/completion masks。该版本还会将 `padding_free=True` 回退为 False，本篇直接关闭，避免把 SFT 的 padding-free 行为迁移过来。`label_smoothing=0.0` 表示基础硬标签设定；不能仅改变这个参数就假定 sigmoid 分支启用了平滑标签。

本篇使用 v1.12.0 的字段，避免把 SFTConfig 的 `assistant_only_loss` 或 `packing` 直接照搬进 DPOConfig。实际完成分数由 DPO 的 completion 边界控制。部署前锁定 TRL、Transformers、PEFT 和 PyTorch 版本，并保存 resolved config。[DPOTrainer tagged source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/dpo_trainer.py) [DPOConfig tagged source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/dpo_config.py)

---

## 22. 怎样读训练日志

至少同时观察四个原始 sequence log probability、margin、loss、completion 长度和 chosen NLL。

| 指标 | 可以帮助判断 | 不能单独证明 |
|---|---|---|
| DPO loss | 对固定偏好数据的拟合 | 实际生成质量提高 |
| implicit reward margin | 相对 reference 的偏好分离 | 绝对质量或跨 reference 可比性 |
| preference accuracy | 正 margin 的比例 | 新问题上的任务成功率 |
| chosen/rejected logp | margin 是怎样变化的 | 仅靠一个方向判定训练健康 |
| 长度与停止率 | 冗长、重复、未终止倾向 | 准确性与任务完成度 |
| target-dev + retention-dev | 目标改善与原能力保持 | 未触碰测试集成绩 |

`rewards/accuracies` 通常使用严格大于判断。policy 等于 reference 时理论 margin 全是 0，严格准确率是 0；这不矛盾于 Bradley–Terry 概率为 0.5。若自定义统计给平局半分，必须说明口径。

两个回答的 policy log probability 可以同时下降，只要 rejected 相对下降更多，DPO margin 仍会改善。应同时检查 chosen NLL 与生成行为，不能只看 margin。

---

## 23. Pair 指标与生成评测分开做

固定 dev preference pairs 用来检查损失和排序能力。真实生成评测则需要给相同 prompts，让 SFT baseline 与 DPO checkpoint 分别作答，再按相同协议评分。

沿用[Checkpoint 评估](2026-09-04-checkpoint-evaluation-harness.md)中的冻结 evaluation harness：

- 固定 prompt IDs、模板、generation config 和评估器版本。
- 同时评估目标任务、基础能力、停止行为、长度与成本。
- 如用 judge，交换回答展示顺序、保留平局和原始判词，抽样人工复核。
- 按 prompt/source group 聚合和 bootstrap，避免把同一 prompt 的多对样本当成独立观测。
- checkpoint 选择使用 target-dev、retention-dev 等开发指标；选择冻结后才打开 sealed test。

DPO 数据必须与这些评估数据隔离。偏好打分高也可能源于评审器的格式或长度偏好，需用任务证据检查。

---

## 24. 保存与恢复需要带上“偏好协议”

除昨天的训练状态外，再保存：

```yaml
objective: offline-dpo-sigmoid
policy_init_revision: fixed-sft-revision
reference_revision: same-fixed-sft-revision
reference_adapter_state: none-for-full-model-baseline
reference_sync: false
beta: 0.1
pair_reduction: mean-over-valid-pairs
sequence_reduction: sum-over-completion-targets
label_smoothing: 0.0
dataset_snapshot: immutable-id
preference_rubric_version: fixed-version
split_manifest_hash: sha256:...
chat_template_hash: sha256:...
tokenizer_revision: fixed-revision
length_policy: reject-whole-pair-if-either-side-exceeds-2048
completion_eos_policy: model-specific-golden-fixture
reference_cache: disabled-for-baseline
checkpoint_global_step: actual-value
pair_ids_and_sampler_state: stored
optimizer_scheduler_rng: stored
```

恢复时要核对 reference revision，而不只是加载 policy checkpoint。否则相同数据可能对应不同的 margin 和训练目标。

模型保存用于推理；精确续训还需要 optimizer、scheduler、RNG、采样器及累积步状态。按昨天约定，在 optimizer-step 边界保存并验证连续运行与断点恢复的一致性。

---

## 25. 常见错误与排查入口

| 现象 | 优先检查 |
|---|---|
| 初始 loss 明显偏离 log 2 | reference/adapter、dropout、模板、缓存 |
| loss 方向反了 | chosen/rejected 顺序、reference 列顺序、负号 |
| padding 长度改变 loss | 位置 mask、shift、position IDs、归约方式 |
| chosen/rejected 截断后相同 | 完整偏好对长度预检、重新标注或过滤 |
| 训练 dev 很好，生成退化 | 长度捷径、chosen NLL、数据污染、judge 偏差 |
| 切换 tokenizer 后 loss 异常 | 缓存失效、EOS、completion 边界 |
| beta 变大却没有更稳 | 学习率/步数共同作用、margin 饱和、过拟合 |
| adapter 训练 reference 漂移 | 可变权重共享、adapter 切换与冻结状态 |

---

## 26. 今日练习与遗留问题

### 本次已完成的教学验证

- 纯 Python DPO loss 的初始化值为 `log(2)`。
- chosen/rejected 交换后 margin 反号，损失方向符合手算。
- 极端 margin 的 softplus 计算保持有限。
- chosen log probability 的有限差分与解析导数一致。
- completion scoring 使用前一个位置的 logit，并包含第一个回答 token。
- 改变 PAD 位置相关 logit、增加左 padding 均不改变合成回答分数。
- EOS 与 PAD 共用 ID 时，真实 EOS 仍计分。
- 零有效 completion、非法 mask、非正 beta 会被拒绝。

### 后续需要在真实模型上验证

1. 抽取三条真实模板样本，逐 token 显示 prompt/completion 边界。
2. 核对 policy/reference 初始化 log probability 和 loss。
3. 与显式 tensor log-softmax/gather/sequence sum 对比 Trainer 分数。
4. 训练一步后，确认 policy 有有限梯度、reference 权重与分数不变。
5. 对同一批数据比较在线 reference 与预计算缓存。
6. 跑小规模 beta × learning-rate 实验，检查目标质量、保留能力和生成长度。

遗留问题：偏好噪声有多大、chosen 是否真的更准确、长度偏差如何量化，以及离线候选分布与当前 policy 的输出分布相差多少。这些要靠数据和实验回答，不能由 DPO 公式直接推出。

---

## 参考资料

- [Rafailov 等：Direct Preference Optimization（原论文）](https://arxiv.org/html/2305.18290v3)
- [DPO 作者实现](https://github.com/eric-mitchell/direct-preference-optimization)
- [TRL v1.12.0：DPOTrainer](https://huggingface.co/docs/trl/v1.12.0/dpo_trainer)
- [TRL v1.12.0：DPOConfig 源码](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/dpo_config.py)
- [TRL v1.12.0：DPOTrainer 源码](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/dpo_trainer.py)
- [TRL v1.12.0：Dataset Formats](https://huggingface.co/docs/trl/v1.12.0/dataset_formats)
- [Transformers：Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)
- [PyTorch：LogSigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html)

## 今日总结

DPO 把“同一个 prompt 下更偏好哪条回答”变成相对于固定 reference 的 log-ratio 分类问题。理解损失公式之后，实现最容易出错的地方仍然是 completion 边界、causal shift、EOS、截断和 reference 身份。

今天建立了一个可手算、可运行合成测试的基线：四个 sequence log probability、一个 margin、稳定 sigmoid loss，再用独立生成评测判断是否获得实际改善。

## 下次衔接建议

继续学习在线强化学习式后训练：从离线 DPO 过渡到 rollout、reward、advantage 与 policy ratio，理解 PPO/GRPO 的基本差异，以及采样、KL、长度和奖励投机为何需要单独监测。
