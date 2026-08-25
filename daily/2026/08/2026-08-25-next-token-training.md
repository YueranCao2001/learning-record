# 2026-08-25：语言模型训练目标——Label Shift、Cross-Entropy 与 Perplexity

## 今日目标

- 理解自回归语言模型为什么把文本训练转化为 next-token prediction。
- 掌握 input、label 和 logits 在时间维度上的对齐关系。
- 区分 label shift、causal mask 与 padding mask。
- 理解 teacher forcing 的训练方式及其与推理阶段的差异。
- 从概率、负对数似然和梯度三个角度理解 cross-entropy。
- 正确计算 token-level loss 与 perplexity。
- 用 NumPy 和 PyTorch 实现最小训练目标，并检查常见的 off-by-one 错误。

## 1. 从生成过程反推训练目标

昨天的生成流程是：

```text
历史 token
→ Transformer
→ 当前最后位置的 logits
→ 采样下一个 token
```

训练时已经拥有完整文本，因此可以把同一序列拆成多个监督样本。

假设 token 序列为：

```text
[BOS, 我, 喜欢, 学习, EOS]
```

模型需要学习：

```text
P(我 | BOS)
P(喜欢 | BOS, 我)
P(学习 | BOS, 我, 喜欢)
P(EOS | BOS, 我, 喜欢, 学习)
```

同一段长度为 5 的序列提供了 4 个 next-token prediction 目标。

## 2. 自回归概率分解

对于 token 序列 `x_1, x_2, ..., x_T`，链式法则给出：

```text
P(x_1, x_2, ..., x_T)
= Π_t P(x_t | x_<t)
```

取对数后：

```text
log P(x_1, ..., x_T)
= Σ_t log P(x_t | x_<t)
```

最大化真实序列的 log probability，等价于最小化负对数似然：

```text
NLL = -Σ_t log P(x_t | x_<t)
```

当输出类别是整个 vocabulary 时，这就是 token-level cross-entropy loss。

## 3. Label Shift：输入与目标错开一位

原始 token ids：

```text
[101, 42, 87, 19, 102]
```

训练对齐方式：

| 位置 | 模型输入 | 监督目标 |
| ---: | --- | --- |
| 0 | 101 / BOS | 42 / 我 |
| 1 | 42 / 我 | 87 / 喜欢 |
| 2 | 87 / 喜欢 | 19 / 学习 |
| 3 | 19 / 学习 | 102 / EOS |

最直接的切片是：

```python
tokens = [101, 42, 87, 19, 102]

input_ids = tokens[:-1]
labels = tokens[1:]
```

因此：

```text
input_ids: [101, 42, 87, 19]
labels:    [ 42, 87, 19,102]
```

这就是 label shift。第 `t` 个输入位置输出的 logits，要与原序列第 `t + 1` 个 token 比较。

## 4. 两种常见的数据接口

### 方式 A：数据侧显式切片

```python
input_ids = tokens[:, :-1]
labels = tokens[:, 1:]

logits = model(input_ids)
loss = cross_entropy(logits, labels)
```

形状通常为：

```text
input_ids: [B, T - 1]
labels:    [B, T - 1]
logits:    [B, T - 1, V]
```

### 方式 B：把完整序列同时作为 inputs 和 labels

一些语言模型库允许：

```python
outputs = model(
    input_ids=tokens,
    labels=tokens,
)
loss = outputs.loss
```

模型内部会执行类似操作：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

两种方式都正确，但不能同时 shift 两次。使用框架封装前必须确认它是否已经在内部完成对齐。

## 5. Causal Mask 与 Label Shift 不是一回事

这两个操作经常同时出现，但解决的问题不同。

### Causal Mask

限制第 `t` 个隐藏状态只能看到当前位置及之前的 token：

```text
position t can attend to positions <= t
```

它防止模型在计算隐藏状态时偷看未来。

### Label Shift

规定第 `t` 个 logits 应预测哪个真实 token：

```text
logits at position t predicts token at position t + 1
```

只做 causal mask 而不 shift labels，会让模型学习复制当前 token；只 shift labels 而没有 causal mask，则训练时可能从注意力中看到答案。

## 6. Teacher Forcing

训练时，每个位置都使用真实历史 token，而不是模型自己刚刚预测的 token：

```text
真实前缀 → 同时预测所有下一个 token
```

这称为 teacher forcing。

优点：

- 一次前向传播可以并行计算全部位置。
- 训练信号稳定，不必在每一步执行离散采样。
- 每个位置都基于正确上下文学习。

推理时则不同：

```text
模型预测 token
→ 将预测结果加入上下文
→ 再预测下一个 token
```

如果早期生成错误，后续上下文也会改变。这种训练与推理输入分布的差异常被称为 exposure bias。

Teacher forcing 不等于把未来 token 暴露给当前位置；causal mask 仍然必须存在。

## 7. Logits 的形状

Transformer 输出每个位置的隐藏状态：

```text
hidden_states: [B, T, D]
```

LM Head 投影到 vocabulary：

```text
logits = hidden_states @ W_vocab + b
logits: [B, T, V]
```

其中：

- `B`：batch size。
- `T`：序列长度。
- `D`：模型隐藏维度。
- `V`：词表大小。

训练使用全部有效位置的 logits；增量推理通常只使用最后一个位置的 logits：

```text
training: logits[:, :, :]
decoding: logits[:, -1, :]
```

## 8. 单个 Token 的 Cross-Entropy

真实类别 token id 为 `y`，模型输出 logits `z`。

Softmax 概率：

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

单个 token 的 loss：

```text
loss = -log p_y
```

等价地：

```text
loss = logsumexp(z) - z_y
```

如果模型给真实 token 更高概率，loss 会下降：

| 真实 token 概率 | Cross-entropy |
| ---: | ---: |
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |

Cross-entropy 没有有限上界。模型越确信一个错误答案，loss 越大。

## 9. 为什么直接从 Logits 计算

不建议先显式 Softmax，再取 log：

```python
# 数值稳定性较差
probabilities = softmax(logits)
loss = -np.log(probabilities[target])
```

更稳定的方法使用 log-sum-exp：

```python
def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return (
        maximum
        + np.log(
            np.exp(values - maximum).sum(
                axis=axis,
                keepdims=True,
            )
        )
    ).squeeze(axis)


def token_cross_entropy(
    logits: np.ndarray,
    target_id: int,
) -> float:
    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 1:
        raise ValueError("期望一维 vocabulary logits")
    if not 0 <= target_id < logits.size:
        raise ValueError("target_id 超出词表范围")

    log_normalizer = logsumexp(logits)
    return float(log_normalizer - logits[target_id])
```

实际框架中的 `cross_entropy` 通常直接接收未归一化 logits，并在内部完成稳定的 `log_softmax + NLL`。

## 10. Batch 与序列上的平均 Loss

对形状 `[B, T, V]` 的 logits 和 `[B, T]` 的 labels：

```text
loss_{b,t}
= -log P(label_{b,t} | prefix_{b,t})
```

如果所有位置都有效：

```text
mean_loss
= Σ_{b,t} loss_{b,t} / (B × T)
```

但实际 batch 常包含 padding，因此分母应该是有效 token 数，而不是张量总元素数。

## 11. Padding Mask 与 ignore_index

两个样本长度不同：

```text
sample A: [BOS, 我, 学习, EOS]
sample B: [BOS, 你好, EOS, PAD]
```

`PAD` 只是为了组成矩形张量，不应该参与语言模型目标。

常用做法是把无效 label 替换为 `ignore_index`，例如 `-100`：

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100
```

之后 cross-entropy 只统计非 `-100` 位置。

注意：

- `attention_mask` 控制哪些输入位置可以参与注意力或属于有效序列。
- `ignore_index` 控制哪些 label 不计入 loss。
- 两者作用不同，通常需要同时正确设置。

## 12. NumPy 最小实现

```python
import numpy as np


def causal_lm_cross_entropy(
    logits: np.ndarray,
    labels: np.ndarray,
    ignore_index: int = -100,
) -> tuple[float, np.ndarray]:
    """
    logits: [B, T, V]
    labels: [B, T]
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    if logits.ndim != 3:
        raise ValueError("logits 必须为 [B, T, V]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels 必须为 [B, T]")

    batch_size, sequence_length, vocabulary_size = logits.shape
    flat_logits = logits.reshape(-1, vocabulary_size)
    flat_labels = labels.reshape(-1)

    valid = flat_labels != ignore_index
    if not np.any(valid):
        raise ValueError("至少需要一个有效 label")

    valid_labels = flat_labels[valid]
    if np.any(valid_labels < 0) or np.any(valid_labels >= vocabulary_size):
        raise ValueError("有效 label 超出词表范围")

    selected_logits = flat_logits[valid]
    maximum = selected_logits.max(axis=-1, keepdims=True)
    shifted = selected_logits - maximum

    log_normalizer = (
        np.log(np.exp(shifted).sum(axis=-1))
        + maximum.squeeze(-1)
    )

    token_losses = (
        log_normalizer
        - selected_logits[
            np.arange(valid_labels.size),
            valid_labels,
        ]
    )

    return float(token_losses.mean()), token_losses
```

测试：

```python
rng = np.random.default_rng(42)

batch_size = 2
sequence_length = 3
vocabulary_size = 5

logits = rng.normal(
    size=(batch_size, sequence_length, vocabulary_size)
)
labels = np.array([
    [1, 2, 3],
    [4, 0, -100],
])

mean_loss, token_losses = causal_lm_cross_entropy(
    logits,
    labels,
)

print(mean_loss)
print(token_losses)

assert np.isfinite(mean_loss)
assert token_losses.shape == (5,)
```

## 13. PyTorch 最小实现

### 数据侧显式 Shift

```python
import torch
import torch.nn.functional as F


tokens = torch.tensor([
    [101, 42, 87, 19, 102],
    [101, 55, 66, 102, 0],
])

input_ids = tokens[:, :-1]
labels = tokens[:, 1:].clone()

# 第二个样本最后一个 label 对应 PAD
labels[1, -1] = -100

logits = model(input_ids)  # [B, T - 1, V]

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1),
    ignore_index=-100,
)
```

### 完整序列后再 Shift

```python
logits = model(tokens)  # [B, T, V]

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = tokens[:, 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

`contiguous()` 不是数学要求，但切片后的张量可能不是连续内存布局；在调用 `view` 前显式处理更稳妥。也可以使用 `reshape`。

## 14. Cross-Entropy 的梯度直觉

对单个位置，令 one-hot 真实分布为 `y`，模型概率为 `p`。

Cross-entropy 对 logits 的梯度是：

```text
∂L / ∂z_i = p_i - y_i
```

因此：

- 对真实 token：梯度为 `p_y - 1`，推动其 logit 增大。
- 对其他 token：梯度为 `p_i`，推动其 logit 减小。
- 模型越确信正确答案，所有梯度越接近 0。
- 模型越确信错误答案，纠正信号越强。

这个简洁结果是 Softmax 与 cross-entropy 经常组合使用的重要原因。

## 15. Perplexity

如果平均 token loss 使用自然对数：

```text
perplexity = exp(mean_cross_entropy)
```

```python
def perplexity_from_loss(mean_loss: float) -> float:
    return float(np.exp(mean_loss))
```

例如：

| 平均 loss | Perplexity |
| ---: | ---: |
| 0.0 | 1.00 |
| 0.693 | 2.00 |
| 1.099 | 3.00 |
| 2.303 | 10.00 |

直觉上，perplexity 可以理解为模型在每一步面对的“有效候选数量”，但这只是帮助理解的近似解释，不代表模型真的在固定数量候选中均匀选择。

越低通常表示模型对评估文本赋予更高概率。

## 16. Perplexity 的比较边界

Perplexity 只有在评估设置一致时才适合比较：

- 相同 tokenizer。
- 相同 tokenization 规则。
- 相同数据集与切分。
- 相同上下文长度与截断方式。
- 相同 padding/忽略策略。
- 相同 loss reduction。

不同 tokenizer 会把同一文本切成不同 token 数，因此 token-level perplexity 不能直接横向比较。

对于有限上下文模型，评估长文本时使用不重叠切块可能丢失跨块上下文。更严谨的评估会使用滑动窗口，并避免重复计算目标 token。

## 17. Sequence Loss 与 Token Loss

训练时常见的 `mean loss` 是对全部有效 token 取平均，而不是先对每个样本平均再对 batch 平均。

两种方式在样本长度不同时权重不同：

### Token 平均

```text
所有有效 token 的 loss 总和 / 有效 token 总数
```

长序列贡献更多 token。

### Sample 平均

```text
先计算每个样本平均 loss
再对样本取平均
```

每个样本权重相同。

语言模型预训练通常使用 token 平均，但记录指标时必须明确 reduction 方式。

## 18. EOS、BOS 与第一个预测

是否使用 `BOS` 取决于 tokenizer 和模型设计。

有 `BOS` 时：

```text
BOS → 预测第一个正文 token
```

没有显式 `BOS` 时，第一个被计算的目标可能依赖拼接方式、文档边界 token 或上一个上下文 token。

`EOS` 通常应该作为预测目标，因为模型需要学会何时停止；但 `EOS` 后的 padding 不应参与 loss。

## 19. 文档拼接与边界

预训练数据常把多篇文档拼接到固定长度 block 中。需要决定：

- 文档之间是否插入 EOS。
- 新文档是否允许关注前一文档。
- 是否跨文档边界计算 next-token loss。
- 剩余不足一个 block 的 token 如何处理。

如果直接无分隔拼接两篇文档，模型会被迫学习一个并不存在的跨文档转移。插入明确边界 token 可以告诉模型内容已经结束。

## 20. 训练与推理的共同部分

训练和推理共享：

```text
token embedding
→ Transformer blocks
→ final normalization
→ LM Head
→ logits
```

之后开始分叉：

| 阶段 | Logits 之后 |
| --- | --- |
| 训练 | 与真实 next token 计算 cross-entropy，反向传播 |
| 推理 | 应用 temperature/top-k/top-p，选出 token，继续生成 |

训练通常处理全部位置并并行计算 loss；带 KV Cache 的推理通常一次只处理新位置。

## 21. 一次完整训练 Step

```text
1. 文本经过 tokenizer 得到 token ids。
2. 拼接、截断或 padding 成 batch。
3. 构造 input_ids、labels 和 masks。
4. Transformer 计算 [B, T, V] logits。
5. 对齐 shift_logits 与 shift_labels。
6. 忽略 PAD 或其他无监督位置。
7. 计算有效 token 的平均 cross-entropy。
8. 反向传播计算梯度。
9. optimizer 更新参数。
10. 清空梯度并进入下一步。
```

今天聚焦第 3 到第 7 步。梯度传播与 optimizer 留到下一篇。

## 22. 常见错误

- inputs 和 labels 没有错开一位，模型被训练成复制当前 token。
- 框架内部已经 shift，外部又 shift 一次。
- 只做 label shift，却忘记 causal mask。
- 认为 causal mask 会自动完成 label shift。
- 把 Softmax 后的概率传入需要 logits 的 cross-entropy。
- 对概率直接取 log，出现 `log(0)` 或数值下溢。
- flatten logits 时把 vocabulary 维度放错位置。
- labels 中存在超出 `[0, V)` 的有效 token id。
- padding token 被计入 loss。
- attention mask 正确，但 labels 没有设置 ignore index。
- EOS 被误当作 padding 全部忽略，模型学不会停止。
- 平均 loss 时除以张量总长度，而不是有效 token 数。
- 对不同 tokenizer 的模型直接比较 perplexity。
- 对长文本独立切块，忽略上下文截断对 perplexity 的影响。
- 训练 loss 很低就认为生成结果一定事实正确。
- 评估时忘记切换到 eval 模式，Dropout 仍然生效。

## 23. 概念检查

1. 为什么一段长度为 `T` 的序列通常提供 `T - 1` 个监督目标？
2. 第 `t` 个 logits 应与哪个 token 比较？
3. causal mask 和 label shift 分别解决什么问题？
4. teacher forcing 为什么可以并行训练全部位置？
5. teacher forcing 是否允许模型看到未来 token？
6. 为什么 cross-entropy 应直接接收 logits？
7. padding 位置为什么不能计入 loss？
8. `attention_mask` 与 `ignore_index` 有什么区别？
9. cross-entropy 对 logits 的梯度为什么是 `p - y`？
10. perplexity 为什么不适合跨 tokenizer 直接比较？
11. 训练时为什么使用全部位置 logits，而增量推理只使用最后位置？
12. EOS 为什么通常应该成为监督目标？

## 24. 今日练习

- [ ] 用 5 个 token 手工构造 `input_ids` 和右移一位的 `labels`。
- [ ] 画出一个 `4 × 4` causal attention mask。
- [ ] 验证当前位置不能看到未来 token，但可以预测下一个 token。
- [ ] 对三个类别的 logits 手算 Softmax 与 cross-entropy。
- [ ] 给全部 logits 加同一个常数，验证 loss 不变。
- [ ] 实现 NumPy batch cross-entropy，并与 PyTorch 结果比较。
- [ ] 加入两个 padding 位置，验证 mean loss 只除以有效 token 数。
- [ ] 故意 double shift，观察 inputs 与 labels 如何错位。
- [ ] 将真实 token 的 logit 增大 1，观察 loss 如何变化。
- [ ] 根据 loss 0.5、1.0、2.0 计算对应 perplexity。
- [ ] 对同一 batch 分别计算 token 平均和 sample 平均，比较结果。
- [ ] 把昨天的采样代码与今天的训练 logits 画成训练/推理分叉图。

## 今日总结

一句话总结：**自回归语言模型通过 teacher forcing 并行预测每个位置的下一个 token，用 label shift 对齐 logits 与真实目标，再以 cross-entropy 最大化真实序列的条件概率。**

关键关系：

```text
原始 tokens
→ 输入与 labels 错开一位
→ causal Transformer 输出 [B, T, V] logits
→ 忽略无效位置
→ token-level cross-entropy
→ mean loss
→ perplexity = exp(mean loss)
```

需要牢记：

- causal mask 防止偷看未来，label shift 指定预测目标。
- Teacher forcing 使用真实历史 token，但仍遵守因果注意力。
- Cross-entropy 直接接收 logits，内部使用数值稳定的 log-softmax。
- 平均 loss 与 perplexity只能基于有效 token。
- Perplexity 是条件概率指标，不是事实性、可控性或生成质量的完整评价。

## 明日衔接建议

继续沿着 loss 向后追踪：学习 `p - y` 如何通过 LM Head、Transformer 和 embedding 反向传播，并理解 gradient accumulation、gradient clipping、AdamW、weight decay、learning-rate warmup 以及一次稳定训练更新的完整流程。
