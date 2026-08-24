# 2026-08-24：从 Logits 到 Token——Temperature、Top-k 与 Top-p

## 今日目标

- 理解最终隐藏状态如何经过 LM Head 变成 vocabulary 上的 logits。
- 区分 logits、概率、log probability 和 token id。
- 掌握 greedy decoding、temperature、top-k 与 top-p 的作用和差异。
- 用 NumPy 实现数值稳定、可复现的单步采样。
- 理解采样策略如何与 KV Cache、停止条件和批量生成配合。

## 1. 完整生成链路

上一篇笔记已经完成 Transformer 解码器与 KV Cache。当前 token 经过最后一层后，会得到隐藏状态：

```text
h_last: [B, D]
```

LM Head 将模型维度 `D` 投影到词表大小 `V`：

```text
logits = h_last W_vocab + b
W_vocab: [D, V]
logits: [B, V]
```

完整单步生成流程是：

```text
hidden state
→ LM Head
→ logits
→ logit processing
→ temperature / top-k / top-p
→ probability distribution
→ token id
→ 更新 KV Cache
→ 下一步解码
```

一些模型会让 LM Head 与输入 embedding 共享权重，称为 weight tying，但这不是所有模型都必须采用的结构。

## 2. Logits 不是概率

logits 可以是任意实数：

```text
[3.2, 1.1, -0.7, 4.5]
```

它们不需要位于 0 到 1，总和也不需要等于 1。越大的 logit 通常表示模型越偏好对应 token，但只有相对差值有意义。

给全部 logits 加相同常数不会改变 Softmax：

```text
softmax(z) = softmax(z + c)
```

因为 Softmax 的分子和分母都会乘以相同的 `exp(c)`。

## 3. 数值稳定的 Softmax

Softmax 把 logits 变成概率：

```text
p_i = exp(z_i) / Σ_j exp(z_j)
```

直接计算 `exp(z)` 可能溢出。利用平移不变性，可以先减去最大值：

```text
p_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
```

```python
import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    finite = np.isfinite(logits)

    if not np.any(finite):
        raise ValueError("至少需要一个有限的 logit")

    maximum = np.max(logits[finite])
    exponentials = np.zeros_like(logits)
    exponentials[finite] = np.exp(logits[finite] - maximum)

    total = exponentials.sum()
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("无法构造有效概率分布")

    return exponentials / total
```

这个版本允许用 `-inf` 屏蔽 token；被屏蔽位置的概率会变成 0。

给定：

```text
logits = [2, 1, 0, -1]
```

Softmax 概率约为：

| token | logit | probability |
| --- | ---: | ---: |
| A | 2 | 0.644 |
| B | 1 | 0.237 |
| C | 0 | 0.087 |
| D | -1 | 0.032 |

## 4. Greedy Decoding

Greedy decoding 每一步选择最大 logit：

```text
token_id = argmax(logits)
```

```python
def greedy_decode(logits: np.ndarray) -> int:
    if logits.ndim != 1:
        raise ValueError("期望一维 vocabulary logits")
    return int(np.argmax(logits))
```

优点：

- 简单、快速。
- 相同 logits 下结果确定。
- 适合需要稳定输出的任务。

局限：

- 只做局部最优选择，不保证整段序列全局最优。
- 容易产生重复或过于保守的文本。
- 无法表达模型分布中的多种合理延续。

## 5. Temperature 调整分布形状

Temperature `τ` 在 Softmax 前缩放 logits：

```text
p_i(τ) = exp(z_i / τ) / Σ_j exp(z_j / τ)
τ > 0
```

效果：

- `0 < τ < 1`：放大 logit 差异，分布更尖锐。
- `τ = 1`：保持原始分布。
- `τ > 1`：缩小 logit 差异，分布更平坦。
- `τ → 0⁺`：概率逐渐集中到最大 logit。

对 `[2, 1, 0, -1]`：

| temperature | 最高 token 概率 | 分布特征 |
| ---: | ---: | --- |
| 0.5 | 约 0.865 | 很尖锐 |
| 1.0 | 约 0.644 | 原始分布 |
| 2.0 | 约 0.455 | 更平坦 |

```python
def probabilities_with_temperature(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature 必须大于 0")
    return stable_softmax(logits / temperature)
```

正 temperature 不改变 logits 的大小顺序，因此不会改变 argmax。它只有配合随机采样时才会改变各候选被选中的频率。

如果需要确定性生成，应显式使用 greedy，而不是设置 `temperature = 0` 后做除法。

## 6. Top-k Sampling

Top-k 只保留 logit 最大的 `k` 个 token，其余位置设为 `-inf`：

```python
def apply_top_k(
    logits: np.ndarray,
    top_k: int | None,
) -> np.ndarray:
    filtered = np.asarray(logits, dtype=np.float64).copy()
    vocabulary_size = filtered.size

    if top_k is None:
        return filtered
    if top_k <= 0:
        raise ValueError("top_k 必须为正数")

    top_k = min(top_k, vocabulary_size)
    keep_indices = np.argpartition(filtered, -top_k)[-top_k:]

    remove_mask = np.ones(vocabulary_size, dtype=bool)
    remove_mask[keep_indices] = False
    filtered[remove_mask] = -np.inf
    return filtered
```

特点：

- 候选数量固定。
- `k = 1` 时，采样等价于 greedy。
- 分布很尖锐时，较大的 `k` 可能保留很多不合理 token。
- 分布很平坦时，较小的 `k` 可能删掉合理候选。

## 7. Top-p / Nucleus Sampling

Top-p 保留累计概率达到阈值 `p` 的最小候选集合：

```text
1. 按概率从高到低排序。
2. 从最高概率开始累加。
3. 保留到累计概率第一次达到或超过 p。
```

例如：

```text
sorted probabilities = [0.50, 0.25, 0.15, 0.07, 0.03]
top_p = 0.80
```

前两项累计 0.75，还不足 0.80；加入第三项后为 0.90，因此保留前三个 token。

```python
def apply_top_p(
    logits: np.ndarray,
    top_p: float | None,
) -> np.ndarray:
    filtered = np.asarray(logits, dtype=np.float64).copy()

    if top_p is None or top_p >= 1.0:
        return filtered
    if top_p <= 0.0:
        raise ValueError("top_p 必须位于 (0, 1]")

    probabilities = stable_softmax(filtered)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probabilities = probabilities[sorted_indices]
    cumulative = np.cumsum(sorted_probabilities)

    cutoff = int(np.searchsorted(cumulative, top_p, side="left"))
    keep_indices = sorted_indices[: cutoff + 1]

    remove_mask = np.ones(filtered.size, dtype=bool)
    remove_mask[keep_indices] = False
    filtered[remove_mask] = -np.inf
    return filtered
```

Top-p 会随分布自动改变候选数量：

- 分布尖锐时，只需保留少量 token。
- 分布平坦时，需要更多 token 才能覆盖相同概率质量。

## 8. Top-k 与 Top-p 的区别

| 策略 | 控制对象 | 候选数量 |
| --- | --- | --- |
| top-k | 排名数量 | 固定为 k |
| top-p | 累计概率质量 | 随分布变化 |

两者可以组合。一个常见教学顺序是：

```text
temperature
→ top-k
→ top-p
→ Softmax
→ sample
```

不同推理库的过滤顺序可能不同，相同参数不一定产生完全相同的结果，因此需要阅读具体实现。

## 9. 完整的单步采样函数

```python
def sample_from_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[int, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 1:
        raise ValueError("期望一维 vocabulary logits")
    if temperature <= 0.0:
        raise ValueError("temperature 必须大于 0")

    filtered = logits / temperature
    filtered = apply_top_k(filtered, top_k)
    filtered = apply_top_p(filtered, top_p)

    probabilities = stable_softmax(filtered)
    token_id = int(
        rng.choice(
            probabilities.size,
            p=probabilities,
        )
    )
    return token_id, probabilities
```

测试：

```python
rng = np.random.default_rng(42)
logits = np.array([2.0, 1.0, 0.0, -1.0])

token_id, probabilities = sample_from_logits(
    logits,
    rng=rng,
    temperature=0.8,
    top_k=3,
    top_p=0.9,
)

print(token_id)
print(probabilities)

assert np.isclose(probabilities.sum(), 1.0)
assert np.count_nonzero(probabilities) <= 3
```

固定随机种子有助于在同一实现中复现实验，但不同硬件、库版本、并行策略和浮点精度仍可能导致结果不同。

## 10. 过滤后为什么要重新归一化

被过滤 token 的 logit 设为 `-inf`：

```text
exp(-inf) = 0
```

剩余 token 的原始概率总和通常小于 1，因此必须重新 Softmax 或除以剩余概率总和，才能得到合法分布。

不能只把被删除位置的概率设为 0，然后直接交给采样器。

## 11. Log Probability 与序列评分

概率连乘容易变成极小数，因此序列评分通常使用 log probability：

```text
log p(token_i) = log_softmax(logits)_i
```

```python
def stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    maximum = np.max(logits)
    shifted = logits - maximum
    log_normalizer = np.log(np.exp(shifted).sum())
    return shifted - log_normalizer
```

一段序列的条件 log probability 是每一步之和：

```text
log P(y₁, ..., y_n | x)
= Σ_t log P(y_t | x, y_<t)
```

长序列累加更多负数，因此直接比较不同长度序列会产生长度偏置。Beam search 等方法可能使用长度归一化或长度惩罚。

## 12. Entropy 与分布不确定性

分布熵：

```text
H(p) = -Σ_i p_i log p_i
```

```python
def entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = probabilities > 0.0
    return float(
        -np.sum(
            probabilities[positive]
            * np.log(probabilities[positive])
        )
    )
```

直觉：

- 概率集中在少量 token：熵低，模型更确定。
- 概率分散在大量 token：熵高，模型更不确定。
- 降低 temperature 通常降低熵。
- 提高 temperature 通常提高熵。

高熵不自动等于“更有创造力”，低熵也不自动等于“更正确”。

## 13. Sampling 参数不是质量旋钮

常见误解：

```text
temperature 越低 → 质量越高
temperature 越高 → 创造力越强
```

实际情况：

- 低 temperature 减少随机性，但可能强化重复或常见错误。
- 高 temperature 增加多样性，也提高选中低概率 token 的机会。
- top-k/top-p 能裁掉尾部，但阈值过严会让输出僵硬。
- 最佳参数取决于模型、提示词、任务与评价目标。

代码生成和事实问答通常偏向保守设置；故事创作和头脑风暴可以允许更高多样性，但仍需要实验验证。

## 14. Logit Processing 的位置

实际系统可能在采样前修改 logits：

- 禁止特殊 token。
- 添加 token bias。
- repetition penalty。
- presence/frequency penalty。
- 强制 JSON 或语法约束。
- 屏蔽当前状态下不允许出现的 token。

常见流程：

```text
LM Head logits
→ 约束与 penalty
→ temperature
→ top-k/top-p
→ Softmax
→ sample
```

顺序会改变最终候选。例如先 top-p 再改变 temperature，与先改变 temperature 再 top-p 可能保留不同 token。

## 15. 选出 Token 后如何继续

```text
1. LM Head 输出 vocabulary logits。
2. 应用约束和采样策略。
3. 选择 token id。
4. 检查 EOS 或其他停止条件。
5. 将 token 作为下一步输入。
6. 为新 token 更新每层 KV Cache。
7. 重复直到结束。
```

采样只决定选择哪个 token，不会替代 Transformer 前向传播，也不会改变 KV Cache 的基本形状。

## 16. 停止条件

生成系统必须有明确停止条件：

- 生成 EOS token。
- 达到 `max_new_tokens`。
- 匹配指定 stop sequence。
- 达到时间或资源预算。
- 结构化输出完成并通过解析。

EOS 通常是单个特殊 token id；stop sequence 可能对应多个 token。文本级匹配需要处理 token 边界和流式输出缓冲。

## 17. Batch 中样本可能不同步结束

批量生成时，有的样本会先产生 EOS，其他样本仍需继续。

实现需要：

- 保存每个样本的 finished mask。
- 避免已结束样本继续产生有效文本。
- 正确处理 padding 和 position id。
- 管理完成样本占用的 KV Cache 槽位。

这也是服务端动态 batching 比单样本教学代码复杂的原因之一。

## 18. 常见错误

- 把 logits 当成已经归一化的概率。
- 直接计算 `exp(logits)`，导致数值溢出。
- 使用 `temperature = 0` 做除法。
- 认为正 temperature 会改变 argmax 排序。
- top-k 过滤后没有重新归一化。
- top-p 没有包含第一个使累计概率越过阈值的 token。
- top-p 在原 token id 顺序上累计，而不是先按概率排序。
- 组合过滤器时忽略操作顺序。
- 使用不可控的全局随机状态，导致实验难以复现。
- 认为固定种子保证跨硬件完全一致。
- 忘记检查 EOS 和最大生成长度。
- 把采样多样性误当成事实准确性。
- 已完成 batch 样本仍继续追加有效 token。

## 19. 概念检查

1. LM Head 的输入和输出维度分别是什么？
2. 为什么给全部 logits 加同一个常数不会改变 Softmax？
3. 为什么 Softmax 前要减去最大 logit？
4. Greedy decoding 为什么不受正 temperature 的影响？
5. temperature 小于 1 和大于 1 分别如何改变分布？
6. top-k 与 top-p 控制的对象有什么不同？
7. top-p 为什么必须先按概率排序？
8. 过滤 token 后为什么需要重新归一化？
9. log probability 为什么比概率连乘更适合序列评分？
10. 采样完成后 KV Cache 如何参与下一步？

## 20. 今日练习

- [ ] 对 `[2, 1, 0, -1]` 手算 Softmax，并与 NumPy 比较。
- [ ] 使用 temperature 0.5、1.0、2.0，打印概率和熵。
- [ ] 使用 top-k 1、2、4，观察候选集合变化。
- [ ] 验证 top-p 0.8 对 `[0.50, 0.25, 0.15, 0.07, 0.03]` 保留前三项。
- [ ] 故意传入 temperature 0，确认函数明确报错。
- [ ] 固定随机种子，运行多次采样并统计 token 频率。
- [ ] 给某个 token 添加负 bias，观察它的概率变化。
- [ ] 实现 finished mask，模拟 batch 中一个样本提前生成 EOS。
- [ ] 把采样函数连接到上一篇的 GQA decode step 之后。

## 今日总结

一句话总结：**LM Head 把最终隐藏状态映射为词表 logits，Softmax 将其变成概率，而 greedy、temperature、top-k 和 top-p 决定如何选择下一个 token。**

核心区别：

- Greedy：直接选择最大 logit，强调确定性。
- Temperature：调整概率分布的尖锐程度。
- Top-k：固定保留排名最高的 k 个候选。
- Top-p：动态保留累计概率达到 p 的最小候选集合。

这些方法是解码策略，不会改变模型参数，也不能单独保证事实正确或生成质量。选出的 token 会进入下一步 Transformer 前向传播，并把新的 K/V 追加到缓存。

## 明日衔接建议

从推理回到训练目标：学习 next-token prediction、label shift、teacher forcing、cross-entropy loss 和 perplexity，理解模型如何通过真实下一个 token 的监督学会产生这些 logits。
