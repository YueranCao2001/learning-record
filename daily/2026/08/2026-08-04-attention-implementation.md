# 2026-08-04：Attention 从公式到最小实现

## 今日目标

- 用 NumPy 实现 Scaled Dot-Product Attention。
- 理解数值稳定的 Softmax 和 Causal Mask。
- 能根据张量形状定位常见矩阵错误。
- 理解 Attention 子层之后为什么还需要残差连接、LayerNorm 和前馈网络。

## 1. 先把公式拆成程序步骤

昨天学习的公式是：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k)V
```

它可以直接拆成四步：

```text
scores  = Q @ K.T
scores  = scores / sqrt(d_k)
weights = softmax(scores)
output  = weights @ V
```

假设：

```text
Q: [n_q, d_k]
K: [n_k, d_k]
V: [n_k, d_v]
```

那么：

```text
scores:  [n_q, n_k]
weights: [n_q, n_k]
output:  [n_q, d_v]
```

关键检查：`K` 与 `V` 的 token 数 `n_k` 必须相同，因为每个 Key 都要对应一个 Value。

## 2. 数值稳定的 Softmax

直接计算 `exp(x)` 可能溢出。例如 `exp(1000)` 已经超出普通浮点数的表示范围。Softmax 对所有元素同时减去同一个常数后结果不变，因此通常先减去最大值：

```python
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

这里必须保留维度 `keepdims=True`，这样减法和除法才能按照预期广播。

## 3. 最小版 Scaled Dot-Product Attention

```python
from math import sqrt

import numpy as np


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 Attention 输出和注意力权重。"""
    d_k = q.shape[-1]
    scores = q @ k.T / sqrt(d_k)

    if mask is not None:
        # True 表示允许关注，False 表示屏蔽。
        scores = np.where(mask, scores, -np.inf)

    weights = softmax(scores, axis=-1)
    output = weights @ v
    return output, weights
```

返回 `weights` 很有用：它既方便学习，也方便检查每个 Query 对各个 Key 分配了多少权重。

## 4. 一个可以运行的例子

```python
q = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])

k = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

v = np.array([
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 3.0],
])

output, weights = scaled_dot_product_attention(q, k, v)

print("weights =\n", weights)
print("row sums =", weights.sum(axis=-1))
print("output =\n", output)
```

应该重点验证：

```text
weights.shape == (2, 3)
output.shape == (2, 2)
weights.sum(axis=-1) ≈ [1, 1]
```

第一条 Query `[1, 0]` 会更偏向与它方向接近的 Key，但输出仍是所有 Value 的加权组合。

## 5. 加入 Causal Mask

自回归生成中，第 `i` 个位置只能看到自己和之前的位置。对于长度为 4 的序列，允许关注的位置为：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

NumPy 可以生成这个下三角矩阵：

```python
sequence_length = 4
causal_mask = np.tril(
    np.ones((sequence_length, sequence_length), dtype=bool)
)
```

把屏蔽位置设成 `-inf` 后，`exp(-inf) = 0`，因此对应 Softmax 权重为 0。

需要注意：如果某一整行全部被屏蔽，那么这一行计算 Softmax 会出现无效值。实际使用时应保证每个 Query 至少能关注一个有效位置，或对这种情况单独处理。

## 6. Attention 之后为什么还要其他模块

Transformer 的一个子层不只是 Attention。以常见的 Pre-LN 结构为例，可以简化成：

```text
X₁ = X + Attention(LayerNorm(X))
X₂ = X₁ + FFN(LayerNorm(X₁))
```

### 残差连接（Residual Connection）

`X + 子层(X)` 为信息和梯度提供更直接的通道。子层只需学习“应该在原表示上增加什么变化”，不必每层都重新构造全部信息。

### LayerNorm

LayerNorm 对单个 token 的特征维度进行归一化，有助于稳定不同层之间的激活尺度。它与 BatchNorm 不同，不依赖同一批次中其他样本的统计量，因此更适合长度变化的序列。

### 前馈网络（FFN）

Attention 负责在不同 token 之间混合信息；FFN 则对每个 token 独立地进行非线性特征变换。典型形式是：

```text
FFN(x) = activation(xW₁ + b₁)W₂ + b₂
```

同一层内，每个位置使用相同的 FFN 参数。

## 7. 常见错误

- 忘记转置 `K`，导致 `Q @ K` 维度不匹配。
- 在错误维度上执行 Softmax；通常应在 Key 维度，也就是最后一维上归一化。
- Mask 的布尔语义前后不一致：有的实现用 `True` 表示保留，有的表示屏蔽。
- 用非常大的正数填充被屏蔽位置，反而让模型只关注 Mask。
- 忘记除以 `√d_k`，使大维度下的 Softmax 容易过度尖锐。
- 只检查输出形状，却没有检查权重每行之和是否接近 1。

## 8. 概念检查

1. 为什么 `scores` 的形状是 `[n_q, n_k]`？
2. Softmax 为什么通常沿最后一维计算？
3. 从每行减去最大值为什么不会改变 Softmax 的结果？
4. Attention 与 FFN 分别在哪个维度上混合信息？
5. 如果 Mask 后某一整行都是 `-inf`，可能出现什么问题？

## 9. 今日练习

- [ ] 把本文代码保存成 Python 文件并运行示例。
- [ ] 使用 `np.testing.assert_allclose` 验证每行权重之和为 1。
- [ ] 构造长度为 4 的 Q、K、V，加入 Causal Mask。
- [ ] 断言注意力权重矩阵上三角区域全部为 0。
- [ ] 尝试删除 `/ sqrt(d_k)`，比较权重分布是否变得更尖锐。
- [ ] 将二维实现扩展为支持批次形状 `[batch, n, d]`。

## 今日总结

一句话总结：**Attention 实现的关键不只是矩阵乘法，还包括缩放、稳定的 Softmax、正确的归一化维度和 Mask 语义。**

Attention 在 token 之间传递信息；残差连接保留原信息并改善优化，LayerNorm 稳定表示尺度，FFN 则为每个 token 提供非线性特征变换。

## 明日衔接建议

在当前实现上加入批次维和多头维，理解 Multi-Head Attention 的拆分、转置、拼接和输出投影。
