# 2026-08-05：Multi-Head Attention

## 今日目标

- 理解多头注意力为什么不等于“把同一次 Attention 重复多遍”。
- 掌握从模型维度拆分多头、并行计算再合并的全过程。
- 理清批次维、序列维、头维和特征维的变化。
- 理解 Causal Mask 在多头张量中的广播方式。
- 用 NumPy 验证实现中的关键不变量。

## 1. 为什么需要多个注意力头

单头 Attention 只使用一组投影矩阵生成 Q、K、V，并在一个表示子空间中计算关系。Multi-Head Attention 使用多组投影，让不同的头能够从不同的表示子空间观察 token 之间的关系。

```text
head_i = Attention(XW_i^Q, XW_i^K, XW_i^V)
MultiHead(X) = Concat(head_1, ..., head_H)W^O
```

每个头拥有自己的 Q、K、V 投影参数。多个头可能学习局部搭配、指代、位置关系或其他模式，但这些职责是训练产生的，并非预先指定。

## 2. 维度约束

设：

```text
B：batch size
N：sequence length
D：模型维度 d_model
H：注意力头数 num_heads
D_h：每个头的维度 head_dim
```

通常要求：

```text
D_h = D / H
D % H == 0
```

例如 `D = 8`、`H = 2`，每个头的维度就是 `D_h = 4`。

输入和投影后的整体 Q、K、V 均为：

```text
X: [B, N, D]
Q: [B, N, D]
K: [B, N, D]
V: [B, N, D]
```

## 3. 拆分多头

先把最后一维 `D` 拆成 `[H, D_h]`：

```text
[B, N, D] → [B, N, H, D_h]
```

然后把头维移动到序列维之前，方便并行矩阵乘法：

```text
[B, N, H, D_h] → [B, H, N, D_h]
```

NumPy 实现：

```python
def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model 必须能被 num_heads 整除")

    head_dim = d_model // num_heads
    x = x.reshape(batch_size, seq_len, num_heads, head_dim)
    return x.transpose(0, 2, 1, 3)
```

`reshape` 只是重新解释最后一维；`transpose` 才真正改变各维度的排列顺序。这两步不能随意交换。

## 4. 多头中的 Scaled Dot-Product Attention

拆分后的形状是：

```text
Q: [B, H, N_q, D_h]
K: [B, H, N_k, D_h]
V: [B, H, N_k, D_h]
```

计算分数时交换 K 的最后两个维度：

```text
Kᵀ:     [B, H, D_h, N_k]
QKᵀ:    [B, H, N_q, N_k]
weights:[B, H, N_q, N_k]
output: [B, H, N_q, D_h]
```

```python
from math import sqrt


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    head_dim = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / sqrt(head_dim)

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    weights = softmax(scores, axis=-1)
    return weights @ v, weights
```

这里除以的是 `√D_h`，而不是 `√D`，因为每个头的点积发生在 `D_h` 维空间中。

## 5. 合并多头

Attention 的输出为 `[B, H, N, D_h]`。合并时反向执行拆分操作：

```text
[B, H, N, D_h]
→ transpose → [B, N, H, D_h]
→ reshape   → [B, N, D]
```

```python
def combine_heads(x: np.ndarray) -> np.ndarray:
    batch_size, num_heads, seq_len, head_dim = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(batch_size, seq_len, num_heads * head_dim)
```

转置后调用 `reshape` 在 NumPy 中通常可以正常工作；在某些框架中，转置后的内存可能不连续，需要先执行类似 `contiguous()` 的操作。

## 6. 输出投影 W^O 的作用

拼接后的各头特征只是排列在一起，还没有进一步混合。输出投影：

```text
Y = Concat(head_1, ..., head_H)W^O
```

把各个头的信息重新组合，并将输出保持在模型维度 `D`，这样才能与残差分支中的输入 `X` 相加。

需要注意：如果只是为了练习形状，可以暂时用单位矩阵或省略输出投影；但完整的 Multi-Head Attention 必须包含 `W^O`。

## 7. Causal Mask 如何广播

注意力分数的形状为：

```text
[B, H, N, N]
```

一个所有样本、所有头共用的 Causal Mask 可以构造成：

```python
seq_len = 4
causal_mask = np.tril(
    np.ones((seq_len, seq_len), dtype=bool)
)[None, None, :, :]
```

其形状是 `[1, 1, N, N]`，NumPy 会把它广播到 `[B, H, N, N]`。两个前导的 `1` 分别对应 batch 维和 head 维。

如果还有 Padding Mask，则要仔细统一布尔语义并组合两个 Mask，例如都约定 `True` 表示允许关注后再使用逻辑与运算。

## 8. 最小可运行示例

```python
import numpy as np

batch_size = 2
seq_len = 4
d_model = 8
num_heads = 2

rng = np.random.default_rng(42)
q = rng.normal(size=(batch_size, seq_len, d_model))
k = rng.normal(size=(batch_size, seq_len, d_model))
v = rng.normal(size=(batch_size, seq_len, d_model))

q_heads = split_heads(q, num_heads)
k_heads = split_heads(k, num_heads)
v_heads = split_heads(v, num_heads)

mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
mask = mask[None, None, :, :]

head_output, weights = attention(q_heads, k_heads, v_heads, mask)
output = combine_heads(head_output)

assert q_heads.shape == (2, 2, 4, 4)
assert weights.shape == (2, 2, 4, 4)
assert output.shape == (2, 4, 8)
np.testing.assert_allclose(weights.sum(axis=-1), 1.0)

future_positions = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
assert np.all(weights[..., future_positions] == 0.0)
```

这个例子没有实现 `W^Q`、`W^K`、`W^V` 和 `W^O` 的训练参数，而是假设 q、k、v 已完成投影；它专门用于验证多头拆分、Attention、Mask 和合并过程。

## 9. 常见错误

- `D` 不能被 `H` 整除，却直接执行 `reshape`。
- 拆分后忘记把头维转到序列维前面。
- 使用普通的 `.T` 反转四维数组的所有轴；应只交换最后两个维度。
- 用 `√D` 缩放每个头，而不是 `√D_h`。
- Softmax 沿 Query 维计算，而不是沿 Key 维计算。
- Causal Mask 缺少 batch/head 占位维，导致错误广播。
- 合并多头时只 `reshape` 不 `transpose`，使 token 与头的数据顺序混乱。
- 忘记输出投影 `W^O`，误以为拼接就是完整的 Multi-Head Attention。

## 10. 概念检查

1. 为什么多头拆分前后总特征维度仍然是 `D`？
2. 为什么 K 只交换最后两个维度？
3. 每个头为什么使用 `√D_h` 缩放？
4. `[1, 1, N, N]` 的 Mask 如何作用于 `[B, H, N, N]`？
5. 输出投影 `W^O` 除了维持形状，还承担什么作用？

## 11. 今日练习

- [ ] 运行本文示例并打印每一步的张量形状。
- [ ] 将 `num_heads` 改为 1、2、4、8，观察形状变化。
- [ ] 故意把 `d_model` 改为 7，确认维度检查能够及时报错。
- [ ] 删除 Causal Mask，比较第一行注意力权重的变化。
- [ ] 添加随机矩阵，实现 `W^Q`、`W^K`、`W^V` 和 `W^O`。
- [ ] 用 PyTorch 的 `torch.nn.MultiheadAttention` 对照输入输出形状。

## 今日总结

一句话总结：**Multi-Head Attention 先把模型特征拆到多个子空间中并行建立 token 关系，再合并各头并通过输出投影融合信息。**

实现时最重要的不是背代码，而是始终写清每个张量的形状，并验证权重归一化、Mask 后未来权重为零、最终输出回到 `[B, N, D]`。

## 明日衔接建议

学习位置编码：先理解为什么纯 Self-Attention 无法区分词序，再推导正弦位置编码，并比较可学习位置嵌入与旋转位置编码的基本思想。
