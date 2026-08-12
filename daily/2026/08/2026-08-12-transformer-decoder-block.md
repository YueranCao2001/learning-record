# 2026-08-12：Transformer 解码器块——串起完整前向传播

## 今日目标

- 理解现代 Pre-Norm Transformer 解码器块的数据流。
- 掌握 RMSNorm、Causal Self-Attention、RoPE、残差连接和 SwiGLU 的分工。
- 区分训练时全序列计算与生成时 KV Cache 增量计算。
- 用 NumPy 搭建形状正确的最小解码器块骨架。

## 1. 先看完整结构

一个常见的 Pre-Norm 解码器块可以写成：

```text
H = X + Attention(RMSNorm(X))
Y = H + SwiGLU(RMSNorm(H))
```

展开 Attention 后：

```text
X
├─ RMSNorm
├─ Q/K/V 线性投影与多头拆分
├─ 对 Q、K 应用 RoPE
├─ QKᵀ / √d_head
├─ Causal Mask → Softmax → 与 V 加权
├─ 合并多头与输出投影
└─ 与 X 做第一次残差相加 → H

H
├─ RMSNorm
├─ SwiGLU 前馈网络
└─ 与 H 做第二次残差相加 → Y
```

输入与输出形状通常都为 `[B, N, D]`，因此多个解码器块可以连续堆叠。

## 2. RMSNorm

RMSNorm 使用特征的均方根归一化，不减去均值：

```text
RMS(x) = √(mean(x²) + ε)
RMSNorm(x) = x / RMS(x) · weight
```

NumPy 实现：

```python
import numpy as np


def rms_norm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return x / rms * weight
```

`weight` 的形状为 `[D]`，通过广播作用于所有 batch 和 token。RMSNorm 与 LayerNorm 的关键区别是：RMSNorm 不执行中心化，也就是不减均值。

## 3. 为什么使用 Pre-Norm

Pre-Norm 把归一化放在子层之前：

```text
X + Sublayer(Norm(X))
```

残差主干 `X` 可以更直接地跨层传递信息和梯度。与 Post-Norm 相比，Pre-Norm 往往更容易稳定训练深层网络，但具体模型仍可能采用不同结构。

“Pre”描述的是 Norm 相对子层的位置，不代表整个模型最终不需要额外归一化；许多模型在所有解码器块之后还有一次最终 Norm。

## 4. Causal Self-Attention 数据流

输入归一化后先投影：

```text
Q = X_norm W_Q
K = X_norm W_K
V = X_norm W_V
```

再拆成多头：

```text
[B, N, D] → [B, H, N, D_h]
```

随后对 Q、K 应用 RoPE，计算：

```text
scores = Q_rot K_rotᵀ / √D_h
weights = softmax(mask(scores))
context = weights V
```

最后合并多头并输出投影：

```text
[B, H, N, D_h] → [B, N, D] → 乘 W_O
```

Causal Mask 保证位置 `i` 只能关注 `j ≤ i` 的 token，防止训练时看到未来答案。

## 5. SwiGLU 前馈网络

普通 FFN 常写为两层线性变换；SwiGLU 使用两个并行投影形成门控：

```text
SwiGLU(x) = (SiLU(xW_gate) ⊙ xW_up)W_down
SiLU(z) = z · sigmoid(z)
```

```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def swiglu(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    gate = silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down
```

如果 `X` 为 `[B, N, D]`，`W_gate` 和 `W_up` 为 `[D, D_ff]`，`W_down` 为 `[D_ff, D]`，最终输出仍为 `[B, N, D]`。

Attention 在 token 之间混合信息；SwiGLU 则对每个 token 独立进行非线性特征变换。两者功能互补。

## 6. 最小解码器块骨架

下面用一个已封装的 `causal_self_attention` 表示注意力细节，突出块级结构：

```python
def decoder_block(
    x: np.ndarray,
    attention_fn,
    attn_norm_weight: np.ndarray,
    ffn_norm_weight: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    attn_input = rms_norm(x, attn_norm_weight)
    attn_output = attention_fn(attn_input)
    hidden = x + attn_output

    ffn_input = rms_norm(hidden, ffn_norm_weight)
    ffn_output = swiglu(ffn_input, w_gate, w_up, w_down)
    return hidden + ffn_output
```

这里有两个重要不变量：

```text
attn_output.shape == x.shape
ffn_output.shape == hidden.shape
```

否则无法执行残差相加。

## 7. 可运行的形状验证

为了只验证块结构，先用恒等 Attention 代替真实注意力：

```python
rng = np.random.default_rng(42)

B, N, D, D_ff = 2, 5, 8, 16
x = rng.normal(size=(B, N, D))

attn_norm_weight = np.ones(D)
ffn_norm_weight = np.ones(D)
w_gate = rng.normal(scale=0.1, size=(D, D_ff))
w_up = rng.normal(scale=0.1, size=(D, D_ff))
w_down = rng.normal(scale=0.1, size=(D_ff, D))


def identity_attention(x: np.ndarray) -> np.ndarray:
    return x


y = decoder_block(
    x,
    identity_attention,
    attn_norm_weight,
    ffn_norm_weight,
    w_gate,
    w_up,
    w_down,
)

assert y.shape == (B, N, D)
assert np.isfinite(y).all()
```

这个实验不验证 Attention 算法，而是验证两个 Norm、两个残差分支和 SwiGLU 的维度能正确衔接。

## 8. 训练时：一次处理完整序列

训练通常并行输入一段完整 token 序列：

```text
X: [B, N, D]
scores: [B, H, N, N]
```

虽然所有位置同时计算，但 Causal Mask 阻止位置看到未来 token。这样既保持自回归目标，又能充分利用矩阵并行计算。

对于长度 `N`，标准 Attention 的分数矩阵包含 `N²` 个位置关系；序列变长时，计算量和显存占用都会快速增长。

## 9. 生成时：KV Cache 增量计算

自回归生成第一个 token 后，过去 token 的 K、V 不会改变，因此可以缓存：

```text
K_cache: [B, H, past_length, D_h]
V_cache: [B, H, past_length, D_h]
```

生成下一个 token 时，只为新 token 计算 Q_new、K_new、V_new：

```text
K_all = concat(K_cache, K_new, sequence_axis)
V_all = concat(V_cache, V_new, sequence_axis)
output_new = Attention(Q_new, K_all, V_all)
```

此时：

```text
Q_new: [B, H, 1, D_h]
K_all: [B, H, past_length + 1, D_h]
```

KV Cache 避免每一步重复计算旧 token 的 K、V，但缓存本身会占用随序列长度增长的显存。

## 10. RoPE 与 KV Cache 的配合

通常在把 K 写入缓存之前就应用 RoPE：

```text
Q_new → 用当前位置旋转
K_new → 用当前位置旋转 → 写入 K_cache
V_new → 不旋转 → 写入 V_cache
```

新 token 的位置应为 `past_length`（若从 0 开始编号）。若每次增量调用都错误地使用位置 0，新的 Q/K 将无法与缓存中的位置形成正确相对关系。

缓存的是旋转前还是旋转后的 K 属于实现选择，但整个系统必须保持一致；常见做法是缓存已经旋转的 K。

## 11. 一个 token 如何流过解码器块

假设正在生成位置 5 的 token，缓存中已有位置 0 到 4：

1. 当前 token embedding 进入 RMSNorm。
2. 投影出当前 token 的 Q、K、V。
3. 使用位置 5 对 Q、K 应用 RoPE。
4. 将 K、V 追加到长度为 5 的缓存中。
5. 当前 Q 关注位置 0 到 5 的所有 K，并汇总 V。
6. 输出投影后与原输入做残差相加。
7. 再经过 RMSNorm 和 SwiGLU。
8. 与中间状态进行第二次残差相加。
9. 输出进入下一个解码器块。

每一层都有自己的 K/V Cache，因为不同层投影得到的 K、V 不同。

## 12. 常见错误

- 把 Pre-Norm 写成 `Norm(X + Sublayer(X))`，混淆了结构顺序。
- RMSNorm 中错误地减去均值，实际写成了 LayerNorm。
- Attention 或 FFN 输出不是 `[B, N, D]`，导致残差相加失败。
- SwiGLU 的 gate 与 up 投影维度不同，无法逐元素相乘。
- 训练完整序列时忘记 Causal Mask，造成未来信息泄漏。
- KV Cache 在错误的维度上拼接；应沿序列维增长。
- 所有层共用同一个 KV Cache，忽略各层 K/V 表示不同。
- 增量生成时 RoPE 位置每步从 0 重启。

## 13. 概念检查

1. Pre-Norm 中 Norm 位于残差相加之前还是之后？
2. RMSNorm 与 LayerNorm 的主要计算差异是什么？
3. Attention 和 SwiGLU 分别负责混合哪类信息？
4. 为什么每个子层输出必须回到模型维度 `D`？
5. KV Cache 节省了哪些重复计算，又增加了什么成本？
6. 为什么每一层需要独立的 KV Cache？

## 14. 今日练习

- [ ] 运行最小解码器块，验证输入输出形状一致。
- [ ] 检查 RMSNorm 后每个 token 的均方根是否接近权重尺度。
- [ ] 打印 SwiGLU 中 gate、up 和 down 各阶段的形状。
- [ ] 把恒等 Attention 替换为之前实现的多头因果注意力。
- [ ] 为注意力加入 RoPE，并验证未来位置权重为 0。
- [ ] 模拟连续生成 3 个 token，记录每一步 KV Cache 的长度。

## 今日总结

一句话总结：**解码器块通过 Attention 在 token 间传递信息，通过 SwiGLU 在 token 内变换特征，并由 RMSNorm 与残差连接稳定地串联两条路径。**

训练时依靠 Causal Mask 并行处理完整序列；生成时则用 KV Cache 保存各层历史 K/V，只对新 token 做增量计算。RoPE 必须使用与缓存长度一致的位置。

## 明日衔接建议

深入学习 KV Cache：计算缓存张量的显存占用，比较 Multi-Head Attention、Multi-Query Attention 和 Grouped-Query Attention 如何减少缓存成本。
