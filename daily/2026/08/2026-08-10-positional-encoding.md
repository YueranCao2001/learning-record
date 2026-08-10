# 2026-08-10：位置编码——让 Attention 理解顺序

## 今日目标

- 理解纯 Self-Attention 为什么无法天然区分词序。
- 掌握正弦位置编码的公式、维度和基本性质。
- 用 NumPy 生成位置编码并验证形状。
- 区分绝对位置嵌入、相对位置方法与 RoPE 的核心思想。

## 1. Self-Attention 为什么需要位置信息

Self-Attention 根据 token 表示计算 Q、K、V，再通过相似度聚合信息。若不提供任何位置信息，它只知道“有哪些 token”，却不知道它们以什么顺序出现。

例如下面两句话包含相似的词，但含义完全不同：

```text
猫追老鼠
老鼠追猫
```

从数学上看，如果用同一个排列矩阵 `P` 重新排列输入 token，未加入位置编码的 Self-Attention 输出也会按相同方式排列：

```text
SelfAttention(PX) = P SelfAttention(X)
```

这种性质叫作**排列等变性**。它不是错误，但意味着模型本身不会把“第一个位置”和“第三个位置”看作不同角色，因此必须额外注入顺序信息。

## 2. 最直接的做法：位置向量相加

设 token embedding 为 `X`，位置表示为 `P`，二者形状相同：

```text
X: [batch, sequence_length, d_model]
P: [1, sequence_length, d_model]
输入 = X + P
```

使用加法而不是拼接有两个直接好处：

- 不改变模型维度 `d_model`，后续层的结构无需变化。
- token 语义与位置从第一层开始共同影响 Q、K、V。

位置向量的尺度不能失控，否则可能淹没 token embedding；实际模型通常通过初始化、归一化或训练过程协调两者。

## 3. 正弦位置编码

经典 Transformer 使用固定的正弦和余弦函数：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

其中：

- `pos` 是 token 的位置，从 0 开始。
- `i` 表示一对特征维度的编号。
- 偶数维使用 sin，奇数维使用 cos。
- 不同维度具有不同波长：低维变化快，高维变化慢。

这类似同时使用多只不同刻度的“时钟”记录位置：短周期维度区分相邻位置，长周期维度表达更大范围的位置变化。

## 4. NumPy 实现

```python
import numpy as np


def sinusoidal_position_encoding(
    sequence_length: int,
    d_model: int,
) -> np.ndarray:
    positions = np.arange(sequence_length)[:, None]
    even_dims = np.arange(0, d_model, 2)
    angle_rates = 1.0 / np.power(10000.0, even_dims / d_model)
    angles = positions * angle_rates[None, :]

    encoding = np.zeros((sequence_length, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding
```

最后一行对余弦部分做了切片，因此奇数 `d_model` 也能正确处理。若项目始终要求偶数模型维度，实现可以更简单，但函数仍应明确检查或支持边界情况。

## 5. 可运行的验证

```python
pe = sinusoidal_position_encoding(sequence_length=6, d_model=8)

assert pe.shape == (6, 8)
np.testing.assert_allclose(pe[0, 0::2], 0.0)
np.testing.assert_allclose(pe[0, 1::2], 1.0)
assert not np.allclose(pe[1], pe[2])

token_embeddings = np.zeros((2, 6, 8))
model_input = token_embeddings + pe[None, :, :]
assert model_input.shape == (2, 6, 8)
```

位置 0 的所有 sin 分量为 0，cos 分量为 1；不同位置的向量不同；增加 batch 维后，位置编码可广播到每个样本。

## 6. 为什么正弦编码能表达相对位置

利用三角恒等式，`sin(pos + k)` 与 `cos(pos + k)` 可以由位置 `pos` 的 sin/cos 值经过与偏移量 `k` 有关的线性组合得到。

因此，对固定偏移 `k`，位置 `pos + k` 的表示与位置 `pos` 的表示存在结构化关系。这不代表模型自动掌握所有相对位置规律，但为学习相对偏移提供了有用的连续信号。

正弦编码还可以直接计算训练长度之外的位置，不需要新增参数。不过，“能够生成编码”不等于模型一定能在更长序列上良好泛化。

## 7. 可学习的绝对位置嵌入

另一种方法是为每个位置维护一个可训练向量：

```text
position_embedding: [max_sequence_length, d_model]
输入 = token_embedding + position_embedding[position]
```

优点：

- 表达灵活，位置表示可针对任务自行学习。
- 实现简单，与普通 embedding 类似。

限制：

- 增加参数量。
- 通常预先设定最大长度。
- 未训练过的位置没有自然定义，长度外推较困难。

## 8. 相对位置与 RoPE

### 相对位置方法

绝对位置编码告诉模型“这个 token 在第几个位置”。相对位置方法更直接地影响位置 `i` 与位置 `j` 之间的注意力分数，让模型关注距离或方向，例如 `i - j`。

这对“相隔多少个 token”比“处于全局第几个位置”更重要的任务很自然。

### RoPE（Rotary Position Embedding）

RoPE 不把位置向量简单加到 token embedding 上，而是在每对特征维度组成的二维平面中，根据位置旋转 Q 和 K。

直觉上：

```text
位置越靠后 → 旋转角度越大
Q 与 K 的点积 → 同时受到相对旋转角度影响
```

因为 Q 和 K 都被旋转，它们的相似度可以自然携带相对位置信息。理解 RoPE 时应先掌握二维旋转矩阵，再扩展到多组不同频率的特征维度。

## 9. 方法对比

| 方法 | 注入位置 | 是否可训练 | 长度外推直觉 |
| --- | --- | --- | --- |
| 正弦位置编码 | 加到输入表示 | 否 | 可计算新位置，但效果不保证 |
| 可学习绝对位置嵌入 | 加到输入表示 | 是 | 通常受最大训练长度限制 |
| 相对位置偏置 | 影响注意力分数 | 通常是 | 更关注位置差 |
| RoPE | 旋转 Q、K | 频率可固定或调整 | 相对位置结构更自然 |

## 10. 常见误区

- 认为 token 在数组中的索引会自动被模型理解；索引本身不会进入矩阵计算。
- 认为正弦编码完全没有长度限制；编码可生成不代表模型能可靠外推。
- 把位置编码拼接到 embedding 后忘记调整所有投影矩阵维度。
- 混淆“排列不变”与“排列等变”：Self-Attention 的输出会随输入一起排列，并非输出完全不变。
- 只记住 RoPE 的名称，却忽略它作用在 Q、K 上而不是直接作用于 V。

## 11. 概念检查

1. 为什么没有位置编码的 Self-Attention 无法区分“猫追老鼠”和“老鼠追猫”？
2. 为什么位置编码通常能直接与 token embedding 相加？
3. 正弦编码中不同维度为什么要采用不同频率？
4. 可学习绝对位置嵌入在超出最大长度时会遇到什么问题？
5. RoPE 为什么主要作用于 Q 和 K？

## 12. 今日练习

- [ ] 运行 NumPy 实现，打印前 6 个位置的编码。
- [ ] 用热力图观察不同位置和维度的周期变化。
- [ ] 分别测试偶数与奇数 `d_model`。
- [ ] 交换两个 token，比较有无位置编码时模型输入的变化。
- [ ] 手算二维向量旋转 90° 的结果，为学习 RoPE 做准备。
- [ ] 查阅所用模型的配置，确认其实际采用哪种位置方法。

## 今日总结

一句话总结：**Self-Attention 擅长建立 token 间关系，但顺序必须由位置机制显式注入。**

正弦位置编码用不同频率的 sin/cos 提供固定的绝对位置；可学习嵌入用参数换取灵活性；相对位置方法和 RoPE 则让注意力计算更直接地感知 token 间的相对距离。

## 明日衔接建议

从二维旋转矩阵出发推导 RoPE，验证旋转后的 Q、K 点积为何主要依赖相对位置，并用 NumPy 实现最小版本。
