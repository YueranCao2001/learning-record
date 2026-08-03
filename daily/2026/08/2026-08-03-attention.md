# 2026-08-03：Attention（注意力机制）

## 今日目标

- 理解 Attention 为什么出现。
- 掌握 Query、Key、Value 的直觉含义。
- 能读懂 Scaled Dot-Product Attention 的公式。
- 手算一个简单示例，并理解 Self-Attention 的输出是什么。

## 1. 为什么需要 Attention

处理序列时，模型往往需要根据当前任务，从输入中挑选最相关的信息。早期的编码器—解码器模型如果把整个输入压缩成一个固定长度向量，长序列中的重要细节容易丢失。

Attention 的核心想法是：**不要让模型对所有信息一视同仁，而是根据当前查询动态分配权重。**

例如翻译 “The animal didn't cross the street because it was too tired” 时，要理解 `it`，模型应更多关注 `animal`，而不是平均关注每个词。

## 2. Query、Key、Value 的直觉

可以把 Attention 想象成一次检索：

- **Query（Q）**：我正在寻找什么？
- **Key（K）**：每条信息用什么特征来接受匹配？
- **Value（V）**：匹配成功后真正取回的内容是什么？

数据库类比：Query 是搜索条件，Key 是索引，Value 是被读取的数据。Q 与 K 越相似，对应 V 获得的权重通常越高。

在 Self-Attention 中，Q、K、V 都由同一组输入向量通过不同的线性变换得到：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

## 3. Scaled Dot-Product Attention

核心公式：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k)V
```

逐步理解：

1. `QKᵀ`：计算每个 Query 与所有 Key 的相似度。
2. `/ √d_k`：缩放分数，防止维度较大时点积过大，使 Softmax 过度饱和、梯度变小。
3. `softmax(...)`：把分数转换为和为 1 的注意力权重。
4. `权重 × V`：对 Value 加权求和，得到融合上下文后的表示。

形状检查（批次维暂时省略）：

```text
Q:       [n_q, d_k]
K:       [n_k, d_k]
V:       [n_k, d_v]
QKᵀ:     [n_q, n_k]
输出:    [n_q, d_v]
```

## 4. 手算示例

假设一个 Query 对两个 Key 的缩放后分数为：

```text
[2, 1]
```

经过 Softmax，近似得到：

```text
[0.731, 0.269]
```

若两个 Value 为：

```text
V₁ = [1, 0]
V₂ = [0, 2]
```

则输出为：

```text
0.731 × [1, 0] + 0.269 × [0, 2]
= [0.731, 0.538]
```

这说明 Attention 的输出不是简单选择某一个 Value，而是按相关程度对它们进行加权组合。

## 5. Self-Attention 与 Multi-Head Attention

### Self-Attention

Self-Attention 让序列中的每个位置关注同一序列的所有位置，从而获得带上下文的信息表示。没有位置编码时，它本身并不知道词序，因此 Transformer 通常还需要加入位置信息。

### Multi-Head Attention

单个注意力头只在一个表示子空间中建立关系。多头注意力并行执行多组 Attention：

```text
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead = Concat(head_1, ..., head_h)W^O
```

不同的头可以学习不同类型的关系，例如局部搭配、指代关系或长距离依赖。这里的“可以”是模型学习出的结果，并非人为规定每个头的职责。

## 6. Mask 的作用

- **Padding Mask**：不让模型关注补齐序列长度所用的无意义 token。
- **Causal Mask**：生成当前位置时屏蔽未来 token，防止模型提前看到答案。

实现时通常把被屏蔽位置的注意力分数设为一个极小值，再执行 Softmax，使其权重接近 0。

## 7. 今日练习

### 概念检查

1. 为什么要除以 `√d_k`？
2. 如果所有注意力分数都相等，Softmax 后的权重是什么样？
3. Self-Attention 中 Q、K、V 的来源是否相同？参数是否相同？
4. Causal Mask 和 Padding Mask 分别解决什么问题？

### 动手任务

- [ ] 不借助框架，用数组实现 Softmax。
- [ ] 用矩阵运算实现 Scaled Dot-Product Attention。
- [ ] 打印注意力权重，验证每一行之和约等于 1。
- [ ] 加入 Causal Mask，确认当前位置无法关注未来位置。

## 8. 今日总结

一句话总结：**Attention 用 Query 与 Key 计算相关性，再用归一化后的相关性对 Value 加权汇总。**

我目前能够解释公式中的每一步，但还需要通过代码练习巩固 Mask、批次维度和多头拆分/拼接。

## 明日衔接建议

实现一个最小版 Scaled Dot-Product Attention，并进一步学习 Transformer 中的残差连接、LayerNorm 与前馈网络。

