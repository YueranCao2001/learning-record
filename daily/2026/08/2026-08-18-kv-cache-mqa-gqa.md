# 2026-08-18：KV Cache——从显存公式到 MQA 与 GQA

## 今日目标

- 理解 KV Cache 为什么能加速自回归生成，以及它没有省掉哪些计算。
- 从缓存张量形状推导显存占用公式。
- 区分 Multi-Head Attention、Multi-Query Attention 和 Grouped-Query Attention。
- 用 NumPy 实现缓存容量估算、预分配写入和最小 GQA 解码步骤。
- 理解 KV Cache 与 RoPE、batch size、上下文长度和推理吞吐之间的关系。

## 1. 为什么生成时会出现重复计算

自回归模型一次生成一个 token。假设提示词已有 `T` 个 token，生成下一个 token 时，新位置需要关注全部历史位置。

如果每一步都把完整前缀重新送入模型，那么旧 token 会重复执行：

- Q、K、V 投影。
- RoPE 位置旋转。
- 每一层的 Attention。
- 每一层的前馈网络。

但模型参数固定时，旧 token 在某一层产生的 K、V 不会因为后面新增 token 而改变。因此可以把它们保存下来，只为新 token 计算新的 Q、K、V。

这就是 KV Cache 的核心：

```text
过去 token 的 K、V：计算一次后缓存
新 token 的 Q、K、V：当前步骤计算
当前 Q：与全部缓存 K 做点积，再对缓存 V 加权
```

## 2. Prefill 与 Decode 是两个不同阶段

推理通常分成两个阶段。

### 2.1 Prefill

Prefill 一次处理完整提示词：

```text
输入长度：P
Q/K/V: [B, H, P, D_h]
Attention scores: [B, H, P, P]
```

这个阶段可以使用大矩阵并行计算，并把每一层的 K、V 写入缓存。其 Attention 计算量随提示词长度近似二次增长。

### 2.2 Decode

Decode 每次只处理一个新 token：

```text
Q_new: [B, H_q, 1, D_h]
K_new: [B, H_kv, 1, D_h]
V_new: [B, H_kv, 1, D_h]
K_cache/V_cache: [B, H_kv, T, D_h]
```

当前 Query 仍需要读取长度为 `T` 的缓存，但不再重新计算旧 token 的 K、V 和中间层表示。

因此 KV Cache 的价值不是让每一步变成常数复杂度，而是避免重复计算整个历史前缀。

## 3. 缓存张量的形状

对某一层，常见缓存布局为：

```text
K_cache: [B, H_kv, T, D_h]
V_cache: [B, H_kv, T, D_h]
```

其中：

- `B`：batch size。
- `H_kv`：Key/Value 头数量。
- `T`：已缓存序列长度。
- `D_h`：每个头的维度。
- 前面的系数 2：同时缓存 K 和 V。

每一层的投影参数不同，因此每一层都必须保存自己的缓存。若模型有 `L` 层，总元素数为：

```text
KV elements = 2 × B × L × T × H_kv × D_h
```

若每个元素占 `s` 字节，总字节数为：

```text
KV bytes = 2 × B × L × T × H_kv × D_h × s
```

## 4. MHA 的显存公式

标准 Multi-Head Attention 中：

```text
H_kv = H_q = H
D = H × D_h
```

代入后：

```text
KV bytes(MHA) = 2 × B × L × T × D × s
```

这说明标准 MHA 的 KV Cache 大小与模型隐藏维度 `D`、层数 `L`、序列长度 `T` 和 batch size `B` 都成线性关系。

例如：

```text
B = 1
L = 32
T = 4096
D = 4096
dtype = FP16/BF16 = 2 bytes
```

则：

```text
KV bytes
= 2 × 1 × 32 × 4096 × 4096 × 2
= 2,147,483,648 bytes
≈ 2 GiB
```

这还没有包含张量元数据、对齐、内存碎片和运行时工作区。

## 5. 为什么 batch 和上下文长度很敏感

缓存对 `B` 和 `T` 都是线性增长：

```text
batch 翻倍 → KV Cache 翻倍
上下文长度翻倍 → KV Cache 翻倍
```

在 beam search 中，每个候选分支通常都需要对应的生成状态，因此 beam width 也可能放大有效 batch。

这解释了一个常见现象：模型权重明明能够装入显存，但并发请求增多或上下文变长后，推理仍会发生显存不足。

## 6. MHA、MQA 与 GQA

三者的主要区别不在 Query 头数量，而在 Key/Value 头如何共享。

### 6.1 Multi-Head Attention

```text
H_q = H_kv
每个 Query 头拥有独立的 K 头和 V 头
```

优点是表达能力直接、结构对称；缺点是 KV Cache 最大。

### 6.2 Multi-Query Attention

```text
H_q > 1
H_kv = 1
全部 Query 头共享同一组 K、V
```

缓存比例相对 MHA 为：

```text
MQA / MHA = 1 / H_q
```

如果有 32 个 Query 头，理论上 K/V 头相关的缓存可缩小到 MHA 的 `1/32`。

### 6.3 Grouped-Query Attention

```text
1 < H_kv < H_q
若 H_q = 32，H_kv = 8
则每 4 个 Query 头共享一组 K、V
```

缓存比例为：

```text
GQA / MHA = H_kv / H_q
```

当 `H_q = 32`、`H_kv = 8` 时，缓存为 MHA 的 `1/4`。GQA 位于 MHA 和 MQA 之间，用更少的 K/V 头换取更低缓存与更少内存读取，同时保留比单一 KV 头更丰富的表示。

## 7. 同一组参数下的缓存对比

继续使用：

```text
B = 1
L = 32
T = 4096
H_q = 32
D_h = 128
dtype = 2 bytes
```

三种结构的理论缓存为：

| 结构 | H_kv | KV Cache |
| --- | ---: | ---: |
| MHA | 32 | 2 GiB |
| GQA | 8 | 512 MiB |
| MQA | 1 | 64 MiB |

这个对比只改变 K/V 头数量。Query 头仍可以保持为 32，因此不能把 MQA/GQA 误解为“整个 Attention 只剩一个头”。

## 8. 用 NumPy 计算缓存容量

```python
def kv_cache_bytes(
    batch_size: int,
    num_layers: int,
    sequence_length: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int = 2,
) -> int:
    return (
        2
        * batch_size
        * num_layers
        * sequence_length
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


def format_binary_size(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise RuntimeError("unreachable")


common = dict(
    batch_size=1,
    num_layers=32,
    sequence_length=4096,
    head_dim=128,
    bytes_per_element=2,
)

for name, num_kv_heads in {
    "MHA": 32,
    "GQA": 8,
    "MQA": 1,
}.items():
    size = kv_cache_bytes(
        **common,
        num_kv_heads=num_kv_heads,
    )
    print(name, format_binary_size(size))
```

预期输出：

```text
MHA 2.00 GiB
GQA 512.00 MiB
MQA 64.00 MiB
```

这个函数适合在阅读模型配置时快速估算缓存下限。需要确认配置中的 `num_attention_heads`、`num_key_value_heads`、`hidden_size`、层数和推理 dtype。

## 9. Query 头如何映射到 KV 头

当 `H_q` 能被 `H_kv` 整除时，每组包含：

```text
group_size = H_q / H_kv
```

例如 `H_q = 8`、`H_kv = 2`：

```text
Query heads 0,1,2,3 → KV head 0
Query heads 4,5,6,7 → KV head 1
```

教学实现可以显式重复 K、V：

```python
import numpy as np


def expand_kv_heads(
    x: np.ndarray,
    num_query_heads: int,
) -> np.ndarray:
    """把 [B, H_kv, T, D_h] 扩展为 [B, H_q, T, D_h]。"""
    _, num_kv_heads, _, _ = x.shape

    if num_query_heads % num_kv_heads != 0:
        raise ValueError("H_q 必须能被 H_kv 整除")

    repeats = num_query_heads // num_kv_heads
    return np.repeat(x, repeats=repeats, axis=1)
```

真实高性能内核通常不会物理复制整份 K、V，否则会抵消缓存节省；它们会通过索引或专用 kernel 完成共享。

## 10. 不要用 concat 反复扩容缓存

最直观的写法是：

```python
k_cache = np.concatenate([k_cache, k_new], axis=2)
```

但每一步都可能重新分配并复制旧缓存。更稳妥的教学方案是预分配最大长度，然后按位置写入：

```python
def append_kv(
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    cache_length: int,
    k_new: np.ndarray,
    v_new: np.ndarray,
) -> int:
    """把 [B, H_kv, S, D_h] 的新 K/V 写入预分配缓存。"""
    if k_new.shape != v_new.shape:
        raise ValueError("K_new 与 V_new 形状必须一致")

    step_length = k_new.shape[2]
    end = cache_length + step_length

    if end > k_cache.shape[2]:
        raise ValueError("KV Cache 容量不足")

    k_cache[:, :, cache_length:end, :] = k_new
    v_cache[:, :, cache_length:end, :] = v_new
    return end


B, H_kv, max_length, D_h = 2, 2, 16, 4
k_cache = np.empty((B, H_kv, max_length, D_h), dtype=np.float32)
v_cache = np.empty_like(k_cache)

cache_length = 0
k_new = np.ones((B, H_kv, 1, D_h), dtype=np.float32)
v_new = np.full_like(k_new, fill_value=2.0)

cache_length = append_kv(
    k_cache,
    v_cache,
    cache_length,
    k_new,
    v_new,
)

assert cache_length == 1
assert np.all(k_cache[:, :, :cache_length, :] == 1.0)
assert np.all(v_cache[:, :, :cache_length, :] == 2.0)
```

预分配需要提前知道最大长度。服务系统还可能使用分页缓存，把缓存切成固定大小的块，以降低碎片并支持不同长度请求。

## 11. 最小 GQA 单步 Attention

下面假设 K 已经应用正确位置的 RoPE，并写入有效缓存区间：

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gqa_decode_step(
    q_new: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    cache_length: int,
) -> np.ndarray:
    """
    q_new: [B, H_q, 1, D_h]
    k_cache/v_cache: [B, H_kv, max_length, D_h]
    """
    k_valid = k_cache[:, :, :cache_length, :]
    v_valid = v_cache[:, :, :cache_length, :]

    num_query_heads = q_new.shape[1]
    k_heads = expand_kv_heads(k_valid, num_query_heads)
    v_heads = expand_kv_heads(v_valid, num_query_heads)

    scale = np.sqrt(q_new.shape[-1])
    scores = np.einsum(
        "bhqd,bhkd->bhqk",
        q_new,
        k_heads,
    ) / scale

    weights = softmax(scores, axis=-1)
    return np.einsum(
        "bhqk,bhkd->bhqd",
        weights,
        v_heads,
    )
```

在单 token 解码且缓存中没有未来位置时，不需要普通训练阶段的上三角 Causal Mask；但批处理中的 padding、无效缓存槽或滑动窗口边界仍需要正确 mask。

## 12. KV Cache 节省了什么

有缓存时，每一步只为新 token 计算当前层的新表示，并复用历史 K/V。

它避免了：

- 对旧 token 重复执行 K/V 投影。
- 对旧 token 重复执行各层前向传播。
- 每一步重新构造完整前缀的中间状态。

但它没有消除：

- 当前 Query 与全部历史 Key 的点积。
- 对全部历史 Value 的加权汇总。
- 随序列增长而增加的缓存读取。
- 新 token 通过全部模型层的计算。

对单层单步 Attention，计算仍近似随当前缓存长度 `T` 线性增长。连续生成很多 token 时，Attention 总工作量仍会累积成二次项。

## 13. 为什么 Decode 常受内存带宽限制

Decode 每一步只有少量新 token，矩阵规模较小，却需要：

- 读取模型权重。
- 读取每一层全部历史 K/V。
- 写入新的 K/V。
- 只产生少量输出。

因此硬件可能没有足够大的矩阵计算来充分利用算力，反而花大量时间搬运数据。减少 `H_kv` 不仅节省容量，也减少每一步需要读取的缓存数据，所以 MQA/GQA 对推理吞吐很重要。

## 14. RoPE 与缓存必须保持位置一致

生成新 token 时：

```text
position = cache_length
Q_new → 使用 position 应用 RoPE
K_new → 使用 position 应用 RoPE → 写入 K Cache
V_new → 不旋转 → 写入 V Cache
```

常见做法是缓存已经旋转后的 K。之后读取缓存时不再重复旋转旧 K。

关键不变量是：

- 缓存中每个 K 的位置只编码一次。
- 新 Q 与新 K 使用同一个当前位置。
- cache length、position id 与实际有效 token 数一致。
- 不同 batch 样本存在 padding 时，不能只依赖一个错误的全局位置。

## 15. 常见的缓存优化方向

### 15.1 减少 KV 头

MQA/GQA 直接减少 `H_kv`，同时降低缓存容量和读取带宽。

### 15.2 降低缓存精度

把 FP16/BF16 缓存量化为更低位宽可以节省显存，但需要考虑量化误差、缩放策略和 kernel 支持。

### 15.3 Sliding Window

只保留最近固定窗口的 K/V，使缓存长度不再无限增长。代价是当前 token 不能直接关注窗口之外的内容，除非模型还有其他长期记忆机制。

### 15.4 Paged KV Cache

把缓存分成固定大小的逻辑块，按需映射到物理显存。它能减少不同请求长度带来的碎片，并支持更灵活的批处理和前缀复用。

### 15.5 Prefix Sharing

当多个请求共享相同提示词前缀时，可以复用前缀缓存。复用前必须确保模型、权重、位置编码和前缀 token 完全一致。

## 16. 设计权衡

KV Cache 是典型的“用空间换时间”：

```text
没有缓存：
显存较少，但不断重复计算历史前缀

使用缓存：
减少重复计算，但显存随 B、L、T、H_kv、D_h 增长
```

MHA、GQA、MQA 则继续在表示能力、缓存容量和推理效率之间取舍。不能只看缓存缩小倍数，还要结合训练方式、模型质量、目标硬件和内核实现判断。

## 17. 常见错误

- 认为 KV Cache 会让每一步 Attention 变成 `O(1)`。
- 只计算一层缓存，忘记乘模型层数 `L`。
- 只计算 K 或 V，忘记公式前面的系数 2。
- 使用隐藏维度 `D` 和 `H_kv × D_h` 时重复计算头数。
- 把 Query 头数量减少误当成 MQA/GQA 的定义。
- 为了实现 GQA，物理复制 K/V 并长期保存，抵消缓存优势。
- 每步使用 `concatenate` 扩容，造成重复分配和复制。
- 把未应用 RoPE 的 K 与已应用 RoPE 的 K 混在同一缓存。
- cache length 与 position id 不一致。
- 预分配缓存后，Attention 没有 mask 掉尚未写入的槽位。
- 忽略 batch、beam width、dtype 和对齐带来的实际显存变化。

## 18. 概念检查

1. 为什么旧 token 的 K、V 可以复用，而新 token 的 Q 不能提前缓存？
2. KV Cache 的元素数量为什么要乘以层数和系数 2？
3. MQA 与 GQA 分别如何共享 K/V 头？
4. 当 `H_q = 32`、`H_kv = 8` 时，每个 KV 头服务多少个 Query 头？
5. KV Cache 为什么没有消除当前 Q 对全部历史 K/V 的访问？
6. 为什么减少 KV 头可以同时改善容量和内存带宽压力？
7. Decode 阶段为什么可能比 Prefill 更难充分利用计算单元？
8. RoPE 的位置为什么必须与 cache length 对齐？

## 19. 今日练习

- [ ] 运行缓存容量函数，复现 MHA 2 GiB、GQA 512 MiB、MQA 64 MiB 的结果。
- [ ] 把 batch size 从 1 改为 8，观察三种结构的缓存变化。
- [ ] 把上下文从 4096 改为 32768，计算线性增长倍数。
- [ ] 使用 `H_q = 8`、`H_kv = 2` 验证 Query 头到 KV 头的分组。
- [ ] 连续写入 5 个 token，检查预分配缓存的有效区间。
- [ ] 故意让 `H_q` 不能被 `H_kv` 整除，确认实现明确报错。
- [ ] 为单步 GQA 加入 padding mask，避免读取无效缓存槽。
- [ ] 比较 FP32、FP16 和 8-bit KV Cache 的理论容量。
- [ ] 模拟两个请求共享相同前缀，思考哪些状态可以复用。

## 今日总结

一句话总结：**KV Cache 保存每一层历史 token 的 K/V，用线性增长的显存换取不再重复计算旧前缀；MQA 和 GQA 通过共享 K/V 头进一步降低缓存容量与读取带宽。**

缓存大小的核心公式是：

```text
2 × B × L × T × H_kv × D_h × bytes_per_element
```

MHA、MQA、GQA 的关键差异由 `H_kv` 决定。KV Cache 不会消除当前 Query 对历史序列的 Attention，因此长上下文生成仍需处理不断增长的历史信息。

## 明日衔接建议

学习从最终隐藏状态到下一个 token 的完整生成流程：LM Head、logits、temperature、greedy decoding、top-k、top-p，以及采样策略如何影响输出的确定性与多样性。
