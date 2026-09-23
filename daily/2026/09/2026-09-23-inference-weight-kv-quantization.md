# 2026-09-23：推理量化——权重、KV Cache、Scale 与质量验收

## 今日目标

上一篇 [PagedAttention 与 KV 块管理](2026-09-22-paged-attention-kv-block-management.md)讨论了缓存的分配与共享。今天进一步降低每个数值的存储成本，同时讨论量化误差与真实部署收益。

完成后应能回答：

1. 权重量化和 KV Cache 量化分别减少什么？
2. scale、零点和截断如何决定重建误差？
3. 为什么分组量化通常需要额外元数据？
4. 为什么低比特不保证推理更快？
5. 如何分开验收数值误差、任务质量与服务性能？

本篇仅运行纯 Python 数值练习，没有量化真实模型、调用低比特 kernel 或测量 GPU 显存。整数列表不是实际打包的低比特张量。

---

## 1. 先分清量化对象

| 对象 | 生命周期 | 主要影响 |
| --- | --- | --- |
| 权重 | 模型加载后通常固定 | 模型存储、权重访存和计算路径 |
| 激活 | 随输入和算子变化 | 中间张量与矩阵乘法精度 |
| KV Cache | 随请求上下文增长 | 长上下文和并发请求的缓存容量 |

W4A16 通常表示权重以 4 bit 表示、激活使用 16 bit，但不能由此推出 KV 也是 4 bit；还必须注明具体格式、分组、累加精度和实现。

权重从 16 bit 降到 4 bit，不意味着整个服务显存变成四分之一。缓存、工作区、未量化层和分配器仍占用资源。

## 2. 均匀仿射量化

一种常见表示是：

```math
q=\operatorname{clip}\left(
\operatorname{round}(x/s)+z,\ q_{\min},q_{\max}
\right),\qquad
\hat{x}=s(q-z)
```

s 是正 scale，z 是整数零点，q 为量化整数，重建值为 x 的近似。

零点让实数零可以准确对应某个整数编码。对称量化通常固定 z=0，并用对称整数范围；非对称方案允许范围偏移。

不能把所有低精度格式都解释为同一种整数线性量化。FP8、FP4 等浮点格式还有指数、尾数和具体编码规则，本文不实现它们。

## 3. 对称量化的范围约定

今天使用 b bit 中的对称子范围：

```math
Q=2^{b-1}-1,\qquad
q\in[-Q,Q],\qquad
s=\frac{\max_i|x_i|}{Q}
```

例如 4 bit 使用 -7 到 7，故意不使用 -8。不是所有实现都这样选择，因此格式描述不能只写“INT4”。

全零组采用 s=1、q=0，避免除以零。没有截断且按最近值取整时，理想实数运算的绝对误差不超过 s/2；浮点计算需要容差。

校准范围之外的新值可能被截断，这时上述误差界不成立。

## 4. 离群值与分组

如果一组包含大量小数值和一个极大值，max-abs scale 被极大值决定，小数值可能全部舍入到零。

可以按通道或更小的组分别保存 scale。对今天的 max-abs 方案，更小的组不会增大各组的理论步长上界，但不应据此宣称任意数据的每个点误差都严格变小：量化网格也发生了变化。

分组越细，需要的 scale 和可能的零点越多，还影响 kernel 布局、访问和反量化成本。组大小不是只由误差决定的参数。

## 5. 校准与任务分布

PTQ 在训练后量化；某些方法用校准样本确定范围或优化重建，另一些规则可直接从权重统计量计算。

[AWQ](https://arxiv.org/abs/2306.00978)是 activation-aware 的低比特 weight-only 方法。它提示我们，权重数值误差不是唯一目标，还要考虑输入激活如何影响误差传播。

校准数据应代表实际任务，但不应使用最终测试答案来挑选最有利配置。记录校准集版本、预处理、长度分布和算法配置；“没有微调”也不表示没有任何数据依赖。

## 6. KV Cache 量化有自己的难点

权重是固定的，KV 由当前请求动态产生，量化或反量化可能处在每步生成的关键路径上。

K 的误差影响 attention 分数，V 的误差影响加权结果。二者的数据分布和合适的分组方向不必相同。

[KIVI](https://arxiv.org/abs/2402.02750)研究了非对称的 K/V 量化设计。本文仅把它作为延伸阅读，不将今天的 max-abs 标量示例称为 KIVI 实现。

若系统保留一段高精度近期缓存，还要把它计入容量。测评必须实际经过写入、存储和读取量化 KV 的路径；只测禁用缓存的整段前向不能验证缓存量化效果。

## 7. 最小实现：分组对称量化

为便于检查，使用标准库，并用 Python 的 round（中点取偶）明确规定舍入方式。生产 kernel 的舍入行为需要单独核对。

```python
import math


def quantize(values, *, bits=4, group_size=4):
    if type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError("bits must be an integer in [2, 8]")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("invalid group size")
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("need nonempty finite values")
    limit = 2 ** (bits - 1) - 1
    groups = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        peak = max(abs(x) for x in group)
        scale = peak / limit if peak else 1.0
        if scale == 0 or not math.isfinite(scale):
            raise ValueError("scale not representable")
        codes = [max(-limit, min(limit, round(x / scale))) for x in group]
        groups.append((scale, codes))
    return groups


def reconstruct(groups):
    return [scale * q for scale, codes in groups for q in codes]


def mse(left, right):
    if not left or len(left) != len(right):
        raise ValueError("incompatible vectors")
    return sum((a - b) ** 2 for a, b in zip(left, right)) / len(left)


# Ideal payload estimate: packed integers and one FP16 scale per group.
def packed_bytes(count, bits, group_size):
    if any(type(x) is not int or x <= 0 for x in (count, bits, group_size)):
        raise ValueError("invalid size")
    return (count * bits + 7) // 8 + ((count + group_size - 1) // group_size) * 2
```

代码返回的 scale 是 Python 浮点数；packed_bytes 只是“每组存一个 FP16 scale”的独立容量估算，并未真的把 scale 转成 FP16。因此代码误差测试没有包含 scale 自身的低精度误差。

估算也忽略每组打包对齐、零点、头信息和 padding，不能当作文件实际大小或显存占用。

## 8. 对照与边界测试

以下代码接在上一段之后执行。

```python
values = [0.1, -0.2, 0.3, -0.4, 8.0, -8.0, 7.0, -7.0]
whole = quantize(values, group_size=8)
grouped = quantize(values, group_size=4)
whole_error = mse(values, reconstruct(whole))
group_error = mse(values, reconstruct(grouped))
assert group_error < whole_error  # This example, not a universal ordering claim.

offset = 0
for scale, codes in grouped:
    original = values[offset:offset + len(codes)]
    assert all(-7 <= q <= 7 for q in codes)
    assert all(abs(x - scale * q) <= scale / 2 + 1e-12
               for x, q in zip(original, codes))
    offset += len(codes)
assert reconstruct(quantize([0.0] * 5, group_size=3)) == [0.0] * 5
assert reconstruct(quantize([2.0, 2.0], group_size=2)) == [2.0, 2.0]
assert len(reconstruct(quantize([1, 2, 3, 4, 5], group_size=3))) == 5
assert quantize(values, group_size=4) == grouped

# Fixed scale calibrated to [-1, 1]: out-of-range input clips severely.
scale, qmax, x = 1 / 7, 7, 3.0
q = max(-qmax, min(qmax, round(x / scale)))
assert math.isclose(scale * q, 1.0)
assert abs(x - scale * q) > scale / 2

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: quantize([]))
reject(lambda: quantize([float("nan")]))
reject(lambda: quantize([float("inf")]))
reject(lambda: quantize([1], bits=1))
reject(lambda: quantize([1], group_size=0))

# 128 values: FP16 baseline=256 bytes; ideal INT4 + FP16 scales=68 bytes.
assert packed_bytes(128, 4, 64) == 68
print(f"whole_group_mse={whole_error:.6f}")
print(f"two_groups_mse={group_error:.6f}")
print(f"ideal_bytes=68, FP16_bytes=256, ratio={256 / 68:.3f}x")
print("All quantization checks passed.")
```

两个分组的误差更小，是这组人工数据的结果，不是某个模型的实测结论。理想压缩比小于 4 倍，因为加入了 scale 存储。

## 9. 低比特不保证更快

容量收益与速度收益要分别测量：

- 读取更少字节可能缓解带宽压力。
- 解包、scale 加载和反量化可能增加工作。
- 硬件与 kernel 是否支持相应格式非常关键。
- batch、上下文长度和并行方式会改变瓶颈。
- 若先展开成完整高精度权重，实际内存收益也可能被抵消。

量化释放的显存可以提高可接纳并发，但更大并发也可能增加排队或尾延迟。不能仅从“模型文件更小”推导服务更快。

## 10. 与 KV 块管理的关系

降低 KV 元素位宽并不改变[昨天的块表](2026-09-22-paged-attention-kv-block-management.md)需要正确映射与计数的要求。

需要额外明确：

- 每块的数据格式、scale 布局和对齐。
- COW 时是否连同量化元数据正确复制。
- 前缀复用是否兼容量化配置、模型和位置条件。
- 精度切换时缓存是否需要失效或重新计算。

不能让一个请求按 FP16 解释另一个请求保存的低比特块。格式是缓存兼容协议的一部分，而不仅是压缩选项。

## 11. 三层验收协议

第一层是数值测试：重建误差、截断率、全零组、常数组、离群值、非有限输入和真实 kernel 对照。

第二层是模型质量：固定评估数据、提示和生成预算，比较任务成功率、困惑度及预定切片；长上下文、数值和格式任务应按实际需求覆盖。权重 MSE 低不保证生成质量不变。

第三层是服务表现：固定硬件和目标负载，测显存峰值、TTFT、生成间隔、吞吐与失败率。沿用 [推理部署验收](2026-09-20-inference-serving-acceptance.md)，不要只报告成功请求速度。

可以先分别量化权重和 KV，再测二者组合，以识别交互。保留高精度 baseline，记录算法、组大小、格式、校准集和运行环境。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
whole_group_mse=0.042602
two_groups_mse=0.005255
ideal_bytes=68, FP16_bytes=256, ratio=3.765x
All quantization checks passed.
```

已检查 UTF-8、代码围栏、公式结构和本地链接；未进行 GitHub 页面视觉渲染验收。

练习只验证分组编码、重建、量化范围、舍入误差界和截断反例。没有实现低比特打包、非对称零点估计、矩阵乘法或真实注意力。

容量估算和数值测试彼此独立；没有证据支持“模型质量保持”或“推理加速”等部署结论。

## 参考资料

- [AWQ：Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)。
- [KIVI：A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)。
- [前篇：PagedAttention 与 KV 块管理](2026-09-22-paged-attention-kv-block-management.md)。

## 今日总结

1. 权重、激活与 KV 是不同量化对象，位宽要分别声明。
2. Scale 与截断决定误差，舍入规则也属于实现协议。
3. 分组带来精度与元数据成本的权衡。
4. 存储更小不必然更快，必须测试实际执行路径。
5. 数值、任务质量和服务性能需要分别验收。

## 下次衔接建议

继续学习投机解码：理解草稿生成、目标模型验证、拒绝采样与接受率，区分保持目标分布的算法性质和实际延迟收益。
