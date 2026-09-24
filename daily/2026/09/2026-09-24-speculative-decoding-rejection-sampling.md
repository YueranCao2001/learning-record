# 2026-09-24：投机解码——草稿、目标验证与拒绝采样

## 今日目标

上一篇 [推理量化](2026-09-23-inference-weight-kv-quantization.md)讨论了减少数值存储成本。今天换一个角度：用便宜的草稿计算减少目标模型的串行生成轮数，同时保持指定的目标采样分布。

完成后应能回答：

1. 为什么目标模型能并行验证已知草稿？
2. 为什么不能简单接受“看起来合理”的 token？
3. 拒绝后为什么要从残差分布采样？
4. 保持分布为什么不等于同 seed 输出相同？
5. 接受率高为什么仍不保证加速？

本篇只验证有限词表上的概率恒等式，没有加载草稿或目标模型，没有测量真实延迟，也没有实现完整多 token 解码器。

---

## 1. 草稿生成与目标验证

设当前已提交前缀为 h。草稿模型先自回归提出若干 token；目标模型再对这段已知候选序列进行带因果 mask 的前向计算，得到各位置的条件分布。

目标模型可以并行计算候选位置的 logits，因为输入候选已经给定；不是打破自回归依赖，也不是知道未来的正确答案。

验证时只能接受连续前缀。某个候选被拒绝后，后面候选的目标分布仍以被拒绝 token 为条件，不能继续作为新前缀下的有效验证结果。

[Leviathan 等（ICML 2023）](https://proceedings.mlr.press/v202/leviathan23a.html)给出了保持目标分布的投机采样算法；[Chen 等（2023）](https://arxiv.org/abs/2302.01318)也研究了 speculative sampling。本文的有限词表推导和代码用于理解单步校正规则。

## 2. 明确 p 和 q 的含义

本文规定：

- p：目标模型在当前前缀上的实际采样分布。
- q：草稿生成候选时实际使用的提议分布。
- x：从 q 采样得到的候选 token。

p 应包含目标解码协议要求的 temperature、top-k、top-p 或约束后的归一化结果。q 也必须对应草稿真正采用的分布，不能拿未处理 logits 的概率冒充。

本篇假设同一有限词表和 token 语义。不同 tokenizer 的映射、多候选树或特殊验证器需要额外机制，不在这里讨论。

## 3. 接受概率

对从 q 得到、因此 q(x) 大于零的候选 x，接受概率是：

```math
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
```

候选为 x 且被接受的联合概率是：

```math
q(x)a(x)=\min(p(x),q(x))
```

目标比草稿更偏好的 token 可以全部接受；草稿过度偏好的 token 需要部分拒绝。这不是比较两个模型独立抽样是否相同，也不是比较候选概率是否高于某个任意阈值。

q(x)=0 的 token 不会被草稿提出，不需要计算相应的比值。

## 4. 拒绝后的残差分布

总接受概率记作 A：

```math
A=\sum_x\min(p(x),q(x))
```

若发生拒绝，从下面的残差分布重新采样：

```math
r(x)=\frac{\max(p(x)-q(x),0)}{1-A}
```

因为 p、q 都归一化，正残差之和等于 1-A。输出 x 的总概率为：

```math
\min(p(x),q(x))+(1-A)r(x)=p(x)
```

A=1 时 p=q，拒绝事件概率为零，无须构造残差分布。

若拒绝后直接从 p 再抽一次，一般会得到“已接受部分 + 拒绝概率乘 p”，并不等于 p。这是最容易忽略的偏差来源。

## 5. 一个三 token 例子

设词表是 a、b、c：

- p = [0.5, 0.3, 0.2]。
- q = [0.2, 0.5, 0.3]。
- 接受概率分别为 [1, 0.6, 2/3]。
- 被接受的概率质量为 [0.2, 0.3, 0.2]，总接受率 0.7。
- 残差只有 a，因此拒绝后补回 a 的概率质量 0.3。

最终分布恰为 [0.5, 0.3, 0.2]。

如果拒绝后直接按 p 抽样，会得到 [0.35, 0.39, 0.26]。即使目标模型参与了验证，错误校正仍会改变分布。

## 6. 多 token 一轮怎样结束

假设草稿提出 K 个 token：

1. 顺序验证候选，保留被接受的连续前缀。
2. 在第一次拒绝处按该位置的残差分布采样一个 token，并丢弃后续草稿。
3. 如果 K 个全部接受，可从目标在完整候选前缀之后的分布再采样一个 token。
4. 遇到 EOS 或输出预算边界时按协议停止，不强行输出额外 token。

目标一次前向通常可以提供验证及全接受后额外采样所需的 logits，但缓存和索引对齐必须正确。不能将“最多 K+1 个输出”理解为每轮一定获得 K+1 个 token。

## 7. 最小实现：用有理数检查概率质量

下面使用 Fraction 做精确算术，避免把浮点误差误判成概率不守恒。代码计算单步的解析输出分布，不进行随机采样。

```python
from fractions import Fraction as F


def validate(dist):
    if not dist or any(type(x) is not F or x < 0 for x in dist):
        raise ValueError("need nonnegative Fraction probabilities")
    if sum(dist) != 1:
        raise ValueError("distribution must sum to one")


def corrected_law(p, q):
    validate(p)
    validate(q)
    if len(p) != len(q):
        raise ValueError("vocabulary mismatch")
    accepted_mass = [min(pi, qi) for pi, qi in zip(p, q)]
    acceptance = sum(accepted_mass)
    residual_mass = [max(pi - qi, F(0)) for pi, qi in zip(p, q)]
    rejection = 1 - acceptance
    assert sum(residual_mass) == rejection
    residual = [x / rejection for x in residual_mass] if rejection else None
    final = [
        mass + (rejection * residual[i] if residual is not None else F(0))
        for i, mass in enumerate(accepted_mass)
    ]
    return acceptance, residual, final


p = [F(1, 2), F(3, 10), F(1, 5)]
q = [F(1, 5), F(1, 2), F(3, 10)]
acceptance, residual, final = corrected_law(p, q)
assert acceptance == F(7, 10)
assert residual == [F(1), F(0), F(0)]
assert final == p

wrong = [min(pi, qi) + (1 - acceptance) * pi for pi, qi in zip(p, q)]
assert wrong == [F(35, 100), F(39, 100), F(26, 100)]
assert wrong != p

# Identical distributions: rejection never happens.
a, r, out = corrected_law(p, p)
assert a == 1 and r is None and out == p

# Disjoint support: every draft is rejected, residual equals the target.
a, r, out = corrected_law([F(1), F(0)], [F(0), F(1)])
assert a == 0 and r == [F(1), F(0)] and out == r

# Exhaust all three-token distributions on a denominator-four grid.
grid = [[F(i, 4), F(j, 4), F(4 - i - j, 4)]
        for i in range(5) for j in range(5 - i)]
for target in grid:
    for draft in grid:
        a, r, out = corrected_law(target, draft)
        assert out == target
        assert 0 <= a <= 1
        if r is not None:
            assert sum(r) == 1 and all(x >= 0 for x in r)

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: corrected_law([], []))
reject(lambda: corrected_law([F(1, 2)], [F(1)]))
reject(lambda: corrected_law([F(-1), F(2)], [F(0), F(1)]))
reject(lambda: corrected_law([F(1)], [F(1), F(0)]))
print(f"acceptance={float(acceptance):.2f}")
print("corrected:", [float(x) for x in final])
print("wrong_resample:", [float(x) for x in wrong])
print(f"exact_grid_pairs={len(grid) ** 2}")
print("All speculative correction checks passed.")
```

有限网格测试不是对完整实现的证明。单步恒等式说明校正原理，多步保持分布还要求每一步使用正确条件前缀和相应分布。

## 8. 保持分布不等于复现同一串文字

两个算法使用的随机数数量、调用顺序和采样路径可能不同。因此即使分布相同，给定同一个 seed 也不一定生成同一序列。

贪心解码可用一致的 argmax 及平局规则验证草稿，目标是匹配目标贪心路径；它与随机采样的分布校正问题不同。

实际系统还会受到浮点误差、kernel 差异、量化和约束处理影响。若目标本身换成量化模型，保证对象是该目标的分布，而不是原始高精度模型。

## 9. 接受率与速度不是一回事

接受率高可以减少浪费的草稿，但还要支付：

- 草稿模型生成成本。
- 目标验证多个位置的成本。
- 两套状态管理与必要的数据传输。
- 被拒绝后缓存回滚及重新推进。
- batch 调度和额外显存压力。

可用一个粗略预算判断：一轮得到的有效输出数，除以草稿、验证和管理的总时间，再与目标逐 token 基线比较。验证多位置的时间并不固定等于一次单 token decode。

K 越大不一定越好；高负载下，原本较空闲的计算资源可能已被连续批处理充分利用，收益会改变。

## 10. 接受率报告需要定义分母

“接受率”可能指接受的草稿 token / 提出的草稿 token，也可能指只在实际验证到的位置上统计。首次拒绝之后的候选是否计入分母会改变数值。

还应记录每轮有效输出数、全接受比例、K 的分布、草稿耗时、验证耗时和最终服务 goodput。

不要用一个高接受率替代端到端性能测量，也不要用容易预测的数据单独代表所有业务输入。

## 11. 与 KV Cache 的关系

目标验证过程中可能为整段草稿写入临时 KV。发生拒绝后，后续候选对应状态不能继续作为已提交前缀使用。

需要区分逻辑长度与已分配物理空间：截断逻辑长度不一定立即释放所有存储，但后续读取必须排除失效位置。修正 token 的缓存状态还需要按引擎协议补齐。

草稿缓存也必须与新的已提交前缀重新对齐。结合 [KV 块管理](2026-09-22-paged-attention-kv-block-management.md)，重点检查引用计数、写时复制和块边界上的回滚。

## 12. 验收顺序与今日练习边界

已在本地执行 Python 示例，全部断言通过。实际输出：

```text
acceptance=0.70
corrected: [0.5, 0.3, 0.2]
wrong_resample: [0.35, 0.39, 0.26]
exact_grid_pairs=225
All speculative correction checks passed.
```

已检查 UTF-8、代码围栏、公式结构和本地链接；未进行 GitHub 页面视觉渲染验收。

先在小词表检查接受与残差规则，再在短序列上验证条件前缀、首拒绝、全接受、EOS 和长度上限。真实随机实现还需进行分布性检查，不能只比较一次输出。

最后固定模型版本、采样配置与负载，测试质量、延迟、吞吐、显存和失败率。不要把公式性质直接当作具体实现已经无偏或已经加速的证据。

今天的代码仅检查单步解析分布和 225 对有限网格分布，没有实现多 token 验证、实际采样、KV 回滚或性能压测。

## 参考资料

- [Leviathan 等：Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)。
- [Chen 等：Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)。
- [前篇：推理量化](2026-09-23-inference-weight-kv-quantization.md)。

## 今日总结

1. 草稿提供候选，目标在正确条件前缀下验证。
2. 接受规则和残差重采样共同保证目标概率质量。
3. 首次拒绝之后的草稿不能继续直接使用。
4. 相同分布不保证同 seed 逐 token 相同。
5. 接受率、有效输出数和真实端到端收益应分别报告。

## 下次衔接建议

继续学习结构化输出与约束解码：理解 JSON schema、语法状态和 token mask，并区分格式合法、字段语义正确与任务完成。
