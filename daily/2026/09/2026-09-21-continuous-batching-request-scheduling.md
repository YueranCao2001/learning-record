# 2026-09-21：连续批处理与请求调度——Prefill、Decode 和 KV 容量

## 今日目标

上一篇 [推理部署验收](2026-09-20-inference-serving-acceptance.md)定义了延迟、吞吐和失败率。今天从调度器角度解释：为什么相同模型和硬件，在不同请求组合下会出现不同的吞吐与尾延迟？

完成后应能回答：

1. 连续批处理与固定请求批次有什么区别？
2. Prefill 和 decode 为什么会互相干扰？
3. 为什么 token 预算不能代替 KV 容量检查？
4. Chunked prefill 如何改变等待和生成间隔？
5. 调度策略怎样验证，而不只展示平均吞吐？

本篇提供纯 Python 单轮调度练习，没有运行真实模型或 GPU。示例只检查资源记账，不预测毫秒延迟，也不复现某个服务框架的完整调度器。

---

## 1. 从请求级批次到迭代级批次

固定批次可以先收集一组请求，再一起处理到整个批次完成。输出短的请求提前结束后，无法继续产生有效工作；若不补入新请求，执行机会可能被浪费。

连续批处理允许在迭代边界移除完成请求、加入其他就绪请求。批次成员可以变化，但每个请求仍需遵守自己的自回归依赖。

[Orca（OSDI 2022）](https://www.usenix.org/conference/osdi22/presentation/yu)提出迭代级调度，以模型迭代而不是整个请求作为调度粒度。

连续批处理不等于任意时刻打断一个 GPU kernel，也不意味着所有请求都必须同长。实际如何打包张量、执行算子和回收资源，还依赖执行引擎。

## 2. Prefill 与 decode 做的工作不同

Prefill 处理 prompt 的一段或全部 token，建立对应的 KV 状态。最终 prompt 位置的 logits 通常可以用于采样首个输出 token。

常规自回归 decode 则使用已有 KV，每轮为每个被选请求推进一个新位置。投机解码、多 token 预测等机制不在今天的范围内。

长 prefill 往往有更多并行计算；小批次 decode 常受到权重或 KV 访存等开销限制，但这不是对所有模型、长度和硬件都成立的固定结论。

一个 prefill token 与一个 decode token 被计入同一预算，并不意味着它们消耗相同的执行时间。

## 3. 为什么长 prompt 会影响正在生成的请求

如果调度器让一个很长的 prefill 占据较长执行时间，已在流式输出的请求就可能等待更久才能执行下一次 decode。

因此需要同时观察：

- 新请求等待多久得到第一个 token。
- 已运行请求的相邻输出间隔是否出现长停顿。
- 长请求和短请求是否被不同程度地延后。
- 队列是否持续增长，而不是只看瞬时吞吐。

客户端的网络 chunk 间隔仍不一定等于服务端 token 间隔，测量边界沿用昨天的区分。

## 4. Chunked prefill：把长 prompt 拆成多轮

可以把长 prompt 的 prefill 分成多个片段，与 decode 交错调度。

小片段可能减少单次阻塞，但会增加调度、执行启动或中间处理次数；大片段可能更高效，却可能拉长其他请求等待。实际效果必须在目标负载上测量。

[SARATHI（2023）](https://arxiv.org/abs/2308.16369)研究了 chunked prefill 与 decode 混合执行。本文仅借此理解机制，不引用其性能数字作为本地实验结果。

“拆块”不保证公平：如果同一个长请求每轮都拿走剩余预算，后面的短 prefill 仍可能长时间等待。

## 5. 三个上限分别约束不同资源

| 约束 | 限制什么 | 常见误解 |
| --- | --- | --- |
| 本轮 token 预算 | 本次计划处理的位置数 | 不是精确的 GPU 时间预算 |
| 请求或序列数上限 | 同时调度或驻留的数量，依实现定义 | 不直接给出显存使用量 |
| KV 容量 | 已缓存和即将写入的位置所占存储 | 不能用本轮 token 数代替 |

假设本轮还能计算 100 个 token，但 KV 已满；新请求可能仍无法加入。反过来，KV 还有空间，也不意味着本轮必须全部填满。

对于普通全注意力模型，一个简化的未分片 KV 字节估算是：

```math
M_{\mathrm{KV}}
\approx 2 L H_{\mathrm{KV}} d_h b \sum_i T_i
```

其中 L 为层数，H_KV 为 KV 头数，d_h 为头维度，b 为每元素字节数，T_i 为缓存位置数。这个估算不含块取整、元数据和分配器开销；分片、滑动窗口、共享前缀或混合架构需要另行处理。

## 6. 接纳不只检查当前时刻

一个请求当前能放下，不保证它持续增长的 KV 总能放下。

系统可以预留未来空间，或采用按需增长并在压力出现时执行抢占、重算、换出或拒绝。不同选择影响利用率、延迟和可靠性。

预留最大输出长度可能浪费容量；按需分配则需要明确资源不足时的处理。无论哪种策略，都不能让超长或长期驻留请求无边界占用资源。

结束、取消和异常请求必须释放相应资源。共享块还需正确管理引用关系，不能简单按请求结束就释放所有关联块。

## 7. 单轮教学策略

今天只实现一个明确的小策略：

1. 先为当前 decode 请求各安排一个 token。
2. decode 请求整体无法放入预算或 KV 时，返回显式错误。
3. 将剩余预算按队列顺序分给 prefill。
4. 每个 prefill 在本轮最多获得 chunk_limit 个 token。

KV 用“位置槽位”表示，一个新位置消耗一个槽位；忽略实际块大小、共享和分片。输入 kv_used 包含此前已驻留的全部位置，包含部分完成的 prefill。

该函数只规划一轮，不模拟完成、释放、抢占、首次采样或真实时延。decode 与 prefill 列表由外部状态机提供，函数不自动转换状态。

## 8. 最小实现与资源不变量

```python
def plan_step(decode_ids, prefills, *, token_budget, kv_used, kv_capacity,
              chunk_limit):
    # prefills: ordered (request_id, remaining_prompt_positions) pairs.
    for value in (token_budget, kv_used, kv_capacity, chunk_limit):
        if type(value) is not int or value < 0:
            raise ValueError("invalid resource value")
    if chunk_limit == 0 or kv_used > kv_capacity:
        raise ValueError("invalid capacity or chunk limit")
    all_ids = list(decode_ids) + [rid for rid, _ in prefills]
    if not all(isinstance(rid, str) and rid for rid in all_ids):
        raise ValueError("invalid request id")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate or overlapping request id")
    if any(type(n) is not int or n <= 0 for _, n in prefills):
        raise ValueError("invalid remaining prefill")
    free = kv_capacity - kv_used
    if len(decode_ids) > min(token_budget, free):
        raise ValueError("decode set needs explicit admission/preemption policy")

    schedule = [(rid, "decode", 1) for rid in decode_ids]
    budget = token_budget - len(decode_ids)
    free -= len(decode_ids)
    for rid, remaining in prefills:
        take = min(remaining, chunk_limit, budget, free)
        if take:
            schedule.append((rid, "prefill", take))
            budget -= take
            free -= take
    used = sum(n for _, _, n in schedule)
    assert used <= token_budget
    assert kv_used + used <= kv_capacity
    return schedule, kv_used + used
```

断言检查的是这个简化模型的记账。真实 KV 以块分配时，一个新增 token 可能不需要新块，也可能触发整个新块分配，不能照抄这里的位置加法作为框架内存管理代码。

## 9. 对照与边界测试

以下代码接在上一段之后运行。

```python
args = dict(token_budget=6, kv_used=4, kv_capacity=10)
small, used_small = plan_step(
    ["d0", "d1"], [("long", 8), ("short", 2)], chunk_limit=2, **args
)
large, used_large = plan_step(
    ["d0", "d1"], [("long", 8), ("short", 2)], chunk_limit=8, **args
)
assert small == [
    ("d0", "decode", 1), ("d1", "decode", 1),
    ("long", "prefill", 2), ("short", "prefill", 2),
]
assert large == [
    ("d0", "decode", 1), ("d1", "decode", 1), ("long", "prefill", 4),
]
assert used_small == used_large == 10

# Plenty of compute budget, but no KV left after the decode reservation.
limited, used = plan_step(
    ["d0", "d1"], [("new", 8)],
    token_budget=100, kv_used=8, kv_capacity=10, chunk_limit=8,
)
assert limited == [("d0", "decode", 1), ("d1", "decode", 1)]
assert used == 10
assert plan_step([], [], token_budget=0, kv_used=0,
                 kv_capacity=0, chunk_limit=1) == ([], 0)

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: plan_step(["d0", "d1"], [], token_budget=1,
                         kv_used=0, kv_capacity=10, chunk_limit=2))
reject(lambda: plan_step(["d0"], [], token_budget=10,
                         kv_used=10, kv_capacity=10, chunk_limit=2))
reject(lambda: plan_step(["d0"], [("d0", 2)], chunk_limit=2, **args))
reject(lambda: plan_step([], [("p", 0)], chunk_limit=2, **args))
reject(lambda: plan_step([], [], chunk_limit=0, **args))

# Exhaustively check resource limits for a small finite input grid.
for capacity in range(1, 9):
    for occupied in range(capacity + 1):
        for budget in range(7):
            for chunk in (1, 2, 4):
                if min(budget, capacity - occupied) < 1:
                    continue
                schedule, after = plan_step(
                    ["d"], [("p0", 5), ("p1", 2)], token_budget=budget,
                    kv_used=occupied, kv_capacity=capacity, chunk_limit=chunk,
                )
                assert sum(n for _, _, n in schedule) <= budget
                assert occupied <= after <= capacity
                assert all(n <= chunk for _, phase, n in schedule
                           if phase == "prefill")

print("small_chunk:", small)
print("large_chunk:", large)
print("KV-limited:", limited)
print("All single-step scheduling checks passed.")
```

小 chunk 的例子给短 prefill 分到了计算机会，大 chunk 的例子没有。但这只证明本轮计划不同，不证明短请求的真实 TTFT 一定改善，更不能把轮数直接当作毫秒。

## 10. Decode 优先也有代价

如果 decode 长期消耗全部预算，新请求的 prefill 可能得不到服务。可以考虑等待时间提升优先级、分配最小 prefill 份额或按服务目标进行接纳控制。

这些措施也有代价：让 prefill 获得更多资源，可能拉长已有请求的生成间隔。需要明确优先保证的是首 token、流式平滑性还是整体完成时间。

队列顺序本身就是策略。测试中不应要求任意重排请求后输出计划不变；今天的实现特意保留 prefill 的输入队列顺序。

## 11. 怎样做真实对照实验

沿用[昨天的验收协议](2026-09-20-inference-serving-acceptance.md)，固定模型、精度、硬件、输出预算和缓存条件，再逐项改变：

- chunk 大小。
- 每轮 token 预算。
- 接纳数量或排队规则。
- 长短 prompt 比例及到达突发程度。

每次记录整体和长度切片的 TTFT、生成间隔、E2E、失败率、goodput、KV 使用和抢占次数。不要只展示总 tokens/s。

本轮 token 数相同不代表运行时间相同；同样平均吞吐也可能伴随不同尾延迟。真实对照还需要预热、重复和足够样本，而不是只跑一次。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际结果：小分块给 long 和 short 各安排 2 个 prefill token；大分块只给 long 安排 4 个；KV 受限例子只安排两个 decode token。

终端确认：`All single-step scheduling checks passed.` 已检查 UTF-8、代码围栏、公式结构和本地链接，未进行 GitHub 页面视觉渲染验收。

练习覆盖 decode 预留、prefill 分块、KV 先耗尽、重复请求拒绝和小范围资源不变量检查。

没有覆盖完整生命周期、饥饿检测、块碎片、前缀缓存、分布式通信、抢占或投机解码。它是资源约束练习，不是可直接上线的调度器。

## 参考资料

- [Orca：A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)。
- [SARATHI：Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)。
- [前篇：推理部署验收](2026-09-20-inference-serving-acceptance.md)。

## 今日总结

1. 连续批处理在迭代边界调整成员，减少固定批次的等待。
2. Prefill 分块提供交错执行机会，但不自动保证公平或低延迟。
3. 计算预算和 KV 容量必须分别检查。
4. Decode 优先改善某些生成体验，也可能延后新请求。
5. 调度策略应通过质量、尾延迟、失败率和资源表现共同验证。

## 下次衔接建议

继续学习 PagedAttention 与 KV 块管理：理解逻辑块、物理块、前缀共享和引用计数，并用小例子检查分配、复用与释放。
