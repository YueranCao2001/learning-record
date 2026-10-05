# 2026年10月5日 RAG 可观测性与故障定位

## 今日目标

昨天的 [增量索引与版本发布](2026-10-04-rag-incremental-index-publication.md)让请求固定语料快照。今天进一步记录一次请求经过了哪些阶段、使用哪个版本、哪里发生等待，以及答案错误时应从哪里查起。

核心结论是：运行成功、响应快速和答案正确是三件不同的事。用追踪关联单次请求，用指标发现整体变化，再用独立质量评估解释答案是否可靠。

本篇使用人工构造的毫秒时间线检查统计逻辑，不接入真实 RAG 服务或遥测平台，不将合成数据当作性能测量。

## 1 追踪与指标分别回答什么

[OpenTelemetry 的追踪说明](https://opentelemetry.io/docs/concepts/signals/traces/)把 Span 作为一次操作的记录单元，包含时间、父级、属性和状态等信息，并通过上下文传播关联分布式操作。

本文据此设计一条教学追踪：请求入口、缓存检查、两路检索、重排和生成。真实多跳、重试或工具调用还会增加分支。

指标回答“这个时间窗口是否整体变慢”；追踪帮助解释“这个请求具体在哪段时间等待”；日志记录适合诊断的离散事件。三者通过受控标识关联，而不是把完整私人问题复制到每一种记录里。

一次追踪也不是因果证明。某阶段耗时很长，需要继续检查输入规模、排队、下游状态和重试，不能直接断定某个模型或索引实现有问题。

## 2 最小记录字段

我会保留以下应用字段；它们是本篇建议，不是完整的 OpenTelemetry 语义规范。

| 位置 | 记录内容 | 诊断用途 |
| --- | --- | --- |
| 请求 | 请求标识、结果类别、起止时间 | 关联全流程与用户等待 |
| 检索 | 快照版本、候选数、过滤后数量 | 判断空结果与版本变化 |
| 生成 | 模型及提示版本、输入输出量 | 区分配置变化和输入膨胀 |
| 缓存 | 原始命中与复核结果 | 区分找到旧条目和安全复用 |
| 重试 | 尝试编号、失败类别、退避时间 | 解释最终成功前的成本 |

请求标识适合放在追踪或日志中；不要把每个不同的请求、问题或文档 ID 都变成聚合指标的标签。

语料版本由服务实际使用的快照提供，不能仅记录发布系统“希望使用”的版本。旧请求使用旧版本本身是正常情况，同一次请求的证据路径意外混用版本才值得告警。

## 3 时间边界要先定义

端到端时间可以从网关接受请求计到响应结束，也可以从客户端发起计到接收完成；两者不可混称。

首 Token 时间应注明从哪个起点开始，以及是否包含查询改写、检索、排队和网络传输。无流式输出或提前拒绝的请求不要伪造首 Token 时间为零。

同一进程内测量经过时间宜采用单调时钟。跨主机时间线还要考虑时钟偏差，不能直接把墙上时钟的负差值当成负延迟。

下面代码假设所有阶段已转换成同一参考系中的整数毫秒，不演示真实时钟同步。

## 4 并行阶段不能直接相加

假设两路检索都在第 10 毫秒开始，分别在第 60 和第 80 毫秒结束。它们的耗时和为 120 毫秒，但这段等待只覆盖 70 毫秒。

将父 Span 和子 Span 一起相加也会重复计算。本文只分析同一个请求下面的一层阶段记录，用时间区间并集计算被这些阶段覆盖的墙上时间。

根请求时长减去区间并集，得到的是“这些记录没有覆盖的时间”，不是自动得到 CPU 自身耗时，也不是已经定位的网络开销。

区间并集同样不是关键路径算法。真正的关键路径分析还需要依赖关系、调度和等待语义。

## 5 最小追踪检查器

代码拒绝重复阶段 ID、错误父级、越界或反向时间，以及本例约定的快照版本混用。阶段名可以重复，例如同一操作的不同尝试，但 ID 必须不同。

```python
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Stage:
    id: str
    parent: str
    name: str
    start: int
    end: int
    generation: str


def analyze(request_id, start, end, generation, stages):
    if (type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        raise ValueError("invalid request interval")
    seen = {request_id}
    intervals = []
    for s in stages:
        if not s.id or s.id in seen or s.parent != request_id:
            raise ValueError("invalid stage identity")
        seen.add(s.id)
        if (type(s.start) is not int or type(s.end) is not int
                or not start <= s.start <= s.end <= end):
            raise ValueError("invalid stage interval")
        if s.generation != generation:
            raise ValueError("mixed generation")
        intervals.append((s.start, s.end))

    covered = 0
    right = start
    for left, stop in sorted(intervals):
        covered += max(0, stop - max(left, right))
        right = max(right, stop)
    elapsed = end - start
    return {
        "elapsed_ms": elapsed,
        "stage_sum_ms": sum(b - a for a, b in intervals),
        "covered_ms": covered,
        "uncovered_ms": elapsed - covered,
    }
```

本例把每个阶段都绑定到同一个语料版本，是为了测试协议。生产中的鉴权、外部工具或异步事件未必使用语料版本，应只对需要一致快照的路径应用对应检查。

缺少阶段不自动意味着阶段耗时为零；可能是缓存跳过、采样丢失或埋点失败。需要结合显式状态判断。代码不会凭空恢复未记录的工作。

## 6 合成时间线与边界测试

以下代码接在上一段之后执行。请求共 220 毫秒，其中两路检索并行，生成在第 110 毫秒开始。

```python
stages = [
    Stage("s1", "r1", "cache", 0, 10, "g2"),
    Stage("s2", "r1", "keyword", 10, 60, "g2"),
    Stage("s3", "r1", "vector", 10, 80, "g2"),
    Stage("s4", "r1", "rerank", 80, 100, "g2"),
    Stage("s5", "r1", "generate", 110, 200, "g2"),
]
result = analyze("r1", 0, 220, "g2", stages)
assert result == {
    "elapsed_ms": 220, "stage_sum_ms": 240,
    "covered_ms": 190, "uncovered_ms": 30,
}
assert analyze("r1", 0, 220, "g2", stages[::-1]) == result
assert analyze("r1", 0, 220, "g2", [])["uncovered_ms"] == 220
assert analyze("r1", 0, 220, "g2", [
    Stage("zero", "r1", "event", 20, 20, "g2")
])["covered_ms"] == 0
assert analyze("r1", 0, 220, "g2", [
    Stage("full", "r1", "operation", 0, 220, "g2"),
    Stage("overlap", "r1", "operation", 10, 30, "g2"),
])["covered_ms"] == 220


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


reject(lambda: analyze("r1", 0, 220, "g2", stages + [stages[0]]))
for bad in (
    replace(stages[0], parent="other"),
    replace(stages[0], start=-1),
    replace(stages[0], end=221),
    replace(stages[0], start=11),
    replace(stages[0], generation="g1"),
    replace(stages[0], end=True),
):
    reject(lambda bad=bad: analyze("r1", 0, 220, "g2", [bad]))
reject(lambda: analyze("r1", 0, 0, "g2", stages))

print(f"elapsed={result['elapsed_ms']} ms, stage_sum={result['stage_sum_ms']} ms")
print(f"covered={result['covered_ms']} ms, uncovered={result['uncovered_ms']} ms")
print("overlap is not extra user waiting time")
print("All trace analysis checks passed.")
```

30 毫秒未覆盖时间来自第 100 到 110 毫秒和第 200 到 220 毫秒。它们可能包含序列化、调度或其他未记录工作；本例没有证据进一步归因。

生成阶段的 90 毫秒也不是“优化生成就一定能减少 90 毫秒”的承诺。瓶颈判断应结合真实依赖和负载实验。

## 7 分位数与失败分母

[Prometheus 的直方图说明](https://prometheus.io/docs/practices/histograms/)指出，不能通过平均各实例预先计算的分位数得到全局分位数；分布聚合与分位数估计需要合适的数据表示。

同样，各阶段的 p95 相加不是端到端 p95。高分位请求可能不是同一批请求，并行关系也使加法失效。应直接测量请求级时长分布。

成功请求、超时、取消和拒绝分别报告，同时保留全部请求数量。只保留成功样本可能让大量超时期间的图表看起来更快。

服务端在超时点停止记录，不代表下游工作已经停止。应跟踪取消是否传递，以及超时后仍消耗的资源；不要把超时值解释为任务本来能够完成的耗时。

## 8 质量与运行状态分开

HTTP 成功或阶段执行完成不代表答案正确。结合 [RAG 回归评估](../09/2026-09-30-rag-evaluation-regression-tests.md)，额外关联检索覆盖、答案正确、引用支持和合理拒答等独立标签。

没有人工或可靠评估器标注时，质量应写成未知，不能默认正确。异步质量评估可以稍后通过请求标识关联，但必须记录评估器版本和覆盖范围。

缓存命中时生成阶段通常被跳过，应标记跳过原因，不能记录成“生成 0 毫秒，模型变快”。比较冷请求与缓存请求时应分组。

## 9 从症状到可检验假设

我会按以下顺序排查：

1. 先确定受影响时间段、请求类型和发布版本，排除流量构成变化。
2. 比较端到端延迟、失败和取消情况，再抽查代表性追踪。
3. 检查排队、重试、检索分支、输入长度与缓存复核结果。
4. 对质量退化固定问题、证据和配置，重放并逐层替换做对照。
5. 修复后比较同口径指标和逐题回归，而不是只展示一条更快请求。

版本切换与错误同时出现只是线索。重放也应在隔离环境中进行，禁用真实发送、删除等副作用，避免诊断产生新的外部影响。

## 10 遥测数据本身也要治理

本文建议默认记录版本、计数、分类和脱敏标识，按需受控采集正文。原始问题、检索材料和输出可能含私人数据，追踪平台同样需要访问控制、保留期限和删除机制。

采样追踪可帮助诊断，但不能直接将偏向慢请求或错误请求的样本比例当作总体错误率。总体计数应有独立且明确的统计口径，并监控遥测丢失。

埋点和导出也会增加开销。应检查队列积压、丢弃数量与导出失败，避免在服务故障时因为遥测堵塞进一步拖慢请求。

## 11 今日练习结果与边界

已在本地依次执行两个 Python 代码块，全部断言通过。实际输出：

```text
elapsed=220 ms, stage_sum=240 ms
covered=190 ms, uncovered=30 ms
overlap is not extra user waiting time
All trace analysis checks passed.
```

UTF-8、代码围栏、本地链接与 README 索引检查通过；未进行 GitHub 页面视觉渲染验收。

本例只处理一层阶段和人工统一时钟，没有构建完整 Span 树、实现关键路径算法或真实分位数聚合，也没有运行模型质量评估。

## 参考资料

- [OpenTelemetry Traces](https://opentelemetry.io/docs/concepts/signals/traces/)。
- [Prometheus Histograms and summaries](https://prometheus.io/docs/practices/histograms/)。
- [前篇 RAG 增量索引与版本发布](2026-10-04-rag-incremental-index-publication.md)。
- [RAG 端到端评估与回归测试](../09/2026-09-30-rag-evaluation-regression-tests.md)。

## 今日总结

1. 指标发现整体变化，追踪帮助定位单次请求，质量评估判断答案。
2. 并行和嵌套时间不能直接相加作为用户等待时间。
3. 未覆盖时间是待调查的区间，不是已经确定的根因。
4. 版本、缓存状态、重试和失败分母决定指标的解释。
5. 遥测也需要权限、隐私、采样口径和开销控制。

## 下次衔接建议

继续学习 RAG 超时预算与降级策略，设计多阶段剩余时间分配、重试上限、取消传播与证据不足时的安全退出。
