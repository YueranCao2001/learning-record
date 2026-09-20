# 2026-09-20：推理部署验收——延迟、吞吐、显存与失败率

## 今日目标

上一篇 [实验追踪与可复现报告](2026-09-18-experiment-tracking-reproducible-reports.md)解决了结果如何追溯。今天进一步讨论：一个离线质量合格的模型，怎样证明它在目标负载下具备服务能力？

完成后应能回答：

1. TTFT、逐 token 延迟和端到端延迟有什么区别？
2. 为什么吞吐提高时，用户体验可能变差？
3. 为什么不能从延迟统计中悄悄删除失败请求？
4. 如何固定工作负载、缓存状态与测量窗口？
5. 如何把质量、性能和可靠性变成联合验收条件？

本篇只运行合成请求日志的 Python 分析，没有部署模型、执行 GPU 压测或测得真实吞吐。示例阈值仅用于教学，不是通用部署标准。

---

## 1. 先写验收协议，再开始压测

模型名称和显卡型号不足以定义一个性能实验。至少冻结：

- 模型、tokenizer、量化方式、服务框架版本。
- 输入长度、输出长度、任务类型和共享前缀分布。
- 解码配置、最大输出长度和停止条件。
- 请求到达过程、并发上限、超时与重试策略。
- 硬件、并行配置、缓存状态、预热与测量阶段。
- 客户端位置、计时边界、失败分类和指标分母。

同一个模型在短问答和长上下文任务上的服务表现不能直接混比。缩短输出虽然可能降低延迟，但可能同时损害质量，因此需要共同验收。

## 2. 三种延迟，三种体验

本文统一采用客户端单调时钟：

- t_send：客户端实际发出请求。
- t_first：收到第一个有效内容 token，而不是空流事件。
- t_last：收到最后一个内容 token。
- t_end：确认请求终止，或者记录超时与失败。

对应定义：

```math
\mathrm{TTFT}=t_{\mathrm{first}}-t_{\mathrm{send}},
\qquad
\mathrm{E2E}=t_{\mathrm{end}}-t_{\mathrm{send}}
```

对输出 N 个 token 且 N 大于 1 的请求：

```math
\mathrm{TPOT}=\frac{t_{\mathrm{last}}-t_{\mathrm{first}}}{N-1}
```

TPOT 是首 token 之后的平均时间，不显示每次停顿。真正的逐 token 间隔分布需要保存各 token 的到达时间。网络流的一个 chunk 可能包含多个 token，不能把 chunk 间隔直接叫 token 间隔。

单 token 响应的 TPOT 没有定义，应标为空值，不能填零。客户端与服务端计时范围不同，不应相减不同时钟上的时间戳。

[vLLM 的逐请求指标文档](https://docs.vllm.ai/en/latest/features/per_request_metrics/)提供了 TTFT 和平均 ITL 等指标定义；使用具体框架时应核对版本与实际计时边界。

## 3. 排队、计算与网络分别观察

TTFT 可能包含服务端排队、prefill、首步解码和网络传输。端到端延迟还包括后续生成与完成事件开销。

如果客户端本身等待并发许可后才发送，t_send 之前的等待不会出现在本文的延迟里。要测用户从提交任务开始的总等待，应额外记录 t_submit。

只观察 GPU kernel 时间会漏掉队列与网络；只观察客户端总时间又不利于定位瓶颈。两套指标都需要，但应保留各自定义。

## 4. 吞吐必须注明计数对象

常见指标包括：

- 成功请求吞吐：成功完成请求数 / 测量时间。
- 输出 token 吞吐：纳入统计的输出 token 数 / 测量时间。
- 到达率：计划或实际发出的请求数 / 到达阶段时长。
- goodput：同时满足指定质量和延迟条件的请求数 / 测量时间。

不要把输入 token、输出 token和失败前产生的 token 混成一个没有解释的 tokens/s。

本文用一批请求从第一个发送到最后一个终止的时间作为窗口。它包含排空阶段，适合解释有限批次日志，不等于稳定负载下的长期容量。

若使用固定墙钟窗口，则必须处理跨窗口请求和未完成请求，不能简单只保留快速完成的请求。

## 5. 开环与闭环负载回答不同问题

闭环：固定数量的客户端，每个收到响应后再发下一个请求。服务变慢时，实际到达率也会下降。

开环：按事先安排的时间发送请求，不等待前一个请求完成。负载可以持续超过服务能力，暴露排队和拒绝行为。

二者都可用，但不能只写“并发为 32”就声称覆盖了真实流量。开环测试还应检查压测客户端是否真的按计划发送，记录计划时间与实际发送时间。

从低负载逐步提高到达率或并发，观察延迟、失败率与吞吐如何变化；不要只展示最大吞吐的一个点。

## 6. 尾延迟与失败率一起报告

成功请求的 p95 延迟很低，可能只是因为最慢请求全部超时。应同时报告：

- 成功请求的延迟分布，并明确这是条件分布。
- 全部已发送请求的成功率、超时率和错误率。
- 所有请求中同时满足质量和时延要求的比例。
- 测量结束时的未完成请求，以及如何处理它们。

失败请求可以保留“到失败终止的耗时”，但不要把它混成成功生成的延迟。不同失败原因应分开，例如限流、服务错误、客户端断连和超时。

[Google SRE 的 SLO 说明](https://sre.google/sre-book/service-level-objectives/)将延迟、错误率和吞吐列为常见服务指标。本文进一步采用联合通过比例，避免只看成功请求掩盖失败。

## 7. 缓存、预热和显存

模型加载、编译和首次 kernel 初始化可能影响冷启动；稳定运行指标不能代替冷启动验收。

同时区分：

- 进程和模型已预热，但目标 prompt 没有命中前缀缓存。
- 已预热且具有预期的缓存命中率。
- 冷启动或缓存失效后的恢复阶段。

重复同一批 prompt 可能提高缓存命中率，导致结果不能代表真实流量。vLLM 的 [benchmark 文档](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)也提醒重复测试可能受前缀缓存影响。

显存报告应说明测量对象：框架 allocated、reserved，还是设备级占用。记录峰值、采样方式与是否发生 OOM；只记录平均显存不足以证明长上下文请求安全。

## 8. 最小实现：从合成日志重算验收指标

约定：全部请求已终止，成功响应至少包含一个内容 token；failed 请求仍进入总分母。quality_ok 来自独立评分流程，不等于 HTTP 成功。

下面用线性插值定义分位数。小样本 p95 仅是描述性统计，不代表尾部估计稳定。

```python
import math


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows, *, ttft_limit=1.0, e2e_limit=3.0):
    if not rows:
        raise ValueError("empty measurement")
    seen, successes, good = set(), [], 0
    for row in rows:
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("invalid or duplicate id")
        seen.add(row["id"])
        if row["status"] not in {"ok", "timeout", "error"}:
            raise ValueError("unknown or unfinished status")
        if type(row["quality_ok"]) is not bool:
            raise ValueError("quality_ok must be bool")
        start, end = row["send"], row["end"]
        if not all(math.isfinite(t) for t in (start, end)) or end < start:
            raise ValueError("invalid request timing")
        if row["status"] != "ok":
            continue
        first, last, count = row["first"], row["last"], row["tokens"]
        if type(count) is not int or count < 1:
            raise ValueError("invalid successful token count")
        if not all(math.isfinite(t) for t in (first, last)):
            raise ValueError("non-finite token time")
        if not start <= first <= last <= end:
            raise ValueError("token time outside request")
        if count == 1 and first != last:
            raise ValueError("single token has inconsistent timestamps")
        ttft, e2e = first - start, end - start
        tpot = (last - first) / (count - 1) if count > 1 else None
        successes.append((ttft, e2e, tpot, count))
        good += row["quality_ok"] and ttft <= ttft_limit and e2e <= e2e_limit
    duration = max(r["end"] for r in rows) - min(r["send"] for r in rows)
    if duration <= 0:
        raise ValueError("non-positive window")
    return {
        "requests": len(rows),
        "successes": len(successes),
        "failure_rate": 1 - len(successes) / len(rows),
        "success_rps": len(successes) / duration,
        "successful_output_tps": sum(r[3] for r in successes) / duration,
        "good_fraction": good / len(rows),
        "goodput_rps": good / duration,
        "success_ttft_p95": percentile([r[0] for r in successes], 0.95),
        "success_e2e_p95": percentile([r[1] for r in successes], 0.95),
        "request_tpot_p95": percentile(
            [r[2] for r in successes if r[2] is not None], 0.95
        ),
    }
```

request_tpot_p95 是“每请求平均 TPOT 的 p95”，不是“所有 token 间隔的 p95”。失败请求产生的部分 token 不计入 successful_output_tps，但失败请求占用的时间仍在窗口内。

代码不实施压测，不读取 GPU，不处理重试去重；它只汇总符合教学 schema 的终止记录。

## 9. 合成数据与边界测试

以下代码接在上一段后执行。

```python
rows = [
    dict(id="a", send=0.0, first=0.5, last=1.5, end=2.0,
         tokens=3, status="ok", quality_ok=True),
    dict(id="b", send=0.0, first=1.5, last=3.5, end=4.0,
         tokens=5, status="ok", quality_ok=True),
    dict(id="c", send=1.0, first=1.2, last=1.2, end=1.3,
         tokens=1, status="ok", quality_ok=False),
    dict(id="d", send=0.0, first=None, last=None, end=5.0,
         tokens=0, status="timeout", quality_ok=False),
]
result = summarize(rows)
assert result["requests"] == 4 and result["successes"] == 3
assert math.isclose(result["failure_rate"], 0.25)
assert math.isclose(result["success_rps"], 0.6)
assert math.isclose(result["successful_output_tps"], 1.8)
assert math.isclose(result["good_fraction"], 0.25)
assert math.isclose(result["goodput_rps"], 0.2)
assert math.isclose(result["success_ttft_p95"], 1.4)
assert math.isclose(result["success_e2e_p95"], 3.8)
assert math.isclose(result["request_tpot_p95"], 0.5)
assert summarize(rows[::-1]) == result
assert summarize([rows[2]])["request_tpot_p95"] is None
assert summarize([rows[3]])["success_e2e_p95"] is None
assert summarize([rows[3]])["failure_rate"] == 1.0

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

reject(lambda: summarize([]))
reject(lambda: summarize(rows + [rows[0]]))
reject(lambda: summarize([dict(rows[0], first=-1)]))
reject(lambda: summarize([dict(rows[0], tokens=0)]))
reject(lambda: summarize([dict(rows[0], end=float("nan"))]))
reject(lambda: summarize([dict(rows[0], status="running")]))
reject(lambda: summarize([dict(rows[2], last=1.25)]))

print(f"failure_rate={result['failure_rate']:.2f}, "
      f"success_rps={result['success_rps']:.2f}")
print(f"successful_output_tps={result['successful_output_tps']:.2f}, "
      f"goodput_rps={result['goodput_rps']:.2f}")
print(f"success_ttft_p95={result['success_ttft_p95']:.2f}s, "
      f"success_e2e_p95={result['success_e2e_p95']:.2f}s")
print("All serving acceptance checks passed.")
```

三个请求技术上成功，只有一个同时质量合格且满足两个延迟门槛。示例展示了“成功吞吐”和“合格吞吐”不是同一个量，不用于推断真实模型容量。

## 10. 联合验收而不是只选最快配置

一个候选部署配置需要同时检查：

1. 固定评估协议下，质量及保留能力满足预定要求。
2. 目标负载下，延迟分布与联合通过比例满足门槛。
3. 失败率、限流、超时和恢复行为可接受。
4. 显存峰值有可解释的安全余量。
5. 运行时长、独立重复和样本量足够支撑结论。

例如量化后更快，但某类任务退化；或者增加 batch 后吞吐更高，但首 token 等待变长。这些都是需要报告的权衡，不能只保留有利指标。

阈值应来自具体产品体验和资源约束，不能从本文合成示例照搬。大规模长期可靠性也不能由一次短压测证明。

## 11. 重试、取消和请求身份

一次用户请求可能对应多次服务尝试。需要同时保存 logical_request_id 与 attempt_id，分别计算用户最终成功率和后端尝试成本。

重试可能提高最终成功率，也可能放大负载和总等待。不能把重试后的成功当成一次快速成功，隐藏前面的等待。

取消与超时之后，还要检查服务端是否停止无用计算、资源能否回收。本文合成日志不验证这些行为，真实验收应单独覆盖。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
failure_rate=0.25, success_rps=0.60
successful_output_tps=1.80, goodput_rps=0.20
success_ttft_p95=1.40s, success_e2e_p95=3.80s
All serving acceptance checks passed.
```

已检查 UTF-8、代码围栏、公式结构和本地链接；未进行 GitHub 页面视觉渲染验收。

练习覆盖失败计入分母、固定批次窗口、单 token 和全失败情况、请求重排不变性，以及重复 ID、无效时间和未完成状态拒绝。

所有性能数字均为人工日志的计算结果，不是服务实测；没有验证网络、调度、显存或真实模型质量。测试代码通过只说明示例在这些断言下符合预期。

## 参考资料

- [vLLM：Per-Request Metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/)。
- [vLLM：Benchmarking CLI 文档](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)。
- [Google SRE：Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)。
- [前篇：实验追踪与可复现报告](2026-09-18-experiment-tracking-reproducible-reports.md)。

## 今日总结

1. TTFT、生成间隔和总延迟对应不同用户体验。
2. 吞吐必须绑定工作负载、窗口和计数定义。
3. 成功请求的延迟不能替代全部请求的可靠性。
4. 缓存、预热、显存和负载发生方式都是实验条件。
5. 部署验收需要质量、性能与失败行为共同通过。

## 下次衔接建议

继续学习连续批处理与请求调度：理解 prefill/decode 混合、KV Cache 容量和长短请求竞争，解释吞吐与尾延迟之间的权衡。
