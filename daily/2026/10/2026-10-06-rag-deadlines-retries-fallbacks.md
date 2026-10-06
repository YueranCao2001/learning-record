# 2026年10月6日 RAG 超时预算与降级策略

## 今日目标

接着昨天的 [可观测性与故障定位](2026-10-05-rag-observability-trace-diagnostics.md)，今天把“知道哪里慢”推进到“决定何时停止等待”：为整个请求设置截止时间，约束各阶段和重试，并为安全退出留出时间。

核心结论是所有阶段共享剩余预算，而不是每进入一个阶段就重新获得完整超时。降级可以减少可选计算，但不能降低权限要求或把证据不足伪装成有依据的答案。

练习采用人工毫秒时钟和确定性预算函数，验证调度决定；真实 RPC 的终止还需要客户端超时、取消传播与服务端协作。

## 1 超时与截止时间

超时表示允许等待的时长；截止时间表示最晚可以等到的时刻。[gRPC Deadlines](https://grpc.io/docs/guides/deadlines/)说明了两者的关系，并强调服务端应用需要停止为已取消请求启动的工作。

假设请求在时刻 0 开始，总预算为 1000 毫秒。检索消耗 300 毫秒后，后续阶段最多共享剩余 700 毫秒，不应再各自获得 1000 毫秒。

本地经过时间采用单调时钟。跨主机传播不能直接发送某个进程的单调时钟读数，应使用框架支持的截止时间传播或经过时间扣减机制。

总预算应覆盖排队、连接建立、检索、生成和响应收尾；只约束模型生成时间并不等于约束端到端延迟。

## 2 为收尾预留时间

设总截止时间为 D，当前时间为 t，收尾预留为 R，某阶段自身上限为 C，则可分配时长为：

```math
B = \max(0, \min(C, D - t - R))
```

R 用于必要的响应整理和资源清理，不是额外赠送的时间。若调度器暂停、系统过载或清理不可中断，仍可能超过预期；预算公式不是实时系统保证。

阶段还可以设最小有用预算：时间太少时，与其启动注定难以完成的重排，不如在已验证的策略下跳过它。这个阈值应通过真实负载测试确定，下面的数值只是合成练习参数。

## 3 哪些阶段可以降级

| 情况 | 可考虑的处理 | 不能做的替代 |
| --- | --- | --- |
| 可选重排预算不足 | 使用通过验收的基础排序 | 假称重排已完成 |
| 一路检索失败 | 检查剩余证据是否充分 | 默认另一条路径已经足够 |
| 证据不完整 | 明确无法可靠回答或只答可支持部分 | 用猜测补齐缺失条件 |
| 权限服务不可用 | 停止敏感数据返回 | 使用过往允许结果绕过校验 |
| 总截止时间已到 | 停止新工作并传播取消 | 再给降级模型完整新预算 |

降级模型、缓存答案和基础排序路径都需要预先验收。缓存即使能更快返回，也要满足 [缓存失效与权限复核](2026-10-03-rag-cache-invalidation.md)规则。

## 4 重试要同时满足三个条件

首先确认错误适合重试，其次确认操作可安全重放，最后确认尝试次数和剩余时间足够。

[gRPC Retry](https://grpc.io/docs/guides/retry/)将可重试状态、尝试上限和退避列为策略组成，并使用抖动分散客户端同时重试的压力。具体框架行为与应用策略应分别核实。

超时并不证明远端没有完成操作。有副作用的调用需要幂等键或状态核对，不能仅凭“没有收到响应”就再次发送。

多个层级各重试三次可能放大下游调用量。应明确哪一层负责重试，将 SDK 内部尝试也纳入预算与监控。

## 5 最小预算规划器

函数返回等待时间和本次调用的上限，None 表示不应启动。attempt 是准备执行的尝试编号，从 1 开始；max_attempts 包含首次调用。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    wait_ms: int
    call_ms: int


def plan_attempt(*, now, deadline, reserve, cap, minimum,
                 attempt, max_attempts, backoff=0,
                 retryable=False, replay_safe=False, cancelled=False):
    values = (now, deadline, reserve, cap, minimum,
              attempt, max_attempts, backoff)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("expected nonnegative integer milliseconds/counts")
    if cap < 1 or minimum < 1 or minimum > cap:
        raise ValueError("invalid stage budget")
    if attempt < 1 or max_attempts < 1:
        raise ValueError("invalid attempt count")
    if cancelled or attempt > max_attempts:
        return None
    if attempt > 1 and not (retryable and replay_safe):
        return None
    wait = backoff if attempt > 1 else 0
    available = deadline - now - reserve - wait
    call = min(cap, available)
    if call < minimum:
        return None
    return Plan(wait, call)


def after_wait(plan, *, now, deadline, reserve, minimum, cancelled=False):
    # Recheck after the actual wait; scheduling may have consumed extra time.
    if cancelled:
        return None
    available = min(plan.call_ms, deadline - now - reserve)
    return available if available >= minimum else None
```

规划器不执行 sleep、不调用服务，也不自动实施取消。实际执行器必须把返回上限传给调用层，并检查结果到达时是否仍在允许期限内。

退避结束后要再次读时钟，不能假设实际只等待了计划中的毫秒数。等待和调用也都应支持取消；发现取消后不再启动下一次尝试。

## 6 合成预算与重试测试

以下代码接在上一段之后执行。共享截止时间为 1000 毫秒，收尾预留为 100 毫秒，检索单次上限为 300 毫秒。

```python
base = dict(deadline=1000, reserve=100, cap=300, minimum=100,
            max_attempts=2)


def make(now, attempt=1, **changes):
    return plan_attempt(**(base | dict(now=now, attempt=attempt) | changes))


first = make(0)
assert first == Plan(0, 300)
retry = make(300, 2, backoff=50, retryable=True, replay_safe=True)
assert retry == Plan(50, 300)
assert make(750, 2, backoff=50, retryable=True,
            replay_safe=True) == Plan(50, 100)
assert make(751, 2, backoff=50, retryable=True, replay_safe=True) is None
assert make(900) is None
assert make(1000) is None
assert make(1001) is None
assert make(0, cancelled=True) is None
assert make(300, 2, retryable=False, replay_safe=True) is None
assert make(300, 2, retryable=True, replay_safe=False) is None
assert make(300, 3, retryable=True, replay_safe=True) is None

# Actual wake-up is later than planned, leaving a smaller call budget.
assert after_wait(retry, now=750, deadline=1000,
                  reserve=100, minimum=100) == 150
assert after_wait(retry, now=801, deadline=1000,
                  reserve=100, minimum=100) is None
assert after_wait(retry, now=350, deadline=1000,
                  reserve=100, minimum=100, cancelled=True) is None

# Retrieval retry ends at 650; optional reranking is skipped at 820.
rerank = make(820, cap=150, minimum=100)
assert rerank is None
generation = make(650, cap=500, minimum=200, max_attempts=1)
assert generation == Plan(0, 250)

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


reject(lambda: make(-1))
reject(lambda: make(0, attempt=0))
reject(lambda: make(0, reserve=True))
reject(lambda: make(0, minimum=301))
reject(lambda: make(0, max_attempts=0))

print(f"first={first}, retry={retry}")
print(f"generation at t=650: {generation}")
print("optional rerank skipped at t=820")
print("All deadline budget checks passed.")
```

生成在 650 毫秒开始时最多得到 250 毫秒，不代表它一定能完成。若模型未在上限内结束，需要终止或取消，并根据已取得的证据和响应状态决定如何退出。

重排在 820 毫秒的检查与生成在 650 毫秒的检查是两个独立场景，不是一条按时间顺序执行的真实请求。

## 7 并行分支与取消传播

并行检索共享同一个总截止时间，不能把两条分支的超时相加作为总等待上限。协调器要明确等待全部结果、任一结果还是满足证据条件后结束。

不再需要的分支应收到取消信号；收到取消不代表工作已立即停止。记录下游确认取消的时间、仍在消耗的资源和清理结果。

对于不能中断的计算，限制并发和准入可以减少无效工作。客户端停止等待之后，还应避免遗留任务继续写入不适用的结果或触发外部副作用。

## 8 已经开始流式输出时

在流式输出开始前，可以用结构化错误或拒答返回失败。输出开始后，不能把半段回答和另一套重试结果直接拼接成一条看似完整的答案。

应区分完成、截断、取消和失败，保留响应标识。客户端据此决定展示“不完整”状态或发起新的请求，而不是将已收到 Token 数大于零视为成功。

答案提前输出也可能先于证据核验完成。需要明确哪些内容可以流式展示，哪些关键事实必须在核验后才能发布。

## 9 降级验收与指标

沿用昨天的追踪字段，增加初始预算、阶段入场剩余时间、实际退避、尝试次数、降级原因和取消结果。

分别统计完整成功、降级成功、合理拒答、超时和取消。比较延迟时还要比较答案正确性、引用支持和覆盖率，不能用更快拒绝所有请求来冒充系统改进。

测试至少覆盖边界时刻、慢连接、排队超时、退避超预算、客户端取消和迟到结果。重放有副作用的场景时使用隔离环境，不连接真实收件人或业务写接口。

## 10 今日练习结果

两个 Python 代码块已在本地依次执行，全部断言通过。实际输出：

```text
first=Plan(wait_ms=0, call_ms=300), retry=Plan(wait_ms=50, call_ms=300)
generation at t=650: Plan(wait_ms=0, call_ms=250)
optional rerank skipped at t=820
All deadline budget checks passed.
```

人工时钟覆盖预算判断和重试资格；真实网络中的中断延迟、取消传播和模型退出行为需要集成测试。配置中的预留与最小预算应以业务要求和负载测试校准，不能直接照搬练习数值。

## 参考资料

- [gRPC Deadlines](https://grpc.io/docs/guides/deadlines/)。
- [gRPC Retry](https://grpc.io/docs/guides/retry/)。
- [前篇 RAG 可观测性与故障定位](2026-10-05-rag-observability-trace-diagnostics.md)。
- [工具调用的可靠执行](../09/2026-09-26-reliable-tool-execution-idempotency.md)。

## 今日总结

1. 所有阶段和重试共享总截止时间，收尾预留也在总预算内。
2. 退避后重新读取时钟，剩余预算不足就不再启动调用。
3. 可重试错误、可安全重放与次数限制必须同时满足。
4. 取消需要执行层配合，停止等待不等于下游已停止工作。
5. 降级减少计算，不减少权限检查和证据要求。

## 下次衔接建议

继续学习 RAG 流量控制与过载保护，区分并发上限、队列长度、限流和熔断，并用合成请求验证准入、拒绝和恢复策略。
