# 2026年10月7日 RAG 流量控制与过载保护

## 今日目标

昨天的 [超时预算与降级策略](2026-10-06-rag-deadlines-retries-fallbacks.md)限制了单次请求的等待时间。今天进一步控制同时进入系统的工作量，避免请求虽然各自有超时，却一起耗尽线程、显存或下游连接。

核心结论是：限制进入速率、限制在途数量、限制等待队列和暂停故障依赖，是不同的控制手段。系统过载时应尽早给出明确结果，而不是无限排队。

练习使用单线程、人工时钟模拟有界队列，验证准入、过期和容量释放，不进行真实压测。

## 1 四种控制分别管什么

| 控制 | 约束对象 | 不能单独保证 |
| --- | --- | --- |
| 速率限制 | 一段时间内进入的请求量 | 慢请求不会堆积 |
| 并发限制 | 同时执行的工作数量 | 排队等待有上限 |
| 有界队列 | 尚未开始的请求数量 | 排队请求仍来得及完成 |
| 熔断 | 对持续失败依赖的调用 | 整个系统没有资源过载 |

例如每秒只放行十个请求，但每个请求执行一分钟，在途请求仍可能很多。并发限制能够限制执行数，但若等待队列无限增长，内存和用户等待仍会失控。

令牌桶通过按速率补充令牌并限制桶容量，允许受控突发；桶容量不是并发上限。等待令牌的请求同样需要等待上限和截止时间。

## 2 不要只按请求数估算容量

[Google SRE 的过载处理章节](https://sre.google/sre-book/handling-overload/)指出，不同查询的资源成本可能差异很大，仅用每秒请求数描述容量容易失真。

RAG 中，短问题命中答案缓存与长上下文生成的成本不同。需要结合输入长度、生成上限、检索扇出、显存和下游配额设置资源预算。

资源估算也会出错，所以估算准入应配合实际用量观测和硬上限。不能因为某个请求被预测为便宜，就允许它无限生成或无限调用工具。

## 3 排队也消耗截止时间

队列满时拒绝新请求，比让它占用大量内存并最终超时更可控。队列未满也不表示应该排队：已经过期的请求应立即退出，出队时还要再次检查。

若已知最小有用执行预算，可以在准入和出队时预留它；实际执行器仍要遵守总截止时间。下面练习只检查是否已经过期，不预测剩余执行时间。

FIFO 按到达顺序服务，简单但不等于租户公平。大量请求可能占满所有位置；生产系统可结合租户配额、分队列和加权调度，并检查低优先级任务是否长期饥饿。

## 4 最小有界队列

准入器维护运行集合和等待队列。每次推进时间，先清除过期等待项，再将可运行请求按顺序补入空闲位置。

运行中的请求只能在实际完成或确认停止后释放位置；到达截止时间并不证明其计算已经停止。

```python
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    id: str
    deadline: int


class Admission:
    def __init__(self, concurrency, queue_limit):
        if (type(concurrency) is not int or concurrency < 1
                or type(queue_limit) is not int or queue_limit < 0):
            raise ValueError("invalid capacity")
        self.limit = concurrency
        self.queue_limit = queue_limit
        self.running = {}
        self.waiting = deque()
        self.now = 0
        self.events = []

    def advance(self, now):
        if type(now) is not int or now < self.now:
            raise ValueError("time must be monotonic")
        self.now = now
        live = deque()
        for request in self.waiting:
            if request.deadline <= now:
                self.events.append((request.id, "expired"))
            else:
                live.append(request)
        self.waiting = live
        while self.waiting and len(self.running) < self.limit:
            request = self.waiting.popleft()
            self.running[request.id] = request
            self.events.append((request.id, "started"))

    def submit(self, request, now):
        if (not isinstance(request.id, str) or not request.id
                or type(request.deadline) is not int or request.deadline < 0):
            raise ValueError("invalid request")
        self.advance(now)
        if (request.id in self.running
                or any(r.id == request.id for r in self.waiting)):
            return "duplicate"
        if request.deadline <= now:
            return "expired"
        if len(self.running) < self.limit:
            self.running[request.id] = request
            self.events.append((request.id, "started"))
            return "started"
        if len(self.waiting) >= self.queue_limit:
            return "rejected"
        self.waiting.append(request)
        return "queued"

    def finish(self, request_id, now):
        # Call only after the worker has actually stopped using its slot.
        if request_id not in self.running:
            raise ValueError("unknown or already finished request")
        self.advance(now)
        del self.running[request_id]
        self.advance(now)
```

这是内存状态机，不是线程安全的信号量，也不包含持久化。并发实现需要保证检查容量、入队和占用位置的原子性。

duplicate 只防止同时在途的重复 ID，不是业务幂等协议。已完成请求再次提交的处理，需要独立的业务状态与幂等键。

## 5 合成准入测试

以下代码接在上一段之后执行。运行容量为 1，队列容量为 2。

```python
gate = Admission(1, 2)
assert gate.submit(Request("a", 100), 0) == "started"
assert gate.submit(Request("b", 5), 0) == "queued"
assert gate.submit(Request("c", 100), 0) == "queued"
assert gate.submit(Request("d", 100), 0) == "rejected"
assert gate.submit(Request("a", 100), 0) == "duplicate"
assert gate.submit(Request("old", 0), 0) == "expired"
assert len(gate.running) == 1 and len(gate.waiting) == 2

gate.advance(5)
assert ("b", "expired") in gate.events
assert [r.id for r in gate.waiting] == ["c"]
assert gate.submit(Request("d", 100), 5) == "queued"
gate.finish("a", 10)
assert set(gate.running) == {"c"}
assert [r.id for r in gate.waiting] == ["d"]
gate.finish("c", 20)
assert set(gate.running) == {"d"}
gate.finish("d", 30)
assert not gate.running and not gate.waiting

# Expired running work retains its slot until termination is confirmed.
slow = Admission(1, 0)
assert slow.submit(Request("slow", 5), 0) == "started"
slow.advance(5)
assert "slow" in slow.running
assert slow.submit(Request("next", 100), 5) == "rejected"
slow.finish("slow", 6)
assert slow.submit(Request("next", 100), 6) == "started"

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


reject(lambda: gate.finish("a", 30))
reject(lambda: gate.advance(29))
reject(lambda: Admission(0, 1))
reject(lambda: Admission(1, -1))
reject(lambda: gate.submit(Request("bad", True), 30))

assert gate.events == [
    ("a", "started"), ("b", "expired"),
    ("c", "started"), ("d", "started"),
]
print("start order=a,c,d; queued b expired before execution")
print("queue overflow rejected; slots reused after completion")
print("expired running work retains its slot until confirmed stopped")
print("All admission control checks passed.")
```

测试中 a 完成后 c 先于 d 启动，b 在截止时刻被移出等待队列。容量释放后的恢复是本地准入恢复，不代表故障依赖已经恢复健康。

定时清理应由事件循环或定时器持续驱动；不能只在新请求到来时推进，否则空闲期间过期记录可能一直保留。

## 6 熔断要有受控恢复

[Microsoft 的熔断模式](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)描述了关闭、打开和半开状态：正常放行，持续失败后暂停，等待后只允许少量探测请求判断是否恢复。

熔断和重试方向不同：重试尝试再次执行，熔断暂时阻止很可能失败的调用。调用方不能对熔断拒绝立即无限重试。

失败分类很重要。参数错误或用户无权限，不应自动计作依赖服务不可用。不同依赖也不宜共用一个毫无区分的失败计数，否则一个故障端点会连带阻止健康端点。

恢复期逐步增加流量，避免刚恢复就再次被积压请求压垮。上面的队列代码没有实现熔断状态机；真实集成还需测试探测并发上限、成功门槛和再次打开。

## 7 队列之外的资源隔离

仅限制顶层请求数，不能限制每个请求创建多少检索分支。模型生成、向量检索和外部工具可以分别设置并发池，限制单次请求的扇出。

不要持有稀缺 GPU 槽位再长时间等待无关的网络请求。资源获取顺序也要统一，避免两个阶段互相占着对方需要的资源。

多实例各允许十个并发，并不等于整个服务只有十个并发。全局配额、实例配额和扩缩容时的分配需要明确；本例只表示单实例容量。

## 8 拒绝不应引发重试风暴

调用方收到过载结果后，应结合剩余截止时间、重试上限和退避决定是否再尝试。请求已经没有有用预算时应停止，而不是换一个实例继续消耗资源。

保留清晰的拒绝原因：容量满、队列满、已过期、依赖熔断或权限拒绝。客户端可见信息应避免泄露内部容量或私人资源状态。

降级路径仍遵守昨天的边界：可以跳过经验证的可选重排，不能跳过授权、采用失效缓存或用猜测补齐关键证据。

## 9 验收不要只看成功请求

分别统计到达、启动、排队、过期、拒绝、完成和取消；结合队列等待、在途数量、资源用量与答案质量判断收益。

只看完成请求的延迟可能掩盖大量拒绝。应观察在服务目标内完成且质量合格的请求量，并把拒绝与超时纳入清晰分母。

压测覆盖持续过载、短时突发、慢依赖和恢复阶段。恢复时检查队列能否排空、并发是否越界、低优先级是否饥饿，以及取消后的资源何时真正释放。

## 10 今日练习结果

两个 Python 代码块已在本地依次执行，全部断言通过。实际输出：

```text
start order=a,c,d; queued b expired before execution
queue overflow rejected; slots reused after completion
expired running work retains its slot until confirmed stopped
All admission control checks passed.
```

合成状态机验证队列与槽位规则；实际容量参数需要结合模型、硬件和业务流量测试。单线程断言不能替代真实并发、故障恢复和分布式配额测试。

## 参考资料

- [Google SRE Handling Overload](https://sre.google/sre-book/handling-overload/)。
- [Microsoft Circuit Breaker pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)。
- [前篇 RAG 超时预算与降级策略](2026-10-06-rag-deadlines-retries-fallbacks.md)。

## 今日总结

1. 速率、并发、队列和熔断分别约束不同风险。
2. 队列应有界，出队时再次检查截止时间。
3. 执行任务真正停止后才能释放运行容量。
4. 恢复应逐步放量，避免积压和重试再次压垮依赖。
5. 衡量保护效果要同时看合格完成量、拒绝率和资源占用。

## 下次衔接建议

对最近的 RAG 工程笔记做一次端到端复盘，建立从语料版本、检索证据到权限、缓存、预算和过载控制的统一验收清单。
