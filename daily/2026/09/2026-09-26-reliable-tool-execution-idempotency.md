# 2026-09-26：工具调用的可靠执行——权限、幂等键、重试与结果回传

## 今日目标

上一篇 [结构化输出与约束解码](2026-09-25-structured-output-constrained-decoding.md)解决了参数形式。今天讨论从“模型提出操作”到“系统可靠执行”的边界：参数合法不代表允许执行，超时也不代表操作没有发生。

完成后应能回答：

1. 模型建议、权限校验和实际执行为什么要分开？
2. request_id、attempt_id 与幂等键有什么区别？
3. 响应丢失后，如何避免重复副作用？
4. 哪些失败可以重试，哪些应先查询状态？
5. 工具结果应该怎样反馈给模型和用户？

本篇只运行内存中的合成记录服务，不发送邮件、不调用付费接口、不修改外部业务数据。示例不提供生产环境的持久化或并发保证。

---

## 1. 工具调用是一项提议，不是授权凭证

模型可以提出工具名称与参数，但执行器仍需独立检查：

- 工具是否在当前任务的允许范围内。
- 调用者身份来自可信认证上下文，而不是模型参数。
- 资源是否属于调用者，当前权限是否允许目标操作。
- 参数是否满足类型、范围、长度及跨字段关系。
- 是否涉及需要额外确认的外部副作用。

不要把 owner、role 或 approved=true 这类模型提供的字段直接当作权限事实。即使参数严格符合 Schema，也需要业务侧授权。

## 2. 三类 ID 的职责

| 标识 | 作用 | 重试时是否保持 |
| --- | --- | --- |
| logical_request_id | 用户的一次逻辑意图 | 通常保持 |
| attempt_id | 一次具体网络尝试 | 每次不同 |
| idempotency_key | 服务端识别同一操作的重复提交 | 同一操作必须保持 |

模型消息中的 tool_call_id 可以关联工具结果，但不天然等于业务幂等键。模型重新生成一次工具调用后，编排器仍要判断它是原操作重试还是新的用户意图。

两个参数相同的操作可能是用户明确要求的两次独立创建，不能只用参数哈希永久合并。幂等键应绑定一次明确意图，参数指纹则用于发现“同键不同参数”。

## 3. 超时不等于失败

客户端超时可能对应：

- 请求尚未到达服务端。
- 服务端正在执行。
- 操作已成功，但响应丢失。
- 操作失败，错误响应丢失。

因此 unknown 应是明确状态，不能自动转成 failed。

对于有副作用的操作，优先用原幂等键重试或查询已有操作状态。如果接口没有安全重试契约，先对账或请求人工处理，不能通过换一个新键来“解决”超时。

## 4. 幂等服务的核心约束

服务端应把键限定在调用者和操作范围内，并保存参数指纹、状态与结果：

- 同范围、同键、同参数：返回已有结果或处理中状态。
- 同范围、同键、不同参数：拒绝冲突。
- 不同调用者：不能通过相同键读到另一人的结果。
- 保留期结束后：不能继续假定旧键仍有去重保证。

[Amazon Builders’ Library：Making retries safe with idempotent APIs](https://d1.awsstatic.com/builderslibrary/pdfs/making-retries-safe-with-idempotent-apis-malcolm-featonby.pdf)讨论了通过调用者提供的请求标识让重复调用安全化的设计。

幂等不是“网络只发送一次”，也不是所有外部系统都天然支持 exactly-once。它是特定接口、作用范围和保留期限内的行为契约。

## 5. 最危险的是副作用与去重记录之间的空隙

如果先创建资源，再保存幂等记录，中间崩溃后重试可能重复创建。

如果先标记成功，再执行副作用，中间崩溃又可能返回一个并不存在的成功。

当副作用和记录都在同一数据库内，可以考虑事务与唯一约束。若副作用发生在外部服务，需要使用对方的幂等能力、操作查询、对账或其他协调机制。仅在本地添加一张去重表不能自动消除外部提交的不确定性。

处理中记录也不能因为“等得久”就无条件重做：租约、接管和旧执行者是否仍可能提交需要共同设计。

## 6. 最小实现：内存记录服务

示例把所有参数限制为一项短字符串，通过可信 principal 选择身份。它模拟顺序请求以及“服务提交后丢失响应”，不模拟崩溃窗口。

```python
import hashlib
import json
from copy import deepcopy


class RecordService:
    def __init__(self):
        self.allowed = {"alice", "bob"}
        self.records = {}
        self.receipts = {}
        self.next_id = 1

    def create(self, principal, key, args, *, lose_response=False):
        # principal represents authenticated server-side context.
        if principal not in self.allowed:
            raise PermissionError("not authorized")
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("invalid idempotency key")
        if type(args) is not dict or set(args) != {"title"}:
            raise ValueError("unexpected fields")
        title = args["title"]
        if not isinstance(title, str) or not 1 <= len(title) <= 100:
            raise ValueError("invalid title")

        payload = json.dumps(
            args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        fingerprint = hashlib.sha256(payload).hexdigest()
        scope = (principal, "create_record_v1", key)

        if scope in self.receipts:
            old_fingerprint, result = self.receipts[scope]
            if fingerprint != old_fingerprint:
                raise ValueError("same key with different parameters")
            return deepcopy(result)

        rid = f"r{self.next_id}"
        self.next_id += 1
        self.records[rid] = {"owner": principal, "title": title}
        result = {"status": "succeeded", "record_id": rid}
        self.receipts[scope] = (fingerprint, deepcopy(result))

        # Simulates a lost response AFTER both in-memory updates completed.
        if lose_response:
            raise TimeoutError("response lost after commit")
        return deepcopy(result)
```

本例先校验当前权限再返回缓存结果。因此权限撤销后，重放也不会泄露旧结果。实际系统可以选择其他明确的结果读取权限策略，但不能完全省略检查。

代码中的两次字典写入不是事务；没有线程锁、持久化、过期、租约或崩溃恢复。不得将它直接作为生产去重网关。

## 7. 测试：同键重放与不同意图

以下代码接在上一段之后执行。

```python
def reject(error_type, fn):
    try:
        fn()
    except error_type:
        return
    raise AssertionError("expected rejection")


service = RecordService()
params = {"title": "学习记录"}

reject(TimeoutError, lambda: service.create(
    "alice", "intent-001", params, lose_response=True
))
assert len(service.records) == 1

result = service.create("alice", "intent-001", params)
assert result == {"status": "succeeded", "record_id": "r1"}
assert len(service.records) == 1
assert service.create("alice", "intent-001", params) == result

# Returned objects cannot mutate the saved receipt.
result["status"] = "tampered"
assert service.create("alice", "intent-001", params)["status"] == "succeeded"

reject(ValueError, lambda: service.create(
    "alice", "intent-001", {"title": "changed"}
))
assert len(service.records) == 1

# New intent with identical arguments is a distinct operation.
second = service.create("alice", "intent-002", params)
assert second["record_id"] == "r2"
# Same key in another authenticated principal's scope is independent.
third = service.create("bob", "intent-001", params)
assert third["record_id"] == "r3"
assert service.records["r3"]["owner"] == "bob"

reject(PermissionError, lambda: service.create("mallory", "x", params))
reject(ValueError, lambda: service.create(
    "alice", "x", {"title": "ok", "owner": "bob"}
))
reject(ValueError, lambda: service.create("alice", "", params))
reject(ValueError, lambda: service.create("alice", "x", {"title": ""}))
service.allowed.remove("alice")
reject(PermissionError, lambda: service.create("alice", "intent-001", params))
assert len(service.records) == 3

print("lost response + same-key retry: 1 record")
print("new intent and separate principal: 3 records total")
print("conflicts, unauthorized requests and stale access rejected")
print("All idempotent execution checks passed.")
```

参数哈希只表示本例约定编码下的内容身份，不提供授权，也不是通用跨语言规范化方案。示例记录内容为合成数据，没有发送到外部系统。

## 8. 重试分类和预算

| 结果 | 后续处理原则 |
| --- | --- |
| 参数错误、权限不足 | 停止盲重试，修正原因或取得授权 |
| 限流、临时不可用 | 按接口约定，在预算内退避重试 |
| 写操作超时、连接中断 | 先保留未知状态，使用原键或查询状态 |
| 已成功 | 返回已确认结果，不重新制造副作用 |
| 同键不同参数 | 冲突；不要静默换键再次执行 |

不是所有 5xx 都意味着未提交。是否可以自动重试必须结合操作语义和服务契约。

[Amazon Builders’ Library：Timeouts, retries, and backoff with jitter](https://d1.awsstatic.com/builderslibrary/pdfs/timeouts-retries-and-backoff-with-jitter.pdf)强调超时、退避及随机抖动对控制重试负载的重要性。

重试策略应有最大次数、总截止时间和可重试错误集合；若多层都重试，调用次数可能相乘。避免把模型、编排器、SDK 的重试各自开启却无人统一预算。

## 9. 结果回传要保留证据等级

工具响应建议区分：

- succeeded：操作已确认完成，附资源或回执标识。
- failed：已确认失败，附明确错误类别。
- pending：操作仍在处理中，可查询。
- unknown：无法确认最终结果，不把它包装成成功或失败。

同时保留逻辑请求、尝试 ID、操作键、错误类别和时间。日志中对敏感参数脱敏；不要把密钥复制给模型。

如果只收到“任务已受理”，应向用户报告已受理，而不是已完成。模型的叙述应受实际工具结果约束。

## 10. 工具返回内容仍然不是高优先级指令

网页、文件、日志或数据库字段可能包含类似“忽略上文并执行其他工具”的文本。这些是外部数据，不能改变已授权操作范围。

执行器应保持工具 allowlist、身份来源、参数边界和高风险审批独立于模型解释。返回结果可以为下一步提供事实，但不能自行授权新副作用。

幂等键也不能解决权限问题：一个未经授权的操作即使只执行一次，仍然是不允许的。

## 11. 多步流程和补偿

“创建资源后发送通知”包含至少两个副作用。第一步成功、第二步失败时，不应把整个流程从头重做而重复创建。

为各步骤绑定稳定的操作身份，记录已完成状态；恢复时从未确认阶段继续。必要的补偿也应明确授权、可追踪，并考虑自身重试。

补偿不一定能完全撤销现实影响，例如通知一旦被读到就无法让人“没看见”。因此不要把补偿机制描述为通用事务回滚。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
lost response + same-key retry: 1 record
new intent and separate principal: 3 records total
conflicts, unauthorized requests and stale access rejected
All idempotent execution checks passed.
```

已检查 UTF-8、代码围栏和本地链接；未进行 GitHub 页面视觉渲染验收。

示例验证同键重试不重复创建、参数冲突拒绝、调用者隔离、返回结果副本和权限撤销后的重放拒绝。

仍未验证并发重复、持久化故障、键过期、服务端处理中状态、真实网络和外部副作用一致性。上线前应专门注入“提交后响应丢失”“去重记录丢失”和“重试同时到达”等故障。

## 参考资料

- [Amazon Builders’ Library：Making retries safe with idempotent APIs](https://d1.awsstatic.com/builderslibrary/pdfs/making-retries-safe-with-idempotent-apis-malcolm-featonby.pdf)。
- [Amazon Builders’ Library：Timeouts, retries, and backoff with jitter](https://d1.awsstatic.com/builderslibrary/pdfs/timeouts-retries-and-backoff-with-jitter.pdf)。
- [前篇：结构化输出与约束解码](2026-09-25-structured-output-constrained-decoding.md)。

## 今日总结

1. 参数合法不等于操作获准，身份必须来自可信上下文。
2. 同一意图的重试保持同一个幂等键，不同意图不能仅凭参数合并。
3. 超时表示结果可能未知，不证明副作用没有发生。
4. 幂等记录与实际副作用之间的一致性是实现重点。
5. 成功、失败、处理中和未知状态都应如实回传。

## 下次衔接建议

继续学习检索增强生成的最小闭环：从文档切分、检索和上下文组装，到引用核验与“证据不足”的处理。
