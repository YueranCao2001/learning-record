# 2026年10月3日 RAG 缓存与失效策略

## 今日目标

接着昨天的 [提示注入与信任边界](2026-10-02-rag-prompt-injection-trust-boundaries.md)，今天检查一个容易绕过权限校验的捷径：直接返回以前生成过的答案。

核心结论是：缓存命中只表示找到了旧结果，不表示结果仍正确、仍适用于当前问题或仍可向当前用户展示。缓存键、版本校验与当前权限检查必须共同决定是否复用。

本篇用纯 Python 字典和人工时钟演示答案缓存，不连接 Redis、不运行模型，也不测量真实加速效果。

## 1 先区分缓存层

| 缓存层 | 保存内容 | 主要变化来源 |
| --- | --- | --- |
| 查询向量 | 查询的嵌入表示 | 嵌入模型、预处理版本 |
| 检索结果 | 候选文档及排序 | 语料、索引、过滤条件 |
| 上下文片段 | 抽取或压缩后的证据 | 原文、压缩规则、权限 |
| 最终答案 | 文本及来源依赖 | 上述变化加模型与提示配置 |

这些是本文用于分析的逻辑层，实际系统不必全部实现。一次答案缓存命中可能绕过检索和生成，因此不能假定下游原有的权限检查还会发生。

KV Cache 则保存推理过程中的状态，不是按业务问题查找最终答案的这类应用缓存。不要把不同缓存的命中率混在一起报告。

## 2 按需加载与失效

[Microsoft 的 Cache-Aside 模式](https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside)描述了先查缓存、未命中再从数据源读取并填充的流程，也要求在数据更新后处理缓存失效。

把它用于 RAG 时，我会先定义允许复用的条件，再决定是否返回命中内容。失效后重新生成的路径也必须遵守权限与证据检查，不能因为是缓存未命中就获得额外权限。

业务数据源和缓存是不同状态，更新流程不能仅依赖“过一会儿自然就好了”。对于权限撤销，允许的生效延迟尤其需要明确。

## 3 缓存键不是只有问题文本

教学实现的键包含租户、用户、原始查询、语料版本、流水线版本和任务范围版本。

流水线版本概括模型、提示、检索参数和输出规则的配置快照；任务范围版本描述本次允许使用的资源范围。两者由可信应用提供，不能由模型自称版本未变。

本例是无对话历史的单轮问答。真实多轮系统还要包含影响答案的历史、语言、时间条件、工具状态等输入，或直接禁用这类答案复用。

逐用户缓存牺牲共享命中率，但便于说明隔离。共享缓存必须证明权限与上下文等价，不能仅因为两个人属于同一租户就共享答案。把查询变成哈希也不等于加密或匿名化。

## 4 TTL 不是撤权机制

[Redis EXPIRE](https://redis.io/docs/latest/commands/expire/)提供键的超时设置。它解决的是时间上的过期，不会理解业务文档的访问控制。

假设答案有效期为 60 秒，用户在第 10 秒被撤权；如果只检查 TTL，剩下 50 秒仍可能返回不该展示的答案。因此本例在每次读取时再次检查所有依赖文档的当前权限。

TTL 也不等于答案的最大陈旧时间：缓存可能来自已经落后的索引，或者命中时不断延长过期时间。本例采用固定到期时间，命中不续期。

## 5 保存完整依赖与版本

缓存条目保存答案、到期时间，以及生成时依赖的文档 ID 和版本。依赖应覆盖实际参与生成的材料，不只是最终显示的引用列表。

只检查旧依赖仍有漏洞：新加入的文档可能推翻旧结论，却不在旧依赖中。教学实现因此使用全语料版本，一旦已发布语料发生增删改就切换版本。

这是偏保守的设计，更新一份无关文档也会导致旧键不再命中。更精细的依赖追踪需要证明能够捕获新增候选和排序变化，不能仅靠已有引用列表替代。

语料版本应对应一致的已发布索引快照。不能把新版本号与仍然旧的索引内容混合使用。

## 6 最小答案缓存

本例假定缓存及条目写入者可信，依赖由生成流水线完整记录，身份和范围来自应用会话。函数只演示读取验收，不自动证明答案有依据。

```python
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Key:
    tenant: str
    user: str
    query: str
    corpus: str
    pipeline: str
    scope: str


@dataclass(frozen=True)
class Document:
    tenant: str
    version: str
    readers: frozenset[str]


@dataclass(frozen=True)
class Entry:
    answer: str
    dependencies: tuple[tuple[str, str], ...]
    expires_at: int


def get_answer(cache, key, documents, allowed_docs, now):
    entry = cache.get(key)
    if entry is None:
        return None, "miss"
    if now >= entry.expires_at:
        return None, "expired"
    # Empty evidence is deliberately not cacheable in this example.
    if not entry.dependencies:
        return None, "invalid"
    for doc_id, version in entry.dependencies:
        doc = documents.get(doc_id)
        if (doc is None or doc_id not in allowed_docs
                or doc.tenant != key.tenant or key.user not in doc.readers):
            return None, "denied"
        if doc.version != version:
            return None, "stale"
    return entry.answer, "hit"
```

这里返回详细原因用于本地测试；外部接口不应通过不同错误暴露私有文档的存在。失效条目没有从字典物理删除，生产实现还需要容量限制、清理和敏感数据保留策略。

now 使用测试中的整数时刻。单进程计时可采用单调时钟；跨进程持久化需要统一的时间与到期语义，不能直接共享某个进程的单调时钟数值。

## 7 合成回归测试

以下代码接在上一段之后执行，模拟语料更新、流水线变化、主体隔离、TTL 边界以及访问撤销。

```python
key = Key("team-a", "alice", "普通设备退货期限？", "c1", "p1", "s1")
docs = {"policy": Document("team-a", "v1", frozenset({"alice"}))}
scope = frozenset({"policy"})
entry = Entry("普通设备支持30天内退货。", (("policy", "v1"),), 60)
cache = {key: entry}
checks = []


def expect(expected, *, k=key, d=None, allowed=scope, now=10, c=None):
    result, reason = get_answer(
        cache if c is None else c, k, docs if d is None else d, allowed, now
    )
    assert reason == expected, (expected, reason)
    assert (result is not None) == (expected == "hit")
    checks.append(reason)


expect("hit")
expect("hit", now=59)
expect("expired", now=60)
expect("expired", now=61)
expect("miss", k=replace(key, user="bob"))
expect("miss", k=replace(key, tenant="team-b"))
expect("miss", k=replace(key, corpus="c2"))
expect("miss", k=replace(key, pipeline="p2"))
expect("miss", k=replace(key, scope="s2"))
expect("miss", k=replace(key, query="定制设备退货期限？"))
expect("denied", d={"policy": replace(docs["policy"], readers=frozenset())})
expect("denied", d={})
expect("denied", allowed=frozenset())
expect("denied", d={"policy": replace(docs["policy"], tenant="team-b")})
expect("stale", d={"policy": replace(docs["policy"], version="v2")})
expect("invalid", c={key: replace(entry, dependencies=())})

# A new relevant document changes the published corpus generation.
docs_v2 = dict(docs, supplement=Document(
    "team-a", "v1", frozenset({"alice"})))
expect("miss", k=replace(key, corpus="c2"), d=docs_v2)

# The entry is still physically present; revocation blocks its reuse.
assert key in cache
assert cache[key].expires_at == 60
assert len(checks) == 17
print(f"checks={len(checks)}, hits={checks.count('hit')}")
print("TTL boundary, identity isolation and version changes checked")
print("revoked access denied while the cached entry still exists")
print("All cache invalidation checks passed.")
```

测试中改变用户导致未命中，并不等于未命中之后允许该用户访问原文。实际重新检索路径必须独立授权。

新增文档用例显式切换语料版本，检验的是版本协议如何生效，不是代码自动发现了新文档。如果发布方忘记更新版本号，该机制就无法可靠发现变化。

## 8 并发与旧结果回填

生产中会出现这样的顺序：请求 A 读取旧语料；语料切换；请求 B 生成新答案；A 较晚完成并尝试写入旧答案。

我的设计要求 A 只能写入它实际使用的旧版本键，不能在完成时取“最新版本号”给旧结果贴新标签。读路径也必须选择正确的已发布版本，不能为了提高命中率回退到旧键。

权限检查与返回之间同样可能发生撤权。需要根据系统要求定义一致性边界，通过事务、版本化授权或返回前复核等机制控制竞态。本篇单线程样例没有实现这些协议，不能承诺瞬时全局撤权。

## 9 不同失效不能一概降级

缓存服务不可用时，可以回源，但仍须通过正常授权与生成流程。权限服务不可用时，不应默认使用过去的允许结果展示敏感答案。

旧答案短暂可用的降级策略，需要明确哪些数据允许陈旧以及最长时间；不能将它默认应用于撤权、删除请求或已确认错误的答案。

“无答案”缓存也会受新增语料影响。本例不缓存空依赖答案；若实现负结果缓存，应绑定查询范围和语料版本，并限制有效期。

语义缓存还增加了“相似问题是否可复用”的判断。普通设备与定制设备、允许与不允许可能向量相近；即使语义阈值通过，版本和权限检查也不能省略。

## 10 评价收益与风险

我会分别记录原始键命中率、通过复核的可用命中率，以及因过期、版本、权限和范围变化被拒绝的次数。

比较缓存开启和关闭时的端到端延迟、后端负载、答案正确性与陈旧答案比例。统计缓存本身和权限复核成本；不要仅用字典查询耗时代表真实请求性能。

安全回归至少覆盖跨用户、跨租户、撤权、删除、语料新增、配置变化和旧请求回填。高命中率不能抵消一次越权返回。

## 11 今日练习结果与边界

已在本地依次执行两个 Python 代码块，17 项检查全部通过。实际输出：

```text
checks=17, hits=2
TTL boundary, identity isolation and version changes checked
revoked access denied while the cached entry still exists
All cache invalidation checks passed.
```

UTF-8、代码围栏、本地链接与 README 索引检查通过；未进行 GitHub 页面视觉渲染验收。

本例没有 Redis、模型推理、并发请求、缓存淘汰或真实权限服务，不构成性能报告或分布式一致性证明。缓存键和版本字段是教学协议，不是所有 RAG 系统的固定标准。

## 参考资料

- [Microsoft Cache-Aside pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside)。
- [Redis EXPIRE](https://redis.io/docs/latest/commands/expire/)。
- [前篇 RAG 提示注入与信任边界](2026-10-02-rag-prompt-injection-trust-boundaries.md)。
- [RAG 端到端评估与回归测试](../09/2026-09-30-rag-evaluation-regression-tests.md)。

## 今日总结

1. 缓存命中不代表结果可用，复用前还需检查身份、版本和权限。
2. TTL 控制时间边界，不能替代撤权和内容失效。
3. 原文依赖与语料版本分别处理已有材料变化和新增材料影响。
4. 旧请求必须绑定旧快照，不能给旧答案贴最新版本标签。
5. 衡量缓存时，要同时看可用命中率、端到端成本和错误复用。

## 下次衔接建议

继续学习 RAG 的增量索引与版本发布，串起文档变更、分块、嵌入更新、索引切换与失败回滚，验证在线请求始终读取一致快照。
