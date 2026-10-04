# 2026年10月4日 RAG 增量索引与版本发布

## 今日目标

昨天的 [RAG 缓存与失效策略](2026-10-03-rag-cache-invalidation.md)要求缓存绑定语料版本。今天继续追问：这个版本怎样构建、验收并发布，才能避免线上请求读到一半新、一半旧的证据？

核心结论是把构建和发布分开：先生成候选快照，完成检查后再切换入口；请求开始时固定版本，后续检索、原文读取与缓存写入都使用同一版本。

本篇用不可变 Python 对象模拟分块索引和版本入口，不调用嵌入模型，不连接搜索服务，也不声称实现了分布式原子提交。

## 1 增量计算与完整发布视图

增量索引的目标是只重算发生变化的部分；它不意味着可以把未完成的中间结果直接暴露给查询。

教学设计中，一个发布版本仍表示完整的可查询视图。实现可以复用未变对象、分段文件或已有向量，不必每次物理复制所有数据。

至少区分三种变化：新增文档、内容更新和删除。更新后分块数可能变少，如果只覆盖相同序号而不删除多余旧块，检索就会继续命中已经不存在的内容。

分块策略、嵌入模型或向量维度变化则属于流水线配置变化，不能默认沿用旧向量。即使维度相同，不同嵌入空间也不一定兼容。

## 2 给发布版本保存清单

我会给每个候选版本记录以下字段：

| 信息 | 用途 |
| --- | --- |
| 文档 ID 与内容版本 | 判断来源是否对应 |
| 分块规则与嵌入配置 | 判断产物是否兼容 |
| 源数据快照或变更水位 | 界定本次包含哪些更新 |
| 文档及分块数量与内容指纹 | 发现遗漏、重复和意外变化 |
| 验收结果与发布编号 | 判断是否可以切换入口 |

这些是本文的设计建议，不是某个数据库统一规定的字段。数量一致仍可能内容错误，指纹一致也不代表访问权限或答案质量通过。

重复处理同一变更应得到相同逻辑结果。若接入乱序事件，要比较可信源版本并保留删除标记，避免晚到的旧更新把已删除文档重新创建。

## 3 先构建再切换

[Elastic 的别名文档](https://www.elastic.co/docs/manage-data/data-store/aliases)说明可以通过别名更换查询目标，并在一次操作中执行多个动作；同时也提醒检查动作失败及部分成功情况。

这启发了本文的“候选快照加活动入口”模型，但别名切换并不自动保证所有外部文档存储、权限系统和缓存一起事务提交。需要应用自己的版本协议。

[Elastic 的重建索引接口](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-reindex)还明确指出，目标索引的映射、分片等设置需事先配置，不能假定复制文档就复制了全部配置。

本篇不执行这些接口，只借官方说明区分复制数据、配置目标和发布入口三项工作。

## 4 在线写入与构建水位

若构建期间源数据仍在更新，需要选定快照水位，并追赶之后的变更至明确的发布边界。仅扫描一遍再切换，可能丢掉扫描期间的新增或删除。

可根据系统能力选择一致快照加变更日志、短暂停写，或带校验的双写。双写中的一侧失败同样需要补偿和对账，并非天然一致。

下面示例的源数据在单次构建期间不变，传入的修改和删除是一个已排序的批次；不模拟变更日志、乱序处理或断点恢复。

## 5 最小快照构建器

代码把每份文档整体重新分块，未变文档复用原对象。更新文档时替换整个对象，因此旧分块不会残留。候选构建异常时，不修改旧快照。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    version: str
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class Snapshot:
    generation: str
    documents: tuple[Document, ...]


def build(base, generation, upserts, deletes=(), fail_on=None):
    if not generation or generation == base.generation:
        raise ValueError("new generation required")
    deleted = set(deletes)
    if deleted.intersection(upserts):
        raise ValueError("conflicting update and delete")
    result = {d.id: d for d in base.documents}
    for doc_id in deleted:
        result.pop(doc_id, None)
    for doc_id, (version, text) in sorted(upserts.items()):
        if doc_id == fail_on:
            raise RuntimeError("simulated build failure")
        chunks = tuple(line for line in text.splitlines() if line.strip())
        if not doc_id or not version or not chunks:
            raise ValueError("invalid document")
        result[doc_id] = Document(doc_id, version, chunks)
    return Snapshot(generation, tuple(result[k] for k in sorted(result)))


class Publisher:
    def __init__(self, initial):
        self.snapshots = {initial.generation: initial}
        self.pointer = (0, initial.generation)

    def stage(self, candidate, expected):
        # expected is an independently supplied source manifest.
        actual = {d.id: d.version for d in candidate.documents}
        if len(actual) != len(candidate.documents) or actual != expected:
            raise ValueError("manifest mismatch")
        if any(not d.chunks for d in candidate.documents):
            raise ValueError("empty chunks")
        if candidate.generation in self.snapshots:
            raise ValueError("generation already exists")
        self.snapshots[candidate.generation] = candidate

    def activate(self, generation, expected_pointer):
        if self.pointer != expected_pointer:
            raise ValueError("stale publisher")
        if generation not in self.snapshots:
            raise ValueError("snapshot not staged")
        self.pointer = (self.pointer[0] + 1, generation)

    def pin(self):
        return self.snapshots[self.pointer[1]]
```

清单校验仅检查文档集合、版本与非空分块，没有验证向量完整性、分块内容指纹或检索质量。真实发布前还要增加这些检查，不能将 stage 成功当成全面验收。

Publisher 是单线程模型。它用比较旧入口再修改新入口表达乐观并发控制的意图；多进程部署必须使用真正原子的条件更新，不能直接照搬两条 Python 语句。

## 6 发布与回滚练习

以下代码接在上一段之后执行。用例包含缩短文档、删除文档、构建失败、清单遗漏、重复版本和过期发布者。

```python
initial = Snapshot("g1", (
    Document("policy", "v1", ("一般规则", "旧附录")),
    Document("retired", "v1", ("待删除文档",)),
))
publisher = Publisher(initial)
request_a = publisher.pin()
old_pointer = publisher.pointer


def reject(error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError("expected rejection")


reject(RuntimeError, lambda: build(
    initial, "failed", {"policy": ("v2", "新规则")}, fail_on="policy"))
assert publisher.pin() is initial

candidate = build(
    initial, "g2", {"policy": ("v2", "新规则")},
    deletes=("retired",),
)
assert candidate.documents == (Document("policy", "v2", ("新规则",)),)
assert initial.documents[0].chunks == ("一般规则", "旧附录")
assert build(initial, "g2", {"policy": ("v2", "新规则")},
             deletes=("retired",)) == candidate

reject(ValueError, lambda: publisher.stage(candidate, {
    "policy": "v2", "missing": "v1"}))
assert publisher.pointer == old_pointer
reject(ValueError, lambda: publisher.activate("g2", old_pointer))
publisher.stage(candidate, {"policy": "v2"})
reject(ValueError, lambda: publisher.stage(candidate, {"policy": "v2"}))
publisher.activate("g2", old_pointer)
request_b = publisher.pin()
assert request_a.generation == "g1"
assert request_b.generation == "g2"
assert request_a.documents[0].chunks == ("一般规则", "旧附录")
assert request_b.documents[0].chunks == ("新规则",)

reject(ValueError, lambda: publisher.activate("g1", old_pointer))
publisher.activate("g1", publisher.pointer)
assert publisher.pin() is initial
assert publisher.pointer == (2, "g1")
# Returning to g1 does not revive an old publication token.
reject(ValueError, lambda: publisher.activate("g2", old_pointer))
reject(ValueError, lambda: build(
    initial, "g3", {"policy": ("v3", "规则")}, deletes=("policy",)))

print("g2 documents=1, chunks=1; deleted document absent")
print("pinned requests: A=g1, B=g2")
print(f"rollback pointer={publisher.pointer}")
print("All snapshot publication checks passed.")
```

回滚到 g1 后发布计数仍递增，防止“版本又回到原来的值”让过期发布者误以为中间没有发生切换。代码只模拟这种检查，不证明真实并发条件下的正确性。

## 7 请求固定版本的范围

一个请求开始时解析活动入口，保存实际版本；多跳检索、原文回查和答案缓存写入继续携带这个版本。不要每一跳重新读取活动别名。

固定版本解决的是同一请求内的内容一致性，不意味着它永远能使用历史权限。当前授权与删除限制仍应独立检查，沿用 [信任边界](2026-10-02-rag-prompt-injection-trust-boundaries.md)中的执行前校验。

原文也需要有版本化副本或可验证的对应内容。若只固定向量索引，却从一个持续被覆盖的原文地址取回正文，仍可能混用新旧证据。

## 8 回滚不等于撤销一切

回滚入口可以恢复旧的检索产物，但不能撤销已发送的答案或外部操作。模型、提示和缓存配置是否兼容旧版本也必须检查。

示例中恢复 g1 会重新出现 retired 文档，它只是普通内容发布回退的演示。若删除源于用户撤权、隐私删除或安全事件，不能照搬这个回滚结果；当前强制禁用列表和权限规则必须优先，必要时禁止回滚到该版本。

损坏版本应标记为不可用并清理相关缓存。简单回到旧版本号可能复用旧答案，因此发布修订号、安全失效标记和内容版本是否进入缓存键，要在协议中明确。

## 9 何时回收旧快照

发布后立刻删除旧产物可能破坏尚未完成的旧请求，也让回滚失去依据。

我会根据请求存活期、引用计数或租约，以及允许的回滚窗口安排回收。超时请求如何终止也要定义，不能无限保留所有旧版本。

保留旧产物同样受数据保留与删除要求约束。回收范围不只是向量，还包括原文副本、压缩摘要、答案缓存和备份中的对应数据。

## 10 发布验收与运行记录

发布前检查源清单、变更水位、重复 ID、已删文档残留、分块与向量对应关系、向量维度和检索可见性。对关键问题运行固定回归集，比较逐题修复与退化。

发布后记录实际使用的快照、延迟、空结果、权限拒绝和错误率。记录构建成功与入口切换成功两种状态；网络超时导致切换结果不明时，应读取真实入口再决定是否重试。

本文设计优先保证版本边界清晰，并未证明吞吐最优。全量重建、增量构建与分段复用应根据数据规模、变更频率和一致性要求比较。

## 11 今日练习结果与边界

已在本地依次执行两个 Python 代码块，全部断言通过。实际输出：

```text
g2 documents=1, chunks=1; deleted document absent
pinned requests: A=g1, B=g2
rollback pointer=(2, 'g1')
All snapshot publication checks passed.
```

UTF-8、代码围栏、本地链接与 README 索引检查通过；未进行 GitHub 页面视觉渲染验收。

没有真实嵌入计算、搜索排序、并发读写、集群故障或权限服务。测试验证的是给定不可变对象和单线程发布模型，不是线上零停机保证。

## 参考资料

- [Elastic Aliases](https://www.elastic.co/docs/manage-data/data-store/aliases)。
- [Elastic Reindex documents](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-reindex)。
- [前篇 RAG 缓存与失效策略](2026-10-03-rag-cache-invalidation.md)。
- [RAG 端到端评估与回归测试](../09/2026-09-30-rag-evaluation-regression-tests.md)。

## 今日总结

1. 增量计算可以复用旧产物，发布仍要呈现完整一致视图。
2. 更新文档必须处理多余旧分块，删除也属于索引变更。
3. 构建、验收和切换入口应分别记录与验证。
4. 请求固定内容版本，但权限不能因此冻结。
5. 回滚和旧版本回收必须兼顾缓存、在途请求与强制删除限制。

## 下次衔接建议

继续学习 RAG 可观测性与故障定位，把查询、检索、重排、生成、缓存和索引版本串入同一次请求追踪，设计延迟和质量退化的排查流程。
