# 2026年10月9日 最小离线 RAG 检索与证据实验

## 今日目标

接着昨天的 [统一验收复盘](2026-10-08-rag-end-to-end-acceptance-review.md)，今天建立一个固定小语料实验，运行检索、权限过滤、证据召回评估和引用原文检查。

核心结论是先验证进入上下文的证据，再评价生成答案。只命中一条相关材料，可能仍然缺少决定结论的例外。

实验采用人工关键词标签和确定性集合交集排序，不包含生成模型。结果描述的是这个小语料的检索行为，不是模型问答准确率。

## 1 固定语料与问题集

语料包含一般退货规则、定制设备例外、保修期限和一份用户无权读取的内部资料。问题覆盖单证据、双证据和无答案场景。

原文、来源版本、关键词标签和读者权限一起保存。问题集另外保存所需证据 ID，检索函数不读取这些答案标注。

关键词由人工指定，相当于一个受控的词项表示。它省去了真实中文分词过程，也使匹配更容易；未来换成实际分词器后必须重新评估。

无答案仅表示当前固定语料中没有标注答案。不能从检索为空推出整个世界不存在答案。

## 2 基线检索协议

先过滤无权读取的片段，再计算查询词与片段词的交集大小。只返回分数大于零的结果，按分数降序、片段 ID 升序排序，保证并列结果可重复。

没有使用 BM25、向量距离或训练模型。集合交集不考虑词频、逆文档频率和语义近义词，适合检查流程，不能代表生产检索效果。

权限过滤在评分前执行。真实系统还需要在原文读取和最终返回时复核当前权限，避免检索之后撤权导致泄露。

## 3 单条命中与完整证据覆盖

[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html)将召回率定义为相关项目中被检索到的比例。

本实验额外检查完整证据覆盖：对有答案问题，全部必需证据是否都进入 Top-k。定制设备问题需要一般规则和例外两段，仅返回例外算部分召回，不算完整覆盖。

无答案题没有必需证据，其证据召回率未定义。代码只对有答案题汇总完整覆盖率，避免将空集合自动算作一次成功召回。

## 4 最小检索实现

每个片段使用完整文本作为来源区间，后续可以扩展为文档内的分块范围。引用检查验证来源存在、版本一致和引文逐字相同。

```python
from dataclasses import dataclass, replace
from fractions import Fraction


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    version: str
    text: str
    terms: frozenset[str]
    readers: frozenset[str]


@dataclass(frozen=True)
class Case:
    id: str
    terms: frozenset[str]
    required: frozenset[str]


def retrieve(chunks, terms, user, k):
    if type(k) is not int or k < 1:
        raise ValueError("positive integer k required")
    ranked = []
    seen = set()
    for chunk in chunks:
        if chunk.id in seen:
            raise ValueError("duplicate chunk id")
        seen.add(chunk.id)
        if user not in chunk.readers:
            continue
        score = len(terms & chunk.terms)
        if score:
            ranked.append((score, chunk.id, chunk))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [chunk for _, _, chunk in ranked[:k]]


def citation_ok(chunk, sources):
    return sources.get((chunk.source, chunk.version)) == chunk.text


def evaluate(cases, chunks, user, k):
    rows = {}
    for case in cases:
        if case.id in rows:
            raise ValueError("duplicate case id")
        hits = retrieve(chunks, case.terms, user, k)
        ids = frozenset(c.id for c in hits)
        recall = (Fraction(len(ids & case.required), len(case.required))
                  if case.required else None)
        complete = case.required <= ids if case.required else None
        rows[case.id] = {"hits": tuple(c.id for c in hits),
                         "recall": recall, "complete": complete}
    answerable = [r for r in rows.values() if r["complete"] is not None]
    coverage = (Fraction(sum(r["complete"] for r in answerable), len(answerable))
                if answerable else None)
    return rows, coverage
```

required 是测试标注，不是线上决策依据。evaluate 可以用它测量覆盖，实际系统不能通过读取测试答案来决定证据是否充分。

citation_ok 不检查语义支持或文档真实性。即使引文逐字正确，也可能与当前问题无关；答案支持检查仍需独立标注。

## 5 构造语料并比较 Top-k

以下代码接在上一段之后执行。原文为合成政策，不对应真实商家的退货承诺。

```python
alice = frozenset({"alice"})
chunks = [
    Chunk("rule", "policy", "v1", "普通设备支持30天内退货。",
          frozenset({"普通设备", "退货"}), alice),
    Chunk("exception", "custom", "v1", "定制设备不适用上述退货规则。",
          frozenset({"定制设备", "退货"}), alice),
    Chunk("warranty", "warranty", "v1", "设备保修期限为12个月。",
          frozenset({"保修", "期限"}), alice),
    Chunk("private", "internal", "v1", "内部退货统计。",
          frozenset({"普通设备", "定制设备", "退货"}), frozenset({"bob"})),
]
sources = {(c.source, c.version): c.text for c in chunks}
cases = [
    Case("ordinary", frozenset({"普通设备", "退货"}), frozenset({"rule"})),
    Case("custom", frozenset({"定制设备", "退货"}),
         frozenset({"rule", "exception"})),
    Case("warranty", frozenset({"保修", "期限"}), frozenset({"warranty"})),
    Case("delivery", frozenset({"配送", "运费"}), frozenset()),
]
top1, coverage1 = evaluate(cases, chunks, "alice", 1)
top2, coverage2 = evaluate(cases, chunks, "alice", 2)
assert coverage1 == Fraction(2, 3)
assert coverage2 == 1
assert top1["custom"]["hits"] == ("exception",)
assert top1["custom"]["recall"] == Fraction(1, 2)
assert top1["custom"]["complete"] is False
assert top2["custom"]["hits"] == ("exception", "rule")
assert top2["delivery"] == {"hits": (), "recall": None, "complete": None}
assert evaluate(cases, chunks[::-1], "alice", 2) == (top2, coverage2)
assert all("private" not in row["hits"] for row in top2.values())
assert all(citation_ok(c, sources) for c in chunks)
assert not citation_ok(replace(chunks[0], text="所有设备均可退货。"), sources)
assert not citation_ok(replace(chunks[0], version="v2"), sources)
assert retrieve(chunks, frozenset({"退货"}), "unknown", 2) == []

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


reject(lambda: retrieve(chunks, cases[0].terms, "alice", 0))
reject(lambda: retrieve(chunks + [chunks[0]], cases[0].terms, "alice", 2))
reject(lambda: evaluate(cases + [cases[0]], chunks, "alice", 2))

print(f"complete evidence coverage: top1={coverage1}, top2={coverage2}")
print(f"custom top1={top1['custom']['hits']}, top2={top2['custom']['hits']}")
print("unanswerable evidence recall is undefined")
print("unauthorized chunks excluded; altered citations rejected")
print("All offline retrieval checks passed.")
```

Top-2 在这个样例中补齐了定制设备问题的证据，但普通设备问题也会带入额外例外片段。扩大 k 会改变上下文成本和干扰，不能据此固定“所有任务都用 Top-2”。

无答案题恰好返回空结果是词项不匹配的结果。真实语料可能有大量相关但不能回答问题的材料，非空检索不应自动触发回答。

## 6 证据输出与生成的接口

后续生成器只接收获准的检索结果，以及每个片段的来源、版本和原文范围。生成器输出的引用 ID 必须属于实际输入集合，不能随意引用语料中没有送入上下文的内容。

先保留原文抽取基线，等引用与证据检查稳定后再接入压缩。否则检索遗漏、压缩失真和生成错误会混在一起，难以归因。

需要同时记录候选集合和最终上下文集合；证据被检索到之后仍可能因为预算限制没有进入模型。

## 7 增加真实难度的顺序

先扩展问题集：近义词、拼写变化、多个产品、否定条件、版本冲突，以及有相关文档但缺少关键事实的问题。

再替换词项表示和排序，比较真实分词、BM25 或向量检索，并固定相同语料、权限和问题标注。每次只改变一层，保留这个确定性基线用于诊断。

若词项排序无法召回所需证据，生成模型也不应被当作补救未知事实的捷径。可以引入经过验证的查询改写或多跳流程，并记录它们增加的成本和失败。

## 8 实验记录与复现

保存语料、权限快照、问题集、词项处理规则、排序规则和 k。测试结果记录逐题命中 ID、召回、完整覆盖及失败原因，不能只保存总分。

人工标注与语料版本一起变化。更新政策后，要重新检查问题的必需证据与答案条件，避免拿旧标注评价新事实。

小语料的满覆盖适合验证流程，不足以证明泛化。扩展开发集后，还需要独立确认集和真实模型回答评估。

## 9 今日练习结果

两个 Python 代码块已在本地依次执行，全部断言通过。实际输出：

```text
complete evidence coverage: top1=2/3, top2=1
custom top1=('exception',), top2=('exception', 'rule')
unanswerable evidence recall is undefined
unauthorized chunks excluded; altered citations rejected
All offline retrieval checks passed.
```

UTF-8 编码、代码围栏、相对链接和 README 索引检查通过；尚未验证 GitHub 页面的实际渲染。

练习运行的是检索与证据检查链路；完整 RAG 还需要接入生成、答案正确性和引用支持评估，并结合预算与权限回归检查。

## 参考资料

- [Introduction to Information Retrieval 检索集合评估](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html)。
- [前篇 RAG 工程端到端复盘与统一验收](2026-10-08-rag-end-to-end-acceptance-review.md)。
- [检索排序与评估](../09/2026-09-28-retrieval-ranking-bm25-rrf-evaluation.md)。
- [长上下文与上下文压缩](2026-10-01-long-context-evidence-compression.md)。

## 今日总结

1. 固定语料和确定性排序让逐题结果能够复现。
2. 相关片段命中与完整证据覆盖是不同的检查。
3. 无答案题的证据召回分母为空，应明确未定义。
4. 权限过滤和引用原文校验可以先独立验证。
5. 检索实验通过后，再逐层接入真实分词、排序与生成。

## 下次衔接建议

将人工词项基线扩展为真实中文文本检索，比较分词、字符 n-gram 与 BM25，并用近义词和否定条件问题分析漏召回及误召回。
