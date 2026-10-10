# 2026年10月10日 中文文本检索与 BM25 对照实验

## 今日目标

接着昨天的 [离线检索实验](2026-10-09-offline-rag-retrieval-evidence-experiment.md)，今天去掉人工片段标签，直接从中文原文生成词项，比较词典分词与字符二元组，再比较交集排序与 BM25。

核心问题是区分表示错误和排序错误：没有共同词项时，改变 BM25 参数不能补出同义词；有共同词项时，高分也不代表文档支持问题中的肯定或否定结论。

实验使用合成政策与 Python 标准库，没有生成模型。词典分词是教学用的最长匹配实现，不是对成熟中文分词器的性能评价。

## 1 查询与文档共用分析器

分词将字符序列转换为词项。查询与文档必须采用一致的处理流程，否则相同文字也可能无法匹配。[Introduction to Information Retrieval 的分词章节](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)强调这种一致性。

本练习先做 Unicode NFKC 规范化，再转小写，最后按非字母数字字符分段。二元组只在每一段内部生成，避免跨标点拼接。它仍会跨越中文词边界，例如“设备支持”包含“备支”，因此不能将每个二元组解释为独立语义。

最长匹配从当前位置选择词典中最长的词，未登录字符退回单字。词典是固定实验输入，不能根据测试答案临时加入词。真实检索要记录分词器版本、词典、规范化规则与停用词配置。

数字、单位、英文型号和否定词都可能影响答案。本练习不删除停用词；随意删除“不”可能让允许与禁止的文本更难区分。

## 2 把表示与排序分开比较

| 配置 | 词项表示 | 排序 | 对照目的 |
| --- | --- | --- | --- |
| word overlap | 最长匹配词典分词 | 不同匹配词的数量 | 文本检索基线 |
| word BM25 | 同一词典分词 | BM25 | 只改变排序 |
| bigram BM25 | 字符二元组 | 同一 BM25 | 只改变表示 |

每次对照保持原文、读者、问题、证据标注和 k 相同。不同分析器的文档长度以各自词项数量计，因此 BM25 分数不宜跨配置直接比较。

## 3 BM25 的词频与长度处理

BM25 为匹配词项累加贡献，考虑文档词频、逆文档频率和文档长度。词频收益逐渐饱和；长度归一化由 b 控制。[BM25 教材章节](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)给出了这些组成部分。

本实现采用该章节公式 86 的 IDF，即 log(N / df)，不使用可能产生负数的概率 IDF，也不加入查询词频权重。查询中的重复词仅计算一次。k1 固定为 1.2，b 固定为 0.75，作为练习配置而非调优结论。

N 是当前读者可见的片段数，df 是包含该词的可见片段数；文档长度和平均长度也从可见集合计算。这避免本练习通过统计量依赖私有片段，但真实系统采用共享还是按权限分区的统计量，需要单独设计和评估。

这种 IDF 下，出现在全部可见片段中的词贡献为零。代码只返回正分结果；空语料直接返回空列表。

## 4 最小文本检索实现

以下两个代码块依次执行。权限先过滤，之后才分词和计算统计量；并列结果按片段 ID 排序。这里只验证检索层的权限，实际读取原文和输出仍应复核权限。

```python
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction


LEXICON = tuple(sorted({"普通设备", "定制设备", "设备", "支持", "退货",
                        "保修", "期限", "个月", "适用", "规则"},
                       key=lambda x: (-len(x), x)))


def spans(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[^\W_]+", text)


def words(text):
    result = []
    for part in spans(text):
        i = 0
        while i < len(part):
            token = next((w for w in LEXICON if part.startswith(w, i)), part[i])
            result.append(token)
            i += len(token)
    return result


def bigrams(text):
    return [part[i:i + 2] for part in spans(text)
            for i in range(len(part) - 1)]


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    readers: frozenset[str]


def search(docs, query, user, analyzer, method, k=2):
    if type(k) is not int or k < 1:
        raise ValueError("positive integer k required")
    if method not in {"overlap", "bm25"}:
        raise ValueError("unknown method")
    if len({d.id for d in docs}) != len(docs):
        raise ValueError("duplicate document id")
    visible = sorted((d for d in docs if user in d.readers), key=lambda d: d.id)
    if not visible:
        return ()
    counts = [Counter(analyzer(d.text)) for d in visible]
    lengths = [sum(c.values()) for c in counts]
    avg = sum(lengths) / len(visible)
    df = Counter(t for c in counts for t in c)
    terms = set(analyzer(query))
    ranked = []
    for d, c, length in zip(visible, counts, lengths):
        if method == "overlap":
            score = len(terms & c.keys())
        else:
            score = 0.0
            for term in sorted(terms & c.keys()):
                tf = c[term]
                norm = 1.2 * (1 - 0.75 + 0.75 * length / avg)
                score += math.log(len(visible) / df[term]) * tf * 2.2 / (tf + norm)
        if score > 0:
            ranked.append((score, d.id))
    return tuple(id_ for _, id_ in sorted(ranked, key=lambda x: (-x[0], x[1]))[:k])
```

二元组分析器对单字符段产生空词项。这是明确的边界限制，不是所有字符检索都必须如此；可以另设单字回退，但那应作为新的配置对照。

## 5 固定问题与边界测试

普通退货问题需要一般规则；定制退货问题同时需要一般规则和例外；保修问题需要期限片段。“质保多久”作为近义表达，其标注仍是保修片段，但分析器没有同义词扩展。

“配送运费”是无答案题。它不参与有答案题的完整覆盖率分母，另外记录是否错误返回片段。否定问题只用于检查检索边界，不把排序相同当作答案相同。

```python
alice = frozenset({"alice"})
docs = [
    Doc("rule", "普通设备支持30天内退货。", alice),
    Doc("exception", "定制设备不适用上述退货规则。", alice),
    Doc("warranty", "设备保修期限为12个月。", alice),
    Doc("private", "普通设备定制设备退货保修期限内部统计。", frozenset({"bob"})),
]
cases = [
    ("ordinary", "普通设备退货", {"rule"}),
    ("custom", "定制设备退货", {"rule", "exception"}),
    ("warranty", "保修期限", {"warranty"}),
    ("synonym", "质保多久", {"warranty"}),
    ("missing", "配送运费", set()),
]
configs = [("word overlap", words, "overlap"),
           ("word BM25", words, "bm25"),
           ("bigram BM25", bigrams, "bm25")]
assert spans("ＡＢＣ，设备") == ["abc", "设备"]
assert bigrams("退货，规则") == ["退货", "规则"]
assert words("定制设备退货") == ["定制设备", "退货"]
assert bigrams("退") == []
for name, analyzer, method in configs:
    rows = {}
    complete = []
    for case, query, required in cases:
        hits = search(docs, query, "alice", analyzer, method)
        rows[case] = hits
        assert "private" not in hits
        assert search(docs[::-1], query, "alice", analyzer, method) == hits
        assert search(docs, query, "unknown", analyzer, method) == ()
        if required:
            complete.append(required <= set(hits))
    assert rows["synonym"] == ()
    assert rows["missing"] == ()
    # 否定词没有语料匹配，不改变这两条查询的排序。
    assert search(docs, "定制设备不退货", "alice", analyzer, method) == rows["custom"]
    print(f"{name}: coverage@2={Fraction(sum(complete), len(complete))}")
    print(f"  custom={rows['custom']}, synonym={rows['synonym']}, missing={rows['missing']}")
assert search([], "退货", "alice", words, "bm25") == ()
assert search([Doc("empty", "。", alice)], "退货", "alice", words, "bm25") == ()
for bad in (0, True):
    try:
        search(docs, "退货", "alice", words, "bm25", bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid k accepted")
print("All text retrieval checks passed.")
```

## 6 今日练习结果

两个 Python 代码块已依次执行，全部断言通过。实际输出：

```text
word overlap: coverage@2=3/4
  custom=('exception', 'rule'), synonym=(), missing=()
word BM25: coverage@2=3/4
  custom=('exception', 'rule'), synonym=(), missing=()
bigram BM25: coverage@2=3/4
  custom=('exception', 'rule'), synonym=(), missing=()
All text retrieval checks passed.
```

三种配置均补齐定制问题的两条证据，但均漏掉近义表达。当前样例不能区分它们的覆盖表现，应扩充词频、长短片段和词边界相关案例，再分析逐题排名。

覆盖率按四道有答案题计算。普通、定制和保修是直接词面问题，近义题专门暴露未覆盖的词项；不能把这个小集合的结果解释为实际业务准确率。

## 7 从失败结果选择下一步

若同一表示下 BM25 与交集排序均漏召回，先检查查询与文档的词项是否相交，再考虑排序参数。本例“质保多久”与“保修期限”缺少共同词项，增加 k 也不会产生正分候选。

字符二元组可以提供局部匹配，但不是同义词理解。它也没有判断“不适用”与“支持”的逻辑关系。检索负责提供候选证据，回答仍需保留原文中的范围、否定和例外。

后续引入同义词扩展时，保留原查询并记录新增词项。分别统计直接问题与近义问题的变化，检查新召回是否带来误召回；使用独立确认集，避免围绕少数展示题反复调参。

## 参考资料

- [Introduction to Information Retrieval 分词](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)。
- [Introduction to Information Retrieval BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html)。
- [前篇 最小离线 RAG 检索与证据实验](2026-10-09-offline-rag-retrieval-evidence-experiment.md)。
- [检索排序与评估](../09/2026-09-28-retrieval-ranking-bm25-rrf-evaluation.md)。

## 今日总结

1. 查询与原文共享分析器，才有一致的词面匹配。
2. 表示与排序分开对照，避免将分词改善误归因于 BM25。
3. BM25 处理词频与长度，不自动补齐同义词或解释否定。
4. 权限、统计量范围和并列排序规则都属于复现配置。
5. 完整证据覆盖与答案正确性仍需分别评价。

## 下次衔接建议

增加受控同义词扩展，比较原查询与扩展查询的召回、误召回和证据完整性，并设计独立确认集检查改进是否泛化。
