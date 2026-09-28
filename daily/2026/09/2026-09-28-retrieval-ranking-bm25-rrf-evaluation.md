# 2026-09-28：检索排序——BM25、向量检索、RRF 与评估

## 今日目标

上一篇 [检索增强生成](2026-09-27-rag-retrieval-evidence-citations.md)区分了检索命中、引用真实和证据支持。今天集中学习：如何产生候选、合并多路排名，以及判断排序是否变好？

完成后应能回答：

1. BM25 与向量检索各依赖什么信号？
2. 为什么不能直接相加不同检索器的原始分数？
3. RRF 如何融合排名，有哪些局限？
4. Recall@k、Hit@k 和 MRR@k 分别反映什么？
5. 如何把检索评估和最终回答评估分开？

本篇只执行人工排名上的融合与评估代码。没有部署 Elasticsearch、计算真实 embedding 或运行重排序模型，不能据此判断任何真实检索器的优劣。

---

## 1. BM25：词频饱和与长度归一化

一个常见 BM25 形式是：

```math
\mathrm{score}(q,d)=\sum_{t\in q}
\mathrm{IDF}(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}
```

其中 f 是词频，|d| 是文档长度，avgdl 是语料平均长度。不同实现还可能对查询重复词、IDF 和字段统计做不同处理，不能仅凭公式名假定分数完全一致。

k1 控制词频饱和，b 控制长度归一化程度。相关解释见 [Elastic 的 BM25 说明](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)。

关键词方法适合精确名称、错误码等场景，但仍依赖分词、词形处理和字段设计。把中文文本简单按空格切分，不是可靠的中文检索基线。

## 2. 向量检索：表示相似，不是事实判断

Bi-encoder 分别编码查询与文档，再用点积或余弦等度量排序。文档向量可以预计算，便于大规模候选召回。

要记录模型版本、查询/文档提示、归一化、最大长度和距离度量。只有在向量按相应方式归一化时，点积与余弦排序才具有对应关系。

近似最近邻还引入索引搜索误差。应区分“embedding 本身没有把证据排前”和“近似搜索漏掉了本应较近的向量”。

语义相近并不保证事实一致：包含否定、例外或不同版本的文档也可能很相似。

## 3. 混合召回先解决候选覆盖

BM25 和向量检索可能互补。可以先分别取候选，按统一 chunk ID 合并，然后融合或重排序。

统一 ID 必须代表相同评估单位。如果一路返回文档、另一路返回段落，需要先明确映射；不能把同一证据的多个副本当作多个独立命中。

权限和版本过滤应在向生成器暴露内容之前完成，并检查过滤后的候选数量。先取很小 top-k 再过滤，可能让候选集不足。

## 4. 为什么不直接相加原始分数

BM25、余弦相似度和 cross-encoder 输出不天然共享尺度。某一路分数范围更大时，未经处理的加权和可能主要由它决定。

可以在开发集研究校准或归一化后的分数融合，也可以使用只依赖名次的方法。选择必须通过固定测试协议验证，而不是选出对当前几题最好看的参数。

## 5. Reciprocal Rank Fusion

今天采用等权 RRF，排名从 1 开始：

```math
\mathrm{RRF}(d)=\sum_{j:d\in L_j}\frac{1}{c+r_j(d)}
```

L_j 是第 j 路候选列表，r_j 是其中的名次，c 为非负平滑常数。没有出现在某一路中的文档，该路贡献为零。

[Elastic 的 RRF 文档](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)给出了排名倒数融合的定义。本文用 c 区别于最终 top-k，避免混淆两个参数。

RRF 不需要对齐原始分数尺度，但丢失了分数间隔信息。多路高度相关时，重复提供同一信号也会增加其影响。候选窗口、常数和并列处理都会影响结果，不存在无条件改善保证。

## 6. 重排序的边界

Cross-encoder 联合处理查询和候选文本，通常比独立向量打分更昂贵，因此常用于较小候选集。参见 [Sentence Transformers：Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)。

若重排序只改变顺序、不引入新文档，它无法找回候选集合之外的证据。截取更小 top-k 后的召回率还可能下降。

要分别记录初始候选数量、重排后的数量、截断长度和模型版本；查询与长候选一起超出输入长度时，关键证据可能已经被裁掉。

## 7. 三个指标，不要混用

对有非空标注相关集合 G_q 的查询 q，取前 k 个结果 R_q：

```math
\mathrm{Recall@k}(q)=\frac{|G_q\cap R_q|}{|G_q|},
\qquad
\mathrm{Hit@k}(q)=\mathbf{1}[G_q\cap R_q\ne\varnothing]
```

RR@k 是前 k 中第一个相关结果名次的倒数；若没有则为零。MRR@k 是各查询 RR@k 的平均。

Recall 衡量相关项覆盖，Hit 只看是否至少命中一个，MRR 关注第一个命中的位置。需要多个证据联合回答的问题，即使 MRR=1，也可能遗漏其他关键证据。

今天按查询等权宏平均。没有相关标注的查询返回单独状态，不擅自算成零分；它们适合另行评估无答案处理，并明确是否确实没有答案还是尚未标注。

## 8. 最小实现：精确 RRF 与排序指标

下面拒绝同一路重复 ID，避免重复计票；不同路出现相同 ID 是正常融合。并列分数用字符串 ID 排序，保证可复现。

```python
from fractions import Fraction as F


def check_ranking(ranking):
    if not all(isinstance(doc, str) and doc for doc in ranking):
        raise ValueError("invalid document id")
    if len(set(ranking)) != len(ranking):
        raise ValueError("duplicate document id")


def rrf(rankings, c=60):
    if type(c) is not int or c < 0:
        raise ValueError("invalid rank constant")
    scores = {}
    for ranking in rankings:
        check_ranking(ranking)
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, F(0)) + F(1, c + rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc)), scores


def metrics(ranking, relevant, k):
    check_ranking(ranking)
    if type(k) is not int or k <= 0:
        raise ValueError("invalid cutoff")
    if type(relevant) is not set:
        raise ValueError("relevant must be a set")
    check_ranking(list(relevant))
    if not relevant:
        return None
    top = ranking[:k]
    hits = len(set(top) & relevant)
    rr = next((F(1, rank) for rank, doc in enumerate(top, 1)
               if doc in relevant), F(0))
    return {"recall": F(hits, len(relevant)),
            "hit": F(int(hits > 0)), "rr": rr}


def macro_average(rows):
    valid = [row for row in rows if row is not None]
    if not valid:
        return {"evaluated": 0, "excluded": len(rows)}
    return {
        "evaluated": len(valid), "excluded": len(rows) - len(valid),
        **{name: sum(row[name] for row in valid) / len(valid)
           for name in ("recall", "hit", "rr")},
    }
```

这里使用有理数只是便于教学断言，不是生产检索性能优化。代码也不检查语料 ID 是否真实存在；实际系统应另外验证索引与标注版本。

## 9. 人工排名与反例测试

以下代码接在上一段之后执行。列表名只代表假设的两路输出，不是实际运行 BM25 或向量模型的结果。

```python
lexical = ["a", "b", "c"]
dense = ["c", "b", "d"]
fused, scores = rrf([lexical, dense])
assert fused == ["c", "b", "a", "d"]
assert scores["c"] == F(1, 63) + F(1, 61)
assert rrf([dense, lexical]) == (fused, scores)
assert rrf([[], lexical])[0] == lexical
assert rrf([]) == ([], {})
assert rrf([["z"], ["a"]])[0] == ["a", "z"]

gold = {"b", "d"}
base = metrics(lexical, gold, 2)
combined = metrics(fused, gold, 2)
assert base == combined == {"recall": F(1, 2), "hit": F(1), "rr": F(1, 2)}

# Fusion can hurt a particular query at a small cutoff.
assert metrics(lexical, {"a"}, 1)["rr"] == 1
assert metrics(fused, {"a"}, 1)["rr"] == 0

rows = [
    metrics(["a", "b"], {"b", "d"}, 2),
    metrics(["c", "a"], {"c"}, 2),
    metrics(["x"], {"z"}, 2),
    metrics(["x"], set(), 2),
]
report = macro_average(rows)
assert report == {
    "evaluated": 3, "excluded": 1,
    "recall": F(1, 2), "hit": F(2, 3), "rr": F(1, 2),
}
assert metrics([], {"a"}, 2)["recall"] == 0
assert metrics(["a"], {"a", "b"}, 10)["recall"] == F(1, 2)
assert macro_average([None]) == {"evaluated": 0, "excluded": 1}

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: rrf([["a", "a"]]))
reject(lambda: rrf([["a"]], c=-1))
reject(lambda: metrics(["a"], {"a"}, 0))
reject(lambda: metrics(["a", "a"], {"a"}, 2))

print("fused:", fused)
print(f"evaluated={report['evaluated']}, excluded={report['excluded']}")
print(f"Recall@2={float(report['recall']):.3f}, "
      f"Hit@2={float(report['hit']):.3f}, MRR@2={float(report['rr']):.3f}")
print("All fusion and retrieval-metric checks passed.")
```

对 gold={"b","d"} 的例子，融合没有改善 top-2 指标；对 gold={"a"}，融合让 top-1 变差。这只是说明融合不是定理式提升，不说明混合检索总体无用。

## 10. 标注与评估单位

如果一个事实出现在同文档的多个重叠 chunk 中，应决定评估的是 chunk、文档还是证据事实覆盖。

把多个重复 chunk 全部标为相关，可能让 Recall 分母变大；反过来只标一个 chunk 又可能把同样有用的其他片段误判为不相关。

记录不完整标注的限制。相关性标签也不代表来源权威、时效正确或足以支持完整答案，需要沿用昨天的证据核验。

## 11. 对照实验怎么安排

固定语料快照、权限过滤、切分方式和测试问题，比较：

1. 词法检索基线。
2. 向量检索基线。
3. 两路融合。
4. 融合后重排序。

先在开发集选择候选窗口、常数和模型，再在未参与选择的测试集评估。按精确名称、同义改写、长查询和多证据问题切片，避免只展示总体平均。

同时记录耗时与实际生成质量。检索指标提高不自动等于答案质量提高；新加入的材料可能冗余、冲突或超过上下文预算。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
fused: ['c', 'b', 'a', 'd']
evaluated=3, excluded=1
Recall@2=0.500, Hit@2=0.667, MRR@2=0.500
All fusion and retrieval-metric checks passed.
```

已检查 UTF-8、代码围栏、公式结构和本地链接；未进行 GitHub 页面视觉渲染验收。

练习覆盖 RRF 名次从 1 开始、跨路累加、稳定并列排序、空列表、重复 ID 拒绝，以及 Recall、Hit 和 MRR 的不同分母。

代码没有实现 BM25、向量编码或 cross-encoder，也没有真实检索质量结论。空 gold 查询在示例中明确排除并计数，实际评估协议必须提前约定。

## 参考资料

- [Elastic：Practical BM25 — The BM25 Algorithm and its Variables](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)。
- [Elastic：Reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)。
- [Sentence Transformers：Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)。
- [前篇：检索增强生成](2026-09-27-rag-retrieval-evidence-citations.md)。

## 今日总结

1. 词法和语义检索提供不同信号，原始分数不能随意相加。
2. RRF 融合名次而非原始分数，但仍有参数与候选窗口选择。
3. 重排序受候选集合上限约束。
4. Recall、Hit 和 MRR 回答不同问题，尤其要注意多证据任务。
5. 排序、证据支持与最终回答质量需要分别评估。

## 下次衔接建议

继续学习 RAG 的查询改写与多跳检索：拆解复合问题、保留原始意图、逐步补充证据，并设置停止条件与检索预算。
