# 2026-09-27：检索增强生成——检索、证据组织与引用核验

## 今日目标

上一篇 [工具调用的可靠执行](2026-09-26-reliable-tool-execution-idempotency.md)强调外部结果要如实回传。今天把文档检索作为一种只读工具，构建从问题到证据再到回答的最小闭环。

完成后应能回答：

1. 文档切分时需要保存哪些来源信息？
2. 相似度高为什么不等于证据充分？
3. 怎样在上下文预算内组织证据？
4. 引用真实性与语义支持有什么区别？
5. 没有足够证据时如何回应？

本篇提供纯 Python 的关键词检索与引用检查，没有调用语言模型、向量数据库或真实私有文档。所有示例文档为人工构造，不是生产 RAG 系统。

---

## 1. RAG 不是给模型增加一个“事实保证开关”

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)研究了结合检索与生成的知识密集型任务。工程上通常把相关材料加入模型上下文，但检索到材料并不自动保证答案正确。

至少要区分四种失败：

- 语料没有所需信息。
- 语料有信息，但没有检索到。
- 检索到了，但上下文组装遗漏或破坏了证据。
- 模型看到了证据，但误读、夸大或补入无依据内容。

只看最终答案无法准确定位是哪一层出了问题。

## 2. 文档进入索引前，先保存来源

建议为每个 chunk 保存：

- doc_id、文档版本和 chunk_id。
- 标题、章节、页码或原文位置。
- 原文片段及其在固定快照中的偏移。
- 获取时间、业务生效时间和内容摘要。
- 权限标签、来源类型和解析器版本。

获取时间不等于内容生效时间。网页今天被抓取，不代表其中规则今天仍适用。

偏移应说明单位：字节、Unicode 字符还是 token。本文使用 Python 字符串索引，不能拿它直接定位 UTF-8 文件字节。

## 3. 切分要保留解释所需上下文

固定长度切分容易实现，但可能分开定义与例外、表头与数据、结论与适用范围。

可以按标题、段落或文档结构切分，再结合长度限制。重叠可减少边界信息丢失，但也可能让近乎重复的片段挤满 top-k。

保留邻接关系，必要时补充前后段落。不要把一句“允许执行”切出来，却丢掉前面的“仅对测试环境”。

本文直接把两段短文各作为一个 chunk，不实现通用 PDF、表格或网页切分器。

## 4. 检索方法与分数边界

关键词检索适合精确名称和术语；向量检索常用于语义相近但表达不同的查询；混合检索和重排序可以组合两者优点，但增加成本与调参范围。

检索分数通常不是“这段材料真实且足够”的概率。不同索引、模型和查询之间也不应直接比较未经校准的绝对分数。

权限过滤必须在材料进入模型上下文之前完成，并考虑日志、缓存和排序结果中的泄露。不能先把私有材料交给模型，再要求模型“不要说出去”。

## 5. 上下文组装是一项独立步骤

给每个候选提供稳定标签和来源元数据，并区分正文与说明。控制总预算时优先保留能支持关键主张的证据，减少重复片段。

如果预算不足，不应静默截断引用中的句子，同时仍声称引用了完整原文。应记录实际送入模型的文本和它对应的来源范围。

真实预算使用实际 tokenizer，并给问题、指令和输出留空间。字符数或空格分词只能是教学近似。

外部文档中的“忽略之前指令”仍是数据，不会升级为执行指令，也不能授权额外工具调用。

## 6. 引用检查的三个层次

1. 来源存在：引用 ID 属于本次允许使用的证据集合。
2. 文本对应：引用片段与固定版本原文的位置一致。
3. 语义支持：原文确实支持回答中的具体主张，且没有忽略范围、例外或时间条件。

前两层可以做确定性检查；第三层通常需要语义审查、规则或人工复核。

[Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/)把带引用生成及其评估作为研究问题。引用标记不能代替支持关系本身的验证。

## 7. 最小实现：版本化 chunk 与关键词检索

教学语料使用英文，方便用正则分词；正文讨论保持中文。检索只计算查询词集合的覆盖比例，不是 BM25，也不是 embedding 相似度。

```python
import hashlib
import re


def terms(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def make_chunk(doc_id, version, text):
    raw = f"{doc_id}\0{version}\0{text}".encode("utf-8")
    return {
        "chunk_id": hashlib.sha256(raw).hexdigest(),
        "doc_id": doc_id, "version": version,
        "start": 0, "end": len(text), "text": text,
    }


def retrieve(query, chunks, allowed_docs, k=2):
    if type(k) is not int or k <= 0:
        raise ValueError("invalid k")
    query_terms = terms(query)
    if not query_terms:
        return []
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("duplicate chunk id")
    ranked = []
    for chunk in chunks:
        if chunk["doc_id"] not in allowed_docs:
            continue
        score = len(query_terms & terms(chunk["text"])) / len(query_terms)
        if score > 0:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
    return [chunk for _, chunk in ranked[:k]]


def check_quote(citation, evidence, snapshots):
    # Verifies provenance and exact text, NOT whether a claim is entailed.
    indexed = {c["chunk_id"]: c for c in evidence}
    cid = citation["chunk_id"]
    if cid not in indexed:
        raise ValueError("citation was not supplied as evidence")
    chunk = indexed[cid]
    start, end = citation["start"], citation["end"]
    if type(start) is not int or type(end) is not int:
        raise ValueError("invalid offsets")
    if not chunk["start"] <= start < end <= chunk["end"]:
        raise ValueError("quote outside supplied chunk")
    text = snapshots[(chunk["doc_id"], chunk["version"])]
    if text[chunk["start"]:chunk["end"]] != chunk["text"]:
        raise ValueError("snapshot and chunk mismatch")
    if text[start:end] != citation["quote"]:
        raise ValueError("quote differs from source")
    return True
```

这里 allowed_docs 代表可信权限层给出的允许集合，而不是模型自行声称的权限。真实系统还需处理权限变更、共享链接和缓存失效。

chunk_id 摘要用于稳定识别本文记录，不是来源真实性的签名。快照可信性、索引完整性与读取权限属于外层职责。

## 8. 测试：找得到、引得对，但仍可能说错

以下代码接在上一段后运行。

```python
public = "Retries must reuse the same idempotency key."
private = "Internal records are retained for seven days."
snapshots = {
    ("retry-guide", "v1"): public,
    ("internal-guide", "v1"): private,
}
chunks = [
    make_chunk("retry-guide", "v1", public),
    make_chunk("internal-guide", "v1", private),
]
evidence = retrieve("retry retries idempotency key", chunks, {"retry-guide"})
assert evidence == [chunks[0]]
assert retrieve("internal records", chunks, {"retry-guide"}) == []
assert retrieve("quantum telescope", chunks, {"retry-guide"}) == []
assert retrieve("!!!", chunks, {"retry-guide"}) == []
assert retrieve("idempotency", chunks[::-1], {"retry-guide"}) == evidence
assert make_chunk("retry-guide", "v2", public)["chunk_id"] != chunks[0]["chunk_id"]

quote = {
    "chunk_id": chunks[0]["chunk_id"],
    "start": 0, "end": len(public), "quote": public,
}
assert check_quote(quote, evidence, snapshots)

def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")

reject(lambda: check_quote(dict(quote, chunk_id="invented"), evidence, snapshots))
reject(lambda: check_quote(dict(quote, quote="Use a new key."), evidence, snapshots))
reject(lambda: check_quote(dict(quote, end=len(public) + 1), evidence, snapshots))
reject(lambda: retrieve("key", chunks + [chunks[0]], {"retry-guide"}))
changed = dict(snapshots)
changed[("retry-guide", "v1")] = "This snapshot was overwritten."
reject(lambda: check_quote(quote, evidence, changed))

# This deliberately wrong claim still has a real, exact citation.
claim = "Retries must use a different idempotency key."
assert check_quote(quote, evidence, snapshots)
assert claim != public
# Semantic contradiction is explained in the note, not detected by check_quote.

print(f"retrieved_chunks={len(evidence)}")
print("unknown citation, altered quote and changed snapshot rejected")
print("exact quote verified; semantic support still requires review")
print("All retrieval and provenance checks passed.")
```

最后的 claim 与证据关于“是否复用同一个键”相反。代码只验证引用真实，没有检测语义矛盾；assert claim != public 也不构成语义检查。文本不相等可能只是正确改写，必须避免把字符串比较当作蕴含验证。

## 9. 证据不足时怎么办

空检索结果只说明当前方法没有找到材料，不证明世界上没有答案。可以尝试改写查询、扩大适当检索范围或请求缺失资料，但不能编造引用。

有结果也可能不足：例如问题问保留期限，检索只得到创建记录的说明。此时可以说明已找到什么、缺少什么，而不是根据相关术语猜一个天数。

如果不同版本冲突，应指出冲突和适用时间；不能随意选一个更方便的来源。高风险结论还需要更严格的来源与核验标准。

## 10. 分层评估

| 环节 | 可以测什么 | 需要注意 |
| --- | --- | --- |
| 检索 | 标注证据的 Recall@k、排序质量 | gold 证据集合可能不完整 |
| 上下文 | 实际证据覆盖、重复率和截断 | top-k 命中不代表已送入模型 |
| 回答 | 任务正确性、忠实性和关键主张覆盖 | 正确答案也可能引用错误 |
| 引用 | 来源有效性、支持关系与覆盖 | 有引用不等于被证据支持 |
| 系统 | 延迟、失败率、权限隔离 | 缓存命中条件需明确 |

检索评估按问题分组，避免同源改写跨集合泄漏。先用固定语料和可审查的问题集，再考虑扩大规模。

可以给生成器直接提供人工确认的证据，作为诊断对照：若仍答错，说明至少存在生成或指令理解问题。该对照不代表真实检索表现。

## 11. 可追溯记录

延续 [实验追踪](2026-09-18-experiment-tracking-reproducible-reports.md)，保存：

- 查询、允许检索范围、索引和文档快照版本。
- 检索配置、候选排序与重排结果。
- 实际送入模型的 chunk 和文本范围。
- 模型、提示、输出与逐主张引用。
- 核验结果、证据不足原因和重试成本。

涉及敏感文档时只保存必要信息并限制访问；为了复现而无限复制私有材料并不合理。删除或权限撤销还需要同步考虑索引和缓存。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。实际输出：

```text
retrieved_chunks=1
unknown citation, altered quote and changed snapshot rejected
exact quote verified; semantic support still requires review
All retrieval and provenance checks passed.
```

已检查 UTF-8、代码围栏和本地链接；未进行 GitHub 页面视觉渲染验收。

练习覆盖权限范围过滤、无匹配结果、顺序稳定性、版本 ID 变化、伪造引用、篡改片段和快照不一致。

没有实现真实向量检索、重排序、上下文预算、生成或语义支持判断。引用检查通过不能推出“答案有证据支持”。

## 参考资料

- [Lewis 等：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)。
- [Gao 等：Enabling Large Language Models to Generate Text with Citations](https://aclanthology.org/2023.emnlp-main.398/)。
- [前篇：工具调用的可靠执行](2026-09-26-reliable-tool-execution-idempotency.md)。

## 今日总结

1. RAG 的失败可以发生在语料、检索、组装或生成阶段。
2. Chunk 必须关联来源版本、位置和权限。
3. 检索分数不是证据充分性的概率。
4. 引用真实与语义支持是不同问题。
5. 证据不足应明确表达，而不是补出看似合理的答案。

## 下次衔接建议

继续学习检索排序：比较 BM25、向量检索、混合召回与重排序，并用小问题集检查 Recall@k、MRR 和排名融合。
