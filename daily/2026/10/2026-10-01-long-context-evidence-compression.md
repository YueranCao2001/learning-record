# 2026年10月1日 长上下文与上下文压缩

## 今日目标

接着上一篇 [RAG 端到端评估与回归测试](../09/2026-09-30-rag-evaluation-regression-tests.md)，今天研究证据进入模型之前的最后一道选择：上下文太长时，删什么、保留什么，以及如何发现压缩改变了答案。

核心结论是：长度缩短只是资源指标，关键事实、适用条件和来源关系保留下来才是质量要求。能放进上下文窗口，不代表模型一定能正确使用所有信息。

本篇以人工编写的证据包演示预算选择和来源校验，不调用模型，不实现自动摘要，也不把字符数称为真实 Token 数。

## 1 长度容量与有效使用不同

[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)在多文档问答和键值检索任务中观察到，改变相关信息的位置会影响所测模型的表现，中部信息尤其可能被忽略。

这说明验收应改变证据位置，而不是只测试固定位置。它不是“所有模型都存在相同程度位置偏差”的定律，也不证明把答案材料放最后就能解决所有任务。

我会把以下变量分开：总长度、相关证据数量、干扰材料数量、证据所在位置，以及多跳证据之间的距离。一次只改变一个因素，更容易定位问题。

## 2 三种缩短方式

| 方法 | 保留的内容 | 主要风险 |
| --- | --- | --- |
| 直接截断 | 开头或结尾的一段文本 | 切断句子、删掉例外或后续修订 |
| 抽取式选择 | 完整的原文片段 | 选择错误、缺少相邻条件或跨段依赖 |
| 生成式摘要 | 改写后的事实描述 | 数字、否定、主体或范围发生变化 |

抽取式不等于无损。原文逐字保留也可能因为漏掉另一段的限制而误导读者。

[LLMLingua](https://aclanthology.org/2023.emnlp-main.825/)研究了预算控制、迭代 Token 级压缩及模型间分布对齐。本篇只借此区分提示压缩这一研究方向，下面的整包选择代码不是论文算法的复现。

提示文本压缩与 KV Cache 压缩不是同一个操作，不能把两者的压缩率或质量结果混用。

## 3 先扣除非证据预算

对一个总上下文容量同时约束输入和输出的接口，可以采用下面的保守预算思路：

```math
B_{\mathrm{evidence}} \leq
W - T_{\mathrm{fixed}} - R_{\mathrm{output}} - M
```

W 是模型和接口允许的总容量，固定输入包含指令、问题、对话模板与工具定义，输出预留为 R，M 为安全余量。若接口分别限制输入和输出，还要分别检查相应限制。

真实部署应对最终组装的请求使用匹配的分词器和计数规则。把各段单独计数后相加不一定等于整体计数；分隔符、来源标签和聊天模板也不能漏算。

下面练习用 Python 的 len 统计序列化文本的 Unicode 码点数，只用于测试选择逻辑。它既不是字节数，也不等于中文字符的真实 Token 开销。

## 4 让规则和例外一起移动

假设资料包含：

- 普通设备支持 30 天内退货。
- 定制设备不适用上述退货规则。

若问题是“定制设备能否 30 天内退货”，只保留第一句会制造错误依据。一个简单防线是将这两段标注为不可拆分的证据包，预算不够就整包跳过，不能悄悄只保留一般规则。

这依赖人工或另一个系统正确识别关系。代码可以保证既定证据包不被拆开，却不能自动发现语料中所有未知例外。

证据包也可能跨文档；要保留每个片段自己的文档版本和范围，不能给整个包随意贴一个来源。

## 5 来源映射与引用边界

每个抽取片段保存 source、version、start、end 和 text。范围采用原文 Unicode 码点的左闭右开区间，校验 text 是否等于该范围的原文。

实际索引若使用字节或 Token 偏移，需要显式转换，不能直接套用字符偏移。换行规范化、文档更新和 OCR 变化也可能使旧范围失效，因此必须绑定版本。

引用映射一致只能证明文字来自所标位置，不能证明原文真实、完整、仍有访问权限或支持当前答案。权限过滤应在压缩前完成，不能先把敏感材料交给摘要模型再期待输出过滤补救。

## 6 最小证据包选择器

按调用者给出的优先级扫描候选；试加入整个证据包，最终序列化长度不超预算才接受。超长包被跳过后继续检查后续包。

这是确定性的贪心策略，不是质量最优解。排名和证据包关系都是外部输入。

```python
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Span:
    source: str
    version: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Bundle:
    id: str
    spans: tuple[Span, ...]


def render(bundles):
    return "\n\n".join(
        f"[{b.id}|{s.source}@{s.version}:{s.start}:{s.end}]\n{s.text}"
        for b in bundles for s in b.spans
    )


def validate(bundles, documents):
    seen = set()
    for b in bundles:
        if not b.id or b.id in seen or not b.spans:
            raise ValueError("invalid or duplicate bundle")
        seen.add(b.id)
        for s in b.spans:
            key = (s.source, s.version)
            if key not in documents:
                raise ValueError("unknown source version")
            original = documents[key]
            if not (0 <= s.start < s.end <= len(original)):
                raise ValueError("invalid source range")
            if original[s.start:s.end] != s.text:
                raise ValueError("source text mismatch")


def select(bundles, documents, budget):
    if type(budget) is not int or budget < 0:
        raise ValueError("invalid budget")
    validate(bundles, documents)
    chosen = []
    for bundle in bundles:
        trial = chosen + [bundle]
        if len(render(trial)) <= budget:
            chosen = trial
    return chosen
```

这里把来源标签和分隔符一起计入字符预算，但没有加入完整聊天请求。真实使用需要在最终请求层再次计数和验收。实现只接受教学中的结构化对象，不是面向不可信输入的完整解析器。

## 7 合成案例与边界测试

以下代码接在上一段之后执行。每个包内的片段顺序固定，输入包的顺序代表优先级。

```python
rule = "普通设备支持30天内退货。"
exception = "定制设备不适用上述退货规则。"
policy = rule + exception
docs = {
    ("policy", "v1"): policy,
    ("background", "v1"): "背景说明。" * 100,
    ("contact", "v1"): "联系售后。",
}
critical = Bundle("returns", (
    Span("policy", "v1", 0, len(rule), rule),
    Span("policy", "v1", len(rule), len(policy), exception),
))
background = Bundle("background", (
    Span("background", "v1", 0, len(docs[("background", "v1")]),
         docs[("background", "v1")]),
))
contact = Bundle("contact", (
    Span("contact", "v1", 0, len(docs[("contact", "v1")]),
         docs[("contact", "v1")]),
))
candidates = [background, critical, contact]
budget = len(render([critical, contact]))
chosen = select(candidates, docs, budget)

assert chosen == [critical, contact]
assert len(render(chosen)) == budget
assert select(candidates, docs, 0) == []
assert select([critical], docs, len(render([critical])) - 1) == []
assert select([critical], docs, len(render([critical]))) == [critical]
assert select(candidates, docs, budget) == chosen

# Naive prefix truncation loses the exception.
prefix = policy[:len(rule)]
assert rule in prefix and exception not in prefix
assert rule in render(chosen) and exception in render(chosen)


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


wrong_text = replace(critical.spans[0], text="所有设备均可退货。")
wrong_version = replace(critical.spans[0], version="v2")
wrong_range = replace(critical.spans[0], end=len(policy) + 1)
for broken in (wrong_text, wrong_version, wrong_range):
    reject(lambda broken=broken: select(
        [Bundle("bad", (broken,))], docs, budget))
reject(lambda: select([critical, critical], docs, budget))
reject(lambda: select(candidates, docs, -1))
reject(lambda: select(candidates, docs, True))

print("selected=" + ",".join(b.id for b in chosen))
print("rule and exception retained together")
print("oversized bundle skipped; later bundles considered")
print("All context compression checks passed.")
```

测试证明了给定包的原子选择、来源一致性和预算边界，没有证明模型一定能正确回答退货问题。例外保留检查依赖这个手工案例，不能用关键词匹配代替通用语义判断。

## 8 摘要还要保存什么

如果后续引入生成式摘要，我会要求每个重要主张关联原文片段，并单独核验：

- 实体与主体是否互换。
- 数值、单位、日期和版本是否变化。
- “不适用”“仅限”“除外”等限制是否丢失。
- 多份来源冲突是否被改写为一致结论。
- 推测是否被写成确定事实。

摘要应标记为派生文本，不能放进原文引号里充当逐字引用。无法确认关键条件时，应回到原文或明确证据不足。

保留原文不意味着必须把原文全部再次塞入提示；可以保存可追溯的外部映射，并在需要时取回有权限的对应片段。

## 9 压缩实验怎样对照

沿用昨天的固定问题集，对相同问题比较不压缩、直接截断、整包抽取和摘要。固定模型、提示、语料版本与解码配置；候选的预算规则也要预先明确。

超过模型容量的全量输入不能冒充可运行基线。可以在可容纳子集做质量对照，再在长输入集合比较实际可执行的策略。

记录输入 Token、压缩耗时、首 Token 延迟、总延迟、费用，以及答案正确性、引用支持和拒答行为。压缩器自身的时间和费用必须计入端到端结果，不能把输入长度下降直接当作同比例加速。

还应单独报告证据保留率和完整证据包覆盖率。保留了两个必要片段中的一个，不能算完整覆盖。

## 10 位置与退化切片

将同一组相关证据分别放在开头、中间、结尾，并改变干扰材料量，比较逐题变化。对多跳题，另测相邻证据与分散证据。

重点审查数字、否定、例外、版本冲突与无答案问题。压缩后原本可回答的问题失去证据时，拒答可能比猜测合理，但仍应计入该压缩策略的质量损失，不能偷偷改成无答案标签来提高得分。

平均值相同不代表没有退化；复用 [逐题回归检查](../09/2026-09-30-rag-evaluation-regression-tests.md)列出修复项和退化项。不要只挑压缩效果好的样例展示。

## 11 今日练习结果与边界

已在本地依次执行两个 Python 代码块，全部断言通过。实际输出：

```text
selected=returns,contact
rule and exception retained together
oversized bundle skipped; later bundles considered
All context compression checks passed.
```

UTF-8、代码围栏、公式括号、本地链接与 README 索引检查通过；未进行 GitHub 页面视觉渲染验收。

本例没有执行分词器、模型推理、自动摘要或真实延迟测试。练习结论仅适用于给定证据包与字符预算，不构成生产效果报告。

## 参考资料

- [Liu 等 Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)。
- [Jiang 等 LLMLingua](https://aclanthology.org/2023.emnlp-main.825/)。
- [前篇 RAG 端到端评估与回归测试](../09/2026-09-30-rag-evaluation-regression-tests.md)。
- [证据组织与引用核验](../09/2026-09-27-rag-retrieval-evidence-citations.md)。

## 今日总结

1. 容量允许与可靠使用是两个问题。
2. 先预留固定输入和输出预算，再分配证据空间。
3. 抽取保留字面内容，也可能删掉决定结论的例外。
4. 来源映射检查与语义支持检查不能互相替代。
5. 压缩收益必须与逐题质量退化和额外成本一起报告。

## 下次衔接建议

继续学习 RAG 中的提示注入与信任边界，区分文档事实和文档中的操作指令，并设计恶意检索内容、越权请求和工具调用的回归案例。
