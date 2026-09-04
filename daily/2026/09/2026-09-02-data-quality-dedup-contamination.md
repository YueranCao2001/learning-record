# 2026-09-02：训练数据治理——质量过滤、去重、污染检测、Provenance 与数据混合

## 今日目标

昨天在[训练数据流水线](2026-09-01-training-data-pipeline.md)中解决了“怎样把正确、唯一、可恢复的 token 持续送到 GPU”。今天继续回答更上游的问题：

> 什么数据值得被模型学习，我们又如何证明训练数据确实符合预期？

完成后应能回答：

1. “数据质量”为什么不是一个脱离任务的绝对分数？
2. 质量过滤、去重、Benchmark 污染、安全、隐私与许可为什么不能混为一谈？
3. 精确哈希、Bloom Filter、MinHash + LSH 和 embedding 去重分别解决什么问题？
4. 为什么去重越彻底不一定越好，duplicate keeper 又会怎样改变数据分布？
5. 如何建立可审计的 Benchmark contamination 检测流程？
6. Provenance 为什么应是一张 lineage DAG，而不只是一个 URL？
7. 数据混合中的池大小、采样权重、实际 token 占比与 loss 权重有什么区别？
8. 如何通过固定预算消融、分层人工审查和运行时指标验证数据决策？

---

## 1. 数据集不是文件夹，而是 Versioned Build Artifact

大模型数据集通常不是“下载一批文本后永久不变”，而是由输入快照、代码、配置和治理决策生成的构建产物：

```text
immutable raw snapshots
  + extraction code
  + normalization config
  + quality filters
  + duplicate graph and keeper policy
  + benchmark registry
  + split policy
  + tokenizer and mixture config
  = versioned training dataset
```

任何一项改变，都可能改变最终 token 分布。可靠的数据集版本至少要能回答：

- 输入来自哪些快照？
- 哪个版本的程序处理了它？
- 每条数据为什么被保留或移除？
- 重复簇中为什么选择这一条作为代表？
- 哪些 Benchmark 命中过？
- 最终进入了哪个 split、shard 和训练 run？
- 是否能从输出反向追踪到原始来源？

---

## 2. 五个相邻但不同的问题

### 质量过滤

文本是否可解析、连贯，并适合目标训练分布。

### 去重

训练数据内部是否有相同或近似内容重复占用训练预算。

### Benchmark Contamination

训练数据是否包含评测题、答案、解析、测试代码或等价改写，导致评测失真。

### 安全与隐私

是否包含 PII、恶意内容、敏感信息或其他风险。

### 来源与许可

数据从哪里来、经过什么转换、来源条款和许可证据是什么。

它们相关，但不能互相代替：

- 文本可以高质量但许可未知；
- 文本可以无重复但含评测答案；
- 文本可以公开访问但仍含 PII；
- 安全过滤掉的内容不应被记作“语言质量差”；
- 去重不能自动完成去污染。

一个好的 pipeline 会分别记录这些信号，再根据用途形成可解释的决策。

---

## 3. 端到端治理蓝图

```text
原始不可变快照
  ↓ 记录来源、捕获时间与 raw hash
解析 / 主文本提取
  ↓ 记录 extractor 与失败原因
保守规范化
  ↓ raw hash + normalized hash
轻量有效性与语言检测
  ↓ 多维 quality / safety / PII / license signals
精确重复分组
  ↓
近重复候选：MinHash / LSH / containment / embedding
  ↓ 回算真实相似度并构建 duplicate graph
Keeper policy
  ↓ 按质量、完整度、时间、许可与稳定 ID 确定代表
按 duplicate cluster / entity 做 train-val-test split
  ↓
训练池对外部 Benchmark Registry 去污染
  ↓
tokenization
  ↓ 按最终有效 token 重新计算数据混合
sampling / packing / sharding
  ↓
审计实际 loss-bearing token 占比
```

顺序不是唯一的。例如先做低成本 exact dedup 可以减少昂贵质量评分的计算，但必须保留完整簇成员，等质量和许可信号齐全后再决定 keeper。

---

## 4. “质量”是多目标约束，不是单个客观分数

至少应区分以下维度：

1. **可解析性**：是否成功抽取主文本，有无乱码或残缺 HTML；
2. **语言自然度**：是否像目标语言中的有效文本；
3. **信息与任务适配度**：是否包含目标能力需要的知识、推理、代码或对话；
4. **多样性与覆盖**：语言、地区、时间、领域和文体是否过于集中；
5. **安全与隐私**：是否含 PII、恶意或敏感内容；
6. **来源与合规状态**：是否具备足够的来源和使用条件证据；
7. **可追溯性**：能否解释数据是如何产生和变换的。

“像 Wikipedia”可能是某个分类器的正样本定义，但不等于所有目标下的普遍高质量。代码、诗歌、论坛、方言和低资源语言都可能偏离这种风格，却对模型能力不可替代。

---

## 5. 第一道质量关：解析与主文本提取

网页原始 HTML 中常含：

- 导航栏和页脚；
- Cookie 声明；
- 广告与推荐链接；
- 评论控件；
- 隐藏文本；
- 重复模板；
- JavaScript / CSS；
- 只有标题而没有正文的页面。

因此“抓到网页”不等于“获得训练文档”。主文本提取器本身就是重要实验变量。它可能：

- 漏掉表格、公式和代码；
- 把菜单当成正文；
- 合并本不相邻的段落；
- 破坏对话轮次；
- 让 boilerplate 在大量页面中重复。

至少记录：

```text
extractor_name
extractor_version
raw_bytes
extracted_chars
extraction_status
reason_code
```

解析失败应隔离并保留原因，不要直接静默丢弃。

---

## 6. 可解释的启发式过滤

常见低成本信号包括：

- 文档字符数和词数；
- 平均词长；
- 字母、数字、标点、空白和异常字符比例；
- 语言识别置信度；
- 重复行、重复段落、重复 n-gram 比例；
- 过短行或极长行比例；
- 句末标点比例；
- HTML / URL / code-like token 比例；
- 特定 boilerplate 模式。

Gopher / MassiveText 展示过一组可解释规则，但这些阈值只能作为历史案例，不能直接跨语言、代码和领域照抄。

例如：

- 中文不能用英语空格分词假设；
- 代码天然有较多符号；
- 诗歌天然有短行；
- 表格天然有重复分隔符；
- 论坛和对话天然有碎片句；
- 公式和低资源语言可能让语言模型困惑度偏高。

每条规则都应输出 `score`、`threshold`、`keep/drop`、`reason_code` 和 `filter_version`，而不是只输出最终布尔值。

---

## 7. Perplexity 不是质量真值

用参考语言模型为文本打 perplexity 分数很常见，但要同时警惕两个方向。

### 高 Perplexity

可能表示：

- 乱码或语言错误；
- 随机关键词堆砌；
- 参考模型不熟悉的代码、数学、专业术语；
- 方言、低资源语言或新知识。

### 低 Perplexity

可能表示：

- 流畅、常见的自然语言；
- 高度模板化的页面；
- 大量重复样板；
- 已被参考模型记忆的文本；
- 信息量很低的常见短句。

所以 perplexity 更像一个需要校准的信号，而不是“越低越好”的排序真理。应按语言、领域、长度和文体分层观察分布。

---

## 8. 模型型质量分类器会继承标签定义

常见做法是：

1. 选一组“高质量”正样本和“低质量”负样本；
2. 用 fastText、小 Transformer 或 embedding 分类器学习区分；
3. 对海量网页打分；
4. 保留高于阈值的文档。

风险在于，分类器学习到的可能不是“质量”，而是：

- Wikipedia 的写作风格；
- 教材中的标题结构；
- 某些域名；
- 较正式的英语；
- 教师模型的偏好；
- 正负样本的时间和主题差异。

若用 LLM 自动标注“教育价值”，还要记录教师模型、prompt、采样参数和标注版本。教师也可能见过 Benchmark，进而把污染信号带入分类器。

正确做法是把模型分数视为一个可审计特征，通过分层人工抽样和 proxy model 消融校准。

---

## 9. 阈值必须按分层数据审计

全局选择 top-k 文档可能让某些小语言或领域几乎消失。建议至少按以下 strata 统计接受率：

- language / script；
- source；
- domain；
- country / region；
- time snapshot；
- document length；
- code / prose / conversation；
- dialect 或其他已知群体属性；
- license status。

需要重点人工检查：

1. 阈值附近的文档；
2. 高分但疑似模板化的文档；
3. 低分但来自稀缺领域的文档；
4. 各分层中的误删与漏保留。

过滤器不仅在删除垃圾，也在重塑模型将来看到的世界分布。

---

## 10. 质量过滤实验必须固定训练预算

比较 Raw、Heuristic、Classifier、Combined 四种数据时，至少固定：

- 模型和 tokenizer；
- 总 consumed tokens；
- global batch、sequence length 与训练 steps；
- 优化器和学习率计划；
- 评测套件；
- 多个随机种子。

同时报告：

- 过滤后池大小；
- unique documents / tokens；
- 平均重复暴露次数；
- 各来源和语言的留存率；
- 平均指标与最差领域指标；
- 数据处理成本。

否则过滤后数据池变小、模型被迫重复更多 epoch，实验会把“质量变化”和“重复暴露”混在一起。

---

## 11. 精确去重首先要定义 Canonicalization

设原始文本为 `x`，规范化函数为 `c(x)`，稳定摘要为：

```math
h_x = H(c(x))
```

一个保守示例：

```python
import hashlib
import unicodedata

def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

raw_hash = sha256_text(raw_text)
normalized_text = canonicalize(raw_text)
normalized_hash = sha256_text(normalized_text)
```

是否 lowercase、删除标点、归一化数字、URL 或空白，都会改变“相同”的定义。对代码、数学和结构化文本，激进规范化尤其危险。

建议同时保存：

- `raw_hash`；
- `normalized_hash`；
- `canonicalizer_version`；
- 原始与规范化长度；
- 必要时在 hash 相同后再次比较规范化字节。

不要使用 Python 内置 `hash()` 作为永久 ID，因为它不保证跨进程和跨版本稳定。

---

## 12. Hash 位数与生日碰撞

如果把一个理想的 `b` 位 hash 用在 `n` 个对象上，至少一次碰撞的近似概率为：

```math
P(\text{collision})
\approx
1-\exp\left(-\frac{n(n-1)}{2\cdot 2^b}\right)
```

当 `n = 10^9`、`b = 64` 时，概率约为 2.7%，已经不能忽略。

因此：

- 64 位值适合某些高性能候选索引，不适合作为唯一永久身份；
- 稳定 provenance ID 更适合 128 或 256 位摘要；
- adversarial 场景还要考虑所选 hash 的安全性；
- 精确 dedup 的最终确认可比较规范化内容，而不只依赖摘要。

---

## 13. Bloom Filter 解决的是成员查询

Bloom Filter 能低内存回答：

> 这个 fingerprint 是否“可能已经出现过”？

它可能有 false positive，但没有 false negative（在标准假设且不删除元素时）。若：

- bit 数为 `m`；
- 插入元素数为 `n`；
- hash 函数数为 `k`；

其假阳性率近似为：

```math
p \approx \left(1-e^{-kn/m}\right)^k
```

Bloom Filter 的用途是流式 exact-membership 候选判断，不是 near-duplicate 算法。需要注意：

- false positive 会误删唯一文档；
- 普通 Bloom Filter 不能列出原始重复对象；
- 普通 Bloom Filter 不支持安全删除；
- 参数、插入顺序和预计误删量都应记录；
- 高风险删除最好用外部精确索引复核。

---

## 14. 近重复：先把文档变成 Shingle 集合

把文档 `D` 转成长度为 `w` 的 word 或 character n-gram 集合 `S(D)`。

两个文档的 Jaccard 相似度为：

```math
J(A,B)
=
\frac{|S(A)\cap S(B)|}
{|S(A)\cup S(B)|}
```

例如 word 5-gram 可以捕捉局部连续短语。`w` 太小会让常见短语制造误报；太大则对轻微编辑更敏感，降低召回。

需要明确记录：

- tokenizer / normalization；
- word 还是 character shingles；
- shingle 宽度；
- 是否使用集合或频次；
- 最短文档策略；
- 目标相似度和人工标注定义。

---

## 15. MinHash 为什么能估计 Jaccard

MinHash 的核心性质是：

```math
P[h_{\min}(A)=h_{\min}(B)] = J(A,B)
```

使用 `k` 个独立 MinHash 值，可估计：

```math
\hat J(A,B)
=
\frac{1}{k}
\sum_{j=1}^{k}
\mathbf{1}[h_j(A)=h_j(B)]
```

`k` 越大，估计方差通常越小，但计算和存储越高。MinHash 的作用是把巨大的 shingle 集合压缩成签名，不是加密 hash，也不适合作为文档永久 ID。

---

## 16. LSH 是候选生成器，不是相似度证明

把 `k = b r` 个 MinHash 值拆成：

- `b` 个 bands；
- 每个 band `r` 行。

如果两文档真实相似度为 `s`，至少一个 band 完全相同、从而成为候选的概率约为：

```math
P_{\text{candidate}}(s)
=
1-\left(1-s^r\right)^b
```

这形成一条 S 曲线，而不是硬阈值。因此：

- LSH 命中可能是假阳性；
- LSH 未命中也可能是假阴性；
- `b`、`r` 与 shingle 定义必须联合调参；
- 候选对最好回算真实 Jaccard、containment 或其他确定性分数；
- 应在人工标注对上测 precision / recall。

FineWeb 的一个公开配置使用 word 5-gram、112 个哈希和 14 × 8 banding，目标约为 75% 相似区域。这是特定数据和目标下的选择，不是通用默认值。

---

## 17. Jaccard 会漏掉“长文档包含短文档”

若短文章 B 几乎完整出现在很长文章 A 中，二者 union 很大，Jaccard 可能不高。

可以补充 containment：

```math
C(A,B)
=
\frac{|S(A)\cap S(B)|}
{\min(|S(A)|,|S(B)|)}
```

当短文几乎被长文包含时，containment 会接近 1。它适合发现：

- 新闻转载中的长前后缀；
- 题目出现在带解析的网页中；
- README 被包含进完整仓库快照；
- 一段文章被拼进聚合页面。

但 boilerplate 和常见许可证也可能形成高 containment，仍需结合长度、位置和内容类型复核。

---

## 18. 重复图的传递性陷阱

把相似候选对作为边，再用 Union-Find 或 connected components 建簇，会出现：

```text
A ~ B
B ~ C
A !~ C
```

A 与 C 仍会通过 B 进入同一 connected component。大簇可能因“相似链”不断扩张。

应监控：

- cluster size 的 P50 / P95 / P99 / max；
- 超大簇所占文档和 token 比例；
- 簇内最小、平均与中心相似度；
- 边数量和跨来源边；
- 链式合并样本。

必要时可以采用中心约束、complete-link 类规则、二次验证或限制簇内距离，而不是无条件传递闭包。

---

## 19. Keeper Policy 会重塑数据分布

每个 duplicate cluster 最终保留哪一条，不应是无意识的实现细节。

可能的排序信号：

1. 许可状态更清晰；
2. 来源 provenance 更完整；
3. 主文本抽取更完整；
4. 质量得分更高；
5. boilerplate 更少；
6. 时间更新或更符合目标 cutoff；
7. 稳定 ID 作为最终 deterministic tie-breaker。

需要记录：

- cluster ID；
- 所有成员；
- keeper ID；
- keeper policy version；
- 每个比较信号；
- 原簇大小；
- 被删除 token 数。

随机 keeper 会让构建不可复现，也可能改变来源、时间和语言分布。

---

## 20. 去重范围不是越全局越好

可选范围包括：

- 单个 shard；
- 单个 crawl snapshot；
- 单一来源；
- 单一语言；
- 全部时间与来源。

全局去重删除最多，但也会改变数据分布。例如多个 crawl 中长期存在的内容可能只保留某一次抓取，keeper 顺序便会控制时间和来源比例。

FineWeb 的消融发现，在其英语 Common Crawl 设置中，跨 96 个 snapshot 的全局 MinHash 去重不如逐 snapshot 去重。这不是“永远不要全局去重”，而是说明：

> 去重范围、keeper 规则和训练分布必须一起做实验，删除数量不是成功指标。

---

## 21. 文档唯一不等于内容没有重复

不同文档中可能共享：

- 导航和页脚；
- 许可证；
- 新闻稿；
- 商品描述；
- 长引用；
- 代码模板；
- Benchmark 题目或答案片段。

因此可以分三层统计：

1. **document-level duplicates**；
2. **paragraph / line-level duplicates**；
3. **span / n-gram repetition**。

但全局删除所有常见行也危险，例如合法的法律条款、代码 import、公式定义和常用句可能被误删。更安全的方式是先统计频率、识别 boilerplate 类型，再决定删除整篇、删除片段或仅降低采样权重。

---

## 22. Semantic Deduplication 的能力与风险

Embedding 相似度能发现：

- 轻度改写；
- 语义重复；
- 模板不同但内容相同的文本；
- 部分跨语言等价内容。

但它也可能把以下内容误判为重复：

- 同一问题的不同观点；
- 同一题目的不同答案；
- 同主题但信息互补的文档；
- 少数群体的相似表达；
- 版本更新前后的重要差异。

因此 semantic dedup 更适合作为高成本候选阶段，而不是“一键删除”。应按语言、领域和类别分层标注 precision / recall，并保留相似证据。

---

## 23. 去重与去污染的目标不同

### 去重

比较训练语料内部：

```text
train document ↔ train document
```

目标是减少冗余、偏置和重复记忆。

### 去污染

比较训练候选与评测注册表：

```text
train document ↔ evaluation example / answer / rationale / test
```

目标是保护评测有效性。

训练数据即使内部完全无重复，也可能恰好包含一份 Benchmark 答案；这仍是污染。反过来，大量重复新闻稿可能与任何评测无关，但依然需要去重。

---

## 24. Benchmark 污染需要分类

至少区分：

### 按内容

- input-only：只有背景或问题；
- input + label：问题与正确答案同时出现；
- rationale leakage：出现解析、证明或推理链；
- code leakage：出现测试代码、隐藏测试或参考实现。

### 按变形

- exact；
- 规范化后 exact；
- near-exact；
- paraphrased；
- translated；
- synthetic reproduction。

### 按训练阶段

- 预训练数据；
- continued pretraining；
- SFT；
- preference / RL 数据；
- 开发人员反复查看 test 后形成的人工反馈。

“某段背景文本出现过”和“完整题目答案出现过”的风险不同，不能只给一个 contamination percentage。

---

## 25. 建立版本化 Eval Registry

每道评测样本建议保存：

```json
{
  "benchmark": "example-bench",
  "version": "v2.1",
  "released_at": "2026-01-15",
  "example_id": "test-0042",
  "input_hash": "...",
  "input_label_hash": "...",
  "rationale_hash": "...",
  "verbalized_prompt_hash": "...",
  "normalization_version": "...",
  "ngrams_version": "...",
  "canary": "...",
  "license_or_terms": "...",
  "matches": []
}
```

为什么要保存完整 verbalized prompt？因为多选题的原始结构化记录和最终送给模型的文本不同；选项顺序、模板和答案字母都可能影响匹配。

Benchmark 更新后必须产生新版本，不能静默覆盖旧 registry。

---

## 26. 污染检测应使用级联

一个可扩展流程：

### 第一层：Canary 与精确匹配

- Benchmark 主动提供的 canary；
- raw / normalized hash；
- 完整 input；
- input + label；
- rationale 和测试代码。

### 第二层：Lexical Candidate Search

- character / token n-gram；
- skip-gram；
- suffix array / inverted index；
- Bloom Filter 作为候选索引；
- containment。

### 第三层：Semantic Candidate Search

- embedding 检索；
- 改写和翻译候选；
- 合成问法候选。

### 第四层：复核

- 回看命中 span；
- 区分 input-only 与 input+label；
- 人工或受控模型复核；
- 抽样未命中项估计 false negative。

最终保存“证据”，而不只是一个布尔标签。

---

## 27. “未检出污染”不等于“无污染”

n-gram 方法会漏掉：

- 同义改写；
- 翻译；
- 选项重排；
- 数值或实体替换；
- 合成数据中的等价题；
- 只泄漏解法而非原题。

Embedding 方法则可能把“同主题但不同题”误报为污染。

Rethinking Benchmark Contamination 的实验说明，改写样本可以绕过字符串检测，同时仍让模型对 Benchmark 过拟合。因此报告中应写：

- “在某方法、某阈值下未检出”；
- 不能写“确认无污染”。

---

## 28. 污染报告必须带上方法学

至少披露：

- Benchmark 名称、版本和 hash；
- 数据时间 cutoff；
- 文本规范化规则；
- 匹配单位：文档、段落还是 span；
- n-gram / skip-gram 定义；
- 阈值；
- 是否要求答案共同出现；
- 命中后删除整篇还是局部片段；
- exact / lexical / semantic 各自命中量；
- 人工复核 precision 与抽样 false negatives；
- clean / flagged 子集大小；
- full eval 与 clean-subset 结果。

只报告“污染率 2%”通常信息不足，因为不同定义可能差一个数量级。

---

## 29. Clean Subset 不能单独证明污染无影响

发现污染后，常见做法是只在未命中的样本上评测。但 clean 与 flagged 子集可能在：

- 难度；
- 主题；
- 长度；
- 来源；
- 题型

上系统性不同。

所以“clean subset 分数接近 full set”并不能建立污染无影响的因果结论。更强的证据是：

1. 使用相同训练 recipe；
2. 构造 contaminated 与 decontaminated 数据；
3. 固定训练 token 和计算量；
4. 多 seed 重训；
5. 比较 full、clean、flagged 与新鲜评测。

如果无法重训，应诚实披露检测局限，避免过度解释。

---

## 30. Split 必须按重复簇或实体分组

若先把文档随机分到 train / validation，再分别在各 split 内 dedup，近重复内容可能跨 split 泄漏。

更安全的做法：

1. 先构建 duplicate clusters；
2. 再把整个 cluster 分到同一 split；
3. 对对话、仓库、用户或时间序列，再增加 entity-level grouping；
4. 最后检查跨 split exact / near duplicate rate。

这与昨天的原则一致：split 必须早于 chunking、windowing 和 packing。外部 Benchmark 则由独立 registry 对训练候选再做去污染。

---

## 31. Provenance 应是一张 Lineage DAG

W3C PROV 用三个核心概念描述来源：

- **Entity**：原始抓取、抽取文档、过滤后记录、token shard；
- **Activity**：抓取、解析、规范化、过滤、去重、tokenization；
- **Agent**：组织、人员、软件或自动标注模型。

一个典型 lineage：

```text
WARC record
  └─[extractor v3]→ extracted document
       └─[normalizer v2]→ normalized document
            ├─[quality model q7]→ score
            ├─[minhash config m4]→ duplicate cluster
            └─[decontam registry e9]→ match evidence
                 └─[build config b12]→ train shard
```

同一输出可以有多个 parent，例如合成对话来自原文、生成模型和 prompt 三者。用 DAG 比单一 `source_url` 更接近真实数据生命周期。

---

## 32. 单条记录的最小 Provenance Schema

```json
{
  "record_id": "stable-id",
  "source": {
    "uri": "https://example.org/page",
    "snapshot": "crawl-2026-08",
    "captured_at": "2026-08-19T10:11:12Z",
    "upstream_record_id": "warc-record-id"
  },
  "digests": {
    "raw_sha256": "...",
    "normalized_sha256": "...",
    "canonicalizer_version": "normalize-v2"
  },
  "lineage": {
    "parent_ids": ["raw-record-id"],
    "extractor_version": "extract-v3",
    "pipeline_commit": "git-sha",
    "config_hash": "..."
  },
  "decisions": [
    {
      "stage": "quality",
      "version": "quality-v7",
      "score": 0.82,
      "threshold": 0.70,
      "keep": true,
      "reason_code": "quality_pass"
    }
  ],
  "dedup": {
    "cluster_id": "cluster-001",
    "method_version": "minhash-v4",
    "is_keeper": true,
    "cluster_size": 7
  },
  "contamination_matches": [],
  "governance": {
    "license_status": "unknown",
    "license_evidence": null,
    "pii_status": "not_scanned"
  },
  "output": {
    "split": "train",
    "shard": "train-0042.parquet",
    "token_count": 1234
  }
}
```

`unknown` 必须保留为一等状态，不能把缺失证据默认解释为 permissive。

---

## 33. Dataset Manifest 记录构建全局事实

数据集级 manifest 至少包含：

- dataset semantic version；
- 创建时间；
- 所有输入 snapshot；
- 输入与输出 checksums；
- pipeline commit、容器和 config hash；
- tokenizer 版本；
- filter / dedup / decontam registry 版本；
- split policy；
- 每阶段按来源、语言、领域统计的 docs / bytes / tokens；
- removal reasons；
- target mixture 与 observed mixture；
- 已知限制；
- license / PII / provenance completeness；
- withdrawal 与 tombstone 策略；
- 下游训练 run IDs。

Croissant 可用 JSON-LD 表达机器可读的数据集元数据、资源、版本、checksum 和 provenance 关系；W3C PROV 可表达 `wasDerivedFrom`、`wasGeneratedBy` 和 `wasAttributedTo` 等 lineage。

---

## 34. Provenance 不是法律结论

必须区分：

- URL 可公开访问；
- `robots.txt` 允许抓取；
- 页面或数据集有明确许可证；
- 许可证覆盖底层所有内容；
- 预期训练用途符合条款和适用法律。

这些不是同一件事。

Provenance 的职责是记录可审计事实和证据，不是自动替代法律判断。对于不确定项应标为 `unknown`、`restricted` 或待审查，并保留证据时间，因为网页条款可能变化。

Data Provenance Initiative 的研究也说明，数据托管页上的许可与 attribution 信息可能缺失或不准确，不能只复制一个顶层标签就结束审计。

---

## 35. Withdrawal 需要反向索引

为了支持删除、纠错和 opt-out，系统应能查询：

```text
source record
  → extracted / normalized records
  → duplicate cluster and keeper
  → output shards and pack IDs
  → dataset versions
  → training runs
```

数据撤回后应：

1. 生成 tombstone；
2. 构建新的 dataset version；
3. 更新 manifest 和 checksums；
4. 标明哪些模型 run 使用过旧版本。

从 shard 中删除记录不会让已经训练的模型自动“遗忘”。是否需要进一步处置是另一个模型治理问题，不能被文件删除掩盖。

---

## 36. 数据混合中至少有四种“权重”

假设有多个领域或来源。

### 池大小占比

过滤去重后，每个域拥有多少 docs / tokens。

### 采样概率

sampler 选择某个域的概率。

### 实际 Consumed-Token Share

经过长度差异、tokenization、packing、截断和 drop 后，真正送入模型的 token 比例。

### Loss Weight

不同域或样本的 token loss 是否再乘额外权重。

这四者通常不相等。还要额外关注 **loss-bearing-token share**，因为 PAD、prompt mask 和边界 mask 不参与优化。

---

## 37. 温度采样如何重平衡领域

设过滤去重后第 `i` 个域有 `n_i` 个有效 token。按自然数据量的比例为：

```math
p_i
=
\frac{n_i}{\sum_j n_j}
```

常见温度形式：

```math
q_i(\alpha)
=
\frac{p_i^\alpha}{\sum_j p_j^\alpha}
=
\frac{n_i^\alpha}{\sum_j n_j^\alpha}
```

其中：

- $`\alpha = 1`$ ：按数据量采样；
- $`0 < \alpha < 1`$ ：拉平分布，提高小域占比；
- $`\alpha \to 0`$ ：趋向领域均匀；
- $`\alpha > 1`$ ：进一步偏向大域，较少用于低资源重平衡。

必须说明 `n_i` 是文档、原始 token 还是最终有效 token。不同定义会产生不同权重。

---

## 38. 小领域的重复暴露

若总训练预算为 `T` 个 token，则第 `i` 域的平均 token 暴露轮次近似为：

```math
E_i
=
\frac{Tq_i}{n_i}
```

例如一个很小的高质量域即使只拿到 5% 采样权重，也可能被重复几十轮。

过度重复可能导致：

- 记忆；
- 域内过拟合；
- 少量错误被放大；
- 隐私风险增加；
- Benchmark 污染影响扩大。

因此除了 `q_i`，还应为每个域记录：

- unique tokens consumed；
- coverage；
- repeat factor；
- 最大样本暴露次数；
- 是否设置 sampling cap。

---

## 39. 审计实际 Loss-Bearing Token 比例

目标权重 `q_i` 不等于实际训练比例。定义：

```math
\hat q_i
=
\frac{\text{domain }i\text{ 的有效 loss token}}
{\text{全部领域的有效 loss token}}
```

偏差可能来自：

- 文档长度；
- tokenizer fertility；
- truncation；
- packing；
- prompt / answer loss mask；
- 分布式 `drop_last`；
- worker 或 shard 不均；
- checkpoint resume 重复；
- 数据错误重试。

可以使用总变差距离：

```math
TV(q,\hat q)
=
\frac12
\sum_i |q_i-\hat q_i|
```

并按固定 step window 与 DP rank 监控，避免长期平均掩盖局部 drift。

---

## 40. DoReMi 不是“高 Loss 域多采样”

DoReMi 使用小 proxy model 和 reference model，通过 group distributionally robust optimization 学习 domain weights。其核心信号是 proxy 相对 reference 的非负 per-token excess loss，而不是单纯追逐最高原始 loss。

直觉上，它更关注：

> reference 已经能学到、但当前 proxy 仍明显落后的领域。

随后对训练轨迹中的 domain weights 做平均，用来重采样训练大模型的数据。

需要警惕：

- domain taxonomy 改变会改变结果；
- proxy 学到的权重不保证完美迁移到大模型；
- reference 和 proxy 的训练 recipe 会影响权重；
- 高噪声域的高 loss 不应被误解为“值得加权”；
- 不能反复窥视最终 test benchmark 来调 mixture。

DoReMi 是一种系统优化方法，不是一个无需验证的固定配方。

---

## 41. “加入高质量来源”也需要消融

Wikipedia、Books、arXiv 或代码通常被称为高质量，但加入它们不保证任何现有 mixture 都会提升。

原因可能包括：

- 与已有强过滤数据重复；
- 抢占固定 token budget；
- 改变能力之间的 Pareto trade-off；
- tokenizer 或格式不匹配；
- 过度重复小型来源；
- keeper / dedup 改变实际比例。

DCLM 的固定 recipe 实验展示过：额外 curated sources 可以帮助较弱数据基线，但也可能降低一个已经很强的过滤基线的平均成绩。

因此 mixture 应通过小型 proxy runs 验证，并保持：

- 固定训练 token 和 compute；
- 固定 tokenizer 与优化设置；
- 多 seed；
- 各域 validation loss；
- 综合与 worst-domain 指标；
- 实际 loss-token share；
- 最终 test 保持未触碰。

---

## 42. 推荐的构建顺序

```text
1. 冻结 raw snapshots 与来源清单
2. 记录 capture metadata、raw hash
3. 解析并记录 extractor 结果
4. 保守规范化，生成 normalized hash
5. 运行低成本 exact duplicate grouping
6. 计算独立的 quality / language / safety / PII / license signals
7. 运行 MinHash / containment / semantic candidate search
8. 回算相似度，构建 duplicate graph
9. 依据版本化 keeper policy 选代表
10. 按 cluster / entity 做内部 split
11. 用版本化 eval registry 对 train candidate 去污染
12. tokenize，并按最终有效 token 重算 mixture
13. 生成 shards、manifest 与反向索引
14. 训练时审计 observed loss-bearing-token mixture
```

流程中先“打分并隔离”，再依据 policy 构建版本；尽量避免不可恢复的原地删除。

---

## 43. 数据治理 Dashboard

| 类别 | 推荐指标 |
|---|---|
| 库存 | raw / kept docs、bytes、tokens、unique docs |
| 解析 | success rate、空文档、乱码、boilerplate ratio |
| 质量 | score 分布、各 strata retention、阈值附近人工 precision |
| 去重 | exact / near / span duplicate rate、removed tokens |
| 簇结构 | size P50 / P95 / P99 / max、超大簇比例、跨来源边 |
| Keeper | keeper 与 removed 的来源、时间、质量、许可分布差异 |
| 污染 | 每 Benchmark 命中样本率、input-only / input+label、方法分布 |
| Provenance | source、time、hash、parent、version、license 字段完整率 |
| Mixture | planned vs observed docs / tokens / loss tokens |
| 暴露 | unique coverage、repeat factor、最大重复次数 |
| 性能 | 每十亿 token 的 CPU/GPU 小时、I/O、存储与处理吞吐 |

所有指标最好按 language、source、domain、time 和 license status 分层，而不是只看全局均值。

---

## 44. 最小正确性测试

### 测试 1：去重 Toy Set

构造：

- exact copy；
- 只改变空白和换行；
- 大小写和标点变化；
- 共享 boilerplate；
- 长文包含短文；
- 轻度改写；
- 同主题但不同答案；
- 翻译版本。

逐项比较 raw hash、normalized hash、Jaccard、containment 和 embedding 相似度。

### 测试 2：LSH 参数 Sweep

绘制：

```math
1-(1-s^r)^b
```

并在人工标注对上统计 precision / recall 与 retained-token curve。命中和未命中都要分层抽样。

### 测试 3：Cluster Split

验证每个 duplicate cluster、conversation、repository 或其他 entity 只出现在一个 split，计算跨 split exact / near duplicate rate。

### 测试 4：污染 Canary

向训练候选注入：

- exact Benchmark；
- 格式变化；
- question + answer；
- paraphrase；
- translation；
- 同主题 negative controls。

分别测 lexical 和 semantic detector 的 precision / recall。

### 测试 5：Keeper 消融

比较 random、earliest、latest、highest-quality、license-clearest 策略，观察保留来源、时间、语言和 proxy model 结果。

### 测试 6：Mixture 审计

比较 natural、domain-uniform 和多个 $`\alpha`$ ，记录 target `q`、observed $`\hat q`$ 、TV、coverage、repeat factor 与各域验证指标。

### 测试 7：Reproducible Build

同一输入、代码、配置与 seed 重跑，比较 manifest、shard checksums 和每阶段计数。

### 测试 8：Withdrawal Drill

模拟撤回一个 source，确认能定位所有派生记录、shards、dataset versions 和 model run，并产生新版本与 tombstone。

---

## 45. 常见错误

### 错误 1：把质量、重复、污染、安全和许可合成一个分数

它们的证据、目标与处置方式不同，应分别记录。

### 错误 2：低 Perplexity 就是高质量

低分也可能来自模板、重复或被参考模型记忆的文本。

### 错误 3：用 Python `hash()` 做永久文档 ID

它不保证跨运行稳定，位数也可能不适合大规模身份。

### 错误 4：把 Bloom Filter 当近重复检测

它解决近似成员查询，不估计语义或 Jaccard 相似度。

### 错误 5：LSH 命中就直接删除

LSH 是概率候选器，应回算相似度并审计 false positives。

### 错误 6：把 MinHash 阈值说成“字面有 75% 完全相同”

它依赖 shingle、Jaccard、签名和 band 参数。

### 错误 7：Connected Component 内任意两条都足够相似

链式合并不保证簇内所有 pair 满足阈值。

### 错误 8：随机选择 Keeper

keeper 会改变时间、来源、质量和许可分布，必须确定且可解释。

### 错误 9：去重越全局、删除越多越好

范围和 keeper policy 都会改变训练分布，应以固定预算消融判断。

### 错误 10：去重等于去污染

一个唯一文档也可能完整包含 Benchmark 答案。

### 错误 11：未被 n-gram 检出就是无污染

改写、翻译和合成样本可以绕过字符串检测。

### 错误 12：只看 Clean Subset 就证明污染无影响

clean 与 flagged 子集可能难度不同，需要更强的训练消融。

### 错误 13：Provenance 就是一个 URL

URL 会变化，也无法表达转换版本、父记录、决策和输出 lineage。

### 错误 14：公开网页自动等于许可明确

公开访问、抓取规则、许可证和训练使用条件不是同一概念。

### 错误 15：配置权重就是实际训练权重

长度、packing、mask 和分布式丢弃都会改变 loss-bearing token 占比。

### 错误 16：高质量小域可以无限过采样

小域会重复暴露，增加记忆、过拟合和污染放大风险。

---

## 46. 概念检查

### 问题 1

为什么“保留率最高”或“删除最多”都不是质量流水线的目标？

**答案：** 二者都没有衡量模型效果、覆盖、公平性、风险与处理成本。真正目标是在明确约束下改善训练分布，并用固定预算实验验证。

### 问题 2

MinHash 与 SHA-256 的角色有什么不同？

**答案：** SHA-256 可作为稳定内容摘要并用于精确匹配；MinHash 是对 shingle 集合的概率签名，用于估计 Jaccard 和生成近重复候选。

### 问题 3

为什么 LSH 候选还要回算真实相似度？

**答案：** banding 是概率检索，会产生 false positives；命中只表示“值得比较”，不等于已满足业务阈值。

### 问题 4

两个文档在同一 connected component，是否保证它们直接相似？

**答案：** 不保证。它们可能通过一条 A~B~C 的边链间接连接。

### 问题 5

为什么 input-only contamination 和 input+label contamination 应分开？

**答案：** 公共背景或问题出现过不一定泄漏答案；完整答案、解析或测试实现通常对评测有效性的威胁更直接。

### 问题 6

为什么过滤去重后必须重算 mixture？

**答案：** 各域删除比例不同，原始池大小和目标权重已不再代表最终有效 token 分布。

### 问题 7

$`\alpha < 1`$ 为什么可能让小域过拟合？

**答案：** 它提高小域的采样比例；当域大小 `n_i` 很小时，`E_i=Tq_i/n_i` 会变大，同一 token 被重复很多轮。

### 问题 8

Provenance 为什么不能自动给出法律结论？

**答案：** 它记录来源、变换和证据；证据如何适用于具体用途还涉及许可证范围、条款与适用法律的解释。

---

## 47. 今日练习

### 练习 1：实现保守 Exact Dedup

同时输出 raw SHA-256、normalized SHA-256、canonicalizer version 和 collision verification。比较不同规范化规则误删了哪些代码或数学样本。

### 练习 2：实现 MinHash + LSH 小实验

对 100 个 toy documents：

1. 构造 shingles；
2. 计算 MinHash；
3. 扫描不同 `b`、`r`；
4. 回算 Jaccard；
5. 与人工 duplicate label 比较 precision / recall。

### 练习 3：分析 Duplicate Graph

用 connected components 建簇，找出 A~B、B~C、A!~C 的链式案例，并比较 random keeper 与 quality-aware keeper 的分布差异。

### 练习 4：建立 Eval Registry

为一个小 Benchmark 保存 input、input+label、rationale 和 verbalized prompt 的多个视图，检测 exact、n-gram、containment 与 paraphrase canary。

### 练习 5：制作 Provenance Manifest

选择 20 条文本，记录从 raw source 到 final shard 的所有 parent、activity、version、reason code 和 checksum。模拟一条数据撤回。

### 练习 6：计算 Mixture 与 Exposure

给定三个域：

```text
A: 1,000,000 tokens
B:   100,000 tokens
C:    10,000 tokens
```

分别计算 $`\alpha = 1.0`$ 、 $`\alpha = 0.5`$ 、 $`\alpha = 0.3`$ 的 `q_i`，并在 `T = 10,000,000` 时计算 `E_i`。解释哪个域最容易被重复记忆。

### 练习 7：做固定预算过滤消融

比较 raw、heuristic、classifier 和 combined 四个版本，固定模型、tokenizer、总训练 token 与 steps，报告两次随机种子、unique coverage、repeat factor、各域 loss 与综合指标。

---

## 48. 参考资料

- [Gopher / MassiveText：Scaling Language Models](https://arxiv.org/abs/2112.11446)
- [FineWeb：The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [FineWeb Dataset Card 与可复现实验资源](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [NVIDIA NeMo Curator：Exact、Fuzzy 与 Semantic Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication)
- [DataComp-LM：数据过滤与固定 Recipe 消融](https://arxiv.org/abs/2406.11794)
- [Rethinking Benchmark Contamination with Rephrased Samples](https://arxiv.org/abs/2311.04850)
- [GPT-3：Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [DoReMi：Optimizing Data Mixtures](https://arxiv.org/abs/2305.10429)
- [W3C PROV Model Primer](https://www.w3.org/TR/prov-primer/)
- [MLCommons Croissant 1.1 Specification](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html)
- [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

---

## 今日总结

今天把训练数据从“文本集合”提升为“可验证、可复现、可治理的构建产物”：

1. 质量、重复、污染、安全和许可是五类不同问题；
2. 质量不是绝对标量，过滤器必须按语言、领域和群体分层校准；
3. 精确 hash、Bloom Filter、MinHash + LSH 和 semantic dedup 角色不同；
4. LSH 只产生候选，duplicate graph 还存在链式合并；
5. 去重范围和 keeper policy 会改变来源、时间与领域分布；
6. 去重不等于去污染，污染还包括答案、解析、改写和翻译；
7. “未检出”只能说明某个检测器没命中，不能证明无污染；
8. Provenance 应记录 Entity、Activity、Agent 与完整 lineage；
9. mixture 配置必须落到实际 loss-bearing token share 和重复暴露上；
10. 所有数据决策都应通过固定训练预算、分层人工审查和可复现构建验证。

最核心的一句话是：

> 好的数据集不是“看起来很干净的一堆文本”，而是每一条保留、删除、去重、去污染和采样决策都有证据，并能追溯到最终训练结果的版本化系统。

## 明日衔接建议

从“如何证明训练数据可靠”继续进入“训练出现异常时如何定位到数据与数值计算”：学习 loss spike、梯度范数异常、混合精度 overflow、NaN / Inf 检测、异常 batch 回溯、自动跳过与 checkpoint 恢复策略。