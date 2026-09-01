# 2026-09-01：训练数据流水线——Packing、Shuffling、Prefetch 与可恢复采样

## 今日目标

昨天学习了如何用 MFU、Roofline 和 Profiler 判断训练为什么慢。今天沿着“GPU 为什么会等待数据”继续向前，建立一条既高吞吐、又不破坏训练语义的数据流水线。

完成后应能回答：

1. 原始文档如何变成送入模型的定长 token batch？
2. padding、拼接连续 token 流和独立样本 packing 有什么区别？
3. EOS、attention mask、loss mask、position ids 分别解决什么问题？
4. DDP rank 与 DataLoader worker 如何分片，才能避免重复或漏样本？
5. shuffle、随机种子和 checkpoint 需要保存到什么程度，才能可靠恢复？
6. `pin_memory`、`prefetch_factor` 和 `non_blocking=True` 为什么不一定让训练更快？
7. 应该用哪些指标和小测试验证流水线，而不是只看 GPU 利用率？

---

## 1. 数据流水线是训练系统的一部分

训练 step 并不只有前向和反向。一个 batch 到达 GPU 前，通常要经历：

```text
原始语料
  ↓
解析 / 规范化 / 质量过滤 / 去重
  ↓
稳定划分 train / validation / test
  ↓
tokenization + 文档边界
  ↓
采样 / 数据混合 / shuffle
  ↓
截断 / 切块 / packing
  ↓
collate 成 batch
  ↓
worker 预取 + pinned memory
  ↓
Host-to-Device copy
  ↓
模型前向、反向与参数更新
```

任何一段跟不上，GPU 都可能等待；任何一段语义错误，即使吞吐很高，也可能训练出错误的模型。

因此数据流水线同时有两个目标：

- **正确性**：没有意外泄漏、重复、漏读、跨样本 attention 或错误 loss。
- **性能**：尽量提高真正参与 loss 的 token 吞吐，并让 I/O、CPU、H2D 与 GPU 计算重叠。

---

## 2. 离线处理与在线加载

可以把工作分成两层。

### 离线层

适合提前完成且可复用的工作：

- 解压、解析与文本规范化；
- 质量过滤、语言识别、去重；
- train / validation / test 划分；
- tokenization；
- 生成稳定的样本 ID、长度和来源元数据；
- 写成 Arrow、Parquet 或自定义二进制 shard；
- 为 shard 建立版本号、校验和与索引。

优点是训练时 CPU 压力小、结果容易复现。缺点是占用存储，改变 tokenizer 或清洗规则后通常需要重新物化。

### 在线层

适合随 epoch 或实验配置变化的工作：

- 数据集混合比例；
- shuffle 与采样；
- 长度 bucketing；
- packing；
- 动态 padding；
- 数据增强；
- rank / worker 分片；
- prefetch 和 H2D。

一个实用原则是：**把确定且昂贵的变换尽量离线化，把需要随机性或实验灵活性的变换留在线上。**

---

## 3. 先建立稳定的样本身份

如果样本只有“当前是第几行”，重新清洗或重新分 shard 后，它的身份就可能变化。更可靠的做法是为文档记录稳定 ID，例如由以下信息构成：

```text
source_id + document_id + content_fingerprint + preprocessing_version
```

其中：

- `source_id` 表示来源；
- `document_id` 表示来源内的稳定标识；
- `content_fingerprint` 用于发现内容变化或重复；
- `preprocessing_version` 记录清洗、分词和切块规则。

稳定 ID 的用途不只是排错，还包括：

- 检查 train / validation 泄漏；
- 统计重复与遗漏；
- 记录失败样本；
- 精确比较恢复训练前后的样本序列；
- 在 worker 数改变时，用样本 ID 派生确定性随机数。

---

## 4. 必须先划分数据，再切块和 packing

错误流程：

```text
所有文档 → 切成窗口 → 随机划分 train / validation
```

同一文档的相邻窗口可能被分到不同集合，造成近乎重复的验证样本。对代码、对话或用户数据，泄漏单位还可能是 repository、conversation、user 或时间段，而不只是单个文本文件。

更安全的流程：

```text
按 document / conversation / repository / user / time 分组
  ↓
稳定划分 train / validation / test
  ↓
分别执行 tokenization、切块与 packing
```

还应做两类审计：

1. 精确 fingerprint 是否跨 split 重复；
2. near-duplicate 是否跨 split 高度相似。

shuffle 只能改变顺序，不能修复数据泄漏。

---

## 5. Tokenization 也是数据契约

复现实验时，只有 tokenizer 名字不够。至少应记录：

- tokenizer 的精确版本或文件 hash；
- vocabulary 和 merges；
- Unicode 规范化、大小写与空白处理；
- BOS、EOS、PAD、UNK 的 ID；
- 是否由 tokenizer 自动添加特殊 token；
- 截断侧与 padding 侧；
- 最大序列长度；
- 预处理代码版本。

一个常见错误是 tokenizer 已经通过 `add_special_tokens=True` 添加 BOS / EOS，而 packing 代码又添加一次，导致边界重复。

离线 tokenization 更稳定且训练时更快；在线 tokenization 更灵活，但必须考虑 CPU 开销、多进程可序列化和随机性。

---

## 6. 文档边界不能只靠“记得它在这里”

设两个文档分词后为：

```text
A = [a1, a2, EOS]
B = [b1, b2, EOS]
```

直接拼接会得到：

```text
[a1, a2, EOS, b1, b2, EOS]
```

EOS 明确告诉模型“上一个文档结束了”，但它本身不自动意味着：

- 后一个文档不能 attention 到前一个文档；
- position ids 会重置；
- `EOS → b1` 这一步不参与 loss；
- kernel 会跳过跨文档的计算。

因此要把四件事分开：

1. **token boundary**：BOS / EOS 等特殊 token；
2. **attention boundary**：哪些 token 可以互相看见；
3. **loss boundary**：哪些 next-token 目标参与优化；
4. **position boundary**：位置编号是否在新样本处重置。

它们可以采用不同策略，不能互相替代。

---

## 7. 三种常见的 batch 组织方式

### 方式 A：Padding

每个样本保持独立，在同一个 batch 内补齐到最长长度：

```text
A: [a1, a2, EOS, PAD, PAD]
B: [b1, b2, b3,  EOS, PAD]
```

优点：

- 语义简单；
- 容易构造 attention mask 和 labels；
- 适合 SFT 与长度差异不大的数据。

缺点：

- PAD token 会占用显存和计算；
- batch 内长度差异越大，浪费越严重。

动态 padding 只补到当前 batch 的最长样本，通常比固定补到全局最大长度更省。

### 方式 B：连续 Token Stream

将大量文档通过 EOS 连接成一个长 token 流，再按固定长度切块：

```text
[a1, a2, EOS, b1, b2, EOS, c1, ...]
              ↓ 每 T 个 token 切一段
```

这常用于预训练，几乎没有 padding，且可能有意允许跨文档上下文。它的训练语义与“每篇文档完全独立”不同，不能悄悄替换。

### 方式 C：独立样本 Packing

把多个短样本放入同一个物理序列，但保留独立边界：

```text
physical row: [A tokens | B tokens | C tokens]
segments:     [0 0 0 0 | 1 1 1 | 2 2 2 2]
```

如果要求样本相互独立，则需要 segment-aware 或 block-diagonal causal attention，并单独处理 position ids 与边界 loss。

---

## 8. EOS 不是 Attention Mask

普通 causal mask 只阻止“看未来”，不会阻止 B 看 A：

```text
        keys →
query   A A A | B B
A       ✓ ✓ ✓ | ✗ ✗
B       ✓ ✓ ✓ | ✓ ✓   ← 普通 causal mask 仍允许 B 看 A
```

独立 packing 需要的逻辑更接近：

```text
        keys →
query   A A A | B B
A       ✓ ✓ ✓ | ✗ ✗
B       ✗ ✗ ✗ | ✓ ✓
```

普通二维 padding mask 只能表示“真实 token / PAD token”，通常不能表达序列内部的多个独立段。实现时常需要：

- sequence / segment IDs；
- 每段长度；
- `cu_seqlens`；
- 真正的 block-diagonal mask；
- 支持 variable-length packed attention 的 kernel。

是否隔离必须依据训练目标决定：

- 通用预训练可能接受 EOS 连接后的跨文档上下文；
- SFT、偏好数据或要求样本独立的任务通常应隔离。

---

## 9. Loss Mask 与 Attention Mask 是两张不同的表

对于 causal LM，常见做法是复制 `input_ids` 为 `labels`，再把不应计入 loss 的位置设为 `ignore_index`，通常是 `-100`。

需要 mask 的位置可能包括：

- PAD；
- prompt 部分，仅训练 answer；
- packing 边界的某些 transition；
- 数据格式中的控制 token；
- 无效或被截断的监督区域。

如果希望 A 与 B 完全独立，还要避免训练 `EOS_A → first_token_B`。在“模型内部自动右移 labels”的实现中，通常要 mask B 的第一个 label，从而去掉这个 transition；具体索引必须用一个两文档 toy example 验证。

特别注意：如果 `pad_token_id == eos_token_id`，不能仅凭 token ID 推断哪里是 padding。必须保留显式 attention mask、sequence lengths 或有效 token mask，否则会把真实 EOS 误删。

---

## 10. Position IDs 也要单独决定

独立 packing 中常见两种位置策略：

### 连续位置

```text
A: 0 1 2 3
B: 4 5 6
```

### 每段重置

```text
A: 0 1 2 3
B: 0 1 2
```

重置 position ids 并不能自动阻断 attention；阻断 attention 也不必然要求重置位置。两者是独立设计选择。

此外，有些线性 attention、卷积或状态空间结构不能从 position ids 推断段边界，此时必须显式传递边界信息，不能照搬标准 Transformer 的假设。

---

## 11. Packing 的语义优化与物理加速不同

即使构造了 block-diagonal mask，如果底层仍计算一个稠密的 `T × T` attention 矩阵，跨样本位置虽然被 mask，FLOPs 仍可能已经花掉。

若一个物理序列包含长度 `L_1, L_2, ..., L_n` 的独立样本：

- 稠密 attention 规模约为 `T^2`，其中 `T = \sum_i L_i`；
- 理想 varlen / block-diagonal attention 规模约为 `\sum_i L_i^2`。

因此 packing 有两个层次：

1. **layout 利用率**：减少 PAD，容纳更多真实 token；
2. **kernel 利用率**：让底层 kernel 真正跳过无效 attention 区域。

不要因为“mask 正确”就假定“计算也省掉了”。

---

## 12. 长文档如何处理

当文档长于 `sequence_length` 时，常见策略包括：

- truncate：简单，但丢失尾部；
- drop：只适合极少数异常样本；
- non-overlapping chunks：覆盖完整、没有重复；
- sliding windows：保留边界上下文，但会重复 token；
- document-aware segmentation：按段落、对话轮次或代码结构切分。

必须记录：

- chunk 来源文档与 chunk index；
- 是否重叠、重叠长度；
- 边界是否添加 EOS；
- labels 中哪些重叠 token 被屏蔽；
- split 是否早于 chunking。

否则样本数增加并不等于新增了同等数量的信息。

---

## 13. Length Bucketing 的收益与代价

把相近长度样本放入同一 batch，可以显著降低 padding ratio。但如果只按长度排序而不再随机化，会让训练顺序与长度强相关。

较稳妥的方法是：

1. 先全局或分 shard shuffle；
2. 在有限窗口内按长度分桶；
3. 在桶内随机组 batch；
4. 对 batch 顺序再次扰动。

需要同时监控：

- padding 降低了多少；
- batch 的长度分布是否过于集中；
- 数据来源与长度是否相关；
- 训练收敛是否受到排序偏差影响。

---

## 14. 数据混合：按样本还是按 Token

假设数据集 A 多为长样本，B 多为短样本。

如果以 50% 概率抽“一个样本”，A 贡献的 token 可能远高于 50%。因此要区分：

- **sample-uniform mixing**：按样本数控制比例；
- **token-uniform mixing**：按 token 数控制比例；
- **loss-token mixing**：按真正参与 loss 的 token 控制比例。

训练日志不应只记录配置中的采样权重，还应统计每个来源实际贡献的：

- 原始样本数；
- non-PAD tokens；
- loss-bearing tokens；
- 丢弃、截断和重复数量。

最终优化的是 token 上的 loss，实际 token 比例比名义样本比例更重要。

---

## 15. Shuffle 有多个层级

“shuffle 数据”可能指完全不同的事情：

1. 文档 / 样本顺序 shuffle；
2. shard 顺序 shuffle；
3. shard 内样本 shuffle；
4. pack 或 batch 顺序 shuffle；
5. 流式 shuffle buffer。

不要把它表述成“把 token 打乱”。自回归样本内部的 token 顺序不能被随机置换，否则语言建模目标会被破坏。

对可随机访问数据集，可以生成近似全局 permutation。对超大流式数据，通常只能：

- 先打乱 shard；
- 使用大小为 `K` 的 buffer；
- 随机弹出 buffer 中一个元素，再补入新元素。

buffer shuffle 是近似随机，不等价于整个数据集的均匀全排列。buffer 越大，随机性通常越好，但占用更多内存，恢复状态也更复杂。

---

## 16. Map-Style 与 Iterable-Style Dataset

### Map-Style

实现 `__len__` 和 `__getitem__`，适合：

- 数据可随机访问；
- 长度已知且稳定；
- 使用 `DistributedSampler`；
- 精确构造 permutation；
- 通过索引定位样本。

### Iterable-Style

实现 `__iter__`，适合：

- 数据来自流式对象存储或网络；
- 数据量极大；
- 在线生成或解码；
- 长度未知或随机访问成本高。

多 worker 使用 `IterableDataset` 时，每个 worker 都有自己的 Dataset 副本。若不显式分片，多个 worker 可能完整重复同一数据流。此时必须使用 `get_worker_info()` 或数据框架提供的 shard 机制。

`len(dataloader)` 对 IterableDataset 通常只是估计值，不能据此断言没有重复或遗漏。

---

## 17. DDP 不会自动替你切分输入

DistributedDataParallel 负责同步梯度，不会自动保证不同 rank 获得不同数据。

对长度稳定的 map-style dataset，可使用 `DistributedSampler`：

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=dp_size,
    rank=dp_rank,
    shuffle=True,
    seed=base_seed,       # 所有 DP rank 使用同一个 seed
    drop_last=True,
)

loader = DataLoader(
    dataset,
    sampler=sampler,
    shuffle=False,        # 指定 sampler 后不要再开 shuffle
    ...
)

for epoch in range(start_epoch, num_epochs):
    sampler.set_epoch(epoch)  # 必须在创建本轮 iterator 前调用
    for batch in loader:
        train_step(batch)
```

若忘记 `set_epoch(epoch)`，每个 epoch 可能重复相同顺序。

在 TP / PP / EP 等多维并行中，应按 **data-parallel replica group** 分片，而不是盲目使用全局 `world_size`。同一个模型副本内部的 TP / PP ranks 通常需要处理同一批数据，而不同 DP replicas 才应拿不同数据。

---

## 18. Rank 分片与 Worker 分片是两层问题

流式加载需要同时考虑：

```text
全局数据
  ├─ DP rank 0
  │    ├─ worker 0
  │    └─ worker 1
  └─ DP rank 1
       ├─ worker 0
       └─ worker 1
```

一种简单的全局 worker ID 是：

```python
global_worker_id = dp_rank * workers_per_rank + worker_id
global_worker_count = dp_size * workers_per_rank

for shard in shards[global_worker_id::global_worker_count]:
    yield from read_shard(shard)
```

优先在文件 / shard 层分片，而不是让所有 worker 打开同一文件后逐行判断，因为前者 I/O 更高效。

还要考虑：

- shard 大小是否严重不均；
- worker 数是否在所有 rank 相同；
- 小 shard 数是否足够分配；
- 故障恢复后 world size 是否变化；
- 热点数据源是否让某些 rank 成为 straggler。

---

## 19. `drop_last` 可能指不同层

`DistributedSampler(drop_last=...)` 与 `DataLoader(drop_last=...)` 不是同一件事。

- sampler 的 `drop_last=False`：为了让各 rank 索引数相同，可能补充额外索引，因此一个 epoch 中可能出现重复样本；
- sampler 的 `drop_last=True`：丢弃不能均分到各 rank 的尾部索引；
- DataLoader 的 `drop_last=True`：丢弃每个采样序列末尾不足一个 batch 的样本。

IterableDataset 多 worker 下，尾部行为还可能发生在每个 worker 副本上。

所以“每个 epoch 恰好看一次所有样本”“所有 rank step 数相同”“完全不重复”这三个目标并不总能同时满足，需要明确优先级并用 ID 审计。

---

## 20. 随机种子需要分层管理

常见随机源包括：

- sampler shuffle；
- Python `random`；
- NumPy；
- PyTorch CPU；
- PyTorch CUDA；
- tokenizer / augmentation；
- packing 与 shuffle buffer；
- 各个 worker。

一个最小 worker 初始化方式：

```python
import random
import numpy as np
import torch

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
```

再把显式 `torch.Generator` 传给 DataLoader：

```python
loader_generator = torch.Generator()
loader_generator.manual_seed(base_seed + dp_rank)
```

注意 sampler 的 shuffle 通常发生在主进程，不由 `worker_init_fn` 控制。

如果希望改变 `num_workers` 后仍得到稳定的样本级增强，最好不要只依赖 worker ID，而是从以下稳定键派生随机性：

```text
(global_seed, epoch, stable_sample_id, transform_name)
```

---

## 21. `persistent_workers=True` 改变了生命周期

开启后，worker 不会在每个 epoch 结束时销毁，能减少进程启动和 Dataset 重建成本。但同时：

- worker 内的 Dataset 状态会继续存在；
- `worker_init_fn` 不会每个 epoch 重跑；
- 自定义的 epoch 级随机状态不会自动更新；
- 文件句柄、缓存和内存也会继续保留。

如果 IterableDataset 的行为依赖 epoch，需要通过共享状态、显式 `set_epoch` 或框架提供的机制把新 epoch 传入 worker，而不能假设初始化函数会再次执行。

---

## 22. `set_epoch` 不等于精确断点续训

`sampler.set_epoch(epoch)` 可以让同一 epoch 的 permutation 可重建，但它无法表示“本 epoch 已经消费到第几个 pack”。

可靠 checkpoint 至少要考虑保存：

- dataset 与 preprocessing 版本；
- epoch；
- 已提交的 global step / microstep；
- sampler permutation 或可重建它的 seed；
- sampler / iterable dataset cursor；
- 当前 shard 与 shard 内 offset；
- shuffle buffer 内容及 RNG；
- 未完成 pack 的 token 与边界状态；
- Python、NumPy、Torch CPU / CUDA RNG；
- DataLoader generator 状态；
- gradient accumulation 的当前位置。

这里“已提交”很重要：prefetch 可能让 loader 的生产游标领先于 GPU 已完成的 batch。若 checkpoint 保存的是“已经预取到哪里”，恢复后可能跳过尚未真正训练的数据。

实践中要区分：

- **bitwise / exact resume**：恢复后下一个样本、pack 和随机操作完全一致；
- **coverage-preserving resume**：允许顺序略变，但尽量不丢失或大量重复数据；
- **epoch-level restart**：从本 epoch 开头重跑，简单但会重复。

精确恢复通常还要求 dataset 版本、world size、worker 数和 packing 配置不变。弹性训练改变拓扑后，往往只能追求覆盖语义，而非逐样本完全一致。

---

## 23. Prefetch 的真实含义

PyTorch 中，`prefetch_factor` 表示每个 worker 预取的 batch 数。粗略的排队量为：

```text
num_workers × prefetch_factor
```

当 `num_workers > 0` 时，默认通常为 2。

增加预取可能隐藏 I/O 或 CPU 抖动，但也会增加：

- 普通 RAM；
- shared memory；
- pinned memory；
- 已经推进但尚未消费的数据状态；
- checkpoint 精确恢复难度。

若 batch 很大，`8 workers × prefetch 4` 就可能在队列里保留约 32 个 batch。参数不是越大越好，应通过 GPU 等待时间、RAM 和吞吐联合测量。

`in_order=False` 允许 batch 不按先进先出返回，某些慢样本场景可能更快，但会降低可复现性，并可能让不均衡数据的到达分布发生偏斜，默认不要为了少量吞吐轻易打开。

---

## 24. `pin_memory` 为什么有用

GPU 从 page-locked（pinned）CPU memory 发起 DMA 传输更合适。常见组合是：

```python
loader = DataLoader(..., pin_memory=True)

for batch in loader:
    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }
```

DataLoader 的 pinning 线程能在训练主线程之外处理常见 Tensor 容器。不要在主循环中临时对每个 batch 调 `tensor.pin_memory()` 并假定一定更快，因为 pin 操作本身可能阻塞。

自定义 batch 类型若不被默认 pinning 逻辑识别，需要实现自己的 `pin_memory()`。

pinned memory 是有限系统资源。预取过大或 batch 过多会占用大量不可分页内存，反而降低整体性能。

---

## 25. `non_blocking=True` 不保证 Copy 与 Compute 重叠

`non_blocking=True` 通常能避免 host 在每次 copy 后立即等待，但要让 H2D 真正和 GPU compute 重叠，还通常需要：

1. 源 Tensor 位于 pinned memory；
2. copy 使用独立的非默认 CUDA stream；
3. GPU 支持并有可用 DMA copy engine；
4. 至少使用双缓冲；
5. 正确建立 stream 依赖，计算不能在 copy 完成前读取数据；
6. 不在 copy 完成前修改或复用 CPU buffer。

概念上的双缓冲：

```text
时间 →
copy stream:     copy batch n+1      copy batch n+2
compute stream:      compute n           compute n+1
```

因此应在 profiler timeline 上确认 overlap，而不是只看到代码里写了 `non_blocking=True` 就宣布成功。

---

## 26. 一个可复现的 Map-Style DataLoader 骨架

```python
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def build_loader(
    dataset,
    *,
    batch_size,
    base_seed,
    dp_rank,
    dp_size,
    num_workers,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=base_seed,
        drop_last=True,
    )

    generator = torch.Generator()
    generator.manual_seed(base_seed + dp_rank)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=causal_lm_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )
    return loader, sampler

loader, sampler = build_loader(...)

for epoch in range(start_epoch, num_epochs):
    sampler.set_epoch(epoch)

    for batch in loader:
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        loss = model(**batch).loss
        loss.backward()
```

这只是安全起点，不是通用最优配置。需要根据 CPU 核数、存储、batch 大小、NUMA、GPU 数量和数据变换成本做基准测试。

---

## 27. 一个最小 Causal LM Collator

下面示例展示动态 padding 与 PAD loss mask；它没有实现多样本 block-diagonal packing：

```python
def causal_lm_collate(samples, pad_token_id, ignore_index=-100):
    max_len = max(len(sample["input_ids"]) for sample in samples)

    input_ids = []
    attention_mask = []
    labels = []

    for sample in samples:
        ids = sample["input_ids"]
        pad_len = max_len - len(ids)

        row_ids = ids + [pad_token_id] * pad_len
        row_mask = [1] * len(ids) + [0] * pad_len
        row_labels = ids + [ignore_index] * pad_len

        input_ids.append(row_ids)
        attention_mask.append(row_mask)
        labels.append(row_labels)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
```

真实项目还需要确认：

- 模型是在内部 shift labels，还是外部已经 shift；
- prompt token 是否参与 loss；
- BOS / EOS 是否重复；
- PAD 与 EOS 是否共用 ID；
- truncation 后最后一个 token 的语义；
- packing 边界的 loss 与 attention 是否正确。

---

## 28. Hugging Face Streaming 的最小心智模型

对于超大数据，可以流式读取并先 shuffle shard，再进行有限 buffer shuffle。概念示例：

```python
from datasets import load_dataset

dataset = load_dataset(
    "some_dataset",
    split="train",
    streaming=True,
)

dataset = dataset.shuffle(
    seed=base_seed,
    buffer_size=10_000,
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)
    for sample in dataset:
        ...
```

需要记住：

- buffer shuffle 不是全局 permutation；
- worker 必须拿到不同 shards；
- shard 数要足够多；
- `take` / `skip` 会影响或锁定 shard 顺序，应先理解操作顺序；
- 流式状态恢复可能无法保留 buffer 内所有已预取样本，是否“精确恢复”要查所用框架的明确保证。

如果本地已有物化的 Arrow Dataset，转换为有多个 shards 的 IterableDataset 往往比直接远程 streaming 更容易发挥多 worker 并行。

---

## 29. Packing 算法应输出的不只是 `input_ids`

一个健壮的 packer 至少应能输出或重建：

- `input_ids`；
- 有效 token mask；
- `labels` / loss mask；
- 每段长度；
- sequence / segment IDs；
- position ids；
- 原始 sample IDs；
- pack ID；
- 被截断、剩余或跨 pack 的状态。

伪代码：

```python
for sample in shuffled_samples:
    tokens = add_boundaries(tokenize(sample))

    while tokens:
        space = sequence_length - current_pack.length
        piece = tokens[:space]

        current_pack.append(
            tokens=piece,
            sample_id=sample.id,
            segment_id=current_segment,
        )
        tokens = tokens[space:]

        if current_pack.is_full():
            yield finalize_masks_labels_and_positions(current_pack)
            current_pack = new_pack()
```

真正困难的部分不是把列表拼满，而是明确：

- 一个样本能否跨 pack；
- 跨 pack 是否补 EOS；
- 新 segment 的 position 是否重置；
- attention 是否跨 segment；
- 边界 transition 是否计入 loss；
- checkpoint 时未完成 pack 如何保存。

---

## 30. 多 Rank 的 Loss 归一化陷阱

如果每个 rank 的有效 loss token 数不同，而每个 rank 先做本地 mean loss，DDP 再等权平均梯度，那么“有效 token 较少的 rank”会被赋予过高权重。

正确目标应是：

```text
global_loss =
    所有 DP ranks 的 token loss 总和
    ÷
    所有 DP ranks 的有效 loss token 总数
```

因此应全局汇总 loss sum 和有效 token count，或对本地 token-sum loss 按全局计数正确缩放。只有当各 rank 的有效 token 数严格相同时，本地 mean 后等权平均才与全局 token mean 等价。

这也是为什么应记录 **loss-bearing tokens**，而不只记录 batch size。

---

## 31. 应该监控哪些利用率

### Layout Utilization

```text
non-PAD tokens / allocated token slots
```

衡量 padding 浪费。

### Supervised Utilization

```text
labels != ignore_index 的 token / allocated token slots
```

SFT 中大量 prompt token 不参与 loss，因此它可能远低于 layout utilization。

### Pack Fill Rate

```text
每个 pack 的真实 token 数 / sequence_length
```

衡量 packer 是否经常留下空洞。

### Effective Token Throughput

```text
所有 DP ranks 的 loss-bearing tokens / wall-clock second
```

这通常比 samples/s 更接近训练真正完成的工作量。

### Attention Useful-Compute Ratio

若稠密 batch 里包含多个独立段，可观察近似比例：

```text
sum(L_i^2) / (B × T^2)
```

它帮助判断“减少 padding”是否真的转化为 attention kernel 的计算节省。

还应记录：

- DataLoader wait time；
- H2D copy time；
- GPU idle gap；
- CPU 利用率；
- 磁盘 / 网络吞吐；
- RAM、shared memory、pinned memory；
- duplicate / drop / truncation rate；
- 各数据来源的实际 loss-token 比例。

---

## 32. 如何定位 GPU 等数据

可以按以下顺序排查：

1. **先在 timeline 上找 GPU 空白**：确认空白发生在取 batch、H2D 还是同步点。
2. **测单独 DataLoader 吞吐**：不运行模型，只迭代并统计 tokens/s。
3. **测纯模型吞吐**：使用预先放在 GPU 上的固定 batch，排除输入流水线。
4. **比较两者**：若 loader 低于模型消耗速度，GPU 一定会被饿。
5. **逐个改变变量**：`num_workers`、batch size、prefetch、pinning、在线 tokenization。
6. **观察资源上限**：CPU、I/O、RAM、`/dev/shm`、pinned memory 和 NUMA。
7. **重新看 timeline**：确认优化是否真的缩小了 gap。

一个非常实用的指标是：

```text
data_wait_ratio = 等待下一个 batch 的时间 / step wall time
```

但要注意 CUDA 异步执行：测量前要明确同步边界，或直接借助 profiler timeline。

---

## 33. DataLoader 参数不是越大越好

### `num_workers`

过少时 CPU 供不上；过多时可能出现：

- 进程调度与上下文切换；
- CPU 过度订阅；
- 每个 worker 复制父进程对象带来的内存增长；
- 文件句柄与 I/O 争用；
- 小文件随机读放大。

### `prefetch_factor`

过小隐藏不了抖动；过大会占用大量内存，并让恢复游标更难定义。

### `persistent_workers`

能减少每轮重启成本，但会保留状态和资源。

### `pin_memory`

可能提高 H2D 效率，但若 batch 很小、CPU 内存紧张或 pipeline 已不受 H2D 限制，收益可能有限甚至为负。

正确方法是固定其他变量，扫描一组配置并同时记录吞吐、等待时间和资源占用，而不是照抄某个“最佳 worker 数”。

---

## 34. 最小正确性测试

性能测试前，先运行小规模确定性测试。

### 测试 1：两文档 Toy Pack

手工构造 A、B 两段，逐位置检查：

- input tokens；
- segment IDs；
- attention；
- labels；
- loss mask；
- position ids；
- `EOS_A → first_B` 是否符合预期。

### 测试 2：跨 Rank 覆盖

收集每个 rank 实际消费的 sample / pack IDs，检查：

- union 是否等于预期集合；
- intersection 是否意外非空；
- duplicate rate；
- drop rate；
- 各 rank 数量是否一致。

### 测试 3：跨 Epoch Shuffle

固定 seed，验证：

- 同一配置重复运行得到相同 epoch 顺序；
- epoch 0 与 epoch 1 的顺序不同；
- 所有 rank 使用相同 sampler seed；
- `set_epoch` 确实在 iterator 创建前调用。

### 测试 4：断点恢复

在 step `K` 保存 checkpoint，恢复后记录接下来 `N` 个 sample / pack IDs，与不中断运行比较。

不仅比较 loss，还要比较数据 ID，因为相似 loss 不能证明样本序列相同。

---

## 35. 性能基准应控制变量

一个推荐实验表：

| 实验 | 改变量 | 观察指标 |
|---|---|---|
| A | `num_workers` | loader tokens/s、CPU、RAM |
| B | `prefetch_factor` | data wait、RAM、shared memory |
| C | `pin_memory` | H2D、step time、pinned RAM |
| D | 在线 vs 离线 tokenization | CPU、I/O、tokens/s |
| E | padding vs bucketing vs packing | layout / supervised utilization |
| F | 稠密 mask vs varlen kernel | attention time、MFU |
| G | buffer size | shuffle 质量、RAM、恢复开销 |

每个实验都应：

1. warm up；
2. 测足够多 step；
3. 使用相同数据范围；
4. 固定模型、序列长度和并行拓扑；
5. 记录均值、P50、P95 与异常点；
6. 至少重复两次确认趋势。

---

## 36. 常见错误

### 错误 1：把 EOS 当作完整隔离

EOS 只是 token。attention、loss 与位置边界必须另行实现。

### 错误 2：二维 padding mask 表示独立 packing

它只能区分有效 token 与 PAD，不能自然表示行内多个独立 segment。

### 错误 3：只重置 position ids

位置重置不会阻止跨样本 attention。

### 错误 4：只看 padding ratio

prompt、边界和被 mask 的 label 也不参与监督，应看 supervised utilization。

### 错误 5：把样本 shuffle 说成 token shuffle

自回归样本内部 token 顺序不能被随机打乱。

### 错误 6：所有 rank 使用不同 sampler seed

`DistributedSampler` 的 ranks 应共享 seed，由 sampler 根据 rank 切分同一个全局顺序。

### 错误 7：忘记 `set_epoch`

每个 epoch 可能重复相同顺序。

### 错误 8：以为 DDP 会自动切数据

DDP 同步梯度，不负责输入去重。

### 错误 9：IterableDataset 多 worker 不分片

每个 worker 可能重复整条流。

### 错误 10：把 `drop_last=False` 理解成绝不重复

DistributedSampler 为对齐长度可能补索引。

### 错误 11：把 `non_blocking=True` 当成 overlap 证明

需要 pinned source、独立 stream、DMA engine、缓冲与依赖，并用 timeline 验证。

### 错误 12：只保存 epoch 就称为精确恢复

sampler offset、shuffle buffer、packer remainder、RNG 和预取状态都可能影响下一个 batch。

---

## 37. 概念检查

### 问题 1

为什么 dynamic padding 能减少浪费，却不等价于 packing？

**答案：** dynamic padding 仍让每个样本占据独立 batch row，只把这一批补到局部最大长度；packing 会把多个样本放入同一个物理序列，还必须定义段边界语义。

### 问题 2

在两个独立文档间加入 EOS 后，B 是否已经不能看 A？

**答案：** 没有。普通 causal attention 仍允许后面的 B 看到前面的 A，除非显式使用 block-diagonal 或 segment-aware attention。

### 问题 3

`sampler.set_epoch(epoch)` 能否让中断在半个 epoch 后精确恢复？

**答案：** 不能。它只帮助重建 epoch 级 shuffle，还需要保存消费位置、packer / buffer 状态和相关 RNG。

### 问题 4

为什么各 rank 做本地 mean loss 可能有偏差？

**答案：** 当有效 token 数不同，等权平均 rank 相当于让每个 rank 权重相同，而不是让每个 token 权重相同。

### 问题 5

`num_workers` 为什么不能一直增加？

**答案：** 最终会受到 CPU、I/O、内存、shared memory、文件锁和调度开销限制，过多 workers 可能互相争用。

### 问题 6

Streaming shuffle 的 buffer 足够大时会发生什么？

**答案：** 随机性通常改善，但内存和 checkpoint 状态增大；只要 buffer 小于完整数据集，就仍不等同于严格的全局 permutation。

---

## 38. 今日练习

### 练习 1：写一个 Pack 可视化器

输入多个短 token 序列，输出：

- physical positions；
- segment IDs；
- position ids；
- attention matrix；
- labels 与 loss mask。

用 A、B 两文档手算并和程序结果逐位置比较。

### 练习 2：审计 DDP 样本集合

用 2 个 ranks、每 rank 2 个 workers 迭代一个带稳定 ID 的小数据集，保存每个消费者看到的 ID，计算 union、intersection、duplicate 和 drop。

### 练习 3：验证 Resume

在第 7 个 batch 后保存状态，恢复并比较后续 20 个 pack IDs。先只保存 epoch，观察失败；再逐步加入 cursor、RNG、buffer 和 packer remainder。

### 练习 4：做 Worker / Prefetch 扫描

固定模型和数据，测试：

```text
num_workers:    0, 2, 4, 8
prefetch_factor: 1, 2, 4
pin_memory:     false, true
```

记录 loader tokens/s、effective tokens/s、data wait、CPU、RAM 与 pinned RAM。

### 练习 5：比较三种有效率

在真实 batch 上同时计算：

- layout utilization；
- supervised utilization；
- pack fill rate。

解释为什么它们可能明显不同。

---

## 39. 参考资料

- [PyTorch：`torch.utils.data`、DataLoader 与 DistributedSampler](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch：Pinned Memory 与 `non_blocking`](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- [Hugging Face Datasets：Streaming、Sharding、Shuffle 与状态恢复](https://huggingface.co/docs/datasets/stream)
- [Hugging Face Datasets：批处理 `map` 与多进程预处理](https://huggingface.co/docs/datasets/process)
- [Hugging Face Transformers：Data Collator 与动态 Padding](https://huggingface.co/docs/transformers/data_collators)
- [Hugging Face Transformers：Padding-Free Training](https://huggingface.co/docs/transformers/main/padding_free)
- [Hugging Face Tokenizers：Tokenizer API](https://huggingface.co/docs/tokenizers/api/tokenizer)

---

## 今日总结

今天建立了从原始语料到 GPU batch 的完整心智模型：

1. 数据流水线要同时追求正确性与吞吐；
2. split 必须早于 chunking 和 packing，稳定 ID 是审计基础；
3. EOS、attention mask、loss mask 和 position ids 是四个独立设计；
4. padding、连续 token 流和独立 packing 具有不同训练语义；
5. packing 减少 PAD，不代表稠密 attention kernel 自动少算；
6. DDP rank 与 DataLoader worker 必须分两层去重分片；
7. `set_epoch`、worker seed 与 checkpoint state 共同决定可复现性；
8. prefetch、pinned memory 和异步 H2D 需要通过 timeline 与 tokens/s 实测；
9. 真正有意义的吞吐是全局 loss-bearing tokens/s；
10. 所有性能优化之前，都应先用样本 ID、toy mask 和 resume 测试证明语义正确。

最核心的一句话是：

> 高吞吐的数据流水线，不是“尽快把任意 batch 喂给 GPU”，而是“可验证地把正确、唯一、可恢复的有效 token 持续送到 GPU”。

## 明日衔接建议

从“如何高效、正确地读取数据”继续深入“什么数据值得被模型学习”：系统学习质量过滤、精确与近似去重、benchmark contamination、数据 provenance、数据混合权重，以及如何用可审计指标衡量数据集质量。