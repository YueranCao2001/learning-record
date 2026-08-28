# 2026-08-28：大模型并行全景——DP、TP、PP、SP 与 EP

## 今日目标

- 建立 Data、Tensor、Pipeline、Sequence/Context 与 Expert Parallel 的统一认知。
- 明确每种并行方式切分哪个维度、复制什么状态、主要节省什么。
- 理解 broadcast、all-reduce、all-gather、reduce-scatter、all-to-all 和 point-to-point。
- 掌握 Tensor Parallel 中 column/row parallel linear 的基本通信路径。
- 理解 Pipeline Parallel 的 micro-batch、bubble 与 1F1B。
- 区分 Sequence Parallel 与 Context Parallel。
- 理解 MoE Expert Parallel 的 token dispatch、load balancing 和 all-to-all。
- 学会按硬件拓扑组合多种并行方式，而不是只看 GPU 数量。

## 1. 为什么需要多种并行

昨天的 ZeRO/FSDP 主要在 data-parallel group 中切分模型状态。但大模型训练还可能遇到：

- 单层矩阵太大，单卡无法计算。
- 模型层数太多，参数无法放入一个设备。
- sequence 太长，activation 和 attention 无法放入单卡。
- MoE experts 太多，expert weights 无法在每卡复制。
- 单纯增加 data parallel 会让每卡模型显存不变。
- 跨节点带宽远低于节点内 GPU 互联。

因此需要沿不同维度切分：

| 策略 | 主要切分维度 |
| --- | --- |
| Data Parallel / DP | batch |
| Tensor Parallel / TP | 单层 tensor |
| Pipeline Parallel / PP | model depth |
| Sequence Parallel / SP | 部分 activation 的 sequence |
| Context Parallel / CP | 全序列上下文 |
| Expert Parallel / EP | MoE experts |
| FSDP/ZeRO | 参数、梯度、optimizer state |

它们不是互斥选项，而是可以组成多维并行。

## 2. Rank、World Size 与 Process Group

分布式进程通常有：

- global rank：整个训练任务中的编号。
- local rank：当前节点内的设备编号。
- world size：总进程数。
- process group：参与某类 collective 的 rank 子集。

例如 16 张 GPU 使用：

```text
TP = 4
PP = 2
DP = 2

world_size = TP × PP × DP = 16
```

每个 rank 可以被看成一个坐标：

```text
(dp_rank, pp_rank, tp_rank)
```

不同通信只在对应 group 内发生：

- DP group：同一个模型切片，不同数据。
- TP group：同一层的不同 tensor shard。
- PP group：同一个样本经过的不同 pipeline stage。

实际加入 CP/EP 后，group 可能折叠或重叠，不能盲目把所有维度简单相乘；应以框架的并行映射为准。

## 3. Collective 与 Point-to-Point

分布式通信可以分为：

### Collective

一个 group 中多个 rank 共同参与：

- broadcast。
- all-reduce。
- all-gather。
- reduce-scatter。
- all-to-all。

### Point-to-point

两个 rank 之间直接通信：

- send。
- recv。
- isend。
- irecv。

Pipeline stage 之间通常以 P2P 发送 activation/gradient；DP、TP、FSDP 和 EP 更依赖 collective。

## 4. Broadcast

一个 source rank 把 tensor 发送给 group 中所有 rank：

```text
before:
rank 0: [A]
rank 1: [?]
rank 2: [?]
rank 3: [?]

broadcast from rank 0

after:
rank 0: [A]
rank 1: [A]
rank 2: [A]
rank 3: [A]
```

常见用途：

- 初始化时同步模型状态。
- 广播配置或随机种子。
- 从 rank 0 分发某个 tensor。

Broadcast 不进行求和，只复制 source 内容。

## 5. All-Reduce

每个 rank 有一个完整形状的 tensor，先归约，再把完整结果发给所有 rank：

```text
rank 0: g0
rank 1: g1
rank 2: g2
rank 3: g3

all-reduce(sum)

every rank:
g0 + g1 + g2 + g3
```

DDP 的典型梯度同步：

```text
local backward
→ all-reduce gradients
→ 每个 rank 得到相同平均梯度
→ 各自执行相同 optimizer update
```

All-reduce 后每个 rank 仍有完整结果，因此它同步数据，但不减少结果的持久存储。

## 6. All-Gather

每个 rank 保存 tensor 的一个 shard，all-gather 后每个 rank 得到所有 shard：

```text
rank 0: [A]
rank 1: [B]
rank 2: [C]
rank 3: [D]

all-gather

every rank:
[A, B, C, D]
```

常见用途：

- FSDP 计算前聚合参数。
- TP 中恢复完整 activation。
- 收集不同 rank 的输出或元数据。

通信后结果复制到每个参与 rank，可能造成临时显存峰值。

## 7. Reduce-Scatter

先归约各 rank 的输入，再把结果切分：

```text
inputs:
rank 0: [a0, b0, c0, d0]
rank 1: [a1, b1, c1, d1]
...

reduce-scatter(sum)

rank 0: [Σa]
rank 1: [Σb]
rank 2: [Σc]
rank 3: [Σd]
```

常见用途：

- FSDP backward 后同步并切分梯度。
- Sequence Parallel 中把归约结果按 sequence 维度分回各 rank。
- 将 all-reduce 拆成 reduce-scatter + all-gather 的阶段。

结果天然是 shard，因此能与分片存储结合。

## 8. All-to-All

每个 rank 把不同数据块发送给不同目标 rank，同时从所有 rank 接收属于自己的块：

```text
rank i input:
[to rank 0, to rank 1, ..., to rank N-1]

all-to-all

rank j output:
[from rank 0, from rank 1, ..., from rank N-1]
```

它不是简单复制或求和，而是重排分布。

典型用途：

- MoE 把 token dispatch 到拥有目标 expert 的 rank。
- 某些 sequence/context 并行布局转换。
- tensor layout 重新切分。

All-to-all 对网络带宽、延迟、负载均衡和消息大小非常敏感。

## 9. Data Parallel

DP 在每个 rank 上复制模型，并切分 batch：

```text
rank 0: full model + batch shard 0
rank 1: full model + batch shard 1
rank 2: full model + batch shard 2
rank 3: full model + batch shard 3
```

每个 rank 独立 forward/backward，之后同步梯度。

优点：

- 实现简单。
- 计算高度独立。
- 通常容易获得高吞吐。
- batch 可以随 GPU 数扩展。

限制：

- 每卡仍需完整模型。
- 每卡通常仍有完整 optimizer state，除非结合 ZeRO/FSDP。
- global batch 变大后可能需要调整 learning rate 和训练配方。

## 10. DDP 与 FSDP 的区别

| 项目 | DDP | FSDP FULL_SHARD |
| --- | --- | --- |
| 参数持久状态 | 每卡完整复制 | 每卡保存 shard |
| 梯度 | all-reduce 后完整 | reduce-scatter 后 shard |
| optimizer state | 每卡完整 | 每卡 shard |
| 计算前参数 | 已经完整 | 按 FSDP unit all-gather |
| 通信特征 | gradient all-reduce | parameter all-gather + gradient reduce-scatter |
| 主要目标 | 吞吐扩展 | 状态显存扩展 |

两者都沿 batch 维度处理不同数据，因此 FSDP 仍属于 fully sharded data parallel，而不是 tensor parallel。

## 11. Tensor Parallel

TP 把一个层内部的大 tensor 切到多个 rank：

```text
一个 Linear / Attention / MLP
→ weight shard 0 on rank 0
→ weight shard 1 on rank 1
→ ...
```

每个 rank 只计算局部矩阵乘法，之后通过 collective 组合结果。

TP 适合：

- 单层权重无法放入单卡。
- hidden size 或 FFN size 很大。
- 希望切分层内 activation。
- 节点内有高带宽互联。

代价：

- 每层都可能通信。
- 通信频率高。
- 跨慢速节点扩展通常困难。
- tensor 维度必须满足切分约束或进行 padding。

## 12. Column-Parallel Linear

线性层：

```text
Y = X W
W: [D_in, D_out]
```

沿输出维度切分：

```text
W = [W_0, W_1, ..., W_n]
```

每个 rank：

```text
Y_i = X W_i
```

局部输出：

```text
Y_i: [B, T, D_out / n]
```

如果下一层能直接消费分片输出，就不必立即 all-gather；如果需要完整 `Y`，则执行 all-gather。

Transformer 中 QKV 投影和 MLP 上投影常可使用 column parallel。

## 13. Row-Parallel Linear

沿输入维度切分：

```text
X = [X_0, X_1, ..., X_n]
W =
[W_0
 W_1
 ...
 W_n]
```

每个 rank 计算局部贡献：

```text
partial_i = X_i W_i
```

完整结果是各局部贡献之和：

```text
Y = Σ_i partial_i
```

因此需要 reduction，常表现为 all-reduce，或结合 sequence parallel 使用 reduce-scatter。

Transformer 中 attention 输出投影和 MLP 下投影常与 row parallel 对应。

## 14. TP 为什么常成对设计

MLP：

```text
X
→ column-parallel up projection
→ activation
→ row-parallel down projection
→ reduction
```

如果 column-parallel 输出直接保持 shard，激活函数可以局部执行；到 row-parallel 输出处再归约。

这样避免在两个线性层之间恢复完整 activation。

设计 TP 时不只看单个矩阵，还要看相邻算子能否保持兼容的 sharded layout。

## 15. Attention 中的 Tensor Parallel

Multi-Head Attention 天然可以按 head 切分：

```text
rank 0: subset of heads
rank 1: subset of heads
...
```

每个 rank 局部计算：

- 自己的 Q/K/V heads。
- 局部 attention。
- 局部 context。

之后通过输出投影和 reduction 合并。

需要注意：

- num_heads 必须能被 TP degree 合理切分。
- GQA 的 query heads 与 KV groups 也要满足布局约束。
- TP 不能自动解决 sequence length 太长的问题。
- head 切分后每卡仍可能需要完整 sequence。

## 16. Sequence Parallel

不同框架对“Sequence Parallel”命名不完全一致。

Megatron/PyTorch TP 语境中的 SP 常指：在 TP group 内，把原本每个 TP rank 都重复保存的某些 activation 沿 sequence 维度切分。

典型对象：

- LayerNorm/RMSNorm 输入输出。
- Dropout。
- residual 相关 activation。
- 不需要完整 hidden dimension 交互的逐 token 算子。

概念流程：

```text
sharded sequence activation
→ all-gather before需要完整输入的 TP region
→ tensor-parallel compute
→ reduce-scatter back to sequence shards
```

收益是减少 TP 中重复 activation，而不是把整个 attention 上下文永久独立切开。

## 17. Context Parallel

Context Parallel 更直接地把长 sequence 分到多个 rank，并让 attention 在跨 rank 上交换所需上下文。

```text
full sequence T

rank 0: tokens [0 : T/N]
rank 1: tokens [T/N : 2T/N]
...
```

每个 rank 只持有部分 query token，但为了计算正确 attention，需要获得其他分片的 K/V 信息，可能使用：

- P2P ring exchange。
- all-gather。
- all-to-all。
- 分层或混合通信。

CP 主要解决长上下文 activation；模型参数通常仍按其他维度复制或切分。

## 18. SP 与 CP 的区别

| 项目 | Sequence Parallel | Context Parallel |
| --- | --- | --- |
| 常见语境 | TP 附属优化 | 长上下文并行 |
| 主要切分 | 部分逐 token activation | 整个 sequence/context |
| Attention | 常仍由 TP region 处理完整上下文 | 显式跨 rank 交换上下文信息 |
| 主要收益 | 减少 TP 重复 activation | 支持更长 sequence |
| 常见通信 | all-gather + reduce-scatter | P2P/all-gather/all-to-all 等 |

术语会随框架变化，阅读配置时应确认它到底切分哪些 tensor。

## 19. Pipeline Parallel

PP 按模型深度切分 layers：

```text
stage 0: embedding + layers 0..7
stage 1: layers 8..15
stage 2: layers 16..23
stage 3: layers 24..31 + LM Head
```

forward：

```text
stage 0 activation
→ send to stage 1
→ send to stage 2
→ send to stage 3
```

backward 反向发送 activation gradient。

每个 stage 只保存自己的参数和 optimizer state，但 stage 之间存在依赖，不能像 DP 那样完全独立计算。

## 20. 为什么需要 Micro-Batch

如果把整个 batch 一次送入 pipeline：

```text
stage 0 工作
其他 stage 等待

stage 1 工作
stage 0 和后续 stage 等待
...
```

把 batch 切成多个 micro-batch 后：

```text
micro-batch 0 进入 stage 1
同时 micro-batch 1 在 stage 0
```

不同 stage 可以处理不同 micro-batch，从而提高设备利用率。

这里的 micro-batch 与 gradient accumulation 常一起出现，但概念不同：

- pipeline micro-batch 用于填充流水线。
- gradient accumulation 决定多少 micro-batch 后更新参数。

## 21. Pipeline Bubble

Pipeline 开始时需要填充，结束时需要排空，这些空闲时间称为 bubble。

粗略直觉：

```text
pipeline stages = P
micro-batches = M

M 越大
→ bubble 相对比例越小
→ activation in flight 可能更多
```

增加 micro-batch 数能提高利用率，但可能增加调度、activation 和通信开销。

性能不能只看 bubble 公式，还要看：

- stage 是否负载均衡。
- 通信能否与计算重叠。
- micro-batch GEMM 是否足够大。
- virtual pipeline/interleaving。
- activation checkpointing。

## 22. GPipe 与 1F1B

### GPipe 风格

```text
先完成所有 micro-batch forward
再完成所有 backward
```

优点是调度直观，缺点是需要保存较多 in-flight activation。

### 1F1B

稳定阶段交替执行：

```text
one forward
one backward
```

可以降低 activation 峰值，并更好地利用 pipeline。

通常还会经历：

- warmup。
- steady state。
- cooldown。

Interleaved 1F1B 让一个物理 rank 承担多个 virtual stages，以减少 bubble 并改善负载均衡，但调度更复杂。

## 23. Pipeline 的负载均衡

如果某个 stage 比其他 stage 慢：

```text
pipeline throughput
≈ slowest stage throughput
```

导致不均衡的因素：

- embedding/LM Head 很大。
- 某些层包含 MoE。
- 首尾 stage 有额外 loss 或数据处理。
- 各层计算量不一致。
- activation 通信量不同。
- virtual stage 划分不合理。

因此不能只按“层数相等”切分，应按参数、FLOPs、activation 和实际 profile 调整布局。

## 24. Expert Parallel

Mixture-of-Experts 层包含多个 experts，但每个 token 只选择少数 experts：

```text
token
→ router scores
→ top-k experts
→ selected expert MLPs
→ combine outputs
```

EP 把不同 experts 放到不同 rank：

```text
rank 0: experts 0..3
rank 1: experts 4..7
...
```

模型总参数可以很大，但每个 token 只激活一小部分计算。

## 25. MoE Token Dispatch

Router 选择 expert 后，token 当前所在 rank 不一定拥有目标 expert。

流程：

```text
local tokens
→ router
→ group tokens by destination expert/rank
→ all-to-all dispatch
→ local expert compute
→ all-to-all combine
→ restore original token order
```

因此 EP 的核心通信通常是 all-to-all。

通信量受以下因素影响：

- token 数。
- hidden size。
- top-k。
- EP degree。
- dispatch dtype。
- expert load balance。
- capacity/token dropping 策略。

## 26. Expert Load Balancing

如果大量 token 都选择同一个 expert：

- 该 rank 计算变慢。
- 其他 experts 空闲。
- all-to-all 消息大小不均。
- buffer 可能溢出。
- 整个 step 被最慢 rank 限制。

常见对策：

- auxiliary load-balancing loss。
- capacity factor。
- token dropping 或 dropless routing。
- expert choice/分组路由。
- router z-loss。
- shared experts。
- 监控每个 expert 的 token 数。

MoE 的稀疏计算优势依赖有效的路由和负载均衡。

## 27. DP、TP、PP、CP、EP 的对照

| 策略 | 切分对象 | 常见通信 | 通信频率 | 主要解决 |
| --- | --- | --- | --- | --- |
| DP | batch | all-reduce | 每次 backward bucket | 吞吐扩展 |
| FSDP | model states | all-gather + reduce-scatter | 每个 FSDP unit | 静态状态显存 |
| TP | layer tensor | all-reduce/all-gather/reduce-scatter | 每层多次 | 单层太大 |
| PP | layers/depth | P2P send/recv | 每个 micro-batch 边界 | 模型太深 |
| SP | sequence activation | all-gather + reduce-scatter | TP region 边界 | 重复 activation |
| CP | context tokens | P2P/gather/all-to-all | 每个 attention layer | 长上下文 |
| EP | experts/tokens | all-to-all | 每个 MoE layer | 大量 experts |

## 28. 3D Parallelism

经典 3D parallelism：

```text
Data Parallel
× Tensor Parallel
× Pipeline Parallel
```

例如 64 GPUs：

```text
TP = 4
PP = 4
DP = 4

4 × 4 × 4 = 64
```

每个 DP replica 是一个由 TP×PP 组成的完整模型；不同 DP replicas 处理不同数据。

长上下文和 MoE 还会加入 CP/EP。现代配置可能成为 4D、5D 甚至更复杂的异构映射。

## 29. 并行维度如何映射到硬件

经验原则：

### 高频 TP 通信

- 尽量放在同一节点或高带宽互联域。
- 每层通信，对延迟和带宽敏感。

### PP

- 通信主要是 stage boundary activation/gradient。
- 可以跨节点，但仍需关注带宽和 stage balance。

### DP/FSDP

- group 可以跨节点。
- 梯度或参数 collective 较大。
- 应利用分层 collective 和 overlap。

### EP

- all-to-all 对拓扑非常敏感。
- 尽量避免把大量细粒度 token dispatch 放在慢速链路上。

并行配置首先是拓扑问题，其次才是整数分解问题。

## 30. 通信量之外还有 Latency

两个通信方案传输总字节相同，也可能性能差异很大。

小消息：

- latency 主导。
- collective 启动次数重要。
- wrap/stage 粒度过细会吃亏。

大消息：

- bandwidth 主导。
- 链路利用率和拓扑重要。
- overlap 更有机会隐藏通信。

因此：

```text
communication cost
≈ latency × number_of_messages
+ bytes / effective_bandwidth
```

这是粗略模型，但能解释“减少字节却没有加速”的情况。

## 31. Communication-Compute Overlap

如果通信依赖当前计算结果，就不能完全提前。

可重叠的典型方式：

- DDP 在部分 gradient ready 后立即 all-reduce bucket。
- FSDP 计算当前 unit 时预取下一 unit 参数。
- TP 将 GEMM 分块，与 all-gather/reduce-scatter overlap。
- PP 在计算一个 micro-batch 时发送另一个。
- EP 把 token dispatch 与其他独立计算重叠。

Overlap 会增加调度复杂度和可能的 buffer 峰值，必须通过 timeline/profile 验证。

## 32. 一个 Collective 教学示例

以下代码只展示语义，要求使用 `torchrun` 初始化多进程环境：

```python
import torch
import torch.distributed as dist


dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(
    "cuda",
    torch.cuda.current_device(),
)

# All-reduce: 每个 rank 最终得到所有 rank 值之和
value = torch.tensor(
    [float(rank + 1)],
    device=device,
)
dist.all_reduce(
    value,
    op=dist.ReduceOp.SUM,
)

expected = world_size * (world_size + 1) / 2
assert value.item() == expected
```

真实训练不会为每个标量单独通信，而会把梯度组织成较大的 buckets，以减少启动开销。

## 33. 如何选择并行策略

### 模型单卡可放下

优先：

```text
DDP
→ mixed precision
→ efficient kernels
```

### 模型状态单卡放不下，但单层可放下

考虑：

```text
FSDP/ZeRO
```

### 单层本身太大

考虑：

```text
TP
```

### 模型层数很多

考虑：

```text
PP
```

### 长 sequence activation OOM

考虑：

```text
activation checkpointing
+ efficient attention
+ SP/CP
```

### MoE experts 太多

考虑：

```text
EP
```

最终通常是先使用必要的最小 model parallel degree，再用 DP 扩展吞吐。

## 34. 为什么不应把 TP/PP 开得过大

更高 model-parallel degree 可以减少每卡模型部分，但会：

- 增加通信。
- 减小局部 GEMM 尺寸。
- 降低 kernel efficiency。
- 增加 pipeline bubble。
- 让负载均衡更困难。
- 增加 checkpoint 和调试复杂度。
- 对网络故障更敏感。

可行配置不等于高效配置。通常先满足显存，再最小化高频 model parallel communication。

## 35. 并行训练的正确性检查

必须确认：

- 所有 DP ranks 初始参数一致。
- 同一个 TP/PP shard 的 checkpoint 映射正确。
- loss reduction 没有多除或少除 world size。
- gradient accumulation 与 DP averaging 一致。
- padding/valid token 数在分布式归一化中正确。
- pipeline micro-batches 没有丢失或重复。
- MoE dispatch/combine 恢复原 token 顺序。
- expert auxiliary loss 在正确 group 中归约。
- random seed 既能复现，又不会让所有 DP rank 使用完全相同 Dropout mask。
- checkpoint 能在相同或新并行布局下恢复。

分布式 bug 可能不报错，只表现为 loss 曲线异常。

## 36. 性能指标

不要只看每 step 时间，还应记录：

- tokens/s。
- samples/s。
- model FLOPs utilization。
- 每个 collective 的时间与字节数。
- 通信计算 overlap。
- pipeline bubble 比例。
- 每个 stage 的时间。
- 每个 expert 的 token 数。
- 每个 rank 的 peak memory。
- straggler rank。
- 网络吞吐与拥塞。
- checkpoint 保存时间。

最慢 rank 决定同步训练 step 的完成时间，因此平均值可能掩盖问题。

## 37. 常见错误

- 把增加 GPU 数等同于每卡显存自动降低。
- 把 FSDP 误认为 tensor parallel。
- 认为 all-reduce 会让每个 rank 只保留 gradient shard。
- 认为 all-gather 后仍然只有本地 shard 大小。
- 混淆 reduce-scatter 与 all-reduce。
- 把 all-to-all 理解为所有 rank 得到相同结果。
- TP 沿错误维度切分，导致后续算子布局不兼容。
- 每层频繁 all-gather 完整 activation，抵消 TP 收益。
- 把 SP 与长上下文 CP 当成完全相同概念。
- 只按层数平均切 PP，不看每层 FLOPs。
- pipeline micro-batch 太少，bubble 很大。
- pipeline micro-batch 太小，GEMM 效率下降。
- EP 忽略 router load balance，出现 hot expert。
- all-to-all 放在低带宽跨节点链路上却不做 profile。
- 为省显存把 TP/PP degree 调到最大，吞吐反而下降。
- 假设所有并行维度总能简单相乘。
- loss 已在 DDP 中平均，外部又除一次 world size。
- 只比较 step time，不比较有效 tokens/s。
- 只看通信字节，不看消息数量和 latency。
- 并行 checkpoint 只保存 rank 0 的局部 shard。

## 38. 概念检查

1. DP、TP 和 PP 分别切分哪个维度？
2. FSDP 为什么仍属于 data parallel？
3. all-reduce 与 reduce-scatter 的输出布局有什么区别？
4. all-gather 为什么可能造成临时显存峰值？
5. all-to-all 为什么适合 MoE token dispatch？
6. column-parallel linear 切分哪个 weight 维度？
7. row-parallel linear 为什么需要对局部结果求和？
8. TP 为什么常限制在高带宽节点内？
9. SP 与 CP 的主要区别是什么？
10. PP 为什么需要把 batch 切成 micro-batches？
11. 1F1B 如何减少 in-flight activation？
12. pipeline throughput 为什么由最慢 stage 限制？
13. EP 的 load imbalance 会造成什么后果？
14. 64 GPUs、TP=4、PP=4 时 DP degree 是多少？
15. 为什么可行的最大 TP/PP 配置不一定最快？

## 39. 今日练习

- [ ] 为 8 个 rank 画出 TP=2、PP=2、DP=2 的坐标。
- [ ] 手算四个 rank 的 all-reduce 结果。
- [ ] 手算四个 shard 的 all-gather 输出。
- [ ] 用一个长度 8 的向量模拟 reduce-scatter。
- [ ] 构造一个 4×4 block matrix，模拟 all-to-all 重排。
- [ ] 对 `Y=XW` 分别画 column-parallel 和 row-parallel。
- [ ] 计算 MLP TP 中每个 rank 的局部 tensor shape。
- [ ] 画出 4-stage、8-micro-batch 的 pipeline 时间线。
- [ ] 比较 GPipe 与 1F1B 的 activation 存活数量。
- [ ] 模拟 16 个 token 路由到 4 个 experts，并统计负载。
- [ ] 故意制造 hot expert，观察最慢 rank 如何限制 step。
- [ ] 为单机 8 GPU 和双机 16 GPU 分别设计 TP/PP/DP。
- [ ] 记录训练 profile 中 all-reduce、all-gather 和 all-to-all 时间。
- [ ] 写一张决策表：哪类 OOM 应优先使用哪种并行方式。

## 40. 参考资料

- [PyTorch：Distributed Communication Package](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch：Large Scale Transformer Tensor Parallel Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [NVIDIA Megatron Core：Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [NVIDIA Megatron Core：Pipeline Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html)
- [NVIDIA Megatron Core：Mixture of Experts](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)

## 今日总结

一句话总结：**大模型并行的本质是选择一个维度切分计算或状态，再用与该布局匹配的 collective 恢复数学上等价的结果。**

并行地图：

```text
batch 太大或需要吞吐
→ DP

重复模型状态太大
→ FSDP/ZeRO

单层 tensor 太大
→ TP
→ 高频 all-reduce/all-gather/reduce-scatter

模型太深
→ PP
→ micro-batches + P2P + 1F1B

sequence activation 太大
→ SP/CP
→ sequence shard communication

MoE experts 太多
→ EP
→ token all-to-all
```

需要牢记：

- all-reduce 返回完整归约结果；reduce-scatter 返回归约后的 shard。
- TP 切层内 tensor，PP 切模型深度，DP 切 batch。
- SP 与 CP 的命名和覆盖范围依框架而异。
- EP 的性能核心是 all-to-all 与 expert load balance。
- 并行配置必须映射到真实网络拓扑。
- 最好的配置是在满足显存后尽量减少高频 model-parallel communication。

## 明日衔接建议

从“如何并行计算”继续到“如何评估并优化系统性能”：学习 FLOPs、Model FLOPs Utilization、roofline model、arithmetic intensity、通信计算重叠、profiling timeline，以及如何判断训练瓶颈来自计算、显存带宽、网络还是数据加载。
