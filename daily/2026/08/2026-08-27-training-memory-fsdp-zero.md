# 2026-08-27：训练显存拆解——Activation Checkpointing、ZeRO 与 FSDP

## 今日目标

- 建立训练显存的完整账本：参数、梯度、optimizer state、激活、临时缓冲和通信缓冲。
- 用“每个参数多少字节”估算不同精度下的静态显存。
- 理解 mixed precision 为什么既能加速，也可能需要额外的 FP32 状态。
- 区分 gradient accumulation、activation checkpointing 与训练 checkpoint。
- 掌握 ZeRO Stage 1/2/3 分别切分什么。
- 理解 FSDP 在 forward/backward 中的 all-gather、reduce-scatter 与 reshard。
- 认识显存、计算量、通信量和 CPU 内存之间的交换关系。
- 能为一个 OOM 问题选择合理的优化顺序，而不是盲目叠加技术。

## 1. 训练显存不只有模型权重

一次训练 step 的 GPU 显存可以粗略拆成：

```text
训练显存
≈ 参数
+ 梯度
+ optimizer states
+ activations
+ temporary tensors
+ communication buffers
+ allocator reserved/fragmentation
```

其中：

- 参数、梯度、optimizer state 通常与参数量 `P` 近似线性相关。
- activation 与 batch size、sequence length、hidden size、层数和算子实现相关。
- temporary tensors 可能只在某个算子执行期间达到峰值。
- reserved memory 不一定都对应当前活跃 tensor。
- 分布式训练还会出现 all-gather、reduce-scatter 等通信缓冲。

只用“模型参数量 × dtype 字节数”无法判断模型能否训练。

## 2. 参数量与数据类型

常见数据类型的元素大小：

| dtype | 每个元素 |
| --- | ---: |
| FP32 | 4 bytes |
| TF32 storage | 仍是 FP32 storage |
| BF16 | 2 bytes |
| FP16 | 2 bytes |
| FP8 | 1 byte，具体格式与状态依实现而异 |
| INT8 | 1 byte，通常用于量化存储或特定训练方案 |

如果模型有 `P` 个参数，仅权重占用：

```text
parameter memory = P × bytes_per_parameter
```

例如 7B 参数：

```text
FP32 weights: 7e9 × 4 ≈ 28 GB
BF16 weights: 7e9 × 2 ≈ 14 GB
```

这里的 GB 是十进制近似值；系统工具可能使用 GiB，因此显示数字会不同。

## 3. FP32 + AdamW 的静态账本

最简单的 FP32 AdamW 估算：

| 状态 | 每参数字节 |
| --- | ---: |
| FP32 parameter | 4 |
| FP32 gradient | 4 |
| FP32 first moment `m` | 4 |
| FP32 second moment `v` | 4 |
| 合计 | 约 16 |

因此：

```text
static model states
≈ 16 × P bytes
```

7B 模型：

```text
7e9 × 16 ≈ 112 GB
```

这还没有包含 activation、临时 tensor 和通信 buffer。

## 4. Mixed Precision 不一定把所有状态减半

一种传统 FP16 训练布局可能包含：

| 状态 | 每参数字节 |
| --- | ---: |
| FP16 parameter | 2 |
| FP16 gradient | 2 |
| FP32 master parameter | 4 |
| FP32 first moment | 4 |
| FP32 second moment | 4 |
| 合计 | 约 16 |

因此 mixed precision 的主要收益可能首先出现在：

- 矩阵乘法吞吐。
- activation 显存。
- 部分参数/梯度和通信量。

而不是让 AdamW 的所有状态自动减半。

实际每参数字节数取决于：

- 是否保留 FP32 master weights。
- 梯度保存 dtype。
- optimizer state dtype。
- optimizer 是否量化。
- 参数是否分片或 offload。
- 框架是否保留额外副本。

估算时必须写清训练实现，不能把“BF16 模型”直接等同于“全部状态都是 2 bytes”。

## 5. BF16 与 FP16

BF16 和 FP16 都是 16-bit，但数值范围不同。

### FP16

- 尾数精度较高。
- 指数范围较小。
- 小梯度更容易下溢。
- 通常配合 GradScaler。

### BF16

- 指数范围接近 FP32。
- 尾数精度较低。
- 常不需要 loss scaling。
- 需要硬件和算子支持。

PyTorch AMP 使用 autocast 让不同算子选择合适精度。不要假设所有算子都安全地运行在低精度，也不要在使用 autocast 时手工把所有 tensor 强制转成半精度。

## 6. Activation 是动态显存的大头

训练时，backward 需要 forward 中的中间结果，因此很多 activation 必须保留。

一个粗略结构：

```text
activations
∝ batch size
  × sequence length
  × hidden size
  × number of layers
```

但真实系数与模型实现有关，可能包含：

- attention 输入和输出。
- Q、K、V。
- MLP 中间激活。
- normalization 统计量。
- Dropout mask。
- residual stream。
- Softmax 或 attention 中间结果。
- autograd 保存的其他 tensor。

推理只需 forward，并且增量解码主要维护 KV Cache；训练需要为 backward 保存更多内容，因此两者显存结构不同。

## 7. Attention 的 T² 问题

标准 attention 概念上计算：

```text
scores = Q K^T
scores shape: [B, H, T, T]
```

如果显式保存完整 score/probability matrix，相关显存随 `T²` 增长。

优化 attention kernel 可以通过分块和重计算避免把完整 `T × T` 矩阵写回显存，使峰值显存更接近线性增长，但数学结果仍然是 attention。

所以：

- causal mask 不会自动消除 `T²` 计算结构。
- 使用 memory-efficient attention/FlashAttention 类 kernel 会改变中间 tensor 的存储方式。
- 算法复杂度、IO 复杂度和实际峰值显存需要分开讨论。

## 8. Temporary Tensor 与 Peak Memory

OOM 发生在峰值，而不是平均值。

可能的峰值来源：

- 大型矩阵乘法 workspace。
- logits `[B, T, V]`。
- cross-entropy 的临时 buffer。
- FSDP 参数 all-gather。
- gradient bucket。
- optimizer step 同时读写多个 state。
- 保存 full state dict。
- evaluation 使用更长序列。
- 第一次迭代的 kernel/autotune 开销。

因此“空闲显存看起来够”不保证下一个算子不会 OOM。

## 9. Allocated、Reserved 与 Fragmentation

CUDA caching allocator 会保留已经申请过的显存块以便复用。

概念上：

- allocated：活跃 tensor 正在使用的显存。
- reserved：allocator 从设备申请并保留的显存。
- free inside reserved：已保留但暂未被活跃 tensor 使用。
- device free：尚未被当前进程或其他进程保留的显存。

`reserved > allocated` 并不等于内存泄漏。

碎片化会让总空闲量看似足够，却找不到满足大型连续申请的合适块。首先应确认是否真的存在仍被引用的 tensor，而不是习惯性调用清缓存。

## 10. 一个简单的显存测量骨架

```python
import torch


torch.cuda.reset_peak_memory_stats()

loss = training_step(batch)
loss.backward()

current_allocated = torch.cuda.memory_allocated()
peak_allocated = torch.cuda.max_memory_allocated()
current_reserved = torch.cuda.memory_reserved()
peak_reserved = torch.cuda.max_memory_reserved()

print({
    "current_allocated_gb": current_allocated / 1e9,
    "peak_allocated_gb": peak_allocated / 1e9,
    "current_reserved_gb": current_reserved / 1e9,
    "peak_reserved_gb": peak_reserved / 1e9,
})
```

为了定位峰值，应分别测量：

1. batch 移入设备后。
2. forward 后。
3. backward 后。
4. optimizer step 中和之后。
5. evaluation。
6. checkpoint 保存。

异步 CUDA 执行可能影响测量时机；严谨 profiling 应使用框架提供的 profiler 和 memory snapshot 工具。

## 11. Gradient Accumulation 节省什么

Gradient accumulation 允许减小每个 micro-batch：

```text
小 micro-batch
× 多次 backward
→ 一次 optimizer update
```

它主要通过减小单次 forward/backward 的 activation 规模降低峰值显存。

它通常不会直接减少：

- 模型参数。
- 完整梯度 tensor 的大小。
- AdamW moments。
- 单个样本的 sequence-length 相关 activation。

如果 batch size 已经是 1，继续 accumulation 不会让单个 micro-batch 更小；此时可能需要 checkpointing、sequence packing 或分片。

## 12. Activation Checkpointing

Activation checkpointing 的核心交换：

```text
少保存 activation
↔ backward 时重新执行部分 forward
```

普通训练：

```text
forward 保存中间结果
→ backward 直接使用
```

checkpointing：

```text
forward 只保存边界输入
→ backward 重新计算内部 activation
→ 再求梯度
```

因此：

- 显存下降。
- 计算量和 step time 增加。
- 参数、梯度和 optimizer state 不会因此变小。

## 13. 最小 Checkpointing 示例

```python
import torch
from torch.utils.checkpoint import checkpoint


class CheckpointedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask):
        def run_block(x, mask):
            return self.block(
                x,
                attention_mask=mask,
            )

        return checkpoint(
            run_block,
            hidden_states,
            attention_mask,
            use_reentrant=False,
        )
```

checkpoint 的函数应该尽量是确定且无副作用的纯计算段。

PyTorch 会处理常见 RNG 状态，使含 Dropout 的重计算尽量保持与普通 forward 一致；但 RNG 保存/恢复本身有开销，跨多个设备或在函数内部移动设备时需要额外谨慎。

## 14. Checkpoint 粒度

粒度过粗：

- 单段重计算很多。
- 边界少，保存 activation 少。
- 计算开销可能较高。

粒度过细：

- 调用和调度开销增加。
- 保存许多边界 tensor。
- 显存收益可能有限。

Transformer 常见选择是：

- 每个 block checkpoint。
- 每若干 block 一组。
- 只 checkpoint MLP 或 attention。
- 使用 selective activation checkpointing，仅重计算特定算子。

最优粒度依赖显存预算和吞吐目标，需要 profiling。

## 15. Activation Checkpoint 与训练 Checkpoint 不同

两个“checkpoint”含义不同：

### Activation checkpointing

- 发生在一次 forward/backward 内。
- 目标是用额外计算换显存。
- 不产生可恢复训练的磁盘文件。

### Training checkpoint

- 保存模型、optimizer、scheduler 和 step。
- 目标是故障恢复或继续训练。
- 消耗磁盘与保存时间。

混用名称很容易导致沟通误解，应明确说 activation checkpoint 或 training checkpoint。

## 16. CPU Offload

可以把状态从 GPU 移到 CPU：

- optimizer state offload。
- parameter offload。
- gradient offload。
- activation offload。

收益：

- GPU 显存下降。
- 可以训练更大模型。

代价：

- 占用 CPU 内存。
- 增加 PCIe/NVLink 传输。
- 可能让 GPU 等待数据。
- pinned memory 和预取策略变重要。
- 多进程可能重复占用 host memory。

Offload 是把瓶颈迁移，不是让状态消失。

## 17. 普通 Data Parallel 的复制

Data Distributed Parallel 的典型方式：

```text
每个 rank:
- 完整模型参数
- 完整梯度
- 完整 optimizer state
- 不同数据 shard
```

backward 后通过 all-reduce 同步梯度。

优点：

- 模型实现简单。
- 每个 rank 独立执行完整 forward/backward。
- 当模型和 optimizer state 能放入单卡时吞吐通常很好。

限制：

- 增加 GPU 数不会降低每张卡上的模型状态显存。
- 大模型可能在单卡上连静态状态都放不下。

## 18. ZeRO 的核心思想

ZeRO 将 data parallel 中每张卡重复保存的模型状态切分到多个 rank。

设 data-parallel world size 为 `N`。

理想情况下，被切分状态的每卡占用约变为：

```text
原状态 / N
```

但真实峰值还包括 activation、临时 all-gather、通信 buffer、padding 和 allocator 开销，因此不会严格等于总显存除以 `N`。

## 19. ZeRO Stage 1/2/3

| Stage | 切分内容 | 仍复制的主要状态 |
| --- | --- | --- |
| ZeRO-1 | optimizer state | 参数、梯度 |
| ZeRO-2 | optimizer state + gradients | 参数 |
| ZeRO-3 | optimizer state + gradients + parameters | 计算时按需聚合参数 |

记忆方法：

```text
Stage 1: O
Stage 2: O + G
Stage 3: O + G + P
```

其中：

- `O`：optimizer states。
- `G`：gradients。
- `P`：parameters。

Stage 越高，静态显存通常越低，但通信、调度和实现复杂度通常更高。

## 20. 用 16 Bytes/Param 做 ZeRO 粗算

假设 FP32 AdamW 的简化静态状态：

```text
parameters: 4P
gradients: 4P
optimizer states: 8P
total: 16P bytes
```

每卡近似：

### DDP / ZeRO-0

```text
16P
```

### ZeRO-1

```text
4P + 4P + 8P/N
```

### ZeRO-2

```text
4P + (4P + 8P)/N
```

### ZeRO-3

```text
(4P + 4P + 8P)/N
= 16P/N
```

这只是静态状态近似，不包含 full-parameter gather、activation、通信 bucket 和峰值临时内存。

## 21. FSDP 是什么

PyTorch FullyShardedDataParallel 使用 fully sharded data parallel 思路，把参数分片到 data-parallel workers。

FULL_SHARD 概念流程：

```text
持久状态：每个 rank 只保存参数 shard

forward 前:
all-gather 当前 FSDP unit 的完整参数

forward:
执行该 unit 计算

forward 后:
reshard / 释放不需要的完整参数

backward 前:
再次 all-gather 所需参数

backward:
计算局部梯度

backward 后:
reduce-scatter 梯度
→ 每个 rank 保留自己的 gradient shard

optimizer:
只更新本地 parameter/optimizer shard
```

FSDP 通过“计算时短暂聚合、计算后重新切分”降低持久显存，但会增加 collective communication。

## 22. All-Reduce、All-Gather 与 Reduce-Scatter

### All-reduce

```text
每个 rank 输入一个完整梯度
→ 求和/平均
→ 每个 rank 得到完整结果
```

常见于 DDP。

### All-gather

```text
每个 rank 输入一个 shard
→ 每个 rank 拼出完整 tensor
```

FSDP 在计算某个 unit 前用于收集完整参数。

### Reduce-scatter

```text
先对各 rank 输入求和/归约
→ 再把结果切成 shard
→ 每个 rank 只得到一片
```

FSDP backward 后用它同步并切分梯度。

理解这三个 collective，是理解分片训练性能的基础。

## 23. FSDP Wrap 粒度

FSDP unit 决定何时 all-gather 和 reshard。

如果只在根模型包一层：

- 一次可能聚合很大的参数集合。
- 峰值 gather memory 高。
- 通信次数少。

如果每个很小模块都单独包：

- 峰值参数窗口小。
- collective 次数多。
- 小消息通信效率可能差。

Transformer 常按 block 自动 wrap，在峰值显存与通信效率之间折中。

## 24. Prefetch 与通信计算重叠

理想目标：

```text
计算当前 block
同时 all-gather 下一个 block
```

或者在 backward 中提前准备下一个需要的参数。

收益取决于：

- 网络带宽和延迟。
- block 计算时间。
- wrap 粒度。
- bucket 大小。
- GPU 是否已被计算饱和。
- 是否出现过度预取导致额外显存峰值。

更激进的 prefetch 不一定更快；它可能把通信隐藏在计算后，也可能提前占满显存。

## 25. FSDP 最小结构示意

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)


def main(local_rank: int):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    base_model = build_model()

    model = FSDP(
        base_model,
        device_id=local_rank,
    )

    # 按官方约束，在包装 FSDP 后创建 optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    batch = next_batch(local_rank)
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main(local_rank=get_local_rank())
```

真实项目通常还要配置：

- auto wrap policy。
- sharding strategy。
- mixed precision policy。
- activation checkpointing。
- distributed checkpoint。
- gradient accumulation。
- CPU offload。
- process group/device mesh。

FSDP 接口持续演进，具体代码应以所使用 PyTorch 版本文档为准。

## 26. Sharding Strategy 的直觉

FSDP 的策略可以在显存与通信之间折中：

- FULL_SHARD：参数、梯度和 optimizer state 都分片。
- SHARD_GRAD_OP：梯度与 optimizer state 分片；参数在计算阶段保持未分片更久，以减少再次 all-gather。
- NO_SHARD：状态复制，行为接近 DDP。
- HYBRID_SHARD：在一组设备内 fully shard，在组间复制。

选择取决于：

- 单机还是多机。
- 节点内与节点间带宽。
- 模型大小。
- 每卡显存。
- 期望吞吐。
- 通信拓扑。

“切分越多越好”不是普遍结论。

## 27. Activation Checkpointing 与 FSDP 可以叠加

它们解决不同部分：

| 技术 | 主要减少 |
| --- | --- |
| Mixed precision | activation、部分状态与通信字节 |
| Gradient accumulation | 单个 micro-batch activation |
| Activation checkpointing | 保存的中间 activation |
| ZeRO/FSDP | 重复的参数、梯度、optimizer states |
| CPU offload | GPU 上的部分状态 |
| Efficient attention | attention 中间结果 |
| Sequence parallel | 沿 sequence 维度的部分 activation |

因此一个大模型配方可能同时使用多种技术。

但叠加后需要关注：

- 重计算增加。
- 通信与计算重叠。
- 参数 all-gather 峰值。
- CPU/GPU 传输。
- checkpoint 保存格式。
- 调试复杂度。
- 实际吞吐是否下降过多。

## 28. 分片 Checkpoint

FSDP/ZeRO 下每个 rank 只拥有部分状态，因此保存方式需要明确：

- full state dict：聚合成完整权重，方便单进程加载，但可能在 rank 0 OOM。
- sharded state dict：每个 rank 保存自己的 shard，扩展性更好。
- local state dict：保留本地分片表示，通常更依赖相同并行布局。

保存 full model/optimizer state 时，聚合动作本身可能造成新的内存峰值。

大规模训练更适合 distributed checkpoint，并把模型格式转换当成单独流程。

## 29. 7B 模型的分片直觉

使用前面的 FP32 AdamW 简化账本：

```text
7B × 16 bytes ≈ 112 GB static states
```

如果理想地在 8 张卡上做 ZeRO-3/FULL_SHARD：

```text
112 GB / 8 ≈ 14 GB static shard per GPU
```

但真实训练每卡还需要：

- 当前计算 unit 的完整参数窗口。
- activation。
- attention/MLP 临时 tensor。
- communication buffer。
- CUDA context 与 allocator reserved memory。
- 输入 batch。
- 可能的未分片模块。

所以不能得出“每张 16 GB GPU 一定能训练 7B”的结论。

## 30. OOM 优化的推荐排查顺序

### 第一步：确认峰值位置

- forward、backward、optimizer step 还是 checkpoint save？
- allocated 与 reserved 各是多少？
- 是否有意外 tensor 被 Python 容器长期引用？
- batch/sequence 是否偶发变长？

### 第二步：降低动态显存

- 减小 micro-batch。
- 使用 gradient accumulation 保持有效 batch。
- 减小 sequence length。
- 使用 efficient attention。
- 开启 activation checkpointing。
- 使用 mixed precision。

### 第三步：降低静态状态

- ZeRO/FSDP。
- optimizer state sharding。
- 低精度 optimizer state。
- CPU/NVMe offload。
- 参数高效微调，而不是全量训练。

### 第四步：重新 profile

优化一项后重新测峰值和吞吐，避免同时改变太多变量而无法判断收益来源。

## 31. 哪种方案解决哪种 OOM

| 症状 | 优先尝试 |
| --- | --- |
| batch 增大就 OOM | 减小 micro-batch + accumulation |
| sequence length 增大后 OOM | efficient attention + activation checkpointing |
| batch=1、短序列仍 OOM | ZeRO/FSDP 或 offload |
| optimizer.step 时 OOM | optimizer state sharding/offload |
| 保存 checkpoint 时 OOM | sharded/distributed state dict |
| backward 峰值很高 | activation checkpointing、调整 FSDP prefetch |
| reserved 远高于 allocated | 排查生命周期、碎片与 allocator 行为 |
| 速度很慢但显存富余 | 减少重计算/offload，增大 micro-batch |

## 32. 常见错误

- 只计算低精度模型权重，忽略梯度和 AdamW moments。
- 认为 BF16 会让全部训练显存自动减半。
- 把 activation checkpointing 当成磁盘训练 checkpoint。
- 认为 gradient accumulation 会减少 optimizer state。
- 只看平均显存，不看单个算子的 peak memory。
- 把 reserved memory 全部当成泄漏。
- 看到 OOM 就频繁清缓存，却不检查仍被引用的 tensor。
- 认为 causal attention mask 已经让 attention 显存变成线性。
- 只比较显存，不记录 tokens/s 和 step time。
- 使用 ZeRO-3 后仍按“每卡有完整参数”理解执行过程。
- 忽略 all-gather 临时完整参数造成的峰值。
- FSDP wrap 粒度过大，参数窗口太大。
- FSDP wrap 粒度过小，collective 数量过多。
- 在 FSDP 包装前创建 optimizer，参数引用不符合当前接口要求。
- 保存 full state dict 时只在训练峰值上再额外聚合。
- CPU offload 后忽略 host RAM 和 PCIe 带宽。
- 使用 activation checkpointing 的函数有不可控副作用。
- checkpoint 重计算时跨设备移动 tensor，却仍假设 RNG 完全一致。
- 把理论 `1/N` 当成真实每卡峰值显存。
- 同时开启所有优化，却没有逐项 profiling。

## 33. 概念检查

1. 为什么模型权重显存远小于完整训练显存？
2. FP32 AdamW 为什么常粗估为 16 bytes/parameter？
3. mixed precision 为什么可能仍保留 FP32 master weights？
4. activation 主要随哪些维度增长？
5. gradient accumulation 主要减少哪部分显存？
6. activation checkpointing 用什么换取显存？
7. activation checkpoint 与 training checkpoint 有什么不同？
8. ZeRO-1、2、3 分别切分哪些状态？
9. DDP 为什么不能降低每卡模型状态显存？
10. FSDP forward 前为什么需要 all-gather？
11. backward 后为什么使用 reduce-scatter？
12. wrap 粒度如何影响峰值显存和通信次数？
13. 为什么 CPU offload 可能显著降低吞吐？
14. 为什么保存 full state dict 也可能 OOM？
15. 理论 `总状态/N` 为什么不是实际峰值显存？

## 34. 今日练习

- [ ] 计算 1B、7B、13B 模型的 FP32 权重大小。
- [ ] 按 16 bytes/parameter 粗算 AdamW 静态状态。
- [ ] 分别计算 8 卡 ZeRO-1/2/3 的每卡静态状态近似。
- [ ] 测量一个小模型 forward、backward、optimizer step 的峰值显存。
- [ ] 改变 batch size，观察 activation 峰值是否近似线性变化。
- [ ] 改变 sequence length，比较普通 attention 的增长趋势。
- [ ] 对每个 Transformer block 启用 activation checkpointing，记录显存和 step time。
- [ ] 对比 BF16 与 FP32 的 peak memory 和吞吐。
- [ ] 画出 FSDP 的 all-gather → compute → reshard 时间线。
- [ ] 用四个 rank 模拟参数 shard，手工执行一次 all-gather。
- [ ] 模拟 reduce-scatter，把完整梯度和切分结果对齐。
- [ ] 设计一个 OOM 排查表，区分 forward/backward/step/save。
- [ ] 比较 full state dict 与 sharded state dict 的保存流程。
- [ ] 为当前模型写一份“显存预算 = 静态状态 + activation + buffer”的估算。

## 35. 参考资料

- [PyTorch：FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch：Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
- [PyTorch：Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [DeepSpeed：ZeRO](https://deepspeed.readthedocs.io/en/stable/zero3.html)

## 今日总结

一句话总结：**训练显存由静态模型状态、动态 activation 和临时缓冲共同决定；checkpointing 通过重计算减少 activation，ZeRO/FSDP 通过跨 rank 切分重复状态降低每卡静态显存。**

完整显存优化地图：

```text
参数/梯度/optimizer state 太大
→ ZeRO/FSDP/offload

activation 太大
→ smaller micro-batch
→ gradient accumulation
→ activation checkpointing
→ efficient attention

低精度可接受
→ autocast BF16/FP16
→ 必要时 GradScaler

多卡切分
→ all-gather parameters
→ compute
→ reduce-scatter gradients
→ local optimizer update
```

需要牢记：

- 16 bytes/parameter 只是特定 FP32 AdamW 布局的粗估。
- mixed precision、accumulation、checkpointing 和 sharding 解决不同显存组成。
- ZeRO Stage 1/2/3 依次切分 optimizer、gradient、parameter。
- FSDP 的低持久显存来自按需聚合与重新分片，不是无需完整参数计算。
- 所有省显存技术都有计算、通信、CPU 内存或工程复杂度代价。
- 最可靠的方法是先测量峰值，再针对具体组成优化。

## 明日衔接建议

从“单个 data-parallel group 内如何切分状态”扩展到完整并行体系：学习 data parallel、tensor parallel、pipeline parallel、sequence parallel 与 expert parallel，并理解 all-reduce、all-gather、reduce-scatter 和 all-to-all 分别服务于哪些通信模式。
