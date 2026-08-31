# 2026-08-31：训练性能诊断——FLOPs、MFU、Roofline 与 Profiling

## 今日目标

- 区分 throughput、latency、FLOP/s、GPU utilization 与 Model FLOPs Utilization。
- 用统一公式计算 tokens/s、每步模型 FLOPs 和 MFU。
- 理解 `6P` FLOPs/token 估算的来源、用途和边界。
- 掌握 Roofline 模型、arithmetic intensity 与 ridge point。
- 区分 compute-bound、memory-bound、communication-bound 和 input-bound。
- 学会从 end-to-end step time 逐层定位数据、CPU、GPU kernel 与网络瓶颈。
- 使用 PyTorch Profiler 构造低干扰的采样窗口和 timeline。
- 理解 Nsight Systems 与 Nsight Compute 分别回答什么问题。
- 建立“测量 → 假设 → 修改 → 复测”的性能优化闭环。

## 1. 性能优化首先要定义目标

“训练更快”可能指不同目标：

- 更高 samples/s。
- 更高有效 tokens/s。
- 更短 optimizer step time。
- 更高 MFU。
- 更低 time-to-train。
- 更低训练成本。
- 更快达到目标 validation loss。
- 更好的多卡 scaling efficiency。

这些指标可能冲突。

例如：

- 增大 batch 能提高 tokens/s，但可能改变优化轨迹。
- activation checkpointing 能避免 OOM，却增加重计算。
- 更高 MFU 不保证更快收敛。
- 训练 loss 相同不代表 wall-clock time 相同。
- 低精度提高吞吐，但需要验证数值稳定和最终质量。

性能优化必须同时守住正确性与模型质量。

## 2. Step Time

一个 optimizer update 的 wall-clock 时间：

```text
step_time
= 从本次 update 开始
  到参数更新完成
```

使用 gradient accumulation 时，应明确：

- micro-step time。
- accumulation window time。
- optimizer update time。

如果比较不同 accumulation 配置，只看 micro-step 会产生误导，应比较相同有效 token 数或相同 optimizer updates。

## 3. Tokens per Second

全局有效 tokens/s：

```text
global_tokens_per_second
= global_valid_tokens_per_update
  / update_time
```

其中：

```text
global_valid_tokens
= Σ_ranks Σ_microbatches non_padding_tokens
```

不要把以下内容无条件计入有效 token：

- padding。
- 被 `ignore_index` 排除的位置。
- 只作为上下文、不计算 loss 的 token。
- 重复计算但不产生新训练目标的滑动窗口前缀。

如果每个 rank 和 batch 长度固定，可近似：

```text
tokens/update
≈ micro_batch
  × sequence_length
  × accumulation_steps
  × data_parallel_size
```

但动态 padding、packing 和 CP/TP/PP 会让这个公式需要调整。

## 4. Throughput 与 Latency

### Throughput

单位时间完成多少工作：

```text
tokens/s
samples/s
optimizer updates/s
```

### Latency

一个工作单元完成需要多久：

```text
micro-step latency
update latency
request latency
```

训练通常更重视 throughput；在线推理还需要关注首 token 延迟和 token 间延迟。

高 throughput 可以通过 batching 获得，但单个样本 latency 可能上升。

## 5. FLOPs 是什么

FLOP 表示一次浮点运算；FLOP/s 表示每秒执行的浮点运算数量。

矩阵乘法：

```text
A: [M, K]
B: [K, N]
C = A @ B
```

每个输出元素包含 `K` 次乘法和约 `K` 次加法，常近似：

```text
FLOPs ≈ 2 M N K
```

不同工具对 fused multiply-add 的计数约定必须一致：有的按 2 FLOPs，有的按 1 个指令事件。比较数字前先确认定义。

## 6. Dense Transformer 的 6P 估算

设模型有 `P` 个参与计算的参数。

对每个 token：

### Forward

权重矩阵大致参与一次乘加：

```text
forward ≈ 2P FLOPs/token
```

### Backward

通常包含：

- 对输入求梯度。
- 对权重求梯度。

粗略再需要 forward 的约两倍：

```text
backward ≈ 4P FLOPs/token
```

所以：

```text
training ≈ 6P FLOPs/token
```

每个 update：

```text
model_FLOPs/update
≈ 6P × global_valid_tokens/update
```

这是方便估算的主项，不是精确 profiler 结果。

## 7. 6P 估算的边界

`6P` 可能忽略或简化：

- Attention 的 `T²` 项。
- Softmax、normalization 和激活函数。
- embedding/LM Head 的实现。
- weight tying。
- activation checkpointing 重计算。
- MoE 只激活部分 experts。
- 稀疏算子。
- padding 与无效 token。
- optimizer step。
- 通信。
- kernel launch 和数据移动。
- 不同 backward 算法。

对于长 sequence，attention 项可能不可忽略；对于 MoE，应区分 total parameters 与 activated parameters。

因此记录 MFU 时必须说明 FLOPs 模型。

## 8. Achieved Model FLOP/s

先计算每步模型理论工作量：

```text
model_FLOPs_per_step
```

全局 achieved model FLOP/s：

```text
achieved_model_FLOP/s
= model_FLOPs_per_step / step_time
```

每 GPU：

```text
achieved_model_FLOP/s_per_GPU
= model_FLOPs_per_step
  / step_time
  / number_of_GPUs
```

示例：

```text
P = 7e9
global tokens/update = 2e6
step_time = 10 s
GPUs = 64

model FLOPs/update
≈ 6 × 7e9 × 2e6
= 8.4e16 FLOPs

per-GPU achieved
≈ 8.4e16 / 10 / 64
≈ 1.31e14 FLOP/s
≈ 131 TFLOP/s
```

## 9. Model FLOPs Utilization

MFU：

```text
MFU
= achieved_model_FLOP/s_per_GPU
  / peak_hardware_FLOP/s_per_GPU
```

如果该 GPU 在目标 dtype 下理论峰值是 300 TFLOP/s：

```text
MFU ≈ 131 / 300 ≈ 43.7%
```

MFU 试图回答：

> 模型算法上“有用的 FLOPs”相对于设备理论峰值利用了多少？

它是端到端指标，会受到：

- kernel efficiency。
- 数据加载。
- CPU launch。
- 通信。
- pipeline bubble。
- activation checkpointing。
- optimizer。
- idle gaps。
- straggler。

## 10. Peak FLOP/s 必须选对

硬件峰值取决于：

- dtype：FP32、TF32、BF16、FP16、FP8 等。
- Tensor Core 还是普通 CUDA Core。
- dense 还是结构化 sparsity 峰值。
- boost clock 与实际 clock。
- GPU 型号和形态。
- 是否使用特定指令路径。

常见错误是：

- 实际 BF16 dense 训练，却用 FP16 sparse 峰值做分母。
- kernel 没走 Tensor Core，却用 Tensor Core 峰值。
- 多种 dtype 混合，仍选一个不匹配的峰值。
- 把整机峰值与单 GPU achieved 值比较。

MFU 的分母必须写入实验记录。

## 11. MFU 与 Hardware FLOPs Utilization

可以区分：

### MFU

```text
模型定义中的有效 FLOPs
/ 理论硬件峰值
```

### HFU

```text
硬件实际执行的全部 FLOPs
/ 理论硬件峰值
```

Activation checkpointing 会重新执行部分 forward：

- 重计算增加真实硬件 FLOPs。
- 模型语义上的 FLOPs 通常不变。
- HFU 可能较高。
- MFU 可能下降，因为 step time 增加。

因此“GPU 很忙”和“模型有效工作效率高”不是完全同一件事。

## 12. MFU 不能替代 Time-to-Train

两个配置：

| 配置 | MFU | tokens/s | 收敛所需 tokens |
| --- | ---: | ---: | ---: |
| A | 高 | 高 | 多 |
| B | 稍低 | 稍低 | 少 |

最终谁先达到目标质量，需要看：

```text
time_to_target
≈ required_training_tokens
  / effective_tokens_per_second
```

训练性能实验还必须验证：

- loss 曲线。
- validation metrics。
- 数值稳定。
- batch size/learning rate 配方。
- 数据顺序与有效 token。

## 13. Roofline 模型

Roofline 给出一个 kernel 的理论性能上界：

```text
achievable_performance
≤ min(
    peak_compute,
    memory_bandwidth × arithmetic_intensity
)
```

横轴：

```text
arithmetic intensity
= FLOPs / bytes_moved
```

纵轴：

```text
achieved FLOP/s
```

图形包含：

- 斜线：memory-bandwidth roof。
- 水平线：compute roof。
- 两者交点：ridge point。

## 14. Ridge Point

```text
ridge_point
= peak_compute_FLOP/s
  / peak_memory_bandwidth_bytes/s
```

如果 kernel 的 arithmetic intensity 小于 ridge point：

```text
memory-bound
```

如果大于 ridge point：

```text
可能 compute-bound
```

“可能”很重要。即使位于 compute-bound 区域，kernel 也可能因为占用率、依赖、分支、tile 形状或 launch overhead 远低于 compute roof。

## 15. Arithmetic Intensity

Arithmetic intensity：

```text
AI = 执行 FLOPs / 从某级内存移动的字节
```

“字节”可能针对：

- HBM/DRAM。
- L2。
- L1/shared memory。
- register file。

传统 Roofline 常以 HBM 流量为主；hierarchical roofline 会分别分析多级 memory hierarchy。

提高 AI 的方法：

- 增加数据复用。
- kernel fusion。
- 使用更大的 GEMM。
- 避免中间 tensor 写回 HBM。
- tiling。
- 增大有效 batch/token 维度。
- 使用更低精度，减少每个元素的字节数。

## 16. GEMM 为什么常有较高 AI

矩阵乘法可以把加载到片上存储的 tile 重复用于多次乘加。

```text
compute grows roughly with M×N×K
data grows roughly with M×K + K×N + M×N
```

当矩阵足够大时，计算增长快于数据移动，因此 AI 较高。

但以下情况可能降低效率：

- M/N/K 很小。
- 维度不适合硬件 tile。
- TP 把局部矩阵切得过小。
- micro-batch 太小。
- sequence packing 效率低。
- 大量小 GEMM。
- MoE expert token 数不均衡。

## 17. Pointwise Operator 为什么常 Memory-Bound

例如：

```text
y = dropout(gelu(x + bias))
```

每个元素只做少量运算，却需要多次读取和写回 tensor。

如果拆成多个 kernels：

```text
read/write for bias add
read/write for gelu
read/write for dropout
```

Kernel fusion 可以：

- 减少中间 tensor。
- 减少 HBM traffic。
- 减少 kernel launch。
- 提高 arithmetic intensity。

但融合后仍需用 profiler 验证，而不是默认一定更快。

## 18. Compute-Bound 的优化方向

典型信号：

- kernel 靠近 compute roof。
- Tensor Core utilization 高。
- HBM bandwidth 没到上限。
- 大型 GEMM 占主要时间。

可能优化：

- 使用合适低精度。
- 让矩阵维度匹配 tile。
- 增大局部 GEMM。
- 减少 TP degree，避免切得过碎。
- 使用优化 library/kernel。
- 合并小 GEMM。
- 改善 load balance。
- 减少不必要重计算。

当 kernel 已接近 compute roof，再优化 memory access 的收益可能有限。

## 19. Memory-Bound 的优化方向

典型信号：

- HBM bandwidth 接近上限。
- achieved FLOP/s 受斜线 roof 限制。
- pointwise/reduction/operator 占时明显。
- 算术强度低。

可能优化：

- kernel fusion。
- 减少中间 tensor materialization。
- 降低 dtype 字节数。
- 提高数据复用。
- efficient attention。
- 减少重复 all-gather 后的内存拷贝。
- 改善 memory layout。
- 避免不必要的 contiguous/cast/copy。

## 20. End-to-End Step 分解

一个训练 update 可以拆成：

```text
data wait
→ host preprocessing/tokenization
→ host-to-device copy
→ forward compute
→ forward communication
→ backward compute
→ gradient communication
→ gradient clipping
→ optimizer step
→ scheduler/logging/checkpoint
→ synchronization/idle
```

真实 timeline 中这些阶段可能重叠。

因此：

```text
step_time
≠ 各阶段独立耗时的简单相加
```

优化重点应放在 critical path，而不是所有事件中累计时间最大的项目。

## 21. GPU Utilization 为什么容易误导

监控工具中的 GPU utilization 通常表示采样窗口内 GPU 是否执行某类工作，不等于：

- Tensor Core 已接近峰值。
- HBM bandwidth 已饱和。
- MFU 很高。
- 所有 GPUs 负载均衡。
- 没有短暂 idle gap。
- 执行的都是模型有效工作。

GPU utilization 99% 仍可能是低效小 kernel、重计算或 busy-wait。

需要结合：

- timeline。
- kernel duration。
- FLOP/s。
- memory bandwidth。
- MFU。
- tokens/s。

## 22. 正确测量 Wall-Clock

CUDA 默认异步执行。错误示例：

```python
start = time.perf_counter()
loss = model(batch).loss
elapsed = time.perf_counter() - start
```

这里 CPU 可能只测到 kernel launch 时间。

简单同步测量：

```python
torch.cuda.synchronize()
start = time.perf_counter()

loss = model(batch).loss
loss.backward()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

但频繁 synchronize 会破坏通信计算 overlap。正式 benchmark 更适合：

- 先 warmup。
- 用 CUDA events。
- 测量完整稳定区间。
- 只在区间边界同步。
- 多次重复并报告分布。

## 23. Warmup Iterations

前几步可能包含：

- CUDA context 初始化。
- kernel lazy loading。
- memory allocator 扩容。
- library autotuning。
- graph/compiler 编译。
- communication group 初始化。
- cache 填充。
- dataset prefetch 尚未稳定。

因此 benchmark 应：

```text
warmup steps
→ measured steps
→ median / mean / percentile
```

不要把第一步与稳定阶段平均在一起。

## 24. PyTorch Profiler 最小窗口

```python
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
)


profiler_schedule = schedule(
    wait=2,
    warmup=2,
    active=3,
    repeat=1,
)

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    schedule=profiler_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./profiler_logs"
    ),
    record_shapes=False,
    profile_memory=True,
    with_stack=False,
) as prof:
    for step, batch in enumerate(train_loader):
        with record_function("train_step"):
            train_step(batch)

        prof.step()

        if step >= 10:
            break
```

schedule 避免对整个长训练持续采集高开销 trace。

`prof.step()` 告诉 profiler 当前 iteration 已结束；忘记调用会让 schedule 不按预期推进。

## 25. Profiler 开销

以下选项会增加开销：

- record shapes。
- stack traces。
- memory profiling。
- 大量 Python annotations。
- 记录过长窗口。
- 多 rank 同时写 trace。
- 高频 tensorboard handler。

Profiler 可能改变：

- kernel launch timing。
- CPU overhead。
- 内存生命周期。
- communication overlap。
- 文件系统负载。

因此 profiler 用于定位结构，稳定吞吐应使用低干扰 benchmark 单独测量。

## 26. record_function 的价值

给关键区域添加语义范围：

```python
with record_function("data_to_device"):
    batch = move_to_device(batch)

with record_function("forward"):
    outputs = model(**batch)

with record_function("backward"):
    outputs.loss.backward()

with record_function("optimizer"):
    optimizer.step()
```

Timeline 中就能快速区分：

- CPU 数据等待。
- H2D。
- forward。
- backward。
- optimizer。
- logging/checkpoint。

范围名称应稳定且层级清晰，避免每一步动态生成大量不同名称。

## 27. Nsight Systems 与 Nsight Compute

### Nsight Systems

回答系统级问题：

- CPU 与 GPU 是否存在空洞？
- kernel launch 是否及时？
- CUDA streams 是否并发？
- NCCL 是否与计算重叠？
- 多 GPU 是否有 straggler？
- H2D 是否阻塞？
- pipeline stages 是否空闲？

### Nsight Compute

回答单个 CUDA kernel 问题：

- 算术强度是多少？
- 接近 compute roof 还是 memory roof？
- Tensor Core 是否有效使用？
- memory throughput 如何？
- occupancy、warp stall 和指令瓶颈是什么？

推荐顺序：

```text
先 Systems 找 critical kernels/gaps
→ 再 Compute 深挖少数关键 kernels
```

不要一开始就对整个训练任务采集所有 kernel 级指标。

## 28. Communication-Bound

典型信号：

- NCCL collective 占 critical path。
- GPU compute 中间出现通信等待。
- 增加 GPU 后 tokens/s 几乎不增长。
- TP/EP degree 增大后局部 GEMM 变小。
- 跨节点比单节点明显退化。
- 某些 rank 进入 collective 更晚。
- all-to-all 消息不均衡。

可能优化：

- 调整并行映射，让高频通信留在高速互联。
- 增大 communication bucket。
- overlap communication and compute。
- 减少 TP/EP degree。
- 使用 sequence parallel/reduce-scatter 避免重复 activation。
- 改善 expert load balance。
- 减少小消息数量。
- 使用分层 collective。
- 修复先到/后到的 straggler。

## 29. Overlap 不能用耗时相加判断

假设：

```text
compute = 80 ms
communication = 40 ms
step = 90 ms
```

不能说 communication 占 `40/90`，因为大部分可能与 compute 重叠。

真正要看：

- communication 在 critical path 上暴露多少。
- collective 是否在正确 stream 启动。
- 上游 gradient 是否及时 ready。
- compute kernel 是否与 NCCL 并发。
- overlap 是否引入额外 contention。

可定义暴露通信时间：

```text
exposed_comm
≈ step_time_with_comm
  - ideal_step_time_if_comm_fully_hidden
```

实际需要通过 timeline 和对照实验估计。

## 30. Input-Bound

典型 timeline：

```text
GPU kernels
→ 长空白
→ 下一步 kernels
```

同时 CPU/DataLoader 活跃或等待 I/O。

常见原因：

- 在线 tokenizer 太慢。
- 文件碎片化、小文件过多。
- 网络存储延迟。
- 解压缩。
- Python 数据增强。
- workers 太少或过多。
- pinned memory 未使用。
- prefetch 不足。
- 每个 rank 读取相同数据。
- checkpoint/logging 阻塞数据线程。

可能优化：

- 离线 tokenization。
- 连续 binary/shard 格式。
- 增加合理 workers。
- pinned memory + non_blocking H2D。
- 双缓冲/prefetch。
- 本地缓存。
- 异步 checkpoint。
- 减少 Python per-sample overhead。
- sequence packing 提高有效 token 比例。

## 31. DataLoader 不是越多 Workers 越快

workers 过多可能造成：

- CPU oversubscription。
- 内存暴涨。
- 文件系统随机读取。
- 进程切换开销。
- 每个 distributed rank 重复创建大量 workers。
- NUMA 跨节点访问。
- pinned memory 压力。

应对 `num_workers` 做 sweep，并记录：

- batch wait time。
- CPU utilization。
- I/O throughput。
- host memory。
- tokens/s。
- GPU idle gaps。

## 32. Optimizer-Bound

大模型 AdamW step 可能明显占时：

- 读取参数。
- 读取梯度。
- 读取/写回一阶矩。
- 读取/写回二阶矩。
- 写回参数。

它的 arithmetic intensity 较低，常偏 memory-bound。

可能优化：

- fused optimizer。
- foreach/multi-tensor implementation。
- optimizer state sharding。
- precision-aware optimizer state。
- CPU offload（显存换传输，未必更快）。
- 降低 optimizer step 频率需要谨慎，等价于改变有效 batch。

如果 backward 很快但 optimizer 区域很长，应单独 profile，而不是只优化 Transformer kernels。

## 33. Logging 与 Checkpoint Bottleneck

同步执行：

```text
GPU → CPU copy
→ serialization
→ filesystem/network write
```

会让所有 ranks 等待。

建议：

- 只记录必要标量。
- 降低日志频率。
- 避免每步调用 `.item()` 造成同步。
- 使用异步 checkpoint。
- 使用 sharded checkpoint。
- 将保存频率与故障恢复需求平衡。
- 测试 rank 0 是否成为 I/O hotspot。
- 把 checkpoint 时间纳入 time-to-train，而不是从 benchmark 中完全隐藏。

## 34. Strong Scaling

固定总问题规模，增加 GPU：

```text
strong_scaling_efficiency(N)
= throughput_N
  / (N × throughput_1)
```

理想值为 1。

随着 N 增大：

- 每 GPU 工作量下降。
- 通信占比上升。
- kernel 变小。
- pipeline/collective latency 更明显。
- scaling efficiency 通常下降。

Strong scaling 回答：同一个 global batch/工作量能否更快完成。

## 35. Weak Scaling

保持每 GPU 工作量不变，增加 GPU，同时增大全局问题规模。

理想状态：

```text
step_time_N ≈ step_time_1
throughput_N ≈ N × throughput_1
```

但 global batch 增长可能改变优化语义，所以系统 scaling 与训练收敛应分别评价。

## 36. Straggler

同步训练的 step 由最慢 rank 决定。

Straggler 来源：

- 数据长度不同。
- MoE expert load imbalance。
- pipeline stage 不均衡。
- GPU 温度/降频。
- 网络拥塞。
- CPU/I/O 抖动。
- 某个 rank 频繁分配内存。
- 后台进程。
- 故障重试或 page fault。

平均 rank 时间可能看不出问题，应记录 per-rank 分布和最大值。

## 37. 性能实验的控制变量

一次实验尽量只改变一个主要因素：

- micro-batch。
- sequence length。
- dtype。
- TP/PP/DP degree。
- activation checkpointing。
- kernel implementation。
- compile。
- DataLoader workers。

同时固定：

- 模型配置。
- 数据和有效 token 数。
- optimizer/scheduler。
- warmup steps。
- 测量窗口。
- GPU clocks/功耗模式（如果环境允许）。
- 节点和网络拓扑。
- 软件版本。

否则无法把速度变化归因到某个修改。

## 38. 一个性能记录函数

```python
def summarize_step(
    *,
    parameter_count: int,
    global_valid_tokens: int,
    step_seconds: float,
    num_gpus: int,
    peak_flops_per_gpu: float,
) -> dict[str, float]:
    model_flops = (
        6.0
        * parameter_count
        * global_valid_tokens
    )

    tokens_per_second = (
        global_valid_tokens
        / step_seconds
    )

    achieved_per_gpu = (
        model_flops
        / step_seconds
        / num_gpus
    )

    mfu = (
        achieved_per_gpu
        / peak_flops_per_gpu
    )

    return {
        "step_seconds": step_seconds,
        "tokens_per_second": tokens_per_second,
        "model_flops_per_step": model_flops,
        "achieved_flops_per_gpu": achieved_per_gpu,
        "mfu": mfu,
    }
```

使用时还应记录：

- `6P` 近似是否适用。
- peak FLOP/s 对应的 dtype/dense/sparse 定义。
- 是否有 activation recomputation。
- 参数量使用 total 还是 active。
- padding 是否排除。

## 39. 从 Timeline 识别模式

### 模式 A：GPU 大片空白

优先检查：

- input pipeline。
- CPU launch。
- 同步调用。
- pipeline bubble。

### 模式 B：NCCL 覆盖 compute 尾部

优先检查：

- collective 启动时机。
- bucket。
- rank arrival time。
- topology。

### 模式 C：大量极短 kernels

优先检查：

- fusion。
- Python dispatch。
- graph compilation。
- micro-batch/矩阵尺寸。

### 模式 D：少数长 GEMM

使用 Nsight Compute 分析：

- Tensor Core。
- tile。
- arithmetic intensity。
- compute roof。

### 模式 E：optimizer 区域很长

检查：

- fused optimizer。
- state dtype。
- memory bandwidth。
- sharding/offload。

## 40. 优化闭环

```text
1. 定义业务/训练目标。
2. 建立可复现 baseline。
3. 记录 correctness 与 quality。
4. 测 end-to-end tokens/s 和 step time。
5. 用 timeline 找 critical path。
6. 对少数关键 kernel/collective 深挖。
7. 提出一个可证伪假设。
8. 一次修改一个主要变量。
9. 重复相同测量窗口。
10. 比较速度、显存、稳定性和质量。
11. 保留有效修改，回退无效修改。
```

性能工程不是“把所有优化 flag 打开”，而是缩短 critical path。

## 41. 常见错误

- 把 GPU utilization 当成 MFU。
- 只看 samples/s，不看有效 tokens/s。
- 把 padding token 计入吞吐。
- 使用 `6P` 却不说明它忽略了什么。
- 用 total MoE parameters 计算每 token FLOPs。
- 选错 dtype/sparsity 的硬件峰值。
- 把 activation recomputation 计入 model FLOPs numerator，却仍称 MFU。
- 只比较 MFU，不看 time-to-target。
- 用 CPU timer 测异步 CUDA，却没有边界同步。
- 把第一步编译/初始化时间算入稳定吞吐。
- 每一步都开启完整 profiler。
- 开启 record_shapes/with_stack 后仍把 profiler 吞吐当 baseline。
- 只看 operator 累计时间，不看 timeline critical path。
- 把通信总时长直接加到 compute 上，忽略 overlap。
- 看到 NCCL 时间长就断言 communication-bound。
- 只 profile rank 0，错过其他 rank 的 straggler。
- 只测单步，不报告方差和 percentile。
- 增加 DataLoader workers，却不看 CPU 与 I/O 饱和。
- 每步频繁 `.item()`，引入隐式同步。
- 同时修改 batch、dtype、parallelism 和 compile，无法归因。
- 优化后未重新验证 loss 和数值稳定。
- 只优化微基准 kernel，端到端 step 没有变快。

## 42. 概念检查

1. tokens/s 与 step time 各自适合比较什么？
2. `6P` FLOPs/token 为什么是粗略值？
3. MFU 的分子和分母分别是什么？
4. MFU 与 HFU 有什么区别？
5. activation checkpointing 为什么可能让 HFU 高而 MFU 下降？
6. 如何选择正确的 peak FLOP/s？
7. Roofline 的两个 roof 分别代表什么？
8. arithmetic intensity 的单位是什么？
9. ridge point 如何计算？
10. GEMM 为什么通常比 pointwise operator 更容易 compute-bound？
11. 为什么 naive CPU timer 会低估 CUDA 时间？
12. PyTorch Profiler 的 schedule 和 `prof.step()` 有什么作用？
13. Nsight Systems 与 Nsight Compute 应按什么顺序使用？
14. 为什么 communication duration 不等于 exposed communication？
15. Strong scaling 与 weak scaling 有什么区别？
16. 为什么同步训练性能由最慢 rank 决定？

## 43. 今日练习

- [ ] 计算一个 `M×K` 与 `K×N` GEMM 的 FLOPs。
- [ ] 用 `6P` 估算 1B、7B、70B 模型每 token 训练 FLOPs。
- [ ] 根据真实有效 token 数计算 tokens/s。
- [ ] 选择正确 dtype 峰值并计算 MFU。
- [ ] 改变 activation checkpointing，比较 MFU、HFU 直觉和 step time。
- [ ] 根据峰值 compute/bandwidth 计算 ridge point。
- [ ] 为一个小 GEMM 粗算 arithmetic intensity。
- [ ] 比较 pointwise unfused 与 fused 的 HBM traffic。
- [ ] 用 CUDA synchronization 纠正 naive timer。
- [ ] 使用 PyTorch Profiler 采集 3 个 active steps。
- [ ] 用 `record_function` 标注 data/forward/backward/optimizer。
- [ ] 从 timeline 找出最长 GPU idle gap。
- [ ] 检查 NCCL collective 是否与 backward 重叠。
- [ ] 对 `num_workers` 做 sweep 并记录 tokens/s。
- [ ] 比较 1、2、4、8 GPUs 的 strong scaling efficiency。
- [ ] 记录每个 rank 的 step time，找出 straggler。
- [ ] 建立实验表：配置、显存、tokens/s、MFU、loss、备注。

## 44. 参考资料

- [PyTorch：Profiler](https://docs.pytorch.org/docs/stable/profiler)
- [PyTorch：Profiler Recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA：Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA：Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
- [NVIDIA：Roofline 与 Arithmetic Intensity](https://developer.nvidia.com/blog/unleashing-power-of-nvidia-ampere-architecture-with-nsight-developer-tools/)

## 今日总结

一句话总结：**训练性能诊断要从端到端有效 tokens/s 出发，用 MFU 衡量模型有效计算，用 Roofline 判断 kernel 的硬件上限，再用 timeline 找到真正暴露在 critical path 上的瓶颈。**

诊断路径：

```text
先确认正确性与有效 token
→ 测稳定 step time / tokens/s
→ 计算 FLOPs 与 MFU
→ timeline 分解 data/compute/comm/idle
→ Systems 找 critical path
→ Compute 分析关键 kernel 的 roofline
→ 一次修改一个变量
→ 复测吞吐、显存、稳定性与质量
```

需要牢记：

- GPU utilization 高不等于 MFU 高。
- `6P` 是 dense Transformer 的粗估，必须说明边界。
- Roofline 上界是 `min(compute roof, bandwidth × AI)`。
- 通信只有暴露在 critical path 的部分才直接增加 step time。
- Profiler 会改变被测程序，吞吐 benchmark 与详细 trace 应分开。
- 性能最终要服务于 time-to-target，而不是追求孤立指标。

## 明日衔接建议

从“GPU 为什么会等数据”深入训练数据流水线：学习 raw text、tokenization、document boundaries、packing、shuffling、sampling、DataLoader prefetch、pinned memory、异步 H2D，以及如何同时保证数据随机性、可恢复性和高有效 token 利用率。
