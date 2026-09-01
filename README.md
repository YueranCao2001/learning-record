# Learning Record

我的每日学习记录。目标不是单纯点亮贡献图，而是通过持续记录、动手练习和周期复盘，建立可检索的个人知识库。

## 使用方式

1. 每天在 `daily/YYYY/MM/` 下新增一篇记录。
2. 记录当天的学习目标、核心概念、练习结果和遗留问题。
3. 将值得长期保留的内容整理到 `topics/`。
4. 每周回顾一次，补充理解并删除错误结论。

## 当前主题

- [Attention：从直觉到公式](daily/2026/08/2026-08-03-attention.md)
- [Attention：从公式到最小实现](daily/2026/08/2026-08-04-attention-implementation.md)
- [Multi-Head Attention：多头拆分、计算与合并](daily/2026/08/2026-08-05-multi-head-attention.md)
- [位置编码：让 Attention 理解顺序](daily/2026/08/2026-08-10-positional-encoding.md)
- [RoPE：用旋转编码相对位置](daily/2026/08/2026-08-11-rope.md)
- [Transformer 解码器块：串起完整前向传播](daily/2026/08/2026-08-12-transformer-decoder-block.md)
- [KV Cache：从显存公式到 MQA 与 GQA](daily/2026/08/2026-08-18-kv-cache-mqa-gqa.md)
- [从 Logits 到 Token：Temperature、Top-k 与 Top-p](daily/2026/08/2026-08-24-logits-sampling.md)
- [语言模型训练目标：Label Shift、Cross-Entropy 与 Perplexity](daily/2026/08/2026-08-25-next-token-training.md)
- [稳定训练一步：反向传播、AdamW、梯度裁剪与 Warmup](daily/2026/08/2026-08-26-backprop-adamw-training-step.md)
- [训练显存拆解：Activation Checkpointing、ZeRO 与 FSDP](daily/2026/08/2026-08-27-training-memory-fsdp-zero.md)
- [大模型并行全景：DP、TP、PP、SP 与 EP](daily/2026/08/2026-08-28-distributed-parallelism-collectives.md)
- [训练性能诊断：FLOPs、MFU、Roofline 与 Profiling](daily/2026/08/2026-08-31-training-performance-roofline-mfu.md)
- [训练数据流水线：Packing、Shuffling、Prefetch 与可恢复采样](daily/2026/09/2026-09-01-training-data-pipeline.md)

## 提交约定

- `learn:` 新增学习记录
- `review:` 补充复盘或修正理解
- `practice:` 添加练习代码或实验
- `docs:` 调整仓库说明与结构
