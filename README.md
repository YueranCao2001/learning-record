# Learning Record

我的每日学习记录。目标不是单纯点亮贡献图，而是通过持续记录、动手练习和周期复盘，建立可检索的个人知识库。

## 使用方式

1. 每天在 `daily/YYYY/MM/` 下新增一篇记录。
2. 记录当天的学习目标、核心概念、练习结果和遗留问题。
3. 在 [topics 主题索引](topics/README.md) 中按内容分类，原笔记保留日期路径。
4. 每周回顾一次，补充理解并删除错误结论。

## 内容推进原则

每日记录继续遵循 [笔记模板](templates/daily-template.md)，保留学习目标、核心概念、示例与练习、遇到的问题、总结和下一步。理论与代码可以展开为编号章节，但不以篇幅或新增术语数量衡量进度。

围绕一个项目连续推进，先检查已有结果和未完成项，再确定当天要解决的问题。[最小 RAG 项目](topics/rag.md) 于2026年10月21日阶段收尾，未完成项保留；按作者安排，自10月22日起转向 [大模型微调实践](topics/post-training.md)，从真实环境、数据与一步训练验证开始。

- 练习记录实际输入、运行方式、输出与失败案例；没有运行的任务明确列为待完成。
- 区分合成测试、真实数据实验和真实模型实验，不把局部断言通过写成系统或能力提升。
- 解释当天新增了什么、哪项理解需要修正，以及修正依据。个人经历和掌握程度只记录已确认的信息，不代替作者编造。
- 新内容优先来自当前项目的问题；回顾旧主题时说明新增证据或修正点，避免重复生成相近教程。
- 每个阶段形成一次结果复盘，未完成项继续保留；没有改善的实验也属于有效学习记录。

## 主题分类

- [Transformer 基础与生成](topics/transformer-foundations.md)
- [训练与数据工程](topics/training-engineering.md)
- [后训练与偏好优化](topics/post-training.md)
- [评估与实验方法](topics/evaluation-experiments.md)
- [推理部署与工具执行](topics/inference-tools.md)
- [RAG 检索与应用工程](topics/rag.md)

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
- [训练数据治理：质量过滤、去重、污染检测、Provenance 与数据混合](daily/2026/09/2026-09-02-data-quality-dedup-contamination.md)
- [训练异常诊断：Loss Spike、梯度范数、NaN/Inf 与 Checkpoint Rollback](daily/2026/09/2026-09-03-training-instability-loss-spike-nan-recovery.md)
- [Checkpoint 评估：Validation Loss、任务指标、Calibration 与 Evaluation Harness](daily/2026/09/2026-09-04-checkpoint-evaluation-harness.md)
- [Supervised Fine-Tuning：Chat Template、Assistant-only Loss、Packing 与 LoRA](daily/2026/09/2026-09-07-supervised-fine-tuning-chat-template-lora.md)
- [Direct Preference Optimization：偏好对、Reference Model、DPO Loss 与长度偏差](daily/2026/09/2026-09-08-direct-preference-optimization.md)

- [在线强化学习后训练：Rollout、Advantage、PPO 与 GRPO](daily/2026/09/2026-09-09-online-rl-ppo-grpo.md)
- [RL Rollout 工程：权重同步、采样 Log Probability、异步队列与 Policy Staleness](daily/2026/09/2026-09-10-rl-rollout-engineering.md)
- [可验证奖励与评估器工程：答案抽取、容差、失败分类与 Reward Hacking](daily/2026/09/2026-09-11-verifiable-rewards-evaluator.md)
- [后训练链路复盘：SFT、DPO、PPO/GRPO 的统一实验协议与验收](daily/2026/09/2026-09-14-post-training-end-to-end-review.md)
- [小模型后训练实验设计：固定 Baseline、数据切分、一步训练与恢复对照](daily/2026/09/2026-09-15-small-model-experiment-design.md)
- [评估结果的统计解释：配对比较、分组 Bootstrap 与选择偏差](daily/2026/09/2026-09-16-paired-evaluation-group-bootstrap.md)
- [错误分析与消融实验：从失败案例到可检验的改进假设](daily/2026/09/2026-09-17-error-analysis-ablation-experiments.md)
- [实验追踪与可复现报告：Run Manifest、内容指纹与结果重算](daily/2026/09/2026-09-18-experiment-tracking-reproducible-reports.md)
- [推理部署验收：延迟、吞吐、显存与失败率](daily/2026/09/2026-09-20-inference-serving-acceptance.md)
- [连续批处理与请求调度：Prefill、Decode 和 KV 容量](daily/2026/09/2026-09-21-continuous-batching-request-scheduling.md)
- [PagedAttention 与 KV 块管理：映射、共享、写时复制和释放](daily/2026/09/2026-09-22-paged-attention-kv-block-management.md)
- [推理量化：权重、KV Cache、Scale 与质量验收](daily/2026/09/2026-09-23-inference-weight-kv-quantization.md)
- [投机解码：草稿、目标验证与拒绝采样](daily/2026/09/2026-09-24-speculative-decoding-rejection-sampling.md)
- [结构化输出与约束解码：Schema、语法状态和 Token Mask](daily/2026/09/2026-09-25-structured-output-constrained-decoding.md)
- [工具调用的可靠执行：权限、幂等键、重试与结果回传](daily/2026/09/2026-09-26-reliable-tool-execution-idempotency.md)
- [检索增强生成：检索、证据组织与引用核验](daily/2026/09/2026-09-27-rag-retrieval-evidence-citations.md)
- [检索排序：BM25、向量检索、RRF 与评估](daily/2026/09/2026-09-28-retrieval-ranking-bm25-rrf-evaluation.md)
- [查询改写与多跳检索：意图保留、证据依赖和停止条件](daily/2026/09/2026-09-29-query-rewriting-multi-hop-retrieval.md)
- [RAG 端到端评估与回归测试](daily/2026/09/2026-09-30-rag-evaluation-regression-tests.md)
- [长上下文与上下文压缩](daily/2026/10/2026-10-01-long-context-evidence-compression.md)
- [RAG 提示注入与信任边界](daily/2026/10/2026-10-02-rag-prompt-injection-trust-boundaries.md)
- [RAG 缓存与失效策略](daily/2026/10/2026-10-03-rag-cache-invalidation.md)
- [RAG 增量索引与版本发布](daily/2026/10/2026-10-04-rag-incremental-index-publication.md)
- [RAG 可观测性与故障定位](daily/2026/10/2026-10-05-rag-observability-trace-diagnostics.md)
- [RAG 超时预算与降级策略](daily/2026/10/2026-10-06-rag-deadlines-retries-fallbacks.md)
- [RAG 流量控制与过载保护](daily/2026/10/2026-10-07-rag-admission-overload-protection.md)
- [RAG 工程端到端复盘与统一验收](daily/2026/10/2026-10-08-rag-end-to-end-acceptance-review.md)
- [最小离线 RAG 检索与证据实验](daily/2026/10/2026-10-09-offline-rag-retrieval-evidence-experiment.md)
- [中文文本检索与 BM25 对照实验](daily/2026/10/2026-10-10-chinese-tokenization-bm25-experiment.md)
- [将文本检索练习整理为共享基线](daily/2026/10/2026-10-11-reusable-retrieval-baseline.md)
- [用诊断案例区分检索表示与排序](daily/2026/10/2026-10-12-retrieval-ranking-diagnostic-cases.md)
- [将原文引用检查接入共享检索项目](daily/2026/10/2026-10-13-source-citation-checks.md)
- [固定生成协议与待审查问题集](daily/2026/10/2026-10-14-generation-contract-review-draft.md)
- [补齐生成试跑来源与评分依据](daily/2026/10/2026-10-15-pilot-sources-and-scoring-rubric.md)
- [使用 Ollama 完成第一道本地生成测试](daily/2026/10/2026-10-16-ollama-first-local-generation.md)
- [验证定制例外与证据不足响应](daily/2026/10/2026-10-17-exceptions-and-insufficient-evidence.md)
- [验证日期版本并冻结五题首次基线](daily/2026/10/2026-10-18-policy-version-first-pass-review.md)
- [引用充分性提示的单因素对照](daily/2026/10/2026-10-19-citation-prompt-single-factor-comparison.md)
- [引用候选提示的日期题退化](daily/2026/10/2026-10-20-citation-prompt-date-regression.md)
- [日期回归重复检查与微调阶段衔接](daily/2026/10/2026-10-21-repeatability-review-and-sft-handoff.md)
- [中文信息抽取微调：首个 LoRA 训练与 HF 同步](daily/2026/10/2026-10-22-sft-extraction-contract-and-preflight.md)
- [中文信息抽取微调：边界样例的固定预算对照](daily/2026/10/2026-10-23-sft-boundary-data-comparison.md)
- [中文信息抽取微调：冻结标注规则与开放审计](daily/2026/10/2026-10-24-sft-annotation-rules-open-audit.md)

## 提交约定

- `learn:` 新增学习记录
- `review:` 补充复盘或修正理解
- `practice:` 添加练习代码或实验
- `docs:` 调整仓库说明与结构
