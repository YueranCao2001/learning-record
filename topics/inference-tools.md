# 推理部署与工具执行

服务验收、调度、KV 块管理、量化、投机解码、结构化输出及可靠工具调用。区分算法性质、系统正确性与实测性能。

## 本地大模型推理服务与压测

2026 年 11 月 23 日完成本地 HTTP、流式客户端和 24 条单并发基线。首轮并发对照与三次复验共 128 条。追加的 64 条等长输入对照比较了 20 和 117 token 输出，长输出组观察到更大的并发吞吐收益。下一步比较客户端并发超过服务槽数时的表现。[阶段报告](../practice/local-inference/stage-report.md) · [日期笔记](../daily/2026/11/2026-11-23-local-inference-http-smoke.md)

## 日期笔记

- 2026-09-20 [推理部署验收：延迟、吞吐、显存与失败率](../daily/2026/09/2026-09-20-inference-serving-acceptance.md)
- 2026-09-21 [连续批处理与请求调度：Prefill、Decode 和 KV 容量](../daily/2026/09/2026-09-21-continuous-batching-request-scheduling.md)
- 2026-09-22 [PagedAttention 与 KV 块管理：映射、共享、写时复制和释放](../daily/2026/09/2026-09-22-paged-attention-kv-block-management.md)
- 2026-09-23 [推理量化：权重、KV Cache、Scale 与质量验收](../daily/2026/09/2026-09-23-inference-weight-kv-quantization.md)
- 2026-09-24 [投机解码：草稿、目标验证与拒绝采样](../daily/2026/09/2026-09-24-speculative-decoding-rejection-sampling.md)
- 2026-09-25 [结构化输出与约束解码：Schema、语法状态和 Token Mask](../daily/2026/09/2026-09-25-structured-output-constrained-decoding.md)
- 2026-09-26 [工具调用的可靠执行：权限、幂等键、重试与结果回传](../daily/2026/09/2026-09-26-reliable-tool-execution-idempotency.md)
- 2026-11-23 [本地推理实作：跑通第一条 HTTP 回复](../daily/2026/11/2026-11-23-local-inference-http-smoke.md)

[返回主题索引](README.md)
