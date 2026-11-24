# 推理部署与工具执行

服务验收、调度、KV 块管理、量化、投机解码、结构化输出及可靠工具调用。区分算法性质、系统正确性与实测性能。

## 本地大模型推理服务与压测

2026 年 11 月 23 日本地推理服务第一阶段收尾。正常流式、放行前超时、排队断开、入槽后断开和排队完成五条路径，在两个独立服务实例均通过回归验收。并发、前缀缓存与取消对照均已归档。[一键验收](../practice/local-inference/README.md#一键回归验收) · [阶段报告](../practice/local-inference/stage-report.md) · [日期笔记](../daily/2026/11/2026-11-23-local-inference-http-smoke.md)

第二阶段开始测固定到达速率下的积压。双槽、两条在途条件下，每秒到达 1 条和 2 条各跑 20 条，客户端排队 P90 分别为 8.12 ms 和 1700.89 ms。[第二阶段复现命令](../practice/local-inference/README.md#固定到达速率测试)

11 月 24 日测 1.25、1.375 和 1.50 条/s，共 392 条回复通过。等时长复测中，1.25 条/s 的等待维持在毫秒级，1.375 条/s 的后半程排队 P90 两轮均上升。1.50 条/s 的延迟增长主要出现在 HTTP 开始之前。[容量对照](../daily/2026/11/2026-11-24-local-inference-capacity.md)

## 日期笔记

- 2026-09-20 [推理部署验收：延迟、吞吐、显存与失败率](../daily/2026/09/2026-09-20-inference-serving-acceptance.md)
- 2026-09-21 [连续批处理与请求调度：Prefill、Decode 和 KV 容量](../daily/2026/09/2026-09-21-continuous-batching-request-scheduling.md)
- 2026-09-22 [PagedAttention 与 KV 块管理：映射、共享、写时复制和释放](../daily/2026/09/2026-09-22-paged-attention-kv-block-management.md)
- 2026-09-23 [推理量化：权重、KV Cache、Scale 与质量验收](../daily/2026/09/2026-09-23-inference-weight-kv-quantization.md)
- 2026-09-24 [投机解码：草稿、目标验证与拒绝采样](../daily/2026/09/2026-09-24-speculative-decoding-rejection-sampling.md)
- 2026-09-25 [结构化输出与约束解码：Schema、语法状态和 Token Mask](../daily/2026/09/2026-09-25-structured-output-constrained-decoding.md)
- 2026-09-26 [工具调用的可靠执行：权限、幂等键、重试与结果回传](../daily/2026/09/2026-09-26-reliable-tool-execution-idempotency.md)
- 2026-11-23 [本地推理实作：跑通第一条 HTTP 回复](../daily/2026/11/2026-11-23-local-inference-http-smoke.md)
- 2026-11-24 [本地推理实作：缩小可持续到达速率范围](../daily/2026/11/2026-11-24-local-inference-capacity.md)

[返回主题索引](README.md)
