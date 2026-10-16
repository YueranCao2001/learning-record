# RAG 检索与应用工程

检索、证据、引用、评估、安全和运行治理。当前实践主线是将已有局部练习收拢到同一个可运行的小项目。

## 日期笔记

- 2026-09-27 [检索增强生成：检索、证据组织与引用核验](../daily/2026/09/2026-09-27-rag-retrieval-evidence-citations.md)
- 2026-09-28 [检索排序：BM25、向量检索、RRF 与评估](../daily/2026/09/2026-09-28-retrieval-ranking-bm25-rrf-evaluation.md)
- 2026-09-29 [查询改写与多跳检索：意图保留、证据依赖和停止条件](../daily/2026/09/2026-09-29-query-rewriting-multi-hop-retrieval.md)
- 2026-09-30 [RAG 端到端评估与回归测试](../daily/2026/09/2026-09-30-rag-evaluation-regression-tests.md)
- 2026-10-01 [长上下文与上下文压缩](../daily/2026/10/2026-10-01-long-context-evidence-compression.md)
- 2026-10-02 [RAG 提示注入与信任边界](../daily/2026/10/2026-10-02-rag-prompt-injection-trust-boundaries.md)
- 2026-10-03 [RAG 缓存与失效策略](../daily/2026/10/2026-10-03-rag-cache-invalidation.md)
- 2026-10-04 [RAG 增量索引与版本发布](../daily/2026/10/2026-10-04-rag-incremental-index-publication.md)
- 2026-10-05 [RAG 可观测性与故障定位](../daily/2026/10/2026-10-05-rag-observability-trace-diagnostics.md)
- 2026-10-06 [RAG 超时预算与降级策略](../daily/2026/10/2026-10-06-rag-deadlines-retries-fallbacks.md)
- 2026-10-07 [RAG 流量控制与过载保护](../daily/2026/10/2026-10-07-rag-admission-overload-protection.md)
- 2026-10-08 [RAG 工程端到端复盘与统一验收](../daily/2026/10/2026-10-08-rag-end-to-end-acceptance-review.md)
- 2026-10-09 [最小离线 RAG 检索与证据实验](../daily/2026/10/2026-10-09-offline-rag-retrieval-evidence-experiment.md)
- 2026-10-10 [中文文本检索与 BM25 对照实验](../daily/2026/10/2026-10-10-chinese-tokenization-bm25-experiment.md)
- 2026-10-11 [将文本检索练习整理为共享基线](../daily/2026/10/2026-10-11-reusable-retrieval-baseline.md)
- 2026-10-12 [用诊断案例区分检索表示与排序](../daily/2026/10/2026-10-12-retrieval-ranking-diagnostic-cases.md)
- 2026-10-13 [将原文引用检查接入共享检索项目](../daily/2026/10/2026-10-13-source-citation-checks.md)
- 2026-10-14 [固定生成协议与待审查问题集](../daily/2026/10/2026-10-14-generation-contract-review-draft.md)
- 2026-10-15 [补齐生成试跑来源与评分依据](../daily/2026/10/2026-10-15-pilot-sources-and-scoring-rubric.md)
- 2026-10-16 [使用 Ollama 完成第一道本地生成测试](../daily/2026/10/2026-10-16-ollama-first-local-generation.md)

## 当前项目阶段

以下状态截至 2026 年 10 月 16 日，以仓库已有记录为依据。单元断言通过只表示对应局部检查通过。共享项目入口为 [practice/rag-baseline](../practice/rag-baseline/README.md)。

| 阶段 | 当前状态 | 下一项产物 |
| --- | --- | --- |
| 检索与原文引用检查 | 原文检查已用于一题真实模型输出，共 37 项本地测试 | 检查其他问题的真实输出，仍不替代语义支持评价 |
| 中文文本检索对照 | v2 新增词频、长度和词边界诊断，保存 Top-1 与 Top-2 共 48 条记录，原题结果不变 | 增加独立真实问题与语义证据标注，不能用构造目标总分评价业务质量 |
| 端到端生成与引用支持 | Ollama 0.40.5 与 Qwen3 8B 已接通，普通退货题一次生成及引用核验通过，助手单题审查记录已保存 | 固定配置逐题运行其余候选，完成独立审查，再对照实际检索链路 |
| 回归与复盘 | 已有规则示例，项目级闭环未完成 | 同一配置下重跑固定集，记录改进与新增退化，补充独立确认题 |

## 后续每日记录的选题

本地后端已确认并完成首题真实调用。下一步扩展按题入口，固定配置运行例外、缺证据等题并审查实际输出；控制上下文结果与实际检索链路分开记录，不用单题成功推断整体质量。

每次实验保留原配置与候选配置、实际运行命令、逐题结果及失败案例。没有改善也应记录，不能为了新增一篇笔记而切换算法或抬高结论。

## 相关主题

- [评估与实验方法](evaluation-experiments.md)，用于设计对照、分析误差与管理结果。
- [推理部署与工具执行](inference-tools.md)，用于后续集成与执行验收。

[返回主题索引](README.md)
