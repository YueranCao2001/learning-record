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

## 当前项目阶段

以下状态截至 2026 年 10 月 12 日，以仓库已有记录为依据。单元断言通过只表示对应局部检查通过。共享项目入口为 [practice/rag-baseline](../practice/rag-baseline/README.md)。

| 阶段 | 当前状态 | 下一项产物 |
| --- | --- | --- |
| 检索与原文引用检查 | 检索已有共享模块与 12 项测试；原文引用检查仍分散在旧笔记 | 接入来源版本、原文及输入证据集合的引用检查 |
| 中文文本检索对照 | v2 新增词频、长度和词边界诊断，保存 Top-1 与 Top-2 共 48 条记录，原题结果不变 | 增加独立真实问题与语义证据标注，不能用构造目标总分评价业务质量 |
| 端到端生成与引用支持 | 尚未在当前项目中接入 | 在明确模型或 API 与预算后，保存实际上下文、答案、引用及人工核验结果 |
| 回归与复盘 | 已有规则示例，项目级闭环未完成 | 同一配置下重跑固定集，记录改进与新增退化，补充独立确认题 |

## 后续每日记录的选题

已补充可区分现有方法的诊断案例，下一步接入已有原文引用检查。诊断目标不是独立质量评估，后续仍需确认题与真实证据标注；在共享项目中迭代，不要每天重建一套相近的检索与评估函数。

每次实验保留原配置与候选配置、实际运行命令、逐题结果及失败案例。没有改善也应记录，不能为了新增一篇笔记而切换算法或抬高结论。

## 相关主题

- [评估与实验方法](evaluation-experiments.md)，用于设计对照、分析误差与管理结果。
- [推理部署与工具执行](inference-tools.md)，用于后续集成与执行验收。

[返回主题索引](README.md)
