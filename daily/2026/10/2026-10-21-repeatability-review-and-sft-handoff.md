# 2026年10月21日 日期回归重复检查与微调阶段衔接

## 今日目标

按 [昨天的待办](2026-10-20-citation-prompt-date-regression.md)，在执行前固定四次顺序，检查日期选版差异是否重复出现。今天不改提示、模型或数据，结束后保留 RAG 阶段状态。

用户已明确从明天10月22日转向大模型微调。今天只完成 RAG 收尾和后续准备清单，不安装训练依赖、下载训练模型或启动微调。

## 核心概念

固定 seed、温度和单一问题的重复运行，用于检查同一设定的输出是否再次出现，不是独立样本，也不能估计一般准确率。停止条件应在运行前固定，不能看到错误就改提示，看到成功就提前结束。

结构与原文核验仍不判断日期结论。旧版适用10月15日之前购买的设备，问题中的10月16日购买应选新版；真实旧版引文不能使错误选择变正确。

## 示例与练习

### 运行前协议

[date-repeat-protocol-v1.json](../../../practice/rag-baseline/data/date-repeat-protocol-v1.json) 在首次调用前保存，固定顺序为 A、B、B、A，其中 A 为 baseline-v1，B 为 citation-v2，各两次，共四次；不追加次数、不自动重试。若服务不可用，停止剩余调用并记录缺测。

继续使用 Qwen3 8B Q4_K_M、Ollama 0.40.5，temperature=0、seed=42、num_ctx=4096、num_predict=512、think=false。问题、证据和 Schema 均不变。该顺序不是随机试验，也没有强制清理缓存。

```powershell
python -X utf8 practice/rag-baseline/ollama_smoke.py --case version_conflict --prompt-version baseline-v1
python -X utf8 practice/rag-baseline/ollama_smoke.py --case version_conflict --prompt-version citation-v2
python -X utf8 practice/rag-baseline/ollama_smoke.py --case version_conflict --prompt-version citation-v2
python -X utf8 practice/rag-baseline/ollama_smoke.py --case version_conflict --prompt-version baseline-v1
```

### 实际结果

| 次序 | 条件 | answer 原文 | 原文引用数 |
| --- | --- | --- | --- |
| 1 | 原提示 | 应采用新版政策。 | 1 |
| 2 | 候选 | 应采用旧版政策。 | 2 |
| 3 | 候选 | 应采用旧版政策。 | 2 |
| 4 | 原提示 | 应采用新版政策。 | 1 |

四次均正常停止、无程序失败，原文核验通过；候选的事实错误仍然存在。完整记录：[第1次](../../../practice/rag-baseline/results/2026-10-21-repeat-01-baseline.json)、[第2次](../../../practice/rag-baseline/results/2026-10-21-repeat-02-candidate.json)、[第3次](../../../practice/rag-baseline/results/2026-10-21-repeat-03-candidate.json)、[第4次](../../../practice/rag-baseline/results/2026-10-21-repeat-04-baseline.json)。

[助手审查](../../../practice/rag-baseline/results/2026-10-21-repeat-review.json) 仅描述当前设定：两次原提示正确、两次候选错误，与昨天一致。不将2/2或0/2写成泛化准确率，也不据此声称已确定模型内部错误机制。

## 遇到的问题

候选的定制引用修复没有跨题保持事实正确，原文通过也没有暴露这种错误。保留 baseline-v1 默认，citation-v2 不推广。原提示日期解释不完整、真实检索链路和独立审查仍未完成，不能把转向新主题写成 RAG 已全部验收。

## 今日总结

原有41项本地测试通过。测试校验程序性质，不会把候选的日期事实错误判作正确。

本次按预定四次顺序完成运行，差异在固定设定下重复出现，所有首次和后续报告均保留。RAG 阶段暂存，不继续自动扩展提示实验。

## 下一步

从2026年10月22日起，主线转到 [后训练与微调](../../../topics/post-training.md)。先落实已有 [SFT 笔记](../09/2026-09-07-supervised-fine-tuning-chat-template-lora.md) 与 [小模型实验设计](../09/2026-09-15-small-model-experiment-design.md) 的真实待办，而不是重复一篇算法综述：确认目标与样例、训练环境、模型及模板，再执行一步训练和保存恢复检查。

需要用户提供的是候选模型仓库 ID（没有也可）、微调目标及几条脱敏数据。Hugging Face 密码、Token、带凭据的完整配置不发送、不写入仓库。若选择需要认证或审批的模型，在本机完成最小权限登录和访问申请即可；凭据与模型训练参数分开管理。
