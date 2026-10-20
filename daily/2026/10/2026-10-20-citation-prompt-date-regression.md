# 2026年10月20日 引用候选提示的日期题退化

## 今日目标

继续 [昨天的引用提示对照](2026-10-19-citation-prompt-single-factor-comparison.md)，先用日期题检查 citation-v2 是否存在副作用。模型、输入、Schema 和生成参数保持不变，不同时修改日期解释要求。

候选首次输出选错版本后，本次再运行一次原提示作为当日控制。两个条件各调用一次，无自动重试，也没有修改提示后挑选更好的结果。

## 核心概念

原文真实、引用充分和答案事实性是不同维度。合法引用可能与当前购买日期不适用，也可能被用于支持错误结论；引用数增加不意味着答案更可靠。

候选提示应先经过回归再决定是否推广。昨天修复定制题的引用缺口，不能抵消今天在另一题上出现的选版错误。默认仍为 baseline-v1，不因局部修复切换。

## 示例与练习

### 输入与执行顺序

问题询问2026年10月16日购买的普通设备应采用哪份政策。旧版仅适用于10月15日之前购买的设备，期限30天；新版适用于10月15日及之后购买，期限7天。证据均为合成政策，今天没有更改其内容。

继续使用 Ollama 0.40.5、Qwen3 8B Q4_K_M，temperature=0、seed=42、num_ctx=4096、num_predict=512、think=false。按以下顺序实际执行：

```powershell
python -X utf8 practice/rag-baseline/ollama_smoke.py --case version_conflict --prompt-version citation-v2
python -X utf8 practice/rag-baseline/ollama_smoke.py --case version_conflict --prompt-version baseline-v1
python -m unittest discover -s practice/rag-baseline -p "test_*.py" -q
```

本次用 Python 3.12.14，41 项本地测试通过，代码与提示文件未改。响应规范化后继续标记 semantic_support=not_evaluated，事实性由助手另行审查。

### 实际输出

| 条件 | answer 原文 | 引用版本 | 输入/输出 token | 客户端耗时 |
| --- | --- | --- | --- | --- |
| citation-v2 候选 | 应采用旧版政策。 | v1 与 v2 | 631 / 270 | 约8.88秒 |
| baseline-v1 控制 | 应采用新版政策。 | 仅 v2 | 551 / 152 | 约1.72秒 |

两次后端均正常停止，JSON 与原文引用核验通过，程序 failure 均为 null。这只表示调用和校验链路没有程序错误，不表示候选答案正确。

候选多引了一份真实旧版，但购买日期为10月16日，按已提供的适用范围应选新版。候选结论与上下文不符。引用两份政策在比较版本时本身可以合理，问题不能简单归因为引文数量多；关键是答案把日期范围判断错了。

控制选择新版有证据支持，但仍未展开运行前评分依据要求的日期推理、7天期限和旧版范围，沿用 [首次基线审查](../../../practice/rag-baseline/results/2026-10-18-version-review.json) 中的说明不完整问题。

### 原始记录与处置

- [候选首次输出](../../../practice/rag-baseline/results/2026-10-20-version-citation-v2.json)
- [当日原提示控制](../../../practice/rag-baseline/results/2026-10-20-version-control.json)
- [助手退化审查](../../../practice/rag-baseline/results/2026-10-20-version-regression-review.json)，independent_review=false。

维持 baseline-v1 默认，citation-v2 保留为未推广候选。旧基线和昨天的局部改善记录不覆盖，也不将它们重新写成从未发生。

## 遇到的问题

昨天只看定制题会得到引用补齐的好结果；日期回归揭示候选还会出现更严重的事实错误。因此应同时记录修复与新增退化，而不是只统计引用数量或原文校验通过数。

本次只有一组当日配对，候选先执行，加载和缓存条件不同。观察到了输出退化，但没有证明内部错误机制、稳定发生率或泛化效应；耗时差异也不作为性能结论。

## 今日总结

候选在日期题中错误选旧版，原提示控制正确选新版。两者都通过原文核验，说明程序通过与答案可靠必须分开报告。候选暂不推广，aggregate_quality_score 保持 null。

## 下一步

先将本次输入、错误答案和预期日期关系纳入事实性回归记录，预先规定小规模重复的次数与顺序，再检查这一差异是否稳定，保留全部输出。不要在每次失败后即时改提示并挑选成功结果。

在候选通过回归门槛前，不把它接入实际检索。后续若继续修改引用条款，另存新版本并同时检查定制题修复和日期题退化；日期说明完整性仍作为独立问题处理。
