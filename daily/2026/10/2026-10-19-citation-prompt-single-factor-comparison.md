# 2026年10月19日 引用充分性提示的单因素对照

## 今日目标

接着 [昨天冻结的五题首次基线](2026-10-18-policy-version-first-pass-review.md)，今天只调整引用充分性要求，检查能否补齐定制题中上述规则的指代依据，并观察普通回答和拒答是否退化。

日期解释完整性不在本次改动范围内。原提示与首次报告保留，候选提示不替换默认配置。

## 核心概念

引用数量不是充分性的直接指标。需要补的是当前答案依赖的证据：例外片段中的上述规则指向普通设备30天规则，因此本题应同时保留一般规则与例外。无关片段不能因增加数量而获得价值。

单因素对照只改变系统提示中的引用条款，模型、输入证据、问题、Schema 和解码参数不变。定制题在当天分别运行控制与候选；其他三题仅运行候选，并与历史首次输出作回归比较，不把它们混称为当日完整配对。

## 示例与练习

### 提示版本与运行

baseline-v1 保留原提示。citation-v2 在其后追加 [引用充分性条款](../../../practice/rag-baseline/prompts/citation-sufficiency-v2.txt)：引用中有上述、该规则等依赖前文的指代时，同时引用解析指代必需的输入证据；不罗列所有证据，也不要求扩写日期解释。

[运行入口](../../../practice/rag-baseline/ollama_smoke.py) 新增 --prompt-version，默认仍为 baseline-v1，报告保存提示版本和指纹。模型保持 Qwen3 8B Q4_K_M，Ollama 0.40.5；temperature=0、seed=42、num_ctx=4096、num_predict=512、think=false。候选与控制使用相同模型 digest。

```powershell
python -m unittest discover -s practice/rag-baseline -p "test_*.py" -q
python -X utf8 practice/rag-baseline/ollama_smoke.py --case custom_exception --prompt-version baseline-v1
python -X utf8 practice/rag-baseline/ollama_smoke.py --case custom_exception --prompt-version citation-v2
python -X utf8 practice/rag-baseline/ollama_smoke.py --case ordinary_return --prompt-version citation-v2
python -X utf8 practice/rag-baseline/ollama_smoke.py --case delivery_missing --prompt-version citation-v2
python -X utf8 practice/rag-baseline/ollama_smoke.py --case related_not_supporting --prompt-version citation-v2
```

本次用 Python 3.12.14 实际执行五次本地调用，每个配置一次，无自动重试。41 项本地测试通过，其中新增两项检查提示的单因素变化及非法版本拒绝。

### 实际对照结果

| 运行 | 实际答案或拒答 | 引用 |
| --- | --- | --- |
| 定制题控制 | 定制设备不适用普通设备的30天退货规则。 | 仅 exception |
| 定制题候选 | 与控制答案相同 | exception 与 rule |
| 普通题候选 | 普通设备可以在30天内退货。 | 仅 rule |
| 运费题候选 | 证据不足，未提供配送运费信息 | 无 |
| 维修题候选 | 证据不足，未提供维修是否免费信息 | 无 |

两个拒答的 answer 为空，表中概括的是 reason。五次后端均正常停止，解析和响应协议核验通过；拒答不强求引用。

定制题候选补上的逐字引文为普通设备支持30天内退货，与原例外引文共同解释上述指代。答案未改变，改善发生在本次引用链，而不是答案内容。

### 原始记录与审查

- [定制题当日控制](../../../practice/rag-baseline/results/2026-10-19-custom-control.json)
- [定制题候选](../../../practice/rag-baseline/results/2026-10-19-custom-citation-v2.json)
- [普通题候选](../../../practice/rag-baseline/results/2026-10-19-ordinary-citation-v2.json)
- [运费题候选](../../../practice/rag-baseline/results/2026-10-19-delivery-citation-v2.json)
- [维修题候选](../../../practice/rag-baseline/results/2026-10-19-related-citation-v2.json)
- [助手对照审查](../../../practice/rag-baseline/results/2026-10-19-citation-ab-review.json)，independent_review=false。

定制题控制输入/输出为461/124 tokens，候选为541/206 tokens。新增条款与额外引文增加了 token 成本，不能只报告引用修复而忽略输出增长。控制包含模型加载，缓存条件也不同，约6.98秒与2.13秒的等待时间不用于宣称候选更快。

助手判断此配对补齐了既定的指代缺口，普通题与两种拒答未出现已观察到的退化。但普通题只有一段输入，无法检验大量干扰证据下的过度引用；单次配对也不证明稳定性或泛化改善。

## 遇到的问题

原提示允许真实但不自足的引文通过。今天没有通过改写答案或后处理补引文来隐藏问题，而是把需要模型完成的引用依赖要求单独加入提示，再核对原始输出。

新提示在发现问题的同一题上设计并验证，属于针对已知错误的探索性修复。独立问题、日期题在新提示下的表现以及实际检索上下文仍待检查，aggregate_quality_score 保持 null。

## 今日总结

今天的当日配对中，候选提示在不改答案的情况下补齐一般规则引用；三项候选回归未显示普通回答或拒答退化。原基线、当日控制和全部候选输出分别保存，默认提示未切换。

## 下一步

先用已存在、未参与本次引用修复的日期题检查 citation-v2，观察有无多余引用，不同时加入日期解释要求。随后将固定候选提示与实际检索上下文对照，区分检索缺证据和生成引用缺证据。

日期说明完整性仍作为另一项独立改动；所有新对照另存输入、原始响应、成本与退化，继续保留首次基线与独立审查待办。
