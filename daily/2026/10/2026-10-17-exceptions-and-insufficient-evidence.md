# 2026年10月17日 验证定制例外与证据不足响应

## 今日目标

接着 [昨天的首题真实生成](2026-10-16-ollama-first-local-generation.md)，增加按题选择入口，使用相同模型、提示与参数，分别验证定制例外、空证据和相关但不支持三种情况。每题只调用一次，保留首次输出。

今天重点区分答案是否被上下文支持、引文是否真实，以及引用是否充分。输入仍是人工指定的合成控制上下文，不含实际检索。

## 核心概念

规则与例外是相互关联的证据。“定制设备不适用上述退货规则”保留了例外，但上述指向哪条规则，需要一般规则补足。输入两段证据不保证输出会引用两段。

空证据和相关但不足的证据也不同。保修期限与维修费用相关，却不能推出免费或收费。拒答需要按输入材料单独审查；eligible_for_answer_review=false 只表示不进入普通答案审查，不代表接口失败或拒答错误。

## 示例与练习

### 固定配置与入口

[ollama_smoke.py](../../../practice/rag-baseline/ollama_smoke.py) 新增 --case，默认仍为普通退货题，未知或重复 ID 在调用模型前拒绝。

继续使用 Ollama 0.40.5、qwen3:8b、Q4_K_M，digest 与昨天一致。temperature=0、seed=42、num_ctx=4096、num_predict=512、think=false，提示和 JSON Schema 均未改。

```powershell
python -m unittest discover -s practice/rag-baseline -p "test_*.py" -q
python -X utf8 practice/rag-baseline/ollama_smoke.py --case custom_exception
python -X utf8 practice/rag-baseline/ollama_smoke.py --case delivery_missing
python -X utf8 practice/rag-baseline/ollama_smoke.py --case related_not_supporting
```

本次用 Python 3.12.14 实际执行。39 项本地测试通过，其中新增两项问题选择测试。三次真实生成均正常停止，解析与对应响应协议校验通过，无自动重试。

### 实际输出

| 问题 | 实际状态与内容 | 输入与输出 token | 客户端耗时 |
| --- | --- | --- | --- |
| 定制例外 | completed：定制设备不适用普通设备的30天退货规则。 | 461 / 124 | 约6.93秒 |
| 配送运费 | insufficient_evidence：未提供相关证据以确定配送运费的具体金额。 | 246 / 40 | 约0.55秒 |
| 免费维修 | insufficient_evidence：没有提供关于保修期间维修是否免费的具体信息。 | 354 / 40 | 约0.43秒 |

两条证据不足响应的 answer 实际为空，表中展示 reason，两者均无引文。定制题只引用 exception v1：

> 定制设备不适用上述退货规则。

原始报告：[定制题](../../../practice/rag-baseline/results/2026-10-17-custom_exception.json)、[运费题](../../../practice/rag-baseline/results/2026-10-17-delivery_missing.json)、[维修题](../../../practice/rag-baseline/results/2026-10-17-related_not_supporting.json)。定制题含约5.51秒加载，后两题复用模型且各命中150个提示缓存 token，不能由三次耗时宣称性能改善。

### 逐题审查

[助手审查记录](../../../practice/rag-baseline/results/2026-10-17-review.json) 独立于原始输出保存，independent_review=false。

定制题答案被完整上下文支持，引文也逐字真实。但按既定评分依据，答案明确提到普通设备30天规则，唯一引文中的上述不能自足解释这部分。引用充分性标为 partial_missing_antecedent，需要一般规则引用补齐指代，而不是将本题记作全部通过。

运费题输入为空，模型没有编造价格；维修题只提供12个月保修期限，模型没有据此断言免费。两条拒答相对于本次控制上下文是恰当的，不代表真实检索系统始终能正确拒答。

## 遇到的问题

单证据问题通过，不能保证双证据问题的引用充分。今天暴露的不是伪造引用，而是指代依赖未被引用链完整表达；现有字符串核验不检查这种关系。

本次不补写遗漏引用，也不修改提示后只展示更好的结果。原始缺口保留为后续对照案例，避免将人工修复当作模型表现。

## 今日总结

今天完成按题入口、三道真实生成和逐题审查。两种证据不足输入上模型拒答，定制题仍有引用充分性缺口。

本地测试、原文一致性、答案支持和引用充分性分开报告，不压成一个通过率。版本冲突题和实际检索链路仍未运行。

## 下一步

先保持配置验证日期版本题，补齐五题候选的首次输出。随后为引用充分性提出明确改动，另存提示版本做单因素对照，检查是否补齐定制引用，同时观察多余引文与拒答退化。

控制上下文结果逐题审查后，再与实际检索上下文对照。保留所有首次报告和失败分类，不用手工改答案冒充模型结果。
