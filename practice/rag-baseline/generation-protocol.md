# 生成接口与评估准备

## 当前接口

[generation_contract.py](generation_contract.py) 提供 build_request 与 validate_response。前者保存查询和实际输入证据，后者检查响应结构、请求绑定和原文引用。[ollama_smoke.py](ollama_smoke.py) 已支持 --case 按问题 ID 发起单次本地调用，未实现独立质量自动评分。

请求字段为 contract_version、request_id、query、context_sha256、evidence。context_sha256 按输入证据的顺序与完整字段计算，应用层保留原请求和证据包，不从模型回传值重建可信上下文。它用于定位记录，不是身份认证或数字签名。

响应仅包含 contract_version、request_id、context_sha256、status、answer、claims、reason。claims 是回答内的字符范围列表，每项包含 start、end、citations；引用格式复用昨天的原文核验协议。模型适配器负责把后端响应映射到协议，并记录原始完成原因，不能把截断当成完成。

| status | 协议要求 | 后续处理 |
| --- | --- | --- |
| completed | 非空答案与带引用的主张，reason 为 null | 原文核验通过后进入答案与语义支持审查 |
| insufficient_evidence | claims 为空，reason 非空 | 独立检查拒答是否合理 |
| failed | claims 为空，reason 非空 | 记录失败，不计作正确答案 |
| truncated | claims 为空，reason 非空 | 保留部分 answer 用于诊断，不计作完整回答 |

eligible_for_answer_review 只表示能够进入后续审查。主张范围合法且引用真实，仍可能包含错误结论、遗漏例外，或者漏标回答中的其他主张；这些需要独立评价。

## 生成前的固定设置

适配器固定提示版本与证据序列，提示只允许根据给定材料回答，不让文档中的操作指令提升权限。实际发送的提示与上下文必须存档，输入不包含 gold 答案、审查意见或诊断目标标注。

生成运行清单需包含后端、模型版本、提示版本、解码配置、输出预算、超时、数据版本及代码指纹。每次记录原始输出、标准化响应、耗时、输入输出 token 数与成本；后端未提供的字段为 null，不填 0。

已选择本机 Ollama 与 qwen3:8b，完成一次控制上下文的普通退货题调用。固定提示和参数后继续逐题运行；未接入付费 API。首次输出和实际参数见 [运行报告](results/2026-10-16-ollama-smoke.json)。

## 待审查问题

[generation-review-draft.json](data/generation-review-draft.json) 是五题候选草案，作者未独立于本项目，也未经过第二次独立审查。它与调试样例共享合成政策，不能当作未见泛化测试集。

审核应逐题检查事实范围、必要证据、允许的替代证据、合理拒答条件，以及哪些主张必须被引用支持。版本冲突题还需要新增明确的来源与生效时间，现有 v1 快照不足以运行该题。

审核完成后冻结新版本，评分依据单独存储且不进入模型输入。分别记录答案正确性、逐主张支持、引用覆盖、拒答与运行失败，保留逐题证据和分母；未知支持状态不默认为通过。

10 月 15 日另存 [试跑候选](data/generation-pilot-candidate-v1.json) 与 [对应来源](data/generation-pilot-sources-v1.json)，补齐购买日期适用范围和逐题评分依据。它们已经过本项目一致性自查，仍待独立审查，不替换原草案，也不构成独立确认集。

试跑候选使用人工选定上下文，便于先诊断生成；不能当作端到端检索成绩。评分依据保留在候选文件，不进入生成请求。

## 重跑本地校验

```powershell
python -m unittest discover -s practice/rag-baseline -p "test_*.py" -v
```

测试响应由手工夹具构造，验证接口规则，不作为模型生成成绩。
