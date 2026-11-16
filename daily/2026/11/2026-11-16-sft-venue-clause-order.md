# 2026-11-16：中文信息抽取微调：地点分句换序对照

## 今日目标

验证把地点移到未定时间前面，能否恢复明确地点的抽取。完成条件是三份权重跑完九组换序对照，逐 token 复现 27 份原序输出，并记录每组换序前后的时间与地点。[上次措辞对照](2026-11-13-sft-venue-time-wording.md)

同日继续验证换序后仍漏抽的三个“后续发布”版本。完成条件是用原提示和追加一句规则的提示各跑六条输入，复现三份原提示的旧输出，并核对三条明确时间对照。

## 核心概念

同源配对固定活动、场地、未定措辞、称谓和分隔符，只交换两个分句的位置。两种顺序的正确答案相同，时间均为 `null`，地点均为场地原文。对每组输入分别核对时间和地点，可以发现恢复地点时是否影响了时间字段。

## 示例与练习

沿用三份标题式通知和“后续发布”“待定”“尚未确定”三种措辞，得到九组配对。三份权重均使用原提示和启用 KV cache 的贪心解码。以下是植物标本课“后续发布”版本的两种顺序。[完整输入](../../../practice/sft-extraction/results/2026-11-16-venue-clause-order-v1/cases.json) · [运行清单](../../../practice/sft-extraction/results/2026-11-16-venue-clause-order-v1/manifest.json)

```text
时间：后续发布｜地点：雁栖手工室
地点：雁栖手工室｜时间：后续发布
```

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 venue_clause_order.py --run 2026-11-16-venue-clause-order-v1
```

| 未定措辞 | 36 步起点 | 统一称谓组 | 保留原称谓组 |
| --- | ---: | ---: | ---: |
| 后续发布 | 2/3 → 3/3 | 0/3 → 3/3 | 0/3 → 0/3 |
| 待定 | 3/3 → 3/3 | 0/3 → 3/3 | 0/3 → 3/3 |
| 尚未确定 | 3/3 → 3/3 | 0/3 → 3/3 | 0/3 → 3/3 |
| 地点正确合计 | 8/9 → 9/9 | 0/9 → 9/9 | 0/9 → 6/9 |

箭头左侧是时间在前，右侧是地点在前。36 步起点恢复了“雁栖手工室”。统一称谓组的九条都恢复完整场地。保留原称谓组仍将三个“后续发布”版本的地点填为 `null`，其余六条均抽对。54 份输出的活动名、时间和 JSON 字段契约均正确。[逐题结果](../../../practice/sft-extraction/results/2026-11-16-venue-clause-order-v1/review.json)

### 同日补充：追加一句字段判断规则

固定保留原称谓组的权重，沿用地点在前、时间在后的顺序。每条未定通知配一条明确时间的对照，活动和场地保持相同。原提示末尾只加一句“逐个判断字段，只把原文未明确给出的字段填为 null。”。两种提示各跑六条，共 12 次生成。[完整输入](../../../practice/sft-extraction/results/2026-11-16-venue-field-prompt-v1/cases.json) · [提示文本](../../../practice/sft-extraction/results/2026-11-16-venue-field-prompt-v1/protocol.json)

| 通知 | 原提示的地点 | 加规则后的地点 | 两种提示在明确时间对照的结果 |
| --- | --- | --- | --- |
| 蓝晒书签制作课 | `null` | `null` | 时间和地点均正确 |
| 植物标本装帧课 | `null` | 雁栖手工室 | 时间和地点均正确 |
| 木版套色体验 | `null` | 青槐版画坊 | 时间和地点均正确 |

三条未定输入在两种提示下的时间均正确为 `null`。三条明确时间对照均保留完整时间和地点，六条整条正确数从 3/6 提升到 5/6。蓝晒课仍漏掉“白榆印制室”，该条明确时间对照能抽出同一场地。三份原提示的旧输出逐 token 复现。[逐题对照](../../../practice/sft-extraction/results/2026-11-16-venue-field-prompt-v1/review.json)

```powershell
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 venue_field_prompt.py --run 2026-11-16-venue-field-prompt-v1
```

## 遇到的问题

保留原称谓组只在换序和换措辞同时发生时恢复场地。上次时间在前时，“待定”和“尚未确定”也会漏抽，因此逐题比较需要同时考虑分句顺序和未定措辞。本轮覆盖三份同模板的合成通知，独立人工复核待完成。

追加规则恢复了其中两条，蓝晒课仍待定位。提示复验覆盖同一份权重、同一模板和三组配对。

## 今日总结

完成条件已达成。54 次只读生成包含 27 份逐 token 复现的原序输出。地点前置后，三份权重分别恢复 1 条、9 条和 6 条地点，27 组配对均正确保留 `time: null`。模型参数哈希前后一致，130 项微调测试通过。7 份报告已同步至私有 Hugging Face 并通过下载哈希核验。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/024b3cb4acefa05566982d6f11a6129038b42575/evaluations/2026-11-16-venue-clause-order-v1)

补充对照已满足完成条件。当天共 66 次只读生成，复现 30 份历史输出。两轮合计同步 13 份报告，均通过下载哈希核验。[HF 补充报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/fe728c5936f40818ae48d3db94908bbefb51cf6e/evaluations/2026-11-16-venue-field-prompt-v1)

## 下一步

只追踪蓝晒课剩下的地点漏抽。在两种提示下比较首个分歧 token，并用同组明确时间版本核对，记录模型在开始输出地点值时的候选概率。
