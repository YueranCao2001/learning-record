# 2026-11-11：中文信息抽取微调：未定时间与地点漏抽的跨来源确认

## 今日目标

检查“时间未定时漏掉明确地点”是否会在新的通知里出现。沿用 11 月 9 日的贪心解码设置，在三种通知表达中比较时间是否明确、分句顺序，以及省略时间分句后的输出。[上次诊断](2026-11-09-sft-venue-omission-diagnostic.md)

同日补充聚焦“手工线装体验”的明确时间漏抽。完成条件是复现两个原始顺序，跑完称谓和顺序的八种组合，并逐条确认哪些改动能恢复时间。

## 核心概念

- 同源配对固定活动和场地，改动时间分句。明确时间与未定时间使用相同顺序，可以比较未定措辞带来的变化。
- 显式写“后续发布”和省略时间分句都应得到 `time: null`，两个输入仍可能产生不同的地点输出。
- 跨来源复验需要按来源读结果。每组的活动、场地和通知模板共同变化，组间结果用于检查复现范围。

## 示例与练习

### 1. 样例与设置

新增标题式通知、段落式通知和报名清单三组虚构样例。每组交叉明确时间 / “后续发布”和时间在前 / 地点在前，再加一条省略时间分句。三组共 15 条新样例，均有明确场地。

例如“蓝晒书签制作课”组固定场地“白榆印制室”。时间明确时应提取“12月30日上午10:15”，时间未定或省略时应返回 `null`。三种时间状态下都应保留地点。

三份权重各评估 15 条新样例和 3 条历史探针。新样例与现有预算集和上次诊断检查了来源组、样例 ID 和文本重合。评估沿用原提示、KV cache 和 128 个新 token 上限。[协议](../../../practice/sft-extraction/data/venue-omission-confirmation-v1.json) · [完整输入](../../../practice/sft-extraction/results/2026-11-11-venue-omission-confirmation-v1/cases.json)

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 venue_omission_confirmation.py --run 2026-11-11-venue-omission-confirmation-v1
```

### 2. 三组新通知的结果

| 固定权重 | 整条正确 | 时间正确 | 地点正确 |
| --- | ---: | ---: | ---: |
| 36 步起点 | 15/15 | 15/15 | 15/15 |
| 统一称谓组 | 14/15 | 15/15 | 14/15 |
| 保留原称谓组 | 12/15 | 14/15 | 13/15 |

三份权重对新样例的活动名、JSON 和字段契约均为 15/15。地点漏抽集中在标题式通知，三种时间状态对应的地点结果如下。

| “蓝晒书签制作课”的输入条件 | 36 步起点 | 统一称谓组 | 保留原称谓组 |
| --- | --- | --- | --- |
| 明确时间，两种顺序 | 2/2 | 2/2 | 2/2 |
| 后续发布，时间在前 | 白榆印制室 | `null` | `null` |
| 后续发布，地点在前 | 白榆印制室 | 白榆印制室 | `null` |
| 省略时间分句 | 白榆印制室 | 白榆印制室 | 白榆印制室 |

段落式通知和报名清单的 10 条样例在三份权重上均抽对场地。其中“手工线装体验”明确给出时间和地点，保留原称谓组在会址排在前面时把“周日下午16:25”漏成了 `null`，地点仍为“芦汀装帧坊”。同源交换分句顺序后，该组模型能抽出完整时间。[逐题评分与错误](../../../practice/sft-extraction/results/2026-11-11-venue-omission-confirmation-v1/review.json)

### 3. 四处漏抽的首个分歧

对新样例中的 4 次漏抽做目标前缀前向。四次都在对应字段值开始时分歧，模型生成了只含冒号的 token，接着生成 `null`。下表是正确前缀下，含字符串起始引号的目标 token 与实际选择 token 的条件概率。

| 模型与样例 | 字段 | 目标 token | 实际选择 token |
| --- | --- | ---: | ---: |
| 统一称谓组，蓝晒课时间在前 | 地点 | 0.223 | 0.777 |
| 保留原称谓组，蓝晒课时间在前 | 地点 | 0.119 | 0.881 |
| 保留原称谓组，蓝晒课地点在前 | 地点 | 0.437 | 0.561 |
| 保留原称谓组，线装课地点在前 | 时间 | 0.468 | 0.531 |

原句的地点漏抽在两组中都明显偏向实际选择 token。交换顺序后仍漏抽的样例与线装课漏时间的样例，候选概率更接近。这些概率来自目标前缀前向，本轮贪心生成使用带 cache 的原设置。[统一称谓组前向记录](../../../practice/sft-extraction/results/2026-11-11-venue-omission-confirmation-v1/canonical-teacher.json) · [保留原称谓组前向记录](../../../practice/sft-extraction/results/2026-11-11-venue-omission-confirmation-v1/target-teacher.json)

### 4. 同日补充验证明确时间漏抽

固定“手工线装体验”“周日下午16:25”“芦汀装帧坊”及其标签，交叉两种分句顺序、“会址 / 地点”和“日程 / 时间”。三份权重各跑 8 条，合计 24 次。两条原始输入的 6 份输出逐 token 复现。

| 场地称谓 | 时间称谓 | 保留原称谓组，时间在前 | 保留原称谓组，地点在前 |
| --- | --- | --- | --- |
| 会址 | 日程 | 周日下午16:25 | `null` |
| 地点 | 日程 | 周日下午16:25 | 周日下午16:25 |
| 会址 | 时间 | 周日下午16:25 | 周日下午16:25 |
| 地点 | 时间 | 周日下午16:25 | 周日下午16:25 |

36 步起点与统一称谓组各为 8/8，保留原称谓组为 7/8。24 份输出的活动名、地点和 JSON 字段契约均正确。在这个通知里，保留原称谓组只在地点在前且使用“会址”和“日程”时漏抽。固定地点在前，单换任一称谓都恢复完整时间。[八种输入](../../../practice/sft-extraction/results/2026-11-11-time-cue-order-diagnostic-v1/cases.json) · [逐题结果](../../../practice/sft-extraction/results/2026-11-11-time-cue-order-diagnostic-v1/review.json)

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 time_cue_order_diagnostic.py --run 2026-11-11-time-cue-order-diagnostic-v1
```

## 遇到的问题

未定时间的影响在新标题式通知上复现，另外两组的明确地点都保留。每种模板只用了一份虚构通知，模板差异、称谓和活动内容同时变化。样例为开放诊断数据，独立人工复核待完成。

补充对照覆盖“手工线装体验”这一条虚构通知。恢复条件限于本次测试的称谓与顺序。

## 今日总结

当天完成 78 次只读生成和 4 次目标前缀前向，两轮共 15 份历史输出逐 token 复现。首轮在一个新来源上观察到未定时间伴随场地漏抽。补充对照已满足完成条件，定位了这条明确时间漏抽的组合，并验证了单换任一称谓的恢复结果。两轮模型参数哈希前后一致，130 项微调测试通过。

11 份实验报告已同步至私有 Hugging Face 实验分支，并通过下载哈希核验。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/84069a3eefdc7cb6289f71a08c4a915224228cf2/evaluations/2026-11-11-venue-omission-confirmation-v1) · [同步证据](../../../practice/sft-extraction/results/2026-11-11-venue-omission-confirmation-v1/hf-audit-upload.json)

补充对照的 7 份报告也已同步并通过下载哈希核验。[HF 补充报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/74c310315d2b2dcc46274278ec5fa9b05cc3b553/evaluations/2026-11-11-time-cue-order-diagnostic-v1) · [同步证据](../../../practice/sft-extraction/results/2026-11-11-time-cue-order-diagnostic-v1/hf-audit-upload.json)

## 下一步

- 下一个问题是标题式通知的地点漏抽是否随活动内容变化而复现。固定模板和称谓，新增两组活动、时间及场地，跑完明确时间、显式未定和省略时间的配对，记录每组的地点输出。
