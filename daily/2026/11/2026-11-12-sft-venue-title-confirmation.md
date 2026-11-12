# 2026-11-12：中文信息抽取微调：标题式通知的地点漏抽复验

## 今日目标

检查同一标题式模板换成两组新活动后，“时间：后续发布”是否仍伴随明确地点漏抽。完成条件是对每组跑完明确时间、显式未定、省略时间三个版本，核验三个原通知版本的历史输出，再记录每份权重的逐题结果。[上次发现](2026-11-11-sft-venue-omission-confirmation.md)

## 核心概念

同源对照固定活动和场地，只改动时间分句。跨活动复验沿用同一个模板和字段称谓，检查漏抽是否出现在不同的内容上。显式未定和省略时间有相同的时间标签，可以观察两种输入的地点输出是否一致。

## 示例与练习

原通知为“蓝晒书签制作课”。两组新活动为“植物标本装帧课”和“木版套色体验”，场地分别是“雁栖手工室”和“青槐版画坊”。沿用原通知的标题格式、“时间 / 地点”称谓、时间在前的顺序和结尾文字。每组的三个版本都应抽出同一个场地名。下面是第一组的未定时间版本。[协议](../../../practice/sft-extraction/data/venue-title-confirmation-v1.json) · [完整输入](../../../practice/sft-extraction/results/2026-11-12-venue-title-confirmation-v1/cases.json)

```text
【植物标本装帧课】
时间：后续发布｜地点：雁栖手工室
请按活动安排参加。
```

沿用三份固定权重、原提示及启用 KV cache 的贪心解码。各跑 6 条新样例和 3 条旧样例，共 27 次生成。[运行清单](../../../practice/sft-extraction/results/2026-11-12-venue-title-confirmation-v1/manifest.json)

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 venue_title_confirmation.py --run 2026-11-12-venue-title-confirmation-v1
```

| 时间条件 | 36 步起点 | 统一称谓组 | 保留原称谓组 |
| --- | ---: | ---: | ---: |
| 明确时间，地点正确数 | 2/2 | 2/2 | 2/2 |
| 显式未定，地点正确数 | 1/2 | 0/2 | 0/2 |
| 省略时间，地点正确数 | 2/2 | 2/2 | 2/2 |
| 六条新样例，整条正确数 | 5/6 | 4/6 | 4/6 |

两份微调权重在两个新活动的显式未定版本均将地点填为 `null`。36 步起点在“植物标本装帧课”上也漏掉地点，在“木版套色体验”上保留了地点。三份权重在六条新样例上均抽对活动名和时间，输出均符合 JSON 字段契约。三个原通知版本在三份权重上的 9 份历史输出逐 token 复现。[逐题结果](../../../practice/sft-extraction/results/2026-11-12-venue-title-confirmation-v1/review.json)

## 遇到的问题

上次 36 步起点在蓝晒课上能抽出地点，换成植物标本课后也出现漏抽。这表明在同一模板下，具体活动文本仍可能影响结果。新样例只有两组虚构通知，活动名、场地名和具体时间一起更换，独立人工复核待完成。

## 今日总结

完成条件已达成。两组新活动均复现了微调权重的地点漏抽。明确时间和省略时间的配对均能抽对地点。三份模型参数哈希前后一致，130 项微调测试通过。7 份报告已同步至私有 Hugging Face 并通过下载哈希核验。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/085e1c008750b84f0c968e8dcc8b96f599987d53/evaluations/2026-11-12-venue-title-confirmation-v1) · [同步证据](../../../practice/sft-extraction/results/2026-11-12-venue-title-confirmation-v1/hf-audit-upload.json)

## 下一步

固定这三组标题式通知，检查只将“后续发布”换成“待定”或“尚未确定”是否能恢复地点。完成条件是三份权重各跑三组通知的三种未定措辞，逐题报告地点及时间的结果。
