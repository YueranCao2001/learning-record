# 2026-11-13：中文信息抽取微调：未定时间措辞与地点漏抽

## 今日目标

固定昨天的三条标题式通知，验证把“后续发布”换成“待定”或“尚未确定”能否恢复明确地点。完成条件是三份权重分别跑完九条输入，逐题核对时间和地点，并复现九份旧输出。[上次复验](2026-11-12-sft-venue-title-confirmation.md)

## 核心概念

同源改写固定通知模板、场地、活动和字段顺序，只替换时间状态的文字。三种说法的时间标签均为 `null`，场地标签仍为明确地点。对比同一模型在同一通知上的输出，可以看出换词是否改变地点抽取。

## 示例与练习

沿用“蓝晒书签制作课”“植物标本装帧课”“木版套色体验”三组，每组保留原句和两个替换版本。以植物标本课为例，时间分句仍在前，地点始终为“雁栖手工室”。沿用原提示、三份权重和启用 KV cache 的贪心解码。[协议](../../../practice/sft-extraction/results/2026-11-13-venue-time-wording-v1/protocol.json) · [完整输入](../../../practice/sft-extraction/results/2026-11-13-venue-time-wording-v1/cases.json)

```text
时间：后续发布｜地点：雁栖手工室
时间：待定｜地点：雁栖手工室
时间：尚未确定｜地点：雁栖手工室
```

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 venue_time_wording.py --run 2026-11-13-venue-time-wording-v1
```

| 未定措辞 | 36 步起点地点正确数 | 统一称谓组地点正确数 | 保留原称谓组地点正确数 |
| --- | ---: | ---: | ---: |
| 后续发布 | 2/3 | 0/3 | 0/3 |
| 待定 | 3/3 | 0/3 | 0/3 |
| 尚未确定 | 3/3 | 0/3 | 0/3 |

36 步起点在植物标本课原句中把地点填为 `null`，两种替换都恢复“雁栖手工室”。起点对另外两组一直正确。两份微调权重在三组、三种措辞下均输出空地点。27 份输出的活动名、时间和 JSON 字段契约均正确，全部错误均为明确地点被漏成 `null`。[逐题结果](../../../practice/sft-extraction/results/2026-11-13-venue-time-wording-v1/review.json)

## 遇到的问题

昨天省略时间分句可以恢复两份微调权重的场地抽取，今天两种同义替换均未恢复。这个结果覆盖三份合成通知、同一模板和三种措辞。新增的六条都是同源改写，独立人工复核待完成。[省略时间对照](2026-11-12-sft-venue-title-confirmation.md)

## 今日总结

完成条件已达成。27 次只读生成包含九份逐 token 复现的旧输出。两种替换都恢复了 36 步起点的一处地点漏抽，两份微调权重的地点漏抽持续存在。130 项微调测试通过，7 份报告已同步至私有 Hugging Face 并核验下载哈希。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/6353c5e72153d7218baa3ffc057432bffa02a05a/evaluations/2026-11-13-venue-time-wording-v1) · [同步证据](../../../practice/sft-extraction/results/2026-11-13-venue-time-wording-v1/hf-audit-upload.json)

## 下一步

验证同样的九条输入把地点移到时间前面后，是否能恢复两份微调权重的地点抽取。固定称谓、措辞和分隔符，逐条对比换序前后的地点和时间。
