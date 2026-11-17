# 2026-11-17：中文信息抽取微调：白榆印制室漏抽的首个分歧

## 今日目标

定位蓝晒书签制作课的地点漏抽。完成条件是复现同组两条通知在两种提示下的四份旧输出，找到漏抽时首个分歧 token，并记录两种时间状态下地点字段开头的候选概率。[上次提示对照](2026-11-16-sft-venue-clause-order.md)

## 核心概念

逐 token 对齐正确答案和生成结果，可以定位第一次分歧。两条序列共享分歧之前的前缀，所以能在同一个输出位置比较候选 token。带 KV cache 的贪心解码记录实际生成时的分数，正确前缀前向记录模型一次处理完整目标序列时的分数。两种计算路径各自保存。

## 示例与练习

固定保留原称谓组的权重和地点在前的顺序。两条通知都写明“地点：白榆印制室”，后一分句分别为“时间：后续发布”和“时间：12月30日上午10:15”。两条输入各用原提示和追加字段判断规则的提示运行一次。每次取实际生成分数，再做一次正确前缀前向。[输入与提示](../../../practice/sft-extraction/results/2026-11-17-venue-location-token-trace-v1/cases.json) · [运行清单](../../../practice/sft-extraction/results/2026-11-17-venue-location-token-trace-v1/manifest.json)

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 venue_location_token_trace.py --run 2026-11-17-venue-location-token-trace-v1
```

两种提示下的漏抽都发生在第 16 个输出 token，记录中的零基索引为 15。此时已生成到地点键名。正确序列的下一 token 为 `":"`，ID 为 `3252`，它包含值的起始引号。两次生成均选择 `":`，ID 为 `788`，随后写入 `null`。明确时间版本在第 26 个输出 token 选择 `3252`，两种提示均逐 token 匹配正确序列。

| 时间状态与提示 | 带缓存解码的字符串起始 token | 带缓存解码的冒号 token | 生成的地点 |
| --- | ---: | ---: | --- |
| 未定，原提示 | 0.4993 | 0.4993 | `null` |
| 未定，追加规则 | 0.4683 | 0.5306 | `null` |
| 明确，原提示 | 0.9666 | 0.0331 | 白榆印制室 |
| 明确，追加规则 | 0.6504 | 0.3481 | 白榆印制室 |

带缓存时，原提示的未定版本在这两个候选之间得分持平，实际选了 `788`。追加规则后，`788` 的分数更高。在正确前缀前向里，追加规则的未定版本给 `3252` 和 `788` 的概率分别是 0.5615 和 0.4373，排序与实际生成相反。这些数值是单个 token 的条件概率。[分数与逐 token 证据](../../../practice/sft-extraction/results/2026-11-17-venue-location-token-trace-v1/review.json)

## 遇到的问题

两条路径对追加规则的未定版本给出了不同的候选排序。正确前缀前向使用完整目标序列和 `use_cache=False`，实际生成逐步使用 KV cache。本轮测量覆盖同一合成通知的一份权重，数值差异的来源和关闭 cache 后的完整生成仍待验证。

## 今日总结

完成条件已达成。四次带缓存生成均逐 token 复现旧输出，另做四次正确前缀前向，定位了地点值开始处的分歧。明确时间对照完整抽出“白榆印制室”，130 项微调测试通过。

三份报告已同步到私有 Hugging Face，并通过下载哈希核验。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/83c2a3fdd1c28e5219abef9ac7a7b68985e4d30a/evaluations/2026-11-17-venue-location-token-trace-v1)

## 下一步

继续追踪同一条未定通知。固定模型和两种提示，仅切换生成时的 KV cache，比较第 16 个 token 的候选概率和完整地点输出。以两种提示均完成有缓存和无缓存的配对为完成条件。
