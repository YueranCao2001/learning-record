# 2026-11-19：DPO 实作：构造并校验第一批偏好对

## 今日目标

从 SFT 地点漏抽诊断切换到 DPO 实作。今天验证一个问题，即按抽取规则构造的一对回答，经过本地聊天模板后是否仍共享同一个提示前缀。完成条件是至少一组 `chosen` 和 `rejected` 有可追溯的偏好理由，分词后共享前缀、回答 token 有差异、两份回答均包含 EOS。[SFT 收尾](2026-11-18-sft-venue-cache-scope.md)

## 核心概念

DPO 数据把同一提示与两份存在相对偏好的回答配成一组。今天沿用已经冻结的抽取规则作为判断依据，选择符合原文的回答作为 `chosen`，只改动一个字段构造 `rejected`。把提示存一次可以保证两个回答复用同一份通知。分词时还要检查两条完整聊天序列的前缀是否逐 token 一致，以便下一步只对回答部分计分。[DPO 原论文](https://arxiv.org/abs/2305.18290v3) · [项目标注规则](../../../practice/sft-extraction/annotation-rules-v1.md)

## 示例与练习

新写六份合成通知，训练侧与开发侧各三组，两侧都包含明确值漏抽、未定文字误抽和抽取边界错误各一组。源组按通知划分，六份源通知互不重复。样例借用了上一阶段出现的错误类型，文本与偏好标签由助手构造。[偏好数据](../../../practice/sft-extraction/data/dpo-preference-pairs-v1.json)

其中一条通知是“树叶压花体验将在青梧自然教室举行，开始时间另行通知。”以下两份回答使用同一个系统指令和通知。场地已明确，`chosen` 保留场地，两个回答都将未定时间填为 `null`。

```json
{
  "chosen": {"event": "树叶压花体验", "time": null, "location": "青梧自然教室"},
  "rejected": {"event": "树叶压花体验", "time": null, "location": null}
}
```

用现有 Qwen3-0.6B tokenizer 和原聊天模板渲染，关闭思考段，末尾保留 EOS。沿用 SFT 的前缀与目标掩码检查。下表的回答长度包含 EOS，前缀长度包含系统指令、用户通知和助理起始模板。[Tokenizer 清单](../../../practice/sft-extraction/results/2026-11-19-dpo-preference-pairs-v1/manifest.json)

| 样例 | 划分 | `rejected` 的错误 | 共享前缀 token | `chosen` / `rejected` token |
| --- | --- | --- | ---: | ---: |
| dpo-01 | train | 明确地点填成空值 | 76 | 20 / 17 |
| dpo-02 | dev | 把“待定”填为地点 | 79 | 23 / 24 |
| dpo-03 | train | 地点多带“地点是” | 86 | 29 / 31 |
| dpo-04 | dev | 明确时间填成空值 | 83 | 23 / 17 |
| dpo-05 | train | 把“后续公布”填为时间 | 81 | 22 / 23 |
| dpo-06 | dev | 时间只保留时刻 | 84 | 29 / 25 |

六组都共享逐 token 一致的前缀，回答在前缀之后分歧。第一组的首个分歧发生在地点键名后，`chosen` 的 token 含字符串起始引号，`rejected` 的 token 在冒号结束，随后写入 `null`。另做四次故障注入，交换标签、复制成相同回答、跨划分复用来源组以及填入原文外的值，校验器均拒绝。[分词与偏好核验](../../../practice/sft-extraction/results/2026-11-19-dpo-preference-pairs-v1/review.json)

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 dpo_pair_preflight.py --run 2026-11-19-dpo-preference-pairs-v1
```

## 遇到的问题

长度差随着错误类型变化。补回被漏掉的字段有时会让 `chosen` 更长，去掉误抽文字或引导词可以让它更短。六组中两种方向各出现三次。标签来自抽取规则与助手编写的合成输入，独立人工复核待完成。

## 今日总结

完成第一批六组偏好对与 tokenizer 校验，通过 130 项现有微调测试。今日只执行数据和模板核验，实际生成次数及优化器更新均为 0。三份数据校验报告已同步到私有 Hugging Face 并验证下载哈希。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/438fabe3bd320f28b9bc7cff4fcb83dba1a3e68e/evaluations/2026-11-19-dpo-preference-pairs-v1)

## 下一步

固定一组偏好对，分别取 policy 和 reference 下两份回答的四个 log probability。用已核验的前缀边界排除提示 token，再复算一个 DPO loss，检查四项的符号和求和范围。[之前的 DPO 理论笔记](../09/2026-09-08-direct-preference-optimization.md)
