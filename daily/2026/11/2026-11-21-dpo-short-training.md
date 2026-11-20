# 2026-11-21：DPO 实作：三步训练与 adapter 重载

本条属于 11 月 21 日计划，于 11 月 20 日提前执行。

## 今日目标

从核验过的 SFT adapter 出发，完成一次真实 DPO 更新。完成条件是训练侧三组偏好对各更新一次，确认 policy 更新、reference 和基座参数通过哈希校验，并验证保存后的 adapter 可以重载。[上次损失核验](2026-11-20-dpo-loss-preflight.md)

## 核心概念

训练用两份回答计算 policy 的对数概率分差，再与 reference 在同一偏好对上的分差比较。reference 始终使用起始权重，可以先计算并保存每组得分。policy 的得分保留梯度，只更新 LoRA 参数。重载验证覆盖推理权重和输出。

## 示例与练习

使用 Qwen3-0.6B 的 BF16 基座，两角色加载相同的 SFT adapter。训练顺序为 `dpo-01`、`dpo-03`、`dpo-05`，每步一个偏好对。AdamW 学习率为 `2e-5`，`beta = 0.1`，种子为 42，梯度裁剪阈值为 1。每个回答仍用原聊天模板，只累计回答和 EOS 的对数概率。三组开发题和 11 条历史回归均留给评估。[运行清单](../../../practice/sft-extraction/results/2026-11-20-dpo-short-run-v1/manifest.json)

| 步骤 | 偏好对 | DPO logit | 损失 | 裁剪前梯度范数 |
| --- | --- | ---: | ---: | ---: |
| 1 | dpo-01 | 0.000000 | 0.693147 | 8.894743 |
| 2 | dpo-03 | -0.000050 | 0.693172 | 7.021533 |
| 3 | dpo-05 | 0.019211 | 0.683588 | 6.456336 |

第一步梯度非零，即使当步的 loss 与同权重起点的 `ln(2)` 一致。每步的可微损失均用独立公式复算。每一步使用不同偏好对，损失表按该步输入解读。

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 dpo_short_run.py --run 2026-11-20-dpo-short-run-v1
```

保存后重新加载基座和 adapter。重载的完整参数哈希与保存前一致，三组开发题的回答对数概率逐值相同，14 条生成逐 token 一致。训练日志记录了 1,146,880 个可训练参数，112 个 LoRA 张量发生变化。基座和 reference 的哈希检查均通过。[训练记录](../../../practice/sft-extraction/results/2026-11-20-dpo-short-run-v1/training.json) · [重载输出](../../../practice/sft-extraction/results/2026-11-20-dpo-short-run-v1/reloaded.json)

## 遇到的问题

三步训练横跨三组不同偏好对，单步损失变化同时受当前样例和前序更新影响。训练闭环以梯度、参数变化和保存后重载为验收依据。偏好分差和实际抽取效果由固定回归核验。

## 今日总结

完成三次真实优化器更新、adapter 保存和推理重载。130 项现有测试通过。权重保存在本机的运行产物目录，三步训练和验收报告已同步到原私有 Hugging Face 实验分支。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/d074220553c98d24fd73a8fa640c3870dc8c7e47/evaluations/2026-11-20-dpo-short-run-v1)

## 下一步

用重载前后已核对的权重完成开发集和开放回归验收。该项与训练同日执行，结果单独记录。[固定回归](2026-11-22-dpo-fixed-regression.md)
