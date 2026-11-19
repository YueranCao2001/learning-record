# 2026-11-20：DPO 实作：核对四项对数概率与初始损失

本条属于 11 月 20 日计划，于 11 月 19 日提前执行。

## 今日目标

用上一条笔记校验过的 `dpo-01` 检查一次 DPO 损失。完成条件是逐项复算 policy 与 reference 对 `chosen` 和 `rejected` 的回答对数概率，核对计分边界，然后检验同权重起点的损失与公式方向。[偏好对笔记](2026-11-19-dpo-preference-pair-preflight.md)

## 核心概念

每个回答的对数概率是其回答 token 的条件对数概率之和，计入 EOS。提示参与前向，为回答提供上下文，提示位置的标签由掩码排除。DPO 比较 policy 和 reference 各自在同一个偏好对上的分差。[DPO 原论文](https://arxiv.org/abs/2305.18290v3)

```math
r_m = \log p_m(y_{\mathrm{chosen}} \mid x) - \log p_m(y_{\mathrm{rejected}} \mid x)
```

```math
L = -\log \sigma\bigl(\beta(r_{\mathrm{policy}} - r_{\mathrm{reference}})\bigr)
```

## 示例与练习

取“树叶压花体验”这一对。正确回答保留明确地点“青梧自然教室”，较差回答将地点填为空值。两个角色分别加载同一份 SFT adapter 和基座，在本机用 BF16 基座做四次只读前向。每个回答核对前 76 个 token 与上次存档完全一致。移动一位对齐预测与标签后，仅给 20 个 chosen token 和 17 个 rejected token 计分，均含 EOS。逐 token 的 FP32 `log_softmax` 总和与独立计算的交叉熵总量符号相反，误差在 `1e-5` 以内。[单对输入](../../../practice/sft-extraction/results/2026-11-20-dpo-loss-preflight-v1/pair.json) · [运行清单](../../../practice/sft-extraction/results/2026-11-20-dpo-loss-preflight-v1/manifest.json)

| 角色 | chosen 对数概率 | rejected 对数概率 | 分差 |
| --- | ---: | ---: | ---: |
| policy | -0.927429 | -0.552467 | -0.374962 |
| reference | -0.927429 | -0.552467 | -0.374962 |

两份权重相同，分差相减得 `0`。设 `beta = 0.1`，`-logsigmoid(0)` 与用 `log1p(exp(0))` 复算都得到 `0.69314718056`。另用合成分差检验方向。相对分差为 `+1` 时损失约 `0.644397`，为 `-1` 时约 `0.744397`。[逐 token 与损失记录](../../../practice/sft-extraction/results/2026-11-20-dpo-loss-preflight-v1/review.json)

```powershell
# 在 practice/sft-extraction 下执行，复跑时使用新目录名
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 dpo_loss_preflight.py --run 2026-11-20-dpo-loss-preflight-v1
```

## 遇到的问题

在这条偏好对上，起始模型给较短的 rejected 更高的整段对数概率。初始 DPO 损失仍等于 `ln(2)`，因为 policy 和 reference 的分差相同。这一步核验的是同权重起点的公式。回答长短在这组样例中与地点值是否完整一起改变，观察范围只有这一组合成回答。

## 今日总结

11 月 20 日计划已提前完成。四次模型前向与两种损失计算一致，权重哈希前后一致，优化器更新 0 次，130 项现有测试通过。

三份核验报告已同步到原私有 Hugging Face 实验分支，下载哈希校验通过。[HF 报告](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/bda2c5ce2e274ebfcfc293421da65f9d6959c951/evaluations/2026-11-20-dpo-loss-preflight-v1)

## 下一步

按调整后的安排，11 月 20 日完成原定 11 月 21 日的短训练与重载，以及原定 11 月 22 日的开发集和开放回归，分别记录实际运行日期。
