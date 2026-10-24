# 后训练与偏好优化

SFT、DPO、PPO 与 GRPO、Rollout 和奖励评估器。当前记录以概念及合成验证为主，真实模型训练的完成状态需要实际产物支持。

## 日期笔记

- 2026-10-24 [中文信息抽取微调：冻结标注规则与开放审计](../daily/2026/10/2026-10-24-sft-annotation-rules-open-audit.md)
- 2026-10-23 [中文信息抽取微调：边界样例的固定预算对照](../daily/2026/10/2026-10-23-sft-boundary-data-comparison.md)
- 2026-10-22 [中文信息抽取微调：首个 LoRA 训练与 HF 同步](../daily/2026/10/2026-10-22-sft-extraction-contract-and-preflight.md)
- 2026-09-07 [Supervised Fine-Tuning：Chat Template、Assistant-only Loss、Packing 与 LoRA](../daily/2026/09/2026-09-07-supervised-fine-tuning-chat-template-lora.md)
- 2026-09-08 [Direct Preference Optimization：偏好对、Reference Model、DPO Loss 与长度偏差](../daily/2026/09/2026-09-08-direct-preference-optimization.md)
- 2026-09-09 [在线强化学习后训练：Rollout、Advantage、PPO 与 GRPO](../daily/2026/09/2026-09-09-online-rl-ppo-grpo.md)
- 2026-09-10 [RL Rollout 工程：权重同步、采样 Log Probability、异步队列与 Policy Staleness](../daily/2026/09/2026-09-10-rl-rollout-engineering.md)
- 2026-09-11 [可验证奖励与评估器工程：答案抽取、容差、失败分类与 Reward Hacking](../daily/2026/09/2026-09-11-verifiable-rewards-evaluator.md)
- 2026-09-14 [后训练链路复盘：SFT、DPO、PPO/GRPO 的统一实验协议与验收](../daily/2026/09/2026-09-14-post-training-end-to-end-review.md)

## 实践衔接

用户已安排自2026年10月22日起，将当前学习主线转向大模型微调。先确认任务、候选模型与脱敏数据，核查训练环境，再用小规模 SFT 做模板与监督边界检查、一步训练、保存恢复和独立评估。已有 LoRA、QLoRA 笔记作为参考，不预先假定显存一定足够。

2026年10月22日完成 [中文信息抽取练习](../practice/sft-extraction/README.md) 的首次真实 GPU 六步 LoRA：`Qwen/Qwen3-0.6B`、3条合成 train + 3条可见 dev。独立训练环境、模板监督边界、基座冻结、本地保存加载均有证据，全部12项测试通过。训练 loss 降低但 dev 整条正确数从2/3降到1/3，adapter 不推广替代基座；这不是泛化准确率。

私有 HF 模型仓库 `YueranCao2001/qwen3-0.6b-zh-extraction-lora` 已创建并上传；commit `def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f` 下载哈希及加载生成与本地一致。Hugging Face 凭据保留在本机，训练数据不随产物上传。

10月23日完成固定六步的数据替换对照：3条旧 train 重复两次 vs 旧3条+新增3条边界样例，各从相同基座和初始 adapter 开始。9条可见合成诊断题的整条匹配为基座5/9、对照3/9、候选9/9，旧3题错误已修复。该题集针对已知失败设计且模板有重叠，不能当作泛化准确率；监督 token 预算和样例曝光/顺序并未匹配。

候选已同步至私有 HF 分支 `boundary-v2-20261023`，commit `e3b24d1115c21cf69feaa5e6e9cf85a98f502c84` 的下载哈希及加载生成通过验证，`main` 保持昨天版本。23项本地测试通过。下一项是标注边界规则和独立人工审查评估，而不是直接推广候选。精确续训和基础能力保留评估仍待完成。

真实模型的模板检查、一步训练、保存恢复与独立评估，不因合成数值测试通过而自动完成。转向训练后先落实 [小模型实验设计](../daily/2026/09/2026-09-15-small-model-experiment-design.md) 中的待办，再扩大算法范围。

10月24日冻结单活动标注规则，对相同基座/候选执行52次只读 GPU 生成，0步训练。旧9题仍为5/9 vs 9/9；新12条开放审计为6/12 vs 11/12，但候选将明确地点“潮汐教室”遗漏为 null。5条基础探针严格通过为2/5 vs 3/5，需区分内容错误与格式问题，不作为综合能力验收。独立人工审查仍为 pending；34项本地测试通过。

审计文件已同步到原 HF 实验分支的 evaluations 子目录（commit `4890a0afc0d0eb3f9561d34f7220e5b0ffeaf6ad`），模型权重和 main 不变。下一项围绕新地点遗漏做最小对照，继续保留人工复核和精确续训待办。

[评估与实验方法](evaluation-experiments.md) 提供这条实践链路的设计与验收入口。

[返回主题索引](README.md)
