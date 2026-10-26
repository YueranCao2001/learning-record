# 后训练与偏好优化

SFT、DPO、PPO 与 GRPO、Rollout 和奖励评估器。当前记录以概念及合成验证为主，真实模型训练的完成状态需要实际产物支持。

## 日期笔记

- 2026-10-26 [中文信息抽取微调：训练中断后的精确续训对照](../daily/2026/10/2026-10-26-sft-exact-training-resume.md)
- 2026-10-25 [中文信息抽取微调：地点遗漏的最小对照](../daily/2026/10/2026-10-25-sft-location-minimal-pairs.md)
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

10月25日完成一个来源组上的六种固定最小对照，共12次 GPU 生成、0步训练。候选原题遗漏被重现；补入明确时间或将“场地设在”改为“地点是”后地点恢复，仅改变时间缺失措辞、分句顺序或缩短开头仍遗漏。不是通用修复或内部机制证明。42项本地测试通过，仅同步 HF 诊断文档（commit `5f582e5a3da1d7a86c4116bb0512b9613f0d2ead`），权重/main 不变。

地点问题暂存为已定位、待审和未通用修复；下一工程项是 optimizer-step 边界的精确续训对照，独立人工审查和更广泛质量评估继续保留。

10月26日完成同机单GPU的四进程精确续训对照：连续4步 vs 前缀2步+完整恢复2步，LoRA dropout=0.1、确定性 math SDPA；loss/梯度/样本/LR记录、adapter、optimizer、scheduler、RNG和步数逐项一致，adapter最大差0。仅重置优化器历史的负对照从更新后分离，最终 adapter 最大差约0.000340273。实际共10个优化器步骤，不是质量提升实验。

50项本地测试通过，仅将12个工程 JSON 同步到 HF engineering 目录（commit `12484003bb33219209a7520022ba85aea5923bf5`），原模型和 main 不变。下一工程项为恢复前身份校验及拒绝不匹配状态；多worker/分布式/任意microstep恢复仍未验证，人工审查与质量待办保留。

[评估与实验方法](evaluation-experiments.md) 提供这条实践链路的设计与验收入口。

[返回主题索引](README.md)
