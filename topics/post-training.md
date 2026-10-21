# 后训练与偏好优化

SFT、DPO、PPO 与 GRPO、Rollout 和奖励评估器。当前记录以概念及合成验证为主，真实模型训练的完成状态需要实际产物支持。

## 日期笔记

- 2026-09-07 [Supervised Fine-Tuning：Chat Template、Assistant-only Loss、Packing 与 LoRA](../daily/2026/09/2026-09-07-supervised-fine-tuning-chat-template-lora.md)
- 2026-09-08 [Direct Preference Optimization：偏好对、Reference Model、DPO Loss 与长度偏差](../daily/2026/09/2026-09-08-direct-preference-optimization.md)
- 2026-09-09 [在线强化学习后训练：Rollout、Advantage、PPO 与 GRPO](../daily/2026/09/2026-09-09-online-rl-ppo-grpo.md)
- 2026-09-10 [RL Rollout 工程：权重同步、采样 Log Probability、异步队列与 Policy Staleness](../daily/2026/09/2026-09-10-rl-rollout-engineering.md)
- 2026-09-11 [可验证奖励与评估器工程：答案抽取、容差、失败分类与 Reward Hacking](../daily/2026/09/2026-09-11-verifiable-rewards-evaluator.md)
- 2026-09-14 [后训练链路复盘：SFT、DPO、PPO/GRPO 的统一实验协议与验收](../daily/2026/09/2026-09-14-post-training-end-to-end-review.md)

## 实践衔接

用户已安排自2026年10月22日起，将当前学习主线转向大模型微调。先确认任务、候选模型与脱敏数据，核查训练环境，再用小规模 SFT 做模板与监督边界检查、一步训练、保存恢复和独立评估。已有 LoRA、QLoRA 笔记作为参考，不预先假定显存一定足够。

目前未选择训练模型、未安装新训练环境、未启动真实微调。Hugging Face 凭据保留在本机，聊天和仓库只记录非敏感模型 ID、训练参数及执行证据。

真实模型的模板检查、一步训练、保存恢复与独立评估，不因合成数值测试通过而自动完成。转向训练后先落实 [小模型实验设计](../daily/2026/09/2026-09-15-small-model-experiment-design.md) 中的待办，再扩大算法范围。

[评估与实验方法](evaluation-experiments.md) 提供这条实践链路的设计与验收入口。

[返回主题索引](README.md)
