# 2026-09-14：后训练链路复盘——SFT、DPO、PPO/GRPO 的统一实验协议与验收

## 今日目标

上一篇 [可验证奖励与评估器工程](2026-09-11-verifiable-rewards-evaluator.md)完成了一个窄任务数值评分器。今天做阶段复盘：将最近几篇的知识串成可执行的实验协议，明确哪些结论已有合成测试支持，哪些仍待真实模型验证。

完成后应能回答：

1. SFT、DPO 与在线 RL 是什么关系，是否必须依次运行？
2. 哪些工件和规则必须跨训练、生成、评分与评估保持一致？
3. loss、mask 和归约为什么不能只靠算法名称表达？
4. 训练前、训练一步后、恢复后各要检查什么？
5. 怎样防止配置缺项、split 泄漏和错误的“测试通过”声明？
6. 怎样设计下一个小实验，避免同时改变多个因素？

本文是项目自定义复盘与教学验证，不是某个框架的安装教程。没有运行真实模型微调、GPU 训练、生成评测或代码沙箱；Python 示例只检查人工构造的配置与数据 ID。

---

## 1. 后训练不是必须全部跑一遍的固定流水线

[SFT](2026-09-07-supervised-fine-tuning-chat-template-lora.md)学习示范回答；[DPO](2026-09-08-direct-preference-optimization.md)学习同题候选之间的偏好；[PPO/GRPO](2026-09-09-online-rl-ppo-grpo.md)则利用采样回答及其奖励更新策略。

```text
固定起点 checkpoint
    ├─ 示范数据 → SFT → 开发集评估
    ├─ 偏好对   → DPO → 开发集评估
    └─ prompt  → rollout → reward → PPO/GRPO → 开发集评估
```

实际项目可以从已有 SFT checkpoint 开始研究 DPO 或在线 RL，不必重新做 SFT；也不要求先 DPO 再 GRPO。顺序是实验选择，不是算法定义。

基础离线 DPO 不需要在每个训练步骤在线生成候选，PPO 则交替采样与优化。GRPO 的重要特点是通过组内奖励估计基准，省去基础 actor-critic 方案中的 learned critic。[DPO 原论文](https://arxiv.org/abs/2305.18290) [PPO 原论文](https://arxiv.org/abs/1707.06347) [DeepSeekMath](https://arxiv.org/abs/2402.03300)

“方法更复杂”不代表适合当前问题。先确定数据和可靠的评价标准，再选择最小可检验方案。

---

## 2. 用一张表对齐算法边界

| 维度 | SFT 教学基线 | 离线 sigmoid DPO | PPO 教学基线 | Outcome GRPO 教学基线 |
|---|---|---|---|---|
| 样本 | prompt 与示范回答 | 同 prompt 的 chosen/rejected | 带行为身份的轨迹 | 同题多条带奖励回答 |
| 训练信号 | 目标 token | 偏好标签 | 时序 advantage | 组内相对 advantage |
| old policy | 不需要 | 不需要 | 本轮采样基准 | 本轮采样基准 |
| reference | 普通 SFT 不需要 | 固定对照 | 启用相应正则时使用 | 启用相应正则时使用 |
| critic | 不需要 | 不需要 | 本篇基线使用 | 基础 outcome 版本不使用 |
| 主要验收点 | labels、shift、mask | 四个 logp、margin | ratio、GAE、value target | 组完整性、零方差、归约 |

表格描述前几篇选定的教学方案，不排除无 KL、不同优势估计或其他损失变体。不能从“GRPO”三个字推导出 beta、归约或 reference 一定如何配置。

同样，LoRA 是参数更新方式，不是第四种训练目标。SFT、DPO 和某些 RL 实现都可以搭配 adapter，但其 reference 身份和保存方式要另行检查。

---

## 3. 实验身份：把“模型版本”展开成工件集合

一份可复盘记录至少包含：

- 初始权重、训练中 checkpoint 和 adapter 状态；
- tokenizer revision 与 chat template hash；
- 数据快照、split manifest、去重组规则；
- loss 类型、系数、mask、归约与截断；
- optimizer、scheduler、精度、累积步和随机状态；
- 在线路径的采样配置、行为版本和 reward revision；
- checkpoint 选择规则与独立评估协议；
- 训练代码 revision 与依赖锁定文件。

工件版本相同不等于行为完全相同：随机种子、采样策略、kernel 路径和硬件都可能影响输出。清单用于定位差异，不承诺跨设备逐位一致。

缓存还要绑定内容身份。比如 reference logp 缓存不能只写“来自某个 SFT 模型”，还要绑定 token IDs、completion mask、EOS 和 scoring reduction。

---

## 4. Mask 是目标定义，不只是 Padding 处理

把同一段对话交给不同训练方式，监督位置可能不同：

- assistant-only SFT 可以覆盖历史多个 assistant 回合。
- DPO 比较共同 prompt 后分叉的两条 completion，历史 assistant 消息属于条件。
- 在线 policy loss 只作用于被定义为动作的 token；工具返回内容不自动成为动作。
- PAD 位置排除，真实 EOS 是否计入由模板和任务协议决定。
- 第一个 completion token 由最后一个 prompt 位置的 logit 预测。

因此“attention mask 正确”不足以证明训练目标正确。attention mask、label mask、completion mask、action mask 含义不同。

最小 golden fixture 应保存 rendered text、token IDs、角色边界、首个目标位置、EOS 与 PAD。先逐 token 检查三条短样本，再对接大数据集；不要等 loss 异常后才猜 token 边界。

---

## 5. 归约决定样本权重

设第 i 个样本包含 Ti 个有效目标，单 token 损失为 l。常见两种归约：

```math
L_{\mathrm{token}}
=\frac{\sum_i\sum_{t=1}^{T_i}l_{i,t}}{\sum_i T_i},
\qquad
L_{\mathrm{sample}}
=\frac1B\sum_i\frac1{T_i}\sum_{t=1}^{T_i}l_{i,t}
```

前者每个有效 token 同权，后者每个样本同权。短长样本混合时二者一般不同。

自行构造两个样本：2 个 token 的 loss 都是 1，6 个 token 的 loss 都是 3：

```text
全 token 平均 = (2 × 1 + 6 × 3) / 8 = 2.5
先样本平均   = (1 + 3) / 2 = 2.0
```

普通 DPO 还存在另一层：先对一条回答的 log probability 求和形成 sequence score，再对偏好对的 loss 平均。不能把它与 token loss mean 混成一个“都取平均”的模糊描述。

跨 GPU 或累积 microbatch 时，局部均值的平均也未必等于全局有效 token 均值。验收时比较完整分子、分母及最终梯度，不仅比较打印出来的 batch loss。

---

## 6. 从单函数测试到端到端证据

分层验收：

| 层级 | 证据 | 不能替代什么 |
|---|---|---|
| 数学 oracle | 手算与纯 Python 一致 | 真实模型梯度 |
| 数据 fixture | token、mask、边界可追溯 | 分布式归约 |
| 一步训练 | 有限梯度、正确冻结、权重变化 | 长期稳定性和质量 |
| 恢复测试 | 同协议连续/恢复运行可比较 | 线上泛化 |
| 独立评估 | 目标能力、保留能力、成本 | 未覆盖任务的保证 |

已有笔记完成了部分 oracle 测试，但尚未完成真实模型整链路实验。今天的配置测试也不能升级成“训练已验收”。

证据应附带执行环境与适用范围。“通过”必须回答通过了哪个测试、使用什么输入、是否涉及真实后端；不能只有一个布尔字段。

---

## 7. 统一验收门槛与停止条件

建议将实验分成五个关口：

1. 数据关：gold/标签有效、split 无已知泄漏、组与样本 ID 唯一。
2. 前向关：模板和 mask 正确；初始化数值符合所选目标预期。
3. 更新关：梯度有限，policy 可更新，冻结对象保持不变。
4. 状态关：权重、缓存、RNG、采样器与恢复边界可追溯。
5. 质量关：独立开发集达标，保留能力和生成行为未出现不可接受退化。

出现 NaN、reference 漂移、组混题或评分基础设施失败时，应按预定规则停止或隔离，不能靠降低日志频率或吞掉异常继续训练。

“训练 loss 下降”只满足部分优化观察，不等于通过质量关。尤其 DPO margin、在线 reward 和 held-out 任务成功率分别回答不同问题。

---

## 8. 纯 Python：教学配置预检器

下面故意只支持两个已经讲清楚的变体：固定 reference 的 sigmoid DPO，以及启用固定 reference KL 的 outcome GRPO。其他变体应扩展 schema 后再使用，不应偷偷复用默认值。

ID 集合由调用方提供，可以代表 prompt/source 去重组；这里只检查精确集合交集，不做语义近重复检测，也不核实实际工件存在。

```python
import math

def validate_experiment(cfg, splits):
    errors = []
    required = (
        "run_id", "init_revision", "tokenizer_revision", "template_hash",
        "dataset_snapshot", "code_revision", "dependency_lock",
        "eval_protocol", "resume_boundary",
    )
    for key in required:
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            errors.append("missing:" + key)

    names = ("train", "dev", "test")
    if any(name not in splits for name in names):
        errors.append("missing_split")
    else:
        groups = {}
        for name in names:
            values = splits[name]
            if not isinstance(values, list) or not values or any(
                not isinstance(v, str) or not v for v in values
            ):
                errors.append("invalid_split:" + name)
                continue
            if len(set(values)) != len(values):
                errors.append("duplicate_group:" + name)
            groups[name] = set(values)
        if len(groups) == 3:
            for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
                if groups[a] & groups[b]:
                    errors.append("split_overlap:" + a + ":" + b)

    beta = cfg.get("beta")
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0:
        errors.append("beta_must_be_positive")
    if not isinstance(cfg.get("reference_revision"), str) or not cfg["reference_revision"]:
        errors.append("missing_reference")
    if cfg.get("reference_frozen") is not True:
        errors.append("reference_not_frozen")
    if cfg.get("resume_boundary") not in ("optimizer_step", "rollout_round"):
        errors.append("unsupported_resume_boundary")

    objective = cfg.get("objective")
    if objective == "dpo_sigmoid":
        if cfg.get("score_reduction") != "completion_sum":
            errors.append("dpo_requires_completion_sum")
        if cfg.get("loss_reduction") != "pair_mean":
            errors.append("dpo_requires_pair_mean")
    elif objective == "grpo_outcome_ref_kl":
        if type(cfg.get("group_size")) is not int or cfg["group_size"] < 2:
            errors.append("invalid_group_size")
        for key in ("reward_revision", "sampling_config_hash", "behavior_logp_contract"):
            if not isinstance(cfg.get(key), str) or not cfg[key]:
                errors.append("missing:" + key)
        if cfg.get("loss_reduction") not in ("response_mean", "global_token_mean"):
            errors.append("unspecified_grpo_reduction")
        if cfg.get("infra_failure_policy") != "retry_or_exclude_group":
            errors.append("unsafe_infra_failure_policy")
        if cfg.get("resume_boundary") != "rollout_round":
            errors.append("teaching_grpo_requires_round_boundary")
    else:
        errors.append("unsupported_objective")
    return sorted(set(errors))
```

这里 beta>0 是所选 schema 的限制，不是在断言所有 GRPO 都必须有 KL。配置验证器应明确拒绝不支持的变体，而不是将其错误地标成已验收。

本篇没有让 checker 读取文件或连接服务，所以 revision/hash 都只是标识字段。真实 preflight 还应验证依赖、hash、工件权限、token fixture 和 reference 对象身份。

---

## 9. 用故障注入验证预检器

```python
from copy import deepcopy

splits = {"train": ["source-1", "source-2"],
          "dev": ["source-3"], "test": ["source-4"]}
cfg = {
    "run_id": "synthetic-dpo",
    "init_revision": "sft-r1",
    "tokenizer_revision": "tok-r1",
    "template_hash": "template-r1",
    "dataset_snapshot": "pairs-r1",
    "code_revision": "code-r1",
    "dependency_lock": "lock-r1",
    "eval_protocol": "eval-r1",
    "resume_boundary": "optimizer_step",
    "objective": "dpo_sigmoid",
    "reference_revision": "sft-r1",
    "reference_frozen": True,
    "beta": 0.1,
    "score_reduction": "completion_sum",
    "loss_reduction": "pair_mean",
}
assert validate_experiment(cfg, splits) == []

for key, value, expected in [
    ("template_hash", "", "missing:template_hash"),
    ("reference_frozen", False, "reference_not_frozen"),
    ("beta", float("nan"), "beta_must_be_positive"),
    ("beta", True, "beta_must_be_positive"),
    ("score_reduction", "token_mean", "dpo_requires_completion_sum"),
    ("objective", "unknown", "unsupported_objective"),
]:
    changed = {**cfg, key: value}
    assert expected in validate_experiment(changed, splits)

leaked = deepcopy(splits)
leaked["dev"].append("source-1")
assert "split_overlap:train:dev" in validate_experiment(cfg, leaked)
duplicated = deepcopy(splits)
duplicated["train"].append("source-1")
assert "duplicate_group:train" in validate_experiment(cfg, duplicated)

grpo = {
    **cfg, "run_id": "synthetic-grpo",
    "objective": "grpo_outcome_ref_kl", "group_size": 4,
    "reward_revision": "numeric-final-v1",
    "sampling_config_hash": "temperature-1-no-filter",
    "behavior_logp_contract": "selected-action-post-transform",
    "loss_reduction": "response_mean",
    "infra_failure_policy": "retry_or_exclude_group",
    "resume_boundary": "rollout_round",
}
assert validate_experiment(grpo, splits) == []
assert "invalid_group_size" in validate_experiment({**grpo, "group_size": 1}, splits)
assert "unsafe_infra_failure_policy" in validate_experiment(
    {**grpo, "infra_failure_policy": "replace_missing_reward_with_zero"}, splits
)
assert "missing:reward_revision" in validate_experiment(
    {**grpo, "reward_revision": ""}, splits
)
assert "missing_split" in validate_experiment(cfg, {"train": ["a"]})

# Reduction example: unequal lengths produce distinct objectives.
losses = [[1.0] * 2, [3.0] * 6]
token_mean = sum(map(sum, losses)) / sum(map(len, losses))
sample_mean = sum(sum(row) / len(row) for row in losses) / len(losses)
assert token_mean == 2.5 and sample_mean == 2.0
print("PASS: manifest, objective constraints, split leakage, failure policy, reduction")
```

这些断言验证的是配置契约。它们不会判断 chosen 是否真的更好，也不会判断 reward 是否被利用；后两者仍要靠数据审查和独立评估。

---

## 10. 最小真实实验：一次只增加一个不确定因素

下一次动手可以选择一个已有小型 SFT checkpoint，固定模板和数据，先跑 DPO 的一条偏好对：

- 更新前检查 policy/reference logp 与初始 loss。
- 一步训练检查有限梯度及 reference 不变。
- 用同一小 batch 比较连续两步与保存恢复后的两步。
- 再扩展到小开发集，观察 loss 与实际生成行为。

这是实验计划，不是本次已执行结果。模型、硬件预算和数据许可尚未选定，不提前宣称真实训练可以在当前机器运行。

如果改走在线 GRPO，先复用上一节的数值评分契约，但需要额外验收 rollout token 身份、组完整性、采样分布与 reward 缺失处理。不要第一步就同时引入异步队列、多 GPU、LoRA 和复杂 judge。

---

## 11. 怎样比较实验而不误读数字

比较两个方法时，至少声明控制变量与不相等的预算：

- 是否从同一初始 checkpoint 出发；
- 数据来源与标注成本是否一致；
- 训练 token、生成 token、GPU 时间是否同时记录；
- evaluation prompt、采样配置和 scorer 是否固定；
- 是否使用相同开发集选 checkpoint；
- 长度、停止率与原有能力是否一起报告。

不同算法的 loss 数值不在同一尺度，不能用“DPO loss 比 SFT 小”判优。在线 reward 提升也不能直接与离线 preference accuracy 比较。

本项目的验收重点是目标任务质量、保留能力与端到端成本，而不是寻找一个跨算法通用的训练 loss 排名。

---

## 12. 本次复盘结论与遗留问题

### 本次完成的教学验证

- 配置预检拒绝缺失模板身份、非法 beta 与未冻结 reference。
- 拒绝 DPO 的错误 sequence score reduction。
- 检测精确 group ID 重复与 split 交集。
- 拒绝在线组大小不足、缺失 reward revision 和缺失 reward 补零策略。
- 用长短样本证明 token mean 与 sample mean 不同。

### 仍未完成的实证工作

- 真正的 tokenizer/模型端到端 fixture；
- 梯度冻结与 optimizer step 验收；
- 真实训练状态恢复；
- 多 GPU 全局归约；
- 独立生成质量、评分器误判率与端到端成本比较。

这张清单的价值在于防止把“理解了公式”和“跑通了实验”混为一谈。下一步应补最小真实证据，而不是继续堆叠未验证的组件。

---

## 参考与回顾索引

- [SFT：模板、assistant-only loss、packing 与 LoRA](2026-09-07-supervised-fine-tuning-chat-template-lora.md)
- [DPO：reference、margin 与长度偏差](2026-09-08-direct-preference-optimization.md)
- [PPO/GRPO：优势、ratio 与 mask](2026-09-09-online-rl-ppo-grpo.md)
- [Rollout：权重、采样身份与异步队列](2026-09-10-rl-rollout-engineering.md)
- [Verifier：答案契约、容差与失败分类](2026-09-11-verifiable-rewards-evaluator.md)
- [DPO 原论文](https://arxiv.org/abs/2305.18290)
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO 原论文](https://arxiv.org/abs/2402.03300)

核对日期：2026-09-14。本文的配置 schema 为教学自定义格式，不是任何 Trainer 的可直接加载配置。

## 今日总结

后训练可靠性来自一致的数据、目标和状态协议。算法公式、token 边界、工件版本、评分规则、恢复与评估需要共同验收，任何一项缺失都可能让漂亮的曲线失去解释力。

今天把已有笔记收束成统一实验清单，并用合成故障验证配置预检逻辑。真实训练结果仍留待后续，不以测试覆盖之外的结论代替证据。

## 下次衔接建议

继续学习小模型后训练实验设计：如何选择固定 baseline、构造无泄漏的小数据集、安排一步训练与恢复测试，以及记录预算和失败结果。

