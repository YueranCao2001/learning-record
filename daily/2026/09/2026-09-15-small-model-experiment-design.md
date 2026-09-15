# 2026-09-15：小模型后训练实验设计——固定 Baseline、数据切分、一步训练与恢复对照

## 今日目标

上一篇 [后训练链路复盘](2026-09-14-post-training-end-to-end-review.md)整理了统一验收条件。今天将清单转成一个可安排的小实验：先验证实现，再讨论质量提升。

完成后应能回答：

1. baseline 应冻结哪些内容？
2. 如何构造小而可审查的数据集，避免同源泄漏？
3. 一步训练、微型过拟合和开发集评估各证明什么？
4. 为什么恢复训练不能只保存权重？
5. 怎样记录失败、预算与选择规则？

本篇提供实验计划及纯 Python 教学测试。没有选择或下载真实模型，没有运行 GPU 微调，也没有声称获得模型能力提升。合成优化器只用于演示状态恢复，不是语言模型训练实现。

---

## 1. 把研究问题写成可证伪的句子

不要把目标写成“跑通 DPO 并提升效果”。它混合了实现正确性与质量结论。

建议拆成两个问题：

- 实现问题：固定一条偏好对，初始化分数、梯度方向与 reference 冻结是否符合预期？
- 质量问题：固定训练预算后，相对同一 SFT baseline，独立开发集上的任务成功率是否改善，且保留能力没有超过预定容限的退化？

第一项通过，不推出第二项成立。小 batch 过拟合可以帮助发现学习路径断开，但不能证明泛化。

本篇以离线 DPO 为后续真实实验候选，减少 rollout 和在线评分的不确定因素；不是断言 DPO 在所有任务上更优。

---

## 2. 固定 Baseline，而不只是模型名称

实验起点清单至少包含：

| 工件 | 需要固定的内容 |
|---|---|
| 模型 | 仓库、不可变 revision、权重类型与许可 |
| adapter | 基座、adapter revision、激活与合并状态 |
| tokenizer | revision、特殊 token、padding 与 EOS 规则 |
| 模板 | chat template、generation prefix、内容 hash |
| 推理 | temperature、过滤、长度、停止条件、seed |
| 环境 | 训练代码、依赖、设备、dtype 与 kernel 设置 |
| 评估 | prompt IDs、scorer revision、输出契约 |

“小模型”只表示相对规模，不自动保证能放进当前机器。先估算 policy、reference、optimizer、activation 和序列长度的预算，再决定模型与训练方式。本次不基于未经检查的显卡情况推荐具体模型。

baseline 必须在训练前用最终评估协议生成一次并保存原始输出。训练后再临时更换模板或采样策略，会把评估差异与训练差异混在一起。

---

## 3. 数据集小，但身份不能简化

一条原问题可能产生多个改写、示范和偏好对。切分单位应是来源组，而不只是行号。

```text
source group
  ├─ original prompt
  ├─ paraphrase
  ├─ chosen/rejected pair 1
  └─ chosen/rejected pair 2
        ↓ 同组进入同一 split
      train / dev / sealed test
```

先做来源追溯与去重组，再切分；否则“不同文本”也可能泄露相同问题和答案。

小数据集应逐条审核：问题是否一致、chosen 是否确实更好、两侧截断后是否仍不同、模板是否保留关键内容。组 ID 检查只能发现已知分组的交集，不能代替语义近重复检测。

dev 用于调参和 checkpoint 选择；test 在协议冻结后才打开。不要因为数据少就反复查看 test 来决定下一次改动。

---

## 4. 稳定 Hash 切分与它的限制

使用固定 namespace 和来源组 ID 计算 SHA-256，可使同组始终进入同一 split，且新增其他组不会移动旧组。

这不同于“把行号随机打乱后按比例截取”：后者可能受输入顺序和新增数据影响。

Hash 分桶得到的是近似比例，不保证小样本各 split 非空，也不保证类别平衡。必要时预先设计按组分层的固定清单。不能反复更换 salt，直到看见更好看的评估成绩。

---

## 5. 纯 Python：稳定切分与 Manifest

```python
import hashlib
import json

def split_for(group_id, salt="pilot-split-v1"):
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("non-empty group ID required")
    if not isinstance(salt, str) or not salt:
        raise ValueError("non-empty salt required")
    payload = json.dumps([salt, group_id], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"

def manifest(rows):
    result, seen = {}, set()
    for row in rows:
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("invalid or duplicate sample ID")
        seen.add(sample_id)
        result[sample_id] = {
            "group_id": row["group_id"],
            "split": split_for(row["group_id"]),
        }
    return result

def manifest_digest(value):
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
```

该 hash 只绑定示例中的 ID 与 split，并不覆盖实际训练文本。真实 manifest 还应包含文本/token 内容 hash、来源和处理版本；文件存在与数据许可也需另查。

---

## 6. 三阶段训练验收

| 阶段 | 范围 | 通过证据 |
|---|---|---|
| 一步测试 | 一条或少量偏好对 | 分数、mask、有限梯度、冻结对象不变 |
| 微型过拟合 | 固定极小训练子集 | 能拟合指定数据，排查学习路径 |
| 小规模试验 | 完整 pilot train + 固定 dev | 独立质量、保留能力与成本 |

一步测试先关闭不必要的随机性，并核对第一个 completion token、EOS、PAD 和 reference 身份。初始化 policy/reference 同权重且计算协议相同时，基础 DPO loss 应接近 log 2；不同目标不能照搬该检查。

微型过拟合允许用于诊断的样本被反复查看，但它们不能再作为独立质量证据。若训练 loss 降低而生成恶化，应检查目标偏差、长度与数据标签，不立即认定优化器坏了。

每升一级前保存原始证据，不把“没有报错”当作验收通过。

---

## 7. 恢复对照：同一个起点，两条路径

```text
路径 A：初始状态 → step 1 → step 2 → step 3 → step 4
路径 B：初始状态 → step 1 → step 2 → 保存/恢复 → step 3 → step 4
```

比较两条路径的样本顺序、学习率、loss、参数、optimizer 状态与 RNG。首先在相同环境和确定性小实验上对照；生产环境的容差应预先声明。

PyTorch 的通用续训示例保存模型和 optimizer 等状态；仅保存推理权重不足以恢复训练动态。[PyTorch 保存与加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

真实后训练还可能需要 scheduler、AMP scaler、sampler 游标、累积梯度与在线 rollout 状态。项目先选择 optimizer-step 边界保存，避免无意承诺 microstep 任意位置恢复。

---

## 8. 教学优化器：证明“丢失状态会改变后续结果”

下面用一个标量参数、动量和随 step 衰减的学习率演示。目标是拟合数值 1，不包含任何语言模型、DPO loss 或自动微分。

```python
from copy import deepcopy

def fresh():
    return {"weight": 0.0, "momentum": 0.0, "step": 0, "cursor": 0}

def advance(state, steps, targets=(1.0, 1.0, 1.0, 1.0)):
    for _ in range(steps):
        target = targets[state["cursor"] % len(targets)]
        gradient = state["weight"] - target
        lr = 0.1 / (1 + state["step"])
        state["momentum"] = 0.9 * state["momentum"] + gradient
        state["weight"] -= lr * state["momentum"]
        state["step"] += 1
        state["cursor"] += 1
    return state

rows = [
    {"sample_id": "a", "group_id": "source-a"},
    {"sample_id": "b", "group_id": "source-a"},
    {"sample_id": "c", "group_id": "source-b"},
]
m = manifest(rows)
assert m["a"]["split"] == m["b"]["split"]
assert manifest(list(reversed(rows))) == m
extended = manifest(rows + [{"sample_id": "d", "group_id": "source-c"}])
assert all(extended[key] == value for key, value in m.items())
assert manifest_digest(manifest(list(reversed(rows)))) == manifest_digest(m)

try:
    manifest(rows + [rows[0]])
except ValueError:
    pass
else:
    raise AssertionError("duplicate sample accepted")

continuous = advance(fresh(), 4)
partial = advance(fresh(), 2)
saved = deepcopy(partial)
restored = advance(deepcopy(saved), 2)
assert restored == continuous
assert saved == partial  # Saved state is not a live mutable alias.

missing_momentum = deepcopy(saved)
missing_momentum["momentum"] = 0.0
assert advance(missing_momentum, 2)["weight"] != continuous["weight"]

reset_step = deepcopy(saved)
reset_step["step"] = 0
assert advance(reset_step, 2)["weight"] != continuous["weight"]

# A distinct target sequence makes the sample cursor observable.
targets = (1.0, -1.0, 2.0, 0.0)
expected = advance(fresh(), 4, targets)
checkpoint = advance(fresh(), 2, targets)
wrong_cursor = deepcopy(checkpoint)
wrong_cursor["cursor"] = 0
assert advance(wrong_cursor, 2, targets)["weight"] != expected["weight"]
print("PASS: stable group split, manifest identity, resume equivalence, missing-state failures")
```

该测试使用内存深拷贝模拟保存，未测试文件序列化、GPU kernel、分布式 checkpoint 或 RNG 恢复。它只说明状态缺失可以影响轨迹，不能作为真实 Trainer 恢复验收报告。

固定 seed 也不能保证跨 PyTorch 版本、平台或 CPU/GPU 得到完全相同结果；应将环境约束和确定性设置写入实验记录。[PyTorch 可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

---

## 9. 预算、停止与失败结果

pilot 计划应先声明资源上限：有效训练 token、optimizer steps、最长运行时间、显存和磁盘预算。达到任一上限按协议结束，不因曲线尚未变好就无限延长。

建议记录以下状态，而不是只有成功/失败：

- preflight_failed：数据或配置没有通过；
- numerical_failed：出现非有限 loss/gradient；
- resource_exhausted：显存、时间或磁盘不足；
- completed_no_gain：正常完成但独立指标没有改善；
- completed_with_gain：在冻结协议下观察到改善；
- inconclusive：样本太少、误差过大或评估故障，证据不足。

失败运行也保存配置、最后有效 step 与诊断摘要。记录不代表保存所有敏感原始数据，应遵守数据权限。

---

## 10. 怎样避免“小样本胜利”

对同一批 dev prompts 保存 baseline 与候选输出，按 prompt/source group 配对比较。检查提升来自哪些题型，以及是否伴随冗长、停止失败或保留能力下降。

不要只报告最好的一次 seed。若预算只能跑一次，应明确结果是单次探索；若多次试验，应保存所有运行和选择规则。

一个参数改动对应一个假设。例如只改变学习率时，保持数据顺序、预算、模板和 scorer 不变。若同时改变 batch、LoRA rank 和数据过滤，结果只能说明组合变化，不能归因给单个因素。

训练后 test 只用于既定最终评估，不用来决定下一轮超参数。继续迭代应回到 dev，并对最终结论保留适当不确定性。

---

## 11. 今日练习与遗留问题

### 本次完成的合成验证

- 同源样本进入相同 split。
- 输入顺序改变不会改变 manifest 内容和规范化 hash。
- 新增其他组不移动既有组。
- 重复 sample ID 被拒绝。
- 标量优化器完整恢复与连续运行一致。
- 丢失动量、重置 step 或错误样本游标会改变后续结果。

### 尚待真实实验

1. 明确硬件预算、模型许可和不可变 checkpoint。
2. 逐条审核 pilot 数据与来源组。
3. 跑 tokenizer golden fixtures 和一步训练。
4. 在真实后端进行保存/恢复对照。
5. 运行固定预算的 pilot，并保存无增益或失败结果。

今日没有真实模型训练成绩。小实验的价值首先是缩小未知范围，再用独立数据回答是否提升能力。

---

## 参考资料

- [后训练链路统一验收](2026-09-14-post-training-end-to-end-review.md)
- [DPO 数值基线](2026-09-08-direct-preference-optimization.md)
- [可验证奖励与失败分类](2026-09-11-verifiable-rewards-evaluator.md)
- [PyTorch：保存与加载模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch：可复现性](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

核对日期：2026-09-15。代码使用 Python 标准库；PyTorch 文档用于核对实验原则，本次未实例化 PyTorch 模型。

## 今日总结

可信的小模型实验从冻结 baseline 和数据身份开始，以分层验收、完整状态恢复和独立评估结束。小数据不意味着可以省略来源切分，小步数也不意味着可以忽略 optimizer 状态。

今天用稳定分组和标量恢复测试把两个关键实验原则变成了可执行检查。真实模型实验仍需独立验收，不能由教学代码推断其已完成。

## 下次衔接建议

继续学习训练结果的统计解释：配对比较、按来源组 bootstrap、置信区间与多次试验选择偏差，区分“观察到分数更高”和“证据足以支持改进”。

