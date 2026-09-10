# 2026-09-10：RL Rollout 工程——权重同步、采样 Log Probability、异步队列与 Policy Staleness

## 今日目标

上一篇 [在线强化学习后训练：PPO 与 GRPO](2026-09-09-online-rl-ppo-grpo.md)建立了 reward、advantage、old policy 和 reference 的数学边界。今天继续研究怎样把这个循环可靠地跑起来：训练侧每更新一次参数，推理侧到底在用哪份权重？生成回来的 log probability 又属于哪个分布？

完成后应能回答：

1. 同步、流水线和全异步 rollout 的数据时序有什么区别？
2. 采样权重版本为何不能只写一个 global step？
3. sampling logp、old-training logp、current logp 如何区分？
4. 有界队列为什么同时影响显存、吞吐和算法数据分布？
5. 怎样处理跨版本回答、未完成组、重试和截断？
6. 怎样用小规模合成测试验收版本与概率协议？

本篇提供项目自定义协议、纯 Python 数值检查和离散事件示例。未运行 GPU rollout、训练服务或分布式权重传输；示例耗时和阈值均为教学构造，不是性能测量或真实超参数建议。

---

## 1. 优化器完成，不等于推理服务已经更新

训练进程的参数更新可能已经结束，但推理 worker 仍持有旧权重。中间还包括参数收集、分片转换、传输、加载、缓存处理和 worker 确认。

把“训练结束”当成“新策略已经可采样”，会让版本记录与实际行为不一致。

```text
trainer revision v17
  → freeze/export manifest
  → inference workers load all required tensors
  → verify completeness and cache epoch
  → every required worker acknowledges v17
  → publish active rollout revision v17
  → admit new requests tagged v17
```

这里的 manifest 至少绑定模型权重、adapter 状态、tokenizer、chat template 和采样规则。版本号是方便排序的标签，内容 hash 才用于区分实际工件；两者不能互相替代。

vLLM 提供训练到推理的权重传输机制，并区分初始化、开始更新、传输和完成阶段。[vLLM Weight Transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/)

本文在此基础上提出更严格的教学验收协议，不把下面的字段名或状态机当成某个框架的现成 API。

---

## 2. 三种流水方式

| 方式 | 样本产生和参数更新 | 优点 | 主要代价 |
|---|---|---|---|
| 同步批次 | 生成整批 → 更新 → 同步权重 | 版本边界最清楚 | 等待与长尾空闲 |
| 有界流水线 | 生成下一批与训练上一批重叠 | 更好利用独立资源 | 有限策略滞后 |
| 全异步队列 | 多生成器持续生产，训练器按需消费 | 调度更灵活 | 混合版本、背压与恢复复杂 |

同卡共置并不自动实现计算重叠；生成 KV cache、训练激活与优化器状态可能争用同一份显存。分离 GPU 可减少这种争用，但增加传输和独立资源预算。

公平比较必须固定总 GPU 数、模型、prompt 分布、有效训练 token 与质量目标，不能只比较训练进程的耗时。

vLLM 的异步指南讨论生成与训练重叠，以及同步权重时的暂停/恢复窗口。[vLLM Async RL](https://docs.vllm.ai/en/latest/training/async_rl/)

---

## 3. 用理想流水模型估算收益上限

假设有两套独立资源，生成一批耗时 Tg，训练一批耗时 Tt，暂时忽略评分、同步、队列和资源争用。

```math
T_{\mathrm{serial}}=N(T_g+T_t)
```

流水线包含启动、稳态与排空：

```math
T_{\mathrm{pipeline}}
=T_g+T_t+(N-1)\max(T_g,T_t)
```

自行构造 N=4、Tg=6 秒、Tt=4 秒：

```text
串行总时间 = 4 × (6 + 4) = 40 秒
流水总时间 = 6 + 4 + 3 × 6 = 28 秒
理想加速比 = 40 / 28 ≈ 1.43
```

无限多批次时，两阶段理想加速上限为两阶段耗时之和除以较慢阶段耗时。实际系统还有 reward 服务、同步和 cache 重建等开销；不应把这个数当作框架实测收益。

特别注意：该时间模型只描述任务排程，没有自动证明下一批来自“训练完成后的最新策略”。实现重叠往往正是通过允许某种滞后来实现的。

---

## 4. Staleness 至少有三种尺度

本文定义版本 lag 为消费样本时的训练版本减去生成版本：

```math
L_v=v_{\mathrm{consume}}-v_{\mathrm{behavior}}
```

还应同时记录：

- wall-clock age：采样开始或完成到消费之间经过多久，明确采用哪个时间点；
- distribution drift：相同动作在当前与行为策略下的 log-ratio；
- queue residence：完成评分入队到出队之间的等待时间。

版本 lag 相同，可能因学习率、更新次数和数据不同产生完全不同的概率漂移。版本号不是 KL，也不应直接拿来替代重要性权重。

本文教学门槛 `max_version_lag=1` 表示最多落后一个已发布版本；它是自定义整数规则。verl 文档中的 `staleness_threshold` 有其样本比例语义，不能把两者当成同一参数。[verl Fully Async Policy Trainer](https://verl.readthedocs.io/en/latest/advance/fully_async.html)

---

## 5. 四种 Log Probability 的身份

| 名称 | 由谁计算 | 用途 |
|---|---|---|
| behavior / sampling logp | 实际生成引擎、实际采样分布 | 说明动作如何被抽到 |
| old-training logp | 训练引擎在冻结旧权重下重算 | 量化引擎差异或作为指定算法旧基准 |
| current-training logp | 正在更新的训练策略 | actor 梯度 |
| reference logp | 固定基准模型 | reference 正则与诊断 |

即使权重文件相同，推理与训练引擎也可能因计算精度和 kernel 路径产生不同概率；TRL 文档专门讨论这一训练—推理差异。[TRL：Training-Inference Mismatch](https://huggingface.co/docs/trl/grpo_trainer#dealing-with-the-training-inference-mismatch)

另一个独立来源是分布变换：temperature、top-p、top-k、重复惩罚、grammar mask 等会改变实际行为分布。返回字段叫 `logprobs`，并不能证明它是变换后的采样概率。

必须核对：所选 token 是否总被返回、logp 位于处理前还是处理后、是否包含被强制生成的特殊 token。测试从 temperature=1、无截断且无额外 processor 的小基线开始，再逐项打开变换。

---

## 6. Ratio 分解是恒等式，不是完整算法

在固定状态与动作上，可以分解：

```math
\frac{\pi_{\mathrm{current}}(a\mid s)}{\mu_{\mathrm{behavior}}(a\mid s)}
=
\frac{\pi_{\mathrm{current}}(a\mid s)}{\pi_{\mathrm{old,train}}(a\mid s)}
\cdot
\frac{\pi_{\mathrm{old,train}}(a\mid s)}{\mu_{\mathrm{behavior}}(a\mid s)}
```

第一个因子反映训练策略更新，第二个因子反映 old-training 与实际行为分布的差异。该等式要求相关分母非零；数值上应先做 logp 相减，再在安全范围内指数化。

两点容易被忽略：

- 对其中一个因子裁剪，再与另一个因子相乘，不等价于对乘积裁剪。
- token ratio 的代数正确，不代表任意异步轨迹都被完全校正；状态前缀分布、优势估计和目标定义仍需检查。

因此本篇不提供一个“把三个 logp 相乘就修好 PPO”的训练配方。我们只用数值测试验证字段身份与代数关系。

---

## 7. Temperature 与支持集：一个两动作反例

构造原始概率 `[0.8, 0.2]`。temperature=2 后：

```math
\mu_i=\frac{p_i^{1/2}}{\sum_j p_j^{1/2}}
```

行为概率变为 `[2/3, 1/3]`。第一个动作的原始 logp 为约 -0.223144，实际采样 logp 为约 -0.405465，两者不是同一数据。

若目标仍是原分布，用行为分布枚举加权，可以恢复目标下的某个函数期望；若把分母误写成原分布，所谓“校正权重”会变成 1，得不到该期望。

若 top-k=1 把第二个动作概率截为 0，而目标分布仍给它正概率，就不满足完整重要性采样所需的支持条件。无论有限次采样多少次，都观测不到被排除的动作，不能从已采样动作恢复完整目标期望。

这不是要求所有训练都禁用过滤，而是要求明确优化的是哪个分布，以及采用了什么近似或支持集处理。

---

## 8. 一条 Rollout 的最小可追溯记录

```json
{
  "run_id": "synthetic-run",
  "prompt_id": "p001",
  "group_id": "p001-round-07",
  "sample_id": "p001-round-07-candidate-0",
  "attempt_id": 0,
  "behavior_version": 17,
  "weights_digest": "immutable-artifact-id",
  "tokenizer_revision": "pinned-tokenizer",
  "template_digest": "pinned-template",
  "sampling_config_digest": "temperature-1-no-truncation",
  "reward_revision": "verifier-v1",
  "prompt_token_ids": [11, 12],
  "completion_token_ids": [21, 22, 2],
  "action_mask": [1, 1, 1],
  "sampling_logp": [-0.4, -0.6, -0.2],
  "finish_reason": "eos",
  "reward_status": "valid",
  "reward": 1.0
}
```

所有 ID、token、分数均为合成示例。实际记录还应包括 request 起止时间、worker ID、生成后端 revision、有效动作长度和全套 reward 分项。

`sample_id` 表示一个逻辑候选，`attempt_id` 表示重试。训练侧应防止同一逻辑候选被成功消费两次；若重试重新采样，必须记录新的随机状态与实际行为版本，并明确哪个 attempt 被采纳。

只存 completion text 不够：解码再重新分词可能不能还原实际生成动作。缓存键必须绑定 token IDs，而不仅是显示文本。

---

## 9. 原子发布与 KV Cache

教学基线选择单回答单版本：一条 completion 从开始到 EOS/截断都使用同一份权重。

建议的同步状态机：

```text
ACTIVE(v)
 → STOP_ADMISSION
 → DRAIN_OR_ABORT_INFLIGHT
 → LOAD(v+1)
 → VERIFY_ALL_WORKERS
 → INVALIDATE_OR_REBUILD_WEIGHT_DEPENDENT_CACHES
 → PUBLISH(v+1)
 → ACTIVE(v+1)
```

规则是“所有必要 worker 成功后再发布”，而不是收到第一个 ACK 就放行。传输中途失败时保持停止接单，恢复一致版本后再服务；不能让部分层新、部分层旧的模型继续生成。

KV cache 来自模型计算。更新权重后，旧 K/V 通常不再代表新模型对相同前缀的状态。不能只改 version 字段却继续复用旧 cache；prefix cache 也应按权重 epoch 隔离或清空。

vLLM 暂停接口提供 wait、abort、keep 等模式，并有 cache 清理选项；具体行为应以锁定版本文档和实际测试为准。[vLLM Pause and Resume](https://docs.vllm.ai/en/latest/training/async_rl/#the-pause-and-resume-api)

本文不默认采用 keep 后跨版本继续生成。若研究 partial rollout，需记录分段或逐 token 行为身份，并处理前缀重算与正确行为概率；它不再满足单回答单版本基线。

---

## 10. 队列背压与组完整性

生产速度持续大于消费速度时，无界队列只会把当前速度差转化为未来等待、内存占用与策略陈旧。

对稳定系统，Little 定律连接平均在系统内数量、吞吐率和平均停留时间：

```math
\bar Q=\lambda\bar W
```

这里三者必须使用相同系统边界与单位。若 arrival rate 长期大于 service rate，没有稳定平均队列长度，不能套公式宣称系统已稳定。

项目基线按完整 prompt group 入队、出队，并同时设置组数与 token 数预算。GRPO 的组内统计要求明确哪些候选共同构成一组。

满队列时可选择暂停生产、拒绝新请求，或按预定规则淘汰完整组。不能只挑先完成的短回答消费，否则数据会偏向低延迟、短输出。

评分失败与真实任务失败分开存。组不完整时先重试或按协议丢弃整组，不随手缩小 G、混入别题回答，或把请求超时记成内容错误 reward=0。

---

## 11. 纯 Python：概率与入队验收器

下面代码是单进程合成检查器，不模拟 RPC、安全隔离、真实 tokenizer、tensor 梯度或持久事务。使用固定长度为 2 的组，验证“协议拒绝了什么”。

```python
import math

def probabilities(logits, temperature=1.0):
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("valid logits and temperature required")
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("finite logits required")
    scaled = [x / temperature for x in logits]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = math.fsum(exps)
    return [x / total for x in exps]

def corrected_expectation(target, behavior, values):
    if not target or len(target) != len(behavior) or len(target) != len(values):
        raise ValueError("aligned distributions required")
    for dist in (target, behavior):
        if any(not math.isfinite(p) or p < 0 for p in dist):
            raise ValueError("invalid probabilities")
        if not math.isclose(math.fsum(dist), 1.0):
            raise ValueError("probabilities must sum to one")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("finite values required")
    if any(p > 0 and q == 0 for p, q in zip(target, behavior)):
        raise ValueError("target support is not covered by behavior")
    return math.fsum(q * (p / q) * v
                     for p, q, v in zip(target, behavior, values) if q > 0)

def accept_group(rows, current_version, max_lag, seen,
                 expected_size=2, token_budget=8):
    # Pure validator: no mutation of seen, so rejection cannot half-consume a group.
    if type(current_version) is not int or type(max_lag) is not int:
        raise ValueError("integer versions required")
    if current_version < 0 or max_lag < 0:
        raise ValueError("nonnegative versions required")
    if expected_size < 2 or len(rows) != expected_size:
        raise ValueError("incomplete group")
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids) or any(i in seen for i in ids):
        raise ValueError("duplicate sample")
    for field in ("group_id", "prompt_id", "behavior_version", "weights_digest",
                  "sampling_config_digest", "reward_revision"):
        if len({r[field] for r in rows}) != 1:
            raise ValueError("mixed group identity")
    total = 0
    for row in rows:
        version = row["behavior_version"]
        if type(version) is not int or not 0 <= version <= current_version:
            raise ValueError("invalid behavior version")
        if current_version - version > max_lag:
            raise ValueError("stale group")
        if row["reward_status"] != "valid" or not math.isfinite(row["reward"]):
            raise ValueError("invalid reward result")
        ids_, logps, mask = row["tokens"], row["logps"], row["mask"]
        if not ids_ or not len(ids_) == len(logps) == len(mask):
            raise ValueError("unaligned token fields")
        if any(type(t) is not int or t < 0 for t in ids_):
            raise ValueError("invalid token ID")
        if any(m not in (0, 1) for m in mask) or not any(mask):
            raise ValueError("invalid action mask")
        valid_count = sum(mask)
        # This baseline only accepts completion tokens followed by optional padding.
        if mask != [1] * valid_count + [0] * (len(mask) - valid_count):
            raise ValueError("noncontiguous completion mask")
        if row["finish_reason"] not in ("eos", "length"):
            raise ValueError("unhandled finish reason")
        if any(not math.isfinite(lp) or lp > 1e-6
               for lp, m in zip(logps, mask) if m):
            raise ValueError("invalid selected-action logp")
        total += valid_count
    if total > token_budget:
        raise ValueError("group exceeds token budget")
    return tuple(ids), total

def pipeline_seconds(batches, generation, training):
    if type(batches) is not int or batches < 1:
        raise ValueError("positive batch count required")
    if any(not math.isfinite(t) or t <= 0 for t in (generation, training)):
        raise ValueError("positive stage times required")
    serial = batches * (generation + training)
    pipeline = generation + training + (batches - 1) * max(generation, training)
    return serial, pipeline
```

该检查器只验证记录内部一致性，不读取工件仓库核实 hash 真假，也不验证 EOS ID。真实系统还要把版本映射到不可变 manifest，并核对 tokenizer、模板、EOS 协议及采样引擎返回语义。

---

## 12. 可运行的合成测试

将本节接在上一节后运行：

```python
from copy import deepcopy

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

p = [0.8, 0.2]
mu = probabilities([math.log(x) for x in p], temperature=2.0)
close(mu[0], 2 / 3)
close(mu[1], 1 / 3)
close(corrected_expectation(p, mu, [1, 0]), 0.8)
close(sum(q * v for q, v in zip(mu, [1, 0])), 2 / 3)
rejects(lambda: corrected_expectation(p, [1, 0], [1, 0]))

# The two factors multiply correctly before clipping.
current, old_train, behavior = 0.5, 0.4, 0.3
close(current / behavior, (current / old_train) * (old_train / behavior))
clip = lambda w: min(max(w, 0.8), 1.2)
assert not math.isclose(
    clip(current / old_train) * (old_train / behavior),
    clip(current / behavior),
)

row = dict(
    sample_id="a", group_id="g1", prompt_id="p1", behavior_version=7,
    weights_digest="weights-v7", sampling_config_digest="t1-no-filter",
    reward_revision="reward-v1", reward_status="valid", reward=1.0,
    tokens=[21, 2], logps=[-0.4, -0.2], mask=[1, 1], finish_reason="eos",
)
rows = [row, {**deepcopy(row), "sample_id": "b", "reward": 0.0}]
seen = set()
accepted, count = accept_group(rows, 8, 1, seen)
assert accepted == ("a", "b") and count == 4 and seen == set()
seen.update(accepted)  # In production, commit this atomically with consumption.
rejects(lambda: accept_group(rows, 8, 1, seen))
rejects(lambda: accept_group(rows, 9, 1, set()))
rejects(lambda: accept_group(rows, 6, 1, set()))
rejects(lambda: accept_group(rows[:1], 8, 1, set()))
rejects(lambda: accept_group(rows, 8, 1, set(), token_budget=3))

def modified(field, value):
    changed = deepcopy(rows)
    changed[1][field] = value
    return changed

for field, value in [
    ("sample_id", "a"), ("group_id", "g2"), ("prompt_id", "p2"),
    ("behavior_version", 8), ("weights_digest", "different"),
    ("sampling_config_digest", "temperature-2"),
    ("reward_revision", "reward-v2"), ("reward_status", "timeout"),
    ("reward", float("nan")), ("mask", [0, 0]), ("mask", [0, 1]),
    ("logps", [-0.4]), ("logps", [-0.4, float("nan")]),
    ("finish_reason", "unknown"),
]:
    rejects(lambda f=field, v=value: accept_group(modified(f, v), 8, 1, set()))

padded = deepcopy(rows)
for r in padded:
    r["tokens"] += [2]
    r["logps"] += [float("nan")]
    r["mask"] += [0]
assert accept_group(padded, 8, 1, set())[1] == 4
# Masked padding may be ignored; a nonfinite active action must fail.

assert pipeline_seconds(4, 6, 4) == (40, 28)
assert pipeline_seconds(1, 6, 4) == (10, 10)
print("PASS: sampling distribution, support, ratio, group identity, lag, mask, pipeline")
```

这些测试证明我们构造的检查器能捕获指定错误，不证明实际训练后端已遵循协议。特别是 scalar logp 不能自己携带“确实来自 v7”的证据，必须与真实请求和工件记录联合检查。

---

## 13. 重试、恢复与“只消费一次”

真实系统经常提供至少一次消息投递：worker 发送成功但没收到 ACK 时会重试。训练侧需要幂等消费，不能靠“网络通常不会重复”保证正确。

建议按完整组处理：

1. 验证 group 身份、版本、候选数和有效性。
2. 记录 reservation，防止其他消费者同时领取。
3. 读取固定 rollout 和固定 advantage 进入预定更新流程。
4. 在 checkpoint 中保存已完成更新位置与消费记录。
5. 恢复时根据同一 checkpoint 恢复模型、优化器、RNG、队列和游标。

数据库里的“已消费”标记与 GPU optimizer step 不是天然原子事务。先标记再崩溃会漏样本，先训练再标记会可能重复训练。项目必须声明恢复边界；教学基线可从采集轮次边界恢复，避免承诺任意时刻的精确续训。

异步完成顺序可能受硬件时序影响，即使种子相同，也不保证整个训练完全重复。应区分逐位复现、样本顺序复现和统计可复现。

---

## 14. 验收指标：看有用吞吐，不只看生成速度

至少同时记录：

| 指标 | 用途 |
|---|---|
| generated / accepted / consumed action tokens | 区分生产量、通过验收量与训练量 |
| 完整组比例、重复率、评分失败率 | 检查数据损耗 |
| version lag 与队列等待分位数 | 检查陈旧数据长尾 |
| sampled/current log-ratio 分位数 | 检查实际概率变化 |
| 同权重跨引擎 logp 差异 | 隔离引擎计算偏差 |
| 权重同步与 cache 重建耗时 | 定位更新暂停成本 |
| EOS 率、截断率、生成长度分布 | 识别长度或终止退化 |
| 独立质量指标与达到目标质量耗时 | 判断优化是否真的有用 |

有效吞吐可定义为“成功用于训练的有效动作 token 数 / 总 wall-clock 时间”，但这个数还要和训练质量一起看。接受更多陈旧样本可能提高该吞吐，却增加达到相同质量所需的总训练量。

推荐后续小实验只改变一个变量：固定模型、数据和总设备预算，对比同步与有界流水线。保留原始样本 ID、队列事件和最终评估协议，不能只保留平均 token/s。

---

## 15. 今日练习与遗留问题

### 本次完成的合成验证

- temperature=2 把两动作分布从 `[0.8, 0.2]` 变成 `[2/3, 1/3]`。
- 重要性加权在支持覆盖时恢复枚举期望；支持缺失时拒绝。
- 概率比可在裁剪前分解，但分段裁剪与乘积裁剪不同。
- 完整组验收拒绝混合版本、混题、重复、评分失败与过期样本。
- 无效 padding 不增加有效动作计数；有效位置的 NaN 被拒绝。
- 四批次理想流水时间为 28 秒，对照串行为 40 秒。

### 待运行的真实实验

1. 用相同权重和同一 token 前缀对比训练/推理 selected-token logp。
2. 给模型工件增加版本 manifest，确认所有 worker ACK 后才接单。
3. 在同步期间注入 worker 失败，确认不会发布半更新模型。
4. 验证权重更新后 prefix/KV cache 的清理或重算策略。
5. 记录队列长尾与版本 lag，而不只看均值。
6. 比较相同质量目标下的端到端成本。

遗留问题：允许多大滞后才不伤质量，哪一层采样 processor 改变了行为分布，以及多 GPU 下数值差异是否足以影响训练稳定性。这些依赖具体模型与实现，需要实验回答。

---

## 参考资料

- [vLLM：Reinforcement Learning from Human Feedback](https://docs.vllm.ai/en/latest/training/rlhf/)
- [vLLM：Weight Transfer](https://docs.vllm.ai/en/latest/training/weight_transfer/)
- [vLLM：Async Reinforcement Learning](https://docs.vllm.ai/en/latest/training/async_rl/)
- [verl：Fully Async Policy Trainer](https://verl.readthedocs.io/en/latest/advance/fully_async.html)
- [TRL：GRPOTrainer，训练与推理差异](https://huggingface.co/docs/trl/grpo_trainer#dealing-with-the-training-inference-mismatch)

核对日期：2026-09-10。这些是滚动文档；本篇没有声称完成特定版本的框架集成。真实实验需锁定 vLLM、训练框架、模型、tokenizer 和 verifier revision。

## 今日总结

Rollout 工程的核心不是单独让生成更快，而是让每个被训练的动作都能回答：由哪个版本、哪个采样分布生成，何时评分，何时消费，发生过多少策略漂移。

今天建立了单回答单版本、完整组验收、有界队列和可追溯 logp 的教学基线。先保证数据身份与恢复边界，再评估异步重叠是否带来有效收益。

## 下次衔接建议

继续学习可验证奖励与评估器工程：答案抽取、单元测试沙箱、reward 分项、超时与重试，以及怎样通过反例测试防止 reward hacking。

