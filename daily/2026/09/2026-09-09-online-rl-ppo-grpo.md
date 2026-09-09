# 2026-09-09：在线强化学习后训练——Rollout、Advantage、PPO 与 GRPO

## 今日目标

上一篇 [DPO：偏好对、Reference Model 与长度偏差](2026-09-08-direct-preference-optimization.md)研究了固定离线候选的相对偏好。今天转向在线后训练：让策略生成新回答，获取奖励，再用这些回答更新策略。

完成后应能回答：

1. policy、old policy、reference、reward model 和 critic 分别负责什么？
2. 为什么 reward 不等于 advantage？
3. PPO 的 ratio 与 clipping 到底约束哪一种变化？
4. GRPO 怎样用同题多回答替代 learned value baseline？
5. 为什么组内全对、全错、长回答和截断都需要单独处理？
6. 怎样用最小数值测试检查损失方向，而不是只看训练 loss？

本篇采用单轮、纯文本、有限长度 completion 的教学设定。代码只运行合成数值计算，不下载模型、不调用评审服务、不执行生成代码，也不宣称完成真实 PPO/GRPO 训练。框架版本和变体很多，本篇先固定数学协议，不提供未经实例化验证的 Trainer 参数拼装。

---

## 1. 从离线偏好对到在线 Rollout

“在线”指训练数据中的回答会随策略更新重新采样，不是必须连接互联网。

```text
prompt batch
    ↓ 用本轮 old policy 生成
completion + token IDs + old log probabilities + stop reason
    ↓ 固定版本的 reward / verifier
reward + validity
    ↓ 计算并冻结
advantage / value target
    ↓ 当前 policy 重算 log probabilities
有限次梯度更新
    ↓ 刷新生成策略，采集下一轮 rollout
```

离线 DPO 可以重复读取相同 chosen/rejected。在线循环则同时改变回答分布，因此数据状态、生成版本和奖励器状态都成为训练可复现性的一部分。

本篇只处理模型自己生成的 completion token。若扩展到工具调用，工具返回文本是环境观测，不应自动当作策略动作计入 policy loss。

---

## 2. 五个角色不要混在一起

| 角色 | 输入与输出 | 本篇更新规则 |
|---|---|---|
| policy | 前缀 → 下一 token 分布 | 梯度更新 |
| old policy | 前缀 → 本轮采样分布 | 同一 rollout 批次更新期间固定 |
| reference | 前缀 → 基准 token 分布 | 整个教学实验固定 |
| reward / verifier | 问题、回答 → 分数与有效性 | 固定版本；不参与 policy 反向传播 |
| critic / value | 回答前缀 → 预计未来回报 | PPO 路径训练；基础 outcome GRPO 不使用 |

old policy 回答“这批数据由谁采集”；reference 回答“希望不要离哪个行为基准太远”。初始化时二者可能权重相同，但刷新频率和用途不同。

工程上 old policy 不一定始终占据一份完整模型显存：可在采样后缓存所选动作的 old log probability，再更新 policy。前提是缓存确实对应采样权重、分词、mask 和采样分布，且更新时不能覆盖。

reference、reward、critic 是否共享 backbone 或使用 adapter 是实现选择；不能由“五个角色”直接推导“五份完整模型”。

---

## 3. 把语言生成看成一段轨迹

状态是 prompt 加已生成前缀；动作是下一 token。这里用 R 表示 reward，用 rho 表示概率比，避免常见记号 r 同时表示两种量。

```math
s_t=(x,y_{\lt t}),\qquad a_t=y_t,\qquad
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid s_t)
```

最终答案验证可以给一个 terminal reward；过程监督则可以在中间步骤给分。两者的信用分配不同，不能只换 reward 函数名字就视作等价。

奖励例子是“数值答案是否正确”，不是“看起来有推理”。若 verifier 只能抽取最终答案，它并没有证明中间推导正确。奖励应同时保存原始分项与解析状态，防止格式奖励掩盖任务失败。

---

## 4. Reward 与 Advantage 的区别

reward 评价轨迹或动作的结果；advantage 评价“相对这个状态下的通常水平好多少”。

```math
A^\pi(s_t,a_t)=Q^\pi(s_t,a_t)-V^\pi(s_t)
```

同样获得 0 分，在所有候选都失败的问题上不一定产生相对差异；在其他候选都成功的问题上则可能明显低于基准。

下面两种方法主要在优势估计方式上分叉：

- PPO 常用 critic 预测状态价值，再构造时序优势。
- outcome GRPO 用同一 prompt 的一组回答奖励做相对比较。

优势进入 actor loss 时视作固定训练信号。若把 critic 或 reward 的计算图意外连入 actor loss，会改变预期优化问题。

---

## 5. PPO 的 GAE：从 TD Residual 向前分配信用

对真实终止轨迹，令最后一个动作之后的 value 为 0。定义：

```math
\delta_t=R_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\qquad
\widehat A_t=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1}
```

d 为真实终止标记，gamma 是折扣，lambda 调节多步残差的组合。GAE 用价值估计降低方差，同时引入与近似价值及 lambda 有关的偏差；不是天然准确的 reward 分解。[GAE 原论文](https://arxiv.org/pdf/1506.02438)

教学例子：两步 reward 为 `[0, 1]`，采样时 value 为 `[0.2, 0.4]`，终止 bootstrap 为 0，gamma=1、lambda=0.95。

```text
最后一步：delta = 1 - 0.4 = 0.6
第一步：delta = 0 + 0.4 - 0.2 = 0.2
advantage = [0.2 + 0.95 × 0.6, 0.6] = [0.77, 0.6]
value target = advantage + old value = [0.97, 1.0]
```

达到 token 上限是截断，不一定是真实终止。若问题定义仍允许继续，应显式选择 bootstrap 或过滤协议；不能无声地把所有长度截断都当作 value=0 的环境终止。本篇 GAE 代码只支持完整终止轨迹，主动避开该歧义。

---

## 6. PPO 的 Ratio 使用 Old Policy，不是 Reference

对缓存轨迹中实际采样的 token：

```math
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}
=\exp\!\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right)
```

这是 token 级概率比；不要先求整个回答的 log probability sum 再指数化，后者是不同的 sequence-level ratio。

第一次更新前，在同一权重和相同计算协议下，ratio 应接近 1。更新后 old log probability 保持不变；若每个 minibatch 都用新 policy 覆盖分母，ratio 会不断被重置，失去衡量本轮策略漂移的作用。

本篇约定 temperature=1、无 top-k/top-p 截断、相同 allowed-token 规则，便于把采样分布与打分分布对齐。真实训练若改变温度、过滤词表或使用不同推理引擎，需要声明行为分布及相应修正；“checkpoint 相同”本身不足以证明 on-policy。

---

## 7. Clipped Surrogate：保守的是目标，不是硬约束

PPO 最大化 clipped objective；交给最小化优化器时再加负号：

```math
J_{\mathrm{clip}}
=\mathbb{E}_t\!
\left[
\min\!\left(
\rho_t\widehat A_t,\,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right],
\qquad
\mathcal L_{\mathrm{actor}}=-J_{\mathrm{clip}}
```

正优势时，过度提高该动作概率不再得到额外收益；负优势时，过度降低概率不再得到额外收益。相反方向的恶化仍保留惩罚。[PPO 原论文，式 7](https://arxiv.org/pdf/1707.06347)

取 epsilon=0.2、优势绝对值为 2，逐项手算：

| advantage | ratio | 原始项 | clipped 项 | 取 min 后 |
|---|---:|---:|---:|---:|
| +2 | 1.5 | 3.0 | 2.4 | 2.4 |
| +2 | 0.5 | 1.0 | 1.6 | 1.0 |
| -2 | 0.5 | -1.0 | -1.6 | -1.6 |
| -2 | 1.5 | -3.0 | -2.4 | -3.0 |

因此不能实现成“所有 ratio 都 clamp 后乘 advantage”。最后一列才是要最大化的项。

裁剪没有把模型参数投影到信赖域，也不保证所有 token ratio 或实际 KL 留在阈值内；共享参数、其他样本和优化器动量都可能继续改变某个动作概率。

---

## 8. PPO 的完整训练还不止 Actor Loss

常见 actor-critic 训练还包含 value loss，可选 entropy bonus；LLM 后训练可能再引入对固定 reference 的 KL 惩罚。本文不把这些不同项统称为“PPO loss”。

一个清楚的实验清单应分别注明：

- actor surrogate 的 token mask、裁剪区间和归约方式；
- value target、value loss 权重与是否 value clipping；
- entropy 的定义、系数和有效位置；
- reference KL 加在 reward shaping 还是直接加在 loss；
- 每批 rollout 的更新次数与提前停止条件。

若 KL 已进入逐 token reward 并影响优势，又无说明地在 loss 加同样 KL，可能重复正则。要从计算图追踪一次完整数据路径。

---

## 9. Outcome GRPO：同题多回答提供相对基准

对每个 prompt 采样 G 个回答，使用各自最终 reward 形成组内优势。教学基线明确采用总体标准差：

```math
\bar R=\frac1G\sum_{i=1}^{G}R_i,\qquad
s=\sqrt{\frac1G\sum_{i=1}^{G}(R_i-\bar R)^2},
\qquad
\widehat A_i=\frac{R_i-\bar R}{s+\eta}
```

同一回答的所有有效 completion token 共享该回答优势。基础 outcome GRPO 不训练 critic，而保留按 token 的 policy ratio 与 clipped surrogate。[DeepSeekMath，4.1 节](https://arxiv.org/html/2402.03300v2#S4.SS1)

这里的总体标准差、稳定项 eta 和零方差处理是本文显式选择，不声称所有框架完全一致。换成样本标准差会改变优势尺度，小 G 时尤其明显。

把同一回答优势广播到 token 不表示“每个 token 都被验证正确”。这只是粗粒度信用分配：正确结论前面的冗余甚至错误步骤也可能收到正信号。

---

## 10. 全对、全错和 G=1

构造一组 reward `[0, 0, 1, 1]`：均值为 0.5，总体标准差为 0.5，忽略极小稳定项时优势为 `[-1, -1, 1, 1]`。

若 reward 是 `[1, 1, 1, 1]` 或 `[0, 0, 0, 0]`，组内没有相对差异。本篇返回全零优势，并记录零方差组比例；不靠微小随机噪声强行制造排序。

零优势意味着该组 reward 驱动的 actor 项没有梯度，不代表 KL、entropy 等其他项也一定为零。全对和全错都可能是零方差，但教学意义完全不同，必须分别统计。

G=1 不能提供这种组内比较，本篇直接拒绝。增大 G 提高采样成本，也不保证得到有差异的正确性标签。若过滤零方差组或动态补采，会改变数据权重和预算，应写入协议。

---

## 11. 两种归约不能混用

令 u 为某个有效 token 的 clipped surrogate 减 KL 惩罚项。回答平均再组平均与全 token 平均分别是：

```math
J_{\mathrm{seq}}
=\frac1G\sum_i\frac1{T_i}\sum_t u_{i,t},
\qquad
J_{\mathrm{token}}
=\frac{\sum_i\sum_t u_{i,t}}{\sum_i T_i}
```

本文 GRPO 数值函数使用前者，复现“每个回答先平均”的教学基线，而非声称它是当前框架推荐默认值。TRL 文档区分多种 loss reduction，并讨论长度归一化和组内标准差引入的偏差。[TRL GRPO：Loss Types](https://huggingface.co/docs/trl/grpo_trainer#loss-types)

自行构造两个回答：第一条有 2 个有效 token，每项为 1；第二条有 6 个有效 token，每项为 -1。

```text
先回答平均：(1 + (-1)) / 2 = 0
全 token 平均：(2 - 6) / 8 = -0.5
```

这不是浮点误差，而是不同的目标权重。分布式训练及梯度累积时，“每个 microbatch 各自平均后再平均”也未必等于全局有效 token 平均。要同时保存分子与计数，再按选定协议归约。

---

## 12. KL：区分 Reference 漂移与单轮策略漂移

old-policy ratio 衡量本轮更新；reference KL 衡量相对固定基准的偏离。这两个诊断不能用一个曲线代替。

在给定状态，完整词表上的 policy-to-reference KL 为：

```math
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=\sum_a\pi_\theta(a\mid s)
\log\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}
```

GRPO 常用所选动作上的非负形式近似。令 d 为 reference logp 减 policy logp：

```math
d=\log\pi_{\mathrm{ref}}(a\mid s)-\log\pi_\theta(a\mid s),
\qquad k_3=\exp(d)-d-1
```

DeepSeekMath 使用这一形式；注意数学上应称非负，分布相同时为零。[DeepSeekMath，式 4](https://arxiv.org/html/2402.03300v2#S4.SS1)

自行验证其期望：当动作确实来自当前 policy、两分布具有适当的共同支持时，ratio 的期望为 1，故 k3 的期望等于上述 KL。来自 old policy 的旧样本在多次更新后不自动满足这一前提；样本值非负也不证明其均值是当前 KL 的无偏估计。

此外，“数值期望正确”不等于“对固定样本直接反传就是精确 KL 梯度”。采样分布本身也依赖 policy。本文代码只验证估计量的数值恒等式，不声称实现无偏 KL 梯度估计器。

---

## 13. Mask、EOS 与完成边界

继续沿用昨天的 causal shift：第一个 completion token 由最后一个 prompt 位置的 logit 预测。缓存 old/ref logp 与 policy 重算必须采用相同 target 坐标。

```text
input:       p0 p1 y0 y1 EOS PAD
action mask:  0  0  1  1   1   0
```

本篇规定：

- 只对模型动作的 completion token 计分。
- 真实 EOS 保留；PAD 即使和 EOS 同 ID，也通过位置 mask 排除。
- EOS 后的填充或无效生成位置不参与 reward-to-token 广播及 loss。
- 空 completion 或零有效动作不返回一个“看起来正常”的 0 loss，而是报错。
- 超长截断单独记录 stop reason；不能事后凭空补 EOS，再假装生成自然结束。
- 不能通过重新 tokenize 解码后的字符串恢复动作，必须保留实际采样 token IDs。

组中丢失或过滤某个回答后，原有均值与标准差是否重算是算法协议的一部分。失败的评分请求不应被静默当成真实 reward=0。

---

## 14. 纯 Python 数值内核

下面只操作已经对齐的标量 log probability，不执行模型前向。有限范围检查用于教学 fail-fast；遇到极端 ratio 应先定位数据或策略漂移，不静默 clamp log-ratio 来掩盖异常。

```python
import math

def require_finite(values):
    if not all(math.isfinite(x) for x in values):
        raise ValueError("finite values required")

def group_advantages(rewards, eta=1e-8):
    if len(rewards) < 2:
        raise ValueError("group size must be at least two")
    require_finite([*rewards, eta])
    if eta <= 0:
        raise ValueError("eta must be positive")
    mean = math.fsum(rewards) / len(rewards)
    variance = math.fsum((r - mean) ** 2 for r in rewards) / len(rewards)
    if variance == 0:
        return [0.0] * len(rewards)
    scale = math.sqrt(variance) + eta
    return [(r - mean) / scale for r in rewards]

def clipped_term(logp, old_logp, advantage, epsilon=0.2):
    require_finite([logp, old_logp, advantage, epsilon])
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0, 1)")
    delta = logp - old_logp
    if abs(delta) > 50:
        raise ValueError("extreme log ratio; inspect rollout consistency")
    ratio = math.exp(delta)
    clipped = min(max(ratio, 1 - epsilon), 1 + epsilon)
    return min(ratio * advantage, clipped * advantage)

def sampled_k3(logp, ref_logp):
    require_finite([logp, ref_logp])
    delta = ref_logp - logp
    if abs(delta) > 50:
        raise ValueError("extreme reference log ratio")
    # expm1 improves precision near zero.
    return math.expm1(delta) - delta

def sequence_mean_loss(logps, old_logps, ref_logps, mask,
                       advantage, beta=0.0, epsilon=0.2):
    n = len(logps)
    if n == 0 or any(len(v) != n for v in (old_logps, ref_logps, mask)):
        raise ValueError("aligned non-empty arrays required")
    if any(m not in (0, 1) for m in mask):
        raise ValueError("binary mask required")
    require_finite([advantage, beta, epsilon])
    if beta < 0 or not 0 < epsilon < 1:
        raise ValueError("invalid coefficients")
    terms = []
    for lp, old, ref, valid in zip(logps, old_logps, ref_logps, mask):
        if not valid:
            continue
        term = clipped_term(lp, old, advantage, epsilon)
        if beta:
            term -= beta * sampled_k3(lp, ref)
        terms.append(term)
    if not terms:
        raise ValueError("no valid action tokens")
    return -math.fsum(terms) / len(terms)

def gae_terminal(rewards, old_values, gamma=1.0, lam=0.95):
    # One complete episode only; last transition is a true terminal.
    if not rewards or len(old_values) != len(rewards) + 1:
        raise ValueError("values must include terminal bootstrap")
    require_finite([*rewards, *old_values, gamma, lam])
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lambda must be in [0, 1]")
    if old_values[-1] != 0:
        raise ValueError("true terminal bootstrap must be zero")
    advantages = [0.0] * len(rewards)
    tail = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * old_values[t + 1] - old_values[t]
        tail = delta + gamma * lam * tail
        advantages[t] = tail
    targets = [a + v for a, v in zip(advantages, old_values[:-1])]
    return advantages, targets
```

为了检查 padding 隔离，函数只读取有效位置；真实模型中若有效 token 的 logits 已经是 NaN，必须立即停止。标量 oracle 无法检查 tensor 的 detach、跨设备通信或模型 dropout，这些仍需集成测试。

---

## 15. 可运行的方向与边界测试

把本节接在上一节代码之后运行：

```python
def close(a, b, tol=1e-7):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)

def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

# Group normalization: ties, order, and common reward shift.
a = group_advantages([0, 0, 1, 1])
for got, expected in zip(a, [-1, -1, 1, 1]):
    close(got, expected)
close(sum(a), 0)
assert group_advantages([1, 1, 1, 1]) == [0.0] * 4
assert group_advantages([0, 0, 0, 0]) == [0.0] * 4
assert group_advantages([1, 0, 1, 0]) == [a[2], a[0], a[3], a[1]]
for x, y in zip(a, group_advantages([7, 7, 8, 8])):
    close(x, y)
rejects(lambda: group_advantages([1]))
rejects(lambda: group_advantages([0, float("nan")]))

# Four sign-sensitive PPO cases; old p=0.2 gives valid probabilities.
old = math.log(0.2)
for adv, ratio, expected in [
    (2, 1.5, 2.4), (2, 0.5, 1.0),
    (-2, 0.5, -1.6), (-2, 1.5, -3.0),
]:
    close(clipped_term(old + math.log(ratio), old, adv), expected)
close(clipped_term(old, old, 2), 2)

# Finite-difference slope of the minimized actor loss in logp coordinates.
def derivative(ratio, adv):
    lp = old + math.log(ratio)
    h = 1e-6
    loss = lambda z: -clipped_term(z, old, adv)
    return (loss(lp + h) - loss(lp - h)) / (2 * h)

close(derivative(1.0, 2), -2)
close(derivative(1.0, -2), 2)
close(derivative(1.5, 2), 0)
close(derivative(0.5, -2), 0)
close(derivative(0.5, 2), -1)
close(derivative(1.5, -2), 3)

# Padding does not change a sequence mean; real EOS remains an action.
base = sequence_mean_loss([-1, -2], [-1, -2], [-1, -2], [1, 1], 1)
padded = sequence_mean_loss(
    [-1, -2, float("nan")], [-1, -2, 999], [-1, -2, 999], [1, 1, 0], 1
)
close(base, -1)
close(base, padded)
rejects(lambda: sequence_mean_loss([-1], [-1], [-1], [0], 1))
rejects(lambda: sequence_mean_loss([-1], [-1], [-1], [2], 1))
rejects(lambda: clipped_term(0, -100, 1))

# Sequence mean and global-token mean are deliberately different.
sequence_average = (1 + (-1)) / 2
token_average = (2 * 1 + 6 * (-1)) / 8
close(sequence_average, 0)
close(token_average, -0.5)

# Exact enumeration checks K3 expectation under the current policy.
p, q = [0.8, 0.2], [0.5, 0.5]
exact_kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
k3_values = [sampled_k3(math.log(pi), math.log(qi)) for pi, qi in zip(p, q)]
close(sum(pi * k for pi, k in zip(p, k3_values)), exact_kl)
assert all(k >= 0 for k in k3_values)
close(sampled_k3(-1, -1), 0)
# Old-policy sampling weights need not give current-policy KL.
old_p = [0.5, 0.5]
assert not math.isclose(sum(w * k for w, k in zip(old_p, k3_values)), exact_kl)

adv, targets = gae_terminal([0, 1], [0.2, 0.4, 0])
for got, expected in zip(adv, [0.77, 0.6]):
    close(got, expected)
for got, expected in zip(targets, [0.97, 1.0]):
    close(got, expected)
rejects(lambda: gae_terminal([1], [0.2, 0.3]))
print("PASS: group advantage, PPO clipping/gradient, mask, reduction, KL, GAE")
```

这些是人工构造数据上的函数测试。finite difference 只验证独立 logp 坐标上的方向，不证明真实神经网络每个 token 的概率都按预期单独变化。真实 EOS 是否正确标记，需要另做 tokenizer golden fixture；本测试仅验证已给定 mask 的归约行为。

---

## 16. 从 Oracle 到真实训练：先检查计算图

下面是协议伪代码，不是可直接执行的框架 API：

```text
freeze reference revision
repeat:
    sync sampler to a recorded policy revision
    collect G completions per prompt
    preserve actual token IDs, masks, stop reasons, sampling logp
    score with pinned verifier; resolve failures explicitly
    compute old logp and advantages with no gradient
    optionally compute PPO old values and fixed value targets
    for a bounded number of update steps:
        recompute current policy logp WITH gradient
        keep old logp, advantage and reference weights detached
        compute the declared masked loss and reduction
        inspect finite loss / gradient / ratio / drift
        optimizer step; stop early if the configured drift rule triggers
    evaluate or collect a new rollout batch
```

真实模型验收建议从一个 prompt、两个候选和一次 optimizer step 开始：

1. 打印实际 token IDs 与 action mask，核对第一个回答 token 和 EOS。
2. 更新前 policy/old ratio 接近 1；policy/reference 若同权重则 sampled KL 接近 0。
3. actor loss 反传后 policy 获得有限梯度，reference 与 old 缓存不变。
4. 优化一步后重新计算 ratio；不能把新分数写回旧缓存。
5. 故意交换组 ID、扰动 padding 和改变生成 revision，让验证器确实失败。
6. 再扩到多 GPU，比较拆 batch 前后分子、有效计数和最终梯度。

不要仅凭 scalar loss 接近 0 判断模型没学到：组内优势均值本来就接近 0，而不同回答的 logp 梯度一般不会相互抵消。

---

## 17. 奖励投机与评估隔离

奖励越容易被优化，就越需要与实际任务成功分开验证。为后续实验预先设计反例：

| 奖励规则 | 可被利用的漏洞 | 验证方法 |
|---|---|---|
| 最终数字匹配 | 同时输出多个答案、解析歧义 | 固定抽取规范；歧义显式失败 |
| 格式标签齐全 | 只学模板，不解决问题 | 单独记录内容正确性 |
| 单元测试通过 | 硬编码公开测试 | 隐藏测试与跨题去重 |
| judge 偏好 | 冗长、迎合、顺序偏差 | 长度分层、交换展示顺序、人工抽查 |
| 执行成功 | 超时、资源滥用、外部副作用 | 隔离执行、资源预算、无外网与凭据 |

生成代码应视为不可信输入；本篇不会执行它。后续代码奖励实验必须先准备隔离沙箱和明确的资源限制。

reward、dev 评估、最终 test 不应共用一组已反复调参的问题。沿用 [Checkpoint 评估协议](2026-09-04-checkpoint-evaluation-harness.md)：开发集选 checkpoint，冻结后才使用保留测试集。

---

## 18. 一组可诊断的训练日志

至少同时保留：

- 原始 reward 各分项、解析失败率、全对/全错/零方差组比例。
- prompt 数、每题实际候选数、有效动作数、生成长度与自然终止率。
- old-policy ratio 的分位数与有效裁剪比例。
- 相对 reference 的 KL；明确精确还是 sampled、采样来源和归约口径。
- advantage 的均值、标准差与极值。
- PPO 路径的 value loss、value target 统计和预测误差。
- 独立任务成功率、保留能力、token 成本和 wall-clock 时间。

“ratio 越界比例”和“clipped 分支真正主导目标的比例”不是同一统计。正优势且 ratio 太小虽然越界，min 仍选择原始项。日志定义必须写出计算条件。

若 reward 上升但独立成功率下降，先检查评分漏洞、污染和长度捷径。若 ratio 一开始就远离 1，先检查生成权重同步、temperature、token 对齐、dropout、精度与缓存，不要先增大 clipping 范围。

---

## 19. 保存 Rollout 协议，而不只是模型权重

```yaml
objective: teaching-outcome-grpo-sequence-mean
reference_revision: immutable-sft-revision
rollout_policy_revision: actual-weight-revision
prompt_manifest: immutable-split-id
group_size: 4
advantage:
  baseline: per-prompt-group-mean
  std: population
  epsilon: 1.0e-8
  zero_variance: zero-advantage-and-log
loss:
  ratio: token-level-current-over-old
  epsilon: 0.2
  reduction: token-mean-per-response-then-response-mean
  reference_kl: k3-sampled-surrogate
  beta: explicitly-set-for-each-experiment
generation:
  temperature: 1.0
  top_k_top_p_filter: disabled
  token_budget: explicitly-set
  eos_and_truncation: versioned-protocol
reward:
  revision: pinned
  raw_components: stored
  parse_failure: separate-from-task-failure
resume:
  optimizer_scheduler_rng: stored
  rollout_tokens_masks_old_logp_rewards: stored
  group_ids_and_update_cursor: stored
```

以上是项目自定义实验清单，不是 Trainer 可直接加载的配置。数值仅用于教学基线，不代表推荐的真实模型超参数。

若在一批 rollout 的多次更新中途恢复，只加载最新 policy 再重新生成数据，并不等价于续上原实验。可选择保存完整 rollout 状态，或只在采集轮次边界做精确恢复；必须说明保证到哪一层。

---

## 20. 今日练习与遗留问题

### 本次完成的教学验证

- GAE 两步示例得到优势 `[0.77, 0.6]` 与 target `[0.97, 1.0]`。
- PPO 四种 advantage/ratio 组合与手算一致。
- 有限差分验证未饱和方向与饱和区间的标量导数。
- 同题奖励常数平移不改变组内优势；平局组返回零。
- padding 不影响已给定 action mask 的 sequence mean。
- 两种 loss reduction 在长短回答上得到不同结果。
- 枚举两个动作验证 k3 在当前 policy 下的期望；换 old 采样权重后不再相等。
- 单候选组、无有效动作、非法 mask 和极端 ratio 被拒绝。

### 后续真实实验

1. 用小模型复核生成 token 与训练打分一致性。
2. 对同一小批 rollout 做一次 PPO/GRPO 更新，检查 detach 和权重变化。
3. 比较不同 G 的有效非零优势组比例与生成成本。
4. 固定训练 token 预算，比较 sequence mean 与 token mean。
5. 将奖励分数与盲评任务成功率、长度和终止率一起绘图。

尚未解决：reward 的有效覆盖率是多少，稀疏正确性奖励能否提供足够区分，采样引擎和训练前向的差异有多大，以及多少次更新后必须重新采样。这些需要实测，不能从损失公式直接推断。

---

## 参考资料

- [Schulman 等：Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
- [Schulman 等：Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438)
- [Shao 等：DeepSeekMath，GRPO 原始定义](https://arxiv.org/html/2402.03300v2)
- [Hugging Face：GRPOTrainer 文档与 Loss Types](https://huggingface.co/docs/trl/grpo_trainer)

核对日期：2026-09-09。论文版本固定；TRL 链接为滚动文档，只用于理解变体差异。真实实现前锁定包版本与源码 revision，不能把文档默认值当成跨版本协议。

## 今日总结

在线后训练把生成、评分和参数更新接成一个循环。最重要的边界是：old policy 用于本轮采样校正，reference 用于基准约束，reward 与 advantage 分开，token mask 与归约方式共同决定实际目标。

今天得到的是可手算、可测试的算法基线，并非训练效果报告。先检查方向、边界与数据身份，再用独立评估判断 reward 的提高是否转化成真实能力。

## 下次衔接建议

继续学习 RL rollout 的工程实现：训练与推理权重同步、采样 log probability、异步队列与 policy staleness，以及如何在吞吐和 on-policy 一致性之间做可测量的取舍。
