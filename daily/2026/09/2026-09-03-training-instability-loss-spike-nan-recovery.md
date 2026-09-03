# 2026-09-03：训练异常诊断——Loss Spike、梯度范数、NaN/Inf 与 Checkpoint Rollback

## 今日目标

昨天在[训练数据治理](2026-09-02-data-quality-dedup-contamination.md)中建立了从原始来源到训练 shard 的完整 lineage。今天继续向训练环节推进：

> 当 loss 突然升高、梯度爆炸或出现 NaN / Inf 时，怎样定位第一个异常点，并在不破坏分布式一致性的前提下安全恢复？

完成后应能回答：

1. 有限的 loss spike、non-finite loss、non-finite gradient 和 optimizer state 损坏有什么区别？
2. 如何判断异常最早发生在 forward、backward 还是 `optimizer.step()` 之后？
3. 为什么 gradient clipping 既不是 NaN 修复，也不能替代根因分析？
4. AMP 下为什么必须 `scale → backward → unscale → inspect/clip → step → update`？
5. 梯度累积期间某个 microbatch 失败，为什么通常要丢弃整个 accumulation window？
6. DDP / FSDP 中某个 rank 独自 `continue` 为什么可能导致 hang 或参数分叉？
7. checkpoint 至少需要保存哪些状态，才能完成可靠 rollback？
8. 如何设计异常状态机、incident capsule 和 fault-injection tests？

---

## 1. Loss Spike 是症状，不是单一故障

看到 loss 曲线突然升高时，不应立刻得出“学习率太大”或“这个 batch 是脏数据”的结论。

同一个现象可能来自：

- 当前 batch 本来就难；
- 有效 token 数或 loss 归一化方式突然变化；
- 数据、label 或 attention mask 错误；
- 梯度累积缩放错误；
- FP16 overflow；
- attention logits 或激活值持续增大；
- optimizer moments 损坏；
- scheduler 恢复错位；
- 某个 rank 数据或状态不同；
- 自定义 kernel、通信或硬件故障。

更好的第一步是沿三条轴分类：

1. **数值有限吗？** finite spike 还是 NaN / Inf；
2. **首次发生在哪个阶段？** forward、backward、optimizer step 之后；
3. **影响范围是什么？** 单样本、单 batch、单 rank，还是全部 ranks。

---

## 2. 四类异常事件必须分开

### A. Loss 非有限

```text
loss = NaN / +Inf / -Inf
```

异常已在 forward 或 loss reduction 中发生。降低 loss scale 只影响 backward，通常不能修复已经非有限的 forward loss。

### B. Gradient 非有限

loss 仍有限，但 backward 产生 NaN / Inf gradient。AMP 的 `GradScaler` 可以在更新前发现并跳过 optimizer step。

### C. 有限但异常大的 Loss / Grad Norm

所有值仍 finite，只是相对历史基线异常。它可能是困难 batch，也可能是即将发散的早期信号。`GradScaler` 不会因为“很大但有限”自动跳步。

### D. Step 后参数或优化器状态损坏

forward 和 backward 可能都正常，但 `optimizer.step()` 后 parameter、Adam first moment 或 second moment 变成 non-finite。此时更新已经执行，`zero_grad()` 不能撤销，通常需要完整 rollback。

这四类事件需要不同的处置，不能统一写成“检测 NaN 后跳过 batch”。

---

## 3. 把一次 Update 看成事务

一个完整 optimizer update 可以抽象为：

```text
PRECHECK
  ↓
FORWARD
  ↓
LOSS REDUCTION
  ↓
BACKWARD / ACCUMULATION
  ↓
UNSCALE
  ↓
GRAD CHECK + CLIP
  ↓
OPTIMIZER STEP
  ↓
POST-STEP CHECK
  ↓
SCHEDULER / EMA / COUNTERS
  ↓
COMMITTED
```

核心不变量：

> 只有确认成功的参数更新才能进入 COMMITTED，并推进 scheduler、EMA、成功更新计数和 latest-good checkpoint 指针。

与此同时还要分别记录：

- `attempt_step`：尝试过多少次 update；
- `committed_step`：真正完成多少次参数更新；
- `data_cursor`：数据加载已经推进到哪里；
- `seen_tokens`：forward 看过多少 token；
- `committed_loss_tokens`：真正参与成功更新的 token。

一次 AMP overflow 可能推进 RNG 和数据游标，却没有推进参数与 committed step。

---

## 4. 先确保 Loss 的定义可比较

Loss 曲线看起来 spike，不一定是模型突然坏了，也可能是统计口径改变。

需要固定或记录：

- reduction 是 `sum` 还是 `mean`；
- 有效 label token 数；
- PAD、prompt 和 boundary mask；
- sequence length；
- gradient accumulation steps；
- DP world size；
- 每个 rank 的有效 token 数；
- 数据域与任务；
- label smoothing 或额外 loss 权重。

对语言模型，日志 loss 应优先按全局有效 token 计算：

\[
L_{\text{global}}
=
\frac{
\sum_r \text{loss\_sum}_r
}{
\sum_r \text{valid\_token\_count}_r
}
\]

若各 rank 先做本地 mean，再等权平均，token 较少的 rank 会被赋予过高权重。

---

## 5. DDP 下反向 Loss 的 Token 归一化

假设 DDP 默认对 `R` 个 ranks 的 gradients 求平均。为了让最终梯度对应全局 token mean，可让 rank `r` 对本地 token loss sum 使用：

\[
L_r^{\text{backward}}
=
\text{local\_loss\_sum}_r
\cdot
\frac{R}{
\text{global\_valid\_tokens}
}
\]

DDP 平均后：

\[
\frac{1}{R}
\sum_r
\nabla L_r^{\text{backward}}
=
\frac{
\sum_r \nabla \text{local\_loss\_sum}_r
}{
\text{global\_valid\_tokens}
}
\]

梯度累积且每个 microbatch 有效 token 数不同时，也要对整个 accumulation window 的分母保持一致。可选方法包括：

- 提前知道整个窗口的 global valid-token count；
- 对 token loss sum 累积梯度，完整窗口后再按全局 token 数缩放；
- 使用框架提供的 token-aware accumulation。

简单地每个 microbatch 除以固定 `grad_accum_steps`，只在各 microbatch 权重确实应相同或有效 token 数稳定时与全局 token mean 等价。

---

## 6. 数据与 Label 是第一排查对象

常见数据问题：

- token ID 超出词表范围；
- labels 超出类别范围；
- `ignore_index` 配错；
- attention mask 与 labels 错位；
- packing 边界错误；
- position ids 异常；
- 全部 labels 都被 mask；
- 样本为空或极端长；
- input 中已有 NaN / Inf 特征；
- 数据增强产生非法值；
- 某个 source 或 shard 被损坏。

特别危险的情况是：

```text
valid_token_count = 0
loss = loss_sum / valid_token_count
```

若所有 label 都是 `ignore_index`，某些 mean reduction 会产生未定义结果。应在进入模型前显式断言有效 token 数大于 0，并记录 batch provenance。

---

## 7. Mask 必须发生在非法运算之前

错误思路：

```python
ratio = numerator / denominator
ratio = torch.where(valid_mask, ratio, 0.0)
```

如果无效位置已执行 `0 / 0`，NaN 可能已经进入 autograd 图；之后把 forward 输出替换为 0，不保证 backward 安全。

更安全的思路：

```python
safe_denominator = torch.where(
    valid_mask,
    denominator,
    torch.ones_like(denominator),
)
ratio = numerator / safe_denominator
ratio = torch.where(valid_mask, ratio, torch.zeros_like(ratio))
```

同理，不要把 `torch.nan_to_num` 当作通用修复。它会隐藏根因，并让错误梯度或状态继续传播。

---

## 8. 数值精度决定可表示范围

### FP32

动态范围和精度较高，但仍可能因非法数学表达式、过大 logits、错误 optimizer state 或算法发散产生 NaN / Inf。

### FP16

动态范围较小，最大有限值约为 65504，更容易 overflow；很小的 gradient 也容易 underflow，因此常配合动态 loss scaling。

### BF16

指数位与 FP32 相同，动态范围更大，通常不需要 loss scaling；但尾数精度更低，而且无法修复算法性 NaN、除零或错误 mask。

混合精度的目标是让适合低精度的算子更快，同时让 numerically sensitive 的计算保留较高精度。它不是“把所有 Tensor 强制转换成 FP16”。

---

## 9. Autocast 与 GradScaler 分别做什么

### `autocast`

根据算子规则选择计算 dtype，主要作用于 forward 和 loss。

### `GradScaler`

在 backward 前放大 loss，使较小 FP16 gradients 更不容易 underflow；更新前再把 gradients 还原，并检查 non-finite gradients。

两者解决的问题不同。官方推荐：

```python
with torch.autocast("cuda", dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
```

不建议把 backward 本身包在 autocast context 中。

---

## 10. AMP 下正确的顺序

```text
autocast forward + loss
  ↓
scaler.scale(loss).backward()
  ↓
scaler.unscale_(optimizer)
  ↓
检查 / 记录真实梯度
  ↓
gradient clipping
  ↓
scaler.step(optimizer)
  ↓
scaler.update()
```

关键规则：

- scaled gradients 不能直接与真实 clipping threshold 比较；
- `unscale_` 每个 optimizer、每次 update 最多调用一次；
- 必须等该 optimizer 的所有累积 gradients 完成后再 `unscale_`；
- 多 optimizer 时分别 `unscale_ / step`，整轮最后只 `update()` 一次；
- `scaler.step()` 发现 Inf / NaN gradient 时会跳过对应 optimizer step；
- `scaler.step()` 的返回值通常继承 `optimizer.step()` 的返回值，常为 `None`，不能据此判断是否更新。

---

## 11. Gradient Accumulation 下的 AMP

在一个有效 batch 由多个 microbatches 构成时：

1. 整个 accumulation window 使用同一个 scale；
2. 每个 microbatch 只做 `scale(loss).backward()`；
3. 非最后一个 microbatch 可使用 DDP / FSDP 的 `no_sync()`；
4. 完整窗口后才 `unscale_`、clip、step、update；
5. 任一 microbatch 失败时，最简单可靠的语义是所有 ranks 丢弃整个窗口。

不能只清除当前 microbatch，因为之前累积的 gradients 已经改变了这次更新的有效样本权重。

还要注意：在 DDP 中，`no_sync()` 应包住 forward 与 backward，而不只是 backward。

---

## 12. 一个 AMP + Accumulation 骨架

```python
from contextlib import nullcontext

use_fp16 = amp_dtype == torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
optimizer.zero_grad(set_to_none=True)

for micro_idx, batch in enumerate(loader):
    update_now = (micro_idx + 1) % grad_accum_steps == 0
    sync_context = nullcontext() if update_now else model.no_sync()

    with sync_context:
        with torch.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            loss = compute_loss(model, batch)
            scaled_objective = loss / grad_accum_steps

        scaler.scale(scaled_objective).backward()

    if not update_now:
        continue

    scaler.unscale_(optimizer)

    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        error_if_nonfinite=False,
    )

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()

    overflow_skip = scale_after < scale_before

    if not overflow_skip:
        scheduler.step()
        committed_step += 1

    optimizer.zero_grad(set_to_none=True)
```

这个骨架假设：

- 单 optimizer；
- 标准动态 `GradScaler`；
- loss 的固定 accumulation 缩放符合目标；
- 普通模型或 DDP；
- `scale_after < scale_before` 可用于监控默认 scaler 的 overflow backoff。

多 optimizer 可能独立跳步；FSDP、ZeRO 和自定义 scaler 需要框架对应的全局 overflow 协议，不能直接照抄。

---

## 13. Forward Non-Finite 要在 Backward 前全局决定

```python
bad = (~torch.isfinite(loss.detach())).any().to(
    device=device,
    dtype=torch.int32,
)

if torch.distributed.is_initialized():
    torch.distributed.all_reduce(
        bad,
        op=torch.distributed.ReduceOp.MAX,
        group=update_group,
    )

if bad.item():
    optimizer.zero_grad(set_to_none=True)
    discard_current_accumulation_window()
    record_incident(batch_metadata)
    continue
```

若某个 rank 独自跳过 backward，而其他 ranks 进入 gradient collective，结果可能是 hang。

forward 已经 non-finite 时，没有完成 gradient overflow 棽测；无条件调用 `scaler.update()` 也不一定合理。Loss scaling 只改变 backward 的梯度尺度，不能修复 forward 中的 `exp` overflow、除零或非法 mask。

---

## 14. 如何定义 Gradient Norm

把所有参数梯度视为一个长向量 `g`，L2 norm 为：

\[
\|g\|_2
=
\sqrt{
\sum_p
\sum_j
g_{p,j}^2
}
\]

Norm clipping 的缩放系数可写成：

\[
c
=
\min\left(
1,
\frac{\tau}{\|g\|_2+\epsilon}
\right)
\]

然后：

\[
g \leftarrow c g
\]

其中 `\tau` 是 `max_norm`。

`torch.nn.utils.clip_grad_norm_` 原地修改 gradients，并返回**裁剪前**的 total norm。因此日志应记录：

- pre-clip norm；
- max norm；
- clip coefficient；
- 是否触发 clipping；
- 各层 norm top-k。

---

## 15. Gradient Clipping 不是 NaN 修复

Clipping 能限制有限 gradient 的更新幅度，但：

- NaN 与任何数相乘仍是 NaN；
- Inf 乘以 0 可能产生 NaN；
- 它无法恢复已经损坏的 Adam moments；
- 它不能纠正错误 labels 或 loss denominator；
- 长期频繁 clipping 可能掩盖过高 LR 或结构性问题。

调试时，普通模型 / DDP 可使用：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    error_if_nonfinite=True,
)
```

这适合 fail-fast。但 AMP 生产流程若在这里抛异常并直接捕获跳过，就可能绕过 `scaler.step()/update()` 的 overflow backoff。应明确采用“调试时抛错”还是“由 scaler 跳步并降低 scale”的策略。

---

## 16. Grad Norm 阈值不能机械复制

梯度范数受以下因素影响：

- loss 是 sum 还是 mean；
- 有效 token 数；
- batch size；
- gradient accumulation；
- DP world size 和 reduction 语义；
- 模型规模；
- optimizer；
- 训练阶段；
- 数据域；
- 参数共享；
- mixed precision 与 master gradients。

“所有模型都 clip 到 1.0”不是理论定律。更可靠的做法是：

1. 先记录健康训练期的 pre-clip norm；
2. 观察 median、P95、P99；
3. 做多个 threshold 的固定预算消融；
4. 统计 clipping rate；
5. 同时观察 validation、update ratio 与收敛。

如果绝大多数 steps 都被 clip，阈值可能过低，或根因仍未解决。

---

## 17. FSDP 的 Global Norm 必须分片感知

普通 `clip_grad_norm_(model.parameters())` 在本 rank 只能看到本地 gradient shards，未必得到全局 norm。

对于 FSDP1 的真正分片策略，应在所有 ranks 上调用根 FSDP 实例：

```python
pre_clip_norm = model.clip_grad_norm_(max_norm)
```

该方法包含 collective，所有相关 ranks 必须共同调用。只有 `NO_SHARD` 等没有 gradient sharding 的情况，普通函数才等价。

FSDP1 的 FP16 overflow 应使用 `ShardedGradScaler` 或框架提供的等价机制，让 `found_inf` 在 ranks 间同步。较新的 DTensor / FSDP2 可能由 dispatch 处理普通 norm utility，但接口行为随版本演进，必须按照当前框架文档验证。

---

## 18. Loss Spike 检测需要动态基线

固定规则：

```text
loss > 10 → spike
```

通常不可靠，因为 loss 会随模型、词表、训练阶段和任务变化。

一种 robust baseline 是滑动 median 与 MAD：

\[
m_t
=
\operatorname{median}(x_{t-w:t})
\]

\[
MAD_t
=
\operatorname{median}(|x_i-m_t|)
\]

\[
z_t^{\text{robust}}
=
\frac{|x_t-m_t|}
{1.4826\cdot MAD_t+\epsilon}
\]

可同时跟踪：

- loss；
- log grad norm；
- update-to-weight ratio；
- rank spread。

注意：

- warmup 与 curriculum 切换会改变基线；
- MAD 接近 0 时要加 `\epsilon`；
- 阈值必须用历史 incident 和误报率校准；
- finite spike 默认应先告警和收集证据，不宜立即永久删除数据。

NaN / Inf 则属于硬告警，不需要统计阈值。

---

## 19. Rank Spread 能发现局部异常

只记录 rank 0 的 loss 会漏掉其他 rank 的坏 batch。

建议汇总：

\[
L_{\min},
\quad
L_{\max},
\quad
L_{\text{mean}},
\quad
L_{\max}-L_{\min}
\]

并记录每个 rank 的：

- loss sum；
- valid tokens；
- grad finite flag；
- pre-clip norm；
- AMP scale；
- committed step；
- data cursor；
- heartbeat。

若某个 rank 的 local loss 很高，但全局 mean 被稀释，rank spread 能更早发现问题。

---

## 20. Update-to-Weight Ratio

梯度范数没有考虑 optimizer moments、学习率和参数尺度。可以补充：

\[
\rho
=
\frac{
\|\Delta\theta\|_2
}{
\|\theta\|_2+\epsilon
}
\]

其中：

\[
\Delta\theta
=
\theta_{\text{after step}}
-
\theta_{\text{before step}}
\]

可按层或参数组记录 `\rho`。它能帮助发现：

- scheduler 恢复后 LR 突变；
- 某个参数组 LR 配错；
- Adam moments 异常；
- weight decay 异常；
- 梯度不大但更新异常。

无需每步复制所有参数；可周期性抽样、按层统计，或由 optimizer 暴露 update norm。

---

## 21. 优化器与 Scheduler 的异常来源

需要检查：

- peak LR 是否过高；
- warmup 是否过短；
- gradient accumulation 改变后 effective batch 是否同步调整；
- resume 时 optimizer step 与 scheduler step 是否错位；
- scheduler 是否在 AMP skipped step 后仍前进；
- Adam `exp_avg` / `exp_avg_sq` 是否 finite；
- optimizer `eps` 与低精度状态是否合理；
- 参数组是否漏配或重复；
- weight decay 是否错误施加在 norm / bias 参数；
- global batch 或 token 数是否在运行中突然改变。

尤其是 scheduler：

> 若它定义在“成功参数更新次数”上，则只有 COMMITTED update 才能 `scheduler.step()`。

---

## 22. 监控 Parameter 与 Optimizer State

分阶段有限性检查：

### Forward 后

- loss；
- logits 的 max / min / finite；
- 关键 activation；
- attention logits、Q/K norm；
- softmax entropy。

### Backward 后、Unscale 后

- global grad norm；
- per-layer grad norm；
- grad max abs；
- non-finite parameter names。

### Step 后

- parameter finite；
- parameter norm；
- update norm；
- Adam first / second moments；
- LR 与 step counter。

不必每步扫描所有状态，因为这会增加同步和带宽。可采用：

- 每步检查 scalar loss 和 global norm；
- 周期性抽样 layers；
- spike 后开启短时高分辨率诊断；
- incident replay 时完整扫描。

---

## 23. Attention 相关信号只是线索

一些研究把训练 spike 与 gradient norm 增长、attention logit 增大或 attention entropy collapse 联系起来，并提出初始化缩放、QK normalization、z-loss 等方法。

这些是重要机制与设计参考，但不能写成：

> 所有 loss spike 都由 attention entropy collapse 引起。

数据错误、loss 归一化、optimizer 状态和分布式分叉都可能产生相似现象。正确顺序仍是先定位首次异常阶段，再验证机制。

---

## 24. 用 Float32 Island 缩小数值问题

如果怀疑某个复杂 loss、`exp/log/div` 或自定义算子在 autocast 下溢出，可以只让可疑区域回到 FP32：

```python
with torch.autocast("cuda", dtype=torch.float16):
    hidden = model.encode(batch)

with torch.autocast("cuda", enabled=False):
    hidden_fp32 = hidden.float()
    loss = sensitive_loss(hidden_fp32, labels)
```

诊断时可以比较：

```text
eager FP32
  → eager AMP
  → compiled AMP
  → single-node distributed
  → full distributed
```

逐层增加变量，比一开始就在完整集群上猜测更容易定位。

---

## 25. `detect_anomaly` 的准确边界

```python
with torch.autograd.detect_anomaly(check_nan=True):
    with torch.autocast("cuda", enabled=False):
        loss = compute_loss(model, batch)

    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError("non-finite forward loss")

    loss.backward()
```

它的主要作用：

- backward 出错时打印产生该 backward node 的 forward traceback；
- `check_nan=True` 时对 backward 产生的 NaN 抛错。

限制：

- 必须把 forward 也放在 context 内，才能获得有用 traceback；
- 性能开销很大，只适合短窗口调试；
- 它不是完整的 forward finite monitor；
- 不能替代显式 Inf 检查；
- 分布式中单 rank 抛错可能让其他 rank 卡在 collective。

优先在单卡、eager、小 batch、必要时关闭 AMP 的重放环境中使用。

---

## 26. Hooks 与分层二分定位

当模型很深时，可以先低成本监控 block 边界：

```python
def finite_hook(name):
    def hook(module, inputs, output):
        tensors = output if isinstance(output, tuple) else (output,)
        for tensor in tensors:
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                raise FloatingPointError(
                    f"non-finite activation after {name}"
                )
    return hook
```

策略：

1. 先每 8 或 16 层检查；
2. 找到第一个异常区间；
3. 在区间内加密检查；
4. 再检查具体算子和输入。

注意 hook 会增加同步和改变性能特征。生产环境可只计算 device-side flags，并在异常时同步；调试环境再开启完整统计。

---

## 27. 分布式训练的核心是一致决策

DDP 同步 gradients，但不会在每轮 optimizer step 后自动把所有参数重新广播成一致状态。它假设：

- 初始参数一致；
- gradient reduction 一致；
- 所有 ranks 以相同方式执行 optimizer update。

如果 rank 3 跳过 `optimizer.step()`，其他 ranks 更新了参数，之后的 gradient all-reduce 不会自动修复参数差异。

因此共同完成一次 optimizer transaction 的所有 ranks 必须一致决定：

- 是否 backward；
- 是否丢弃 accumulation window；
- 是否 optimizer step；
- 是否 scheduler step；
- 是否保存 checkpoint；
- 是否 rollback / terminate。

---

## 28. 哪个 Process Group 需要同步

在只有 DDP 时，通常是 data-parallel group。

在 TP / PP / DP / EP 混合并行中，异常决定必须覆盖所有共同参与本次更新的 ranks：

- pipeline 最后一 stage 才直接拥有 loss，但异常状态要传播给其他 stages；
- tensor-parallel ranks 共同构成一层计算；
- expert-parallel dispatch 可能把异常 token 送到不同 ranks；
- data-parallel replicas 共同定义梯度更新。

不能盲目只在 global rank 0 决定，也不能只在发现异常的本地 subgroup `continue`。具体 update group 应由并行拓扑显式定义。

---

## 29. 通信或 CUDA 错误不一定能进程内恢复

以下错误可能已经破坏 collective 顺序或 CUDA context：

- 某 rank OOM；
- NCCL timeout；
- 某 rank 提前抛异常；
- illegal memory access；
- 自定义 CUDA kernel crash；
- 硬件 Xid / ECC error。

即使 Python `try/except` 捕获到异常，继续训练也可能再次 hang 或使用损坏状态。更可靠的策略通常是：

1. 让 launcher 终止整组 workers；
2. 保留故障元数据；
3. 从已提交的 last-known-good checkpoint 重启；
4. 对同一 batch 使用有上限的重试策略。

“可捕获异常”不等于“当前进程状态可安全继续”。

---

## 30. 异常 Batch 必须能被重放

计算前记录轻量 batch manifest：

```json
{
  "attempt_step": 120431,
  "committed_step": 120428,
  "microstep": 2,
  "rank": 3,
  "worker": 5,
  "sample_ids": ["...", "..."],
  "source_shards": ["train-0421.parquet"],
  "source_offsets": [8192, 8193],
  "content_hashes": ["...", "..."],
  "pack_id": "...",
  "segment_lengths": [712, 336, 1000],
  "sequence_length": 2048,
  "valid_loss_tokens": 1982,
  "label_min": 0,
  "label_max": 127999,
  "learning_rate": 0.00018,
  "amp_dtype": "float16",
  "loss_scale": 65536,
  "dataset_version": "...",
  "tokenizer_hash": "...",
  "config_hash": "...",
  "code_commit": "..."
}
```

默认不要把受版权、隐私或 PII 约束的原文直接写入 incident log；稳定 ID、hash 和受控访问位置通常更安全。

梯度累积时应保留最近 `N` 个 microbatch manifests 的 ring buffer，因为导致异常的可能是当前窗口中更早的 microbatch。

---

## 31. Skip Batch 不是免费的操作

跳过异常 batch 会改变：

- 数据分布；
- RNG 状态；
- dropout；
- 数据游标；
- sampling exposure；
- effective token budget；
- scheduler 对齐；
- gradient accumulation 组成。

某些 forward 还可能改变：

- BatchNorm buffers；
- MoE routing statistics；
- cache；
- 自定义模块内部状态。

因此“skip 等于什么都没发生”是错误的。至少应记录：

- 为什么跳；
- 哪些 sample IDs；
- 所有 ranks 是否共同跳；
- 是否丢弃整个 accumulation window；
- 是否计入 seen / committed tokens；
- 是否将在未来重试；
- skip budget 是否超限。

---

## 32. 有限 Spike 不应默认永久删除 Batch

一个高 loss batch 可能包含：

- 稀有语言；
- 新知识；
- 困难代码；
- 长推理链；
- 分布边缘样本。

自动删除所有高 loss 样本，会让模型主动回避困难数据，并产生训练偏差。

更稳妥的分级处置：

- finite、单次 spike：记录并继续，必要时 clipping；
- finite、持续 spike：触发高分辨率诊断；
- loss non-finite：同步丢弃窗口并重放；
- grad non-finite：由 AMP 全局跳步，降低 scale；
- step 后 state non-finite：停止并 rollback；
- 同一 batch 重复失败：进入 quarantine 或 fail-fast，由预先声明的 policy 决定。

所有自动跳过都要有总量和连续次数上限，避免训练在“看似运行”时丢掉大量数据。

---

## 33. Checkpoint 不只是模型权重

可靠恢复至少需要：

- model state；
- optimizer state；
- LR scheduler state；
- GradScaler state；
- EMA state（若有）；
- committed optimizer step；
- attempt / consumed-token counters；
- epoch 与 microstep；
- gradient accumulation 状态；
- Python RNG；
- NumPy RNG；
- Torch CPU RNG；
- 每个 rank 的 CUDA RNG；
- sampler / DataLoader cursor；
- streaming shuffle buffer；
- packer remainder；
- dataset、tokenizer、config 和 code version；
- 并行拓扑。

只保存 seed 不等于保存 RNG 当前 state；只保存 epoch 不等于保存 mid-epoch 数据位置。

---

## 34. 最安全的 Checkpoint 边界

AMP 官方建议在以下时点保存：

- 下一 iteration 开始前；
- 或本 iteration 的 `scaler.update()` 完成后。

更具体地，一个清晰边界是：

```text
optimizer/scaler step decision complete
  ↓
scaler.update()
  ↓
成功时 scheduler.step() / EMA update
  ↓
zero_grad()
  ↓
保存 next-data cursor 与完整 state
```

尽量不要在 accumulation 中间保存，因为普通 model `state_dict` 不包含 `.grad`。若必须精确恢复 mid-window，还要保存：

- accumulated gradients；
- current microstep；
- 当前 scale；
- 已消费 microbatch IDs；
- no_sync / reduction 状态。

工程上通常不值得，直接只在完整 update boundary 保存更简单可靠。

---

## 35. 恢复时的对象构造与 RNG 顺序

一个稳妥顺序：

1. 用匹配配置构造 model；
2. 构造 optimizer；
3. 构造 scheduler；
4. 构造 scaler；
5. 加载 model；
6. 加载 scheduler state；
7. 加载 optimizer state；
8. 加载 scaler state；
9. 恢复 counters 与 data state；
10. 所有初始化和加载结束后，最后恢复 RNG；
11. 创建下一 iterator 并执行 smoke test。

PyTorch 的 optimizer 文档提醒，应先初始化 scheduler，再加载 optimizer state，否则 scheduler 初始化可能覆盖已加载的 learning rates。

每个 rank 的 RNG 与 data cursor 可能不同，不能只保存 rank 0 的这部分状态。

---

## 36. Transactional Checkpoint Commit

多文件、分布式或异步 checkpoint 不能因为“目录存在”就视为可恢复。

推荐协议：

```text
1. 生成唯一且不可变的 checkpoint_id
2. 所有 ranks 写 shards / metadata
3. 等待同步 save 完成，或 async Future.result() 成功
4. 校验文件、metadata、step 和必要 checksums
5. 可选：加载小样本做 smoke test
6. 写 COMMITTED marker
7. 原子更新 LATEST-GOOD 指针
8. 按保留策略清理更旧版本
```

对象存储未必支持 POSIX atomic rename，因此可依赖不可变 ID、manifest、COMMITTED marker 和原子元数据服务，而不是假设 rename 一定原子。

至少保留两个 last-known-good checkpoints，避免最新一个静默损坏。

---

## 37. Async Checkpoint 的额外风险

异步保存会把 checkpoint 从主训练路径移开，但需要处理：

- staging CPU buffer 的额外内存；
- pinned memory 压力；
- 同时存在多个异步请求；
- 训练 state 在保存过程中继续变化；
- 异常延迟到 Future 才暴露；
- 作业退出前 future 尚未完成。

必须：

- 限制并发 save 数；
- 跟踪每个 Future；
- 只有 Future 成功后才能提升 LATEST-GOOD；
- job 退出前完成或显式取消；
- 对失败 checkpoint 标记 INVALID；
- 不让 forensic snapshot 覆盖 good checkpoint。

---

## 38. Rollback 必须恢复完整事务状态

### 不需要完整 Rollback 的典型情况

单次 FP16 gradient overflow 在 `optimizer.step()` 前被 `GradScaler` 捕获，参数和 optimizer state 没有更新。同步跳步、降低 scale 并记录 incident 通常足够。

### 需要 Rollback 的情况

- parameter 已 non-finite；
- Adam moments 已损坏；
- 不同 ranks 参数或 step 分叉；
- loss 持续发散；
- scheduler / optimizer 恢复错位；
- 通信器或 CUDA context 不可信；
- checkpoint 之后的状态无法证明一致。

此时应恢复：

```text
model + optimizer + scheduler + scaler + RNG
+ data pipeline + counters + parallel state
```

只加载 model 权重会保留损坏的 optimizer moments 和错误 scheduler 位置。

---

## 39. 一个安全恢复状态机

```text
HEALTHY
  ↓
ACCUMULATING
  ├─ PRECHECK_FAILED ─→ DISCARD_WINDOW
  ├─ FORWARD_NONFINITE → DISCARD_WINDOW / REPLAY
  └─ READY_TO_COMMIT
        ├─ GRAD_NONFINITE → AMP_SKIP
        ├─ OPTIMIZER_STEP
        │      ├─ POSTCHECK_OK → COMMITTED → HEALTHY
        │      └─ POSTCHECK_FAILED → RECOVERING
        └─ PROCESS / COLLECTIVE ERROR → RESTARTING

RECOVERING / RESTARTING
  ├─ load LAST-KNOWN-GOOD
  ├─ validate all ranks
  ├─ bounded replay or quarantine
  └─ HEALTHY or HALTED
```

状态机不变量：

- 共同 update group 中所有 ranks 同步转移；
- 只有 COMMITTED 推进成功更新状态；
- retry 有上限；
- checkpoint promotion 是原子决策；
- emergency / forensic snapshot 标为 BAD 或 SUSPECT；
- 连续同一 batch 失败最终 quarantine 或 halt，不能无限循环。

---

## 40. Incident Capsule

每次严重异常生成一个小型、不可变的 incident capsule：

```text
incident_id
timestamp
job / host / rank topology
attempt_step / committed_step / microstep
first_bad_stage
loss / grad norm / scale / LR
rank-wise finite flags
batch manifest ring buffer
model / optimizer sampled statistics
last-known-good checkpoint ID
current suspect checkpoint ID
software / driver / library versions
NCCL / CUDA / hardware logs
reproduction command and outcome
disposition: continued / skipped / rolled back / halted
```

它连接了昨天的数据 provenance 与今天的训练状态：从异常指标可以追到 sample IDs，再追到原始来源、过滤决策与 shard。

---

## 41. 推荐诊断顺序

### 第一步：确定首次异常阶段

```text
input → loss → grad → optimizer state → next forward
```

### 第二步：排除统计假象

检查全局 token-normalized loss、valid tokens 和 rank spread。

### 第三步：重放异常 Batch

使用稳定 IDs 和相同 data / tokenizer / config，先在单卡 eager FP32 重放。

### 第四步：逐层增加变量

```text
FP32 → AMP → compile → DDP/FSDP → full topology
```

### 第五步：缩小到具体层

使用 `detect_anomaly`、block finite hooks、per-layer grad norm 与 FP32 island。

### 第六步：检查更新是否真实发生

核对 optimizer step、scheduler step、scaler scale、parameter / state finite 和 rank checksums。

### 第七步：决定 Skip 或 Rollback

更新前被安全拦截可同步跳步；状态已污染则恢复完整 last-known-good。

---

## 42. 训练稳定性 Dashboard

| 类别 | 推荐指标 |
|---|---|
| Loss | global token mean、rank min/max/spread、domain loss |
| 数据 | sample IDs、source、seq length、valid tokens、pack segments |
| Gradient | pre-clip global norm、layer top-k、clip coefficient、clip rate |
| Update | parameter norm、update norm、update/weight ratio |
| AMP | scale、scale decrease、skipped optimizer steps |
| 数值 | loss/logits/activation/grad/parameter/state finite flags |
| Optimizer | LR、step、Adam moment norms 与 finite |
| 事务 | attempt、committed、discarded windows、retry count |
| 分布式 | rank heartbeats、step/scale checksum、collective errors |
| 恢复 | checkpoint age、save duration、validation、rollback count |
| 系统 | GPU memory、Xid/ECC、NCCL timeout、kernel errors |

应同时保留低成本常驻监控与异常后的短时高分辨率监控。

---

## 43. 最小 Fault-Injection Tests

### 测试 1：单 Rank NaN Loss

只在 rank 1 注入 NaN，验证：

- 所有 ranks 同步丢弃窗口；
- 没有 collective hang；
- parameters、optimizer、scheduler 和 committed step 不变。

### 测试 2：Inf Gradient

在 DDP 和 FSDP 中分别向一个 rank / shard 注入 Inf gradient，验证全局跳步和 scaler backoff。

### 测试 3：Accumulation 中途失败

在第 2 个 microbatch 注入异常，确认整个窗口 gradients 被清空，scale 只在 update boundary 按 policy 更新一次。

### 测试 4：Unscale 与 Clip 顺序

构造已知 scaled gradients，验证 pre-clip norm 与未使用 AMP 的参考结果一致。

### 测试 5：Post-Step Corruption

在 `optimizer.step()` 后向 parameter 或 Adam moment 注入 NaN，确认系统不继续训练，而是恢复完整 known-good state。

### 测试 6：残缺异步 Checkpoint

异步保存中途终止一个 rank，验证 LATEST-GOOD 仍指向旧 checkpoint，残缺目录没有 COMMITTED marker。

### 测试 7：Exact Resume

保存于 step `K`，继续 `N` 步；重新加载再跑 `N` 步，比较：

- batch / pack IDs；
- LR；
- scaler；
- committed counters；
- 参数；
- 在声明的确定性范围内的 loss。

### 测试 8：Token 归一化

让各 rank 有效 token 数不同，验证结果等于全局 loss sum / global valid tokens，而不是 rank means。

---

## 44. 常见错误

### 错误 1：看到 Loss Spike 就删除 Batch

有限 spike 可能是有价值的困难数据，应先诊断。

### 错误 2：Loss 有限，所以 Gradient 一定有限

backward 的导数表达式仍可能 overflow 或产生非法值。

### 错误 3：对 NaN 使用 `nan_to_num` 后继续 Step

这会静默改变优化目标，并可能污染 optimizer state。

### 错误 4：在 Scaled Gradients 上 Clip

阈值也被 loss scale 放大，失去原本含义。

### 错误 5：每个 Microbatch 都 `unscale_ / update`

会破坏同一 accumulation window 的一致 scale，并违反 `unscale_` 调用约束。

### 错误 6：用 `scaler.step() is None` 判断跳步

多数 optimizer 正常 step 也返回 `None`。

### 错误 7：Gradient Clipping 能修复 Non-Finite

它只能限幅有限梯度，不能恢复 NaN、Inf 或损坏的 moments。

### 错误 8：FSDP 使用本地普通 Grad Norm

本地 shard norm 不等于全局 norm，应使用分片感知接口。

### 错误 9：某个 Rank 发现 NaN 后独自 `continue`

其他 ranks 可能卡在 collective，或参数开始分叉。

### 错误 10：AMP Skip 后仍推进 Scheduler

若 scheduler 按成功更新计数，学习率会与 optimizer step 错位。

### 错误 11：跳过 Microbatch 但保留本窗口旧梯度

这会改变有效 batch 组成和权重，并造成 ranks 不一致。

### 错误 12：只保存 Model 就称为完整恢复

optimizer、scheduler、scaler、RNG 和 data cursor 都决定后续轨迹。

### 错误 13：目录存在就是 Checkpoint 完整

多 rank / 异步保存可能留下半写目录，必须有提交标记与校验。

### 错误 14：Emergency Snapshot 覆盖 Latest-Good

故障后的状态可能已经损坏，应单独标为 forensic / suspect。

### 错误 15：捕获 CUDA / NCCL 异常后原进程继续

collective 顺序或 CUDA context 可能已不可恢复，通常应整组重启。

### 错误 16：`torch.utils.checkpoint` 是训练状态 Checkpoint

它是 activation checkpointing：通过 backward 时重算 forward 节省显存，不负责保存 optimizer 或训练状态。

---

## 45. 概念检查

### 问题 1

为什么 forward loss 已经 NaN 时，单纯降低 GradScaler scale 通常无效？

**答案：** loss scale 作用于 backward 前的 loss 放大，不会改变已经发生的 forward 除零、`exp` overflow 或非法 mask。

### 问题 2

`clip_grad_norm_` 返回的是 clip 前还是 clip 后 norm？

**答案：** 返回 clip 前 total norm，同时在原地修改 gradients。

### 问题 3

为什么 AMP 下必须先 `unscale_` 再 clip？

**答案：** scaled gradients 的 norm 被 scale 放大，直接 clip 会使用错误阈值。

### 问题 4

梯度累积的第 3 个 microbatch 失败，能否只忽略它并保留前两个？

**答案：** 技术上可定义复杂的新语义，但会改变 batch 权重并增加跨 rank 不一致风险；默认最可靠的是共同丢弃整个窗口。

### 问题 5

DDP 下一轮 gradient all-reduce 会自动修复上轮不同 ranks 的参数分叉吗？

**答案：** 不会。DDP 假设参数和 optimizer update 一致，不会每轮重新广播全部参数。

### 问题 6

单次 GradScaler overflow 一定要回滚 checkpoint 吗？

**答案：** 不一定。如果它在 optimizer step 前被安全检测并全局跳过，参数和 optimizer state 未改变，降低 scale 后继续通常足够。

### 问题 7

为什么 step 后发现 parameter NaN 必须完整 rollback？

**答案：** 更新可能已经部分修改参数和 optimizer moments，`zero_grad` 只能清梯度，不能撤销状态变化。

### 问题 8

为什么 checkpoint 最好在完整 update boundary 保存？

**答案：** 可以避免保存半截 accumulated gradients、未完成 scaler 状态和模糊的数据游标语义。

### 问题 9

为什么只恢复 seed 不能 exact resume？

**答案：** RNG 已经推进到某个内部 state，数据 sampler、shuffle buffer 和 packer 也有游标；seed 只能重新开始序列。

---

## 46. 今日练习

### 练习 1：构造四种异常

分别注入：

- NaN loss；
- finite loss + Inf gradient；
- finite but huge gradient；
- optimizer step 后 parameter NaN。

验证监控能区分首次异常阶段。

### 练习 2：实现 Robust Spike Detector

维护 loss 与 log grad norm 的 rolling median / MAD，标记 soft warning；NaN / Inf 直接 hard fail。用 warmup 和 domain switch 数据观察误报。

### 练习 3：验证 AMP 顺序

比较：

```text
clip scaled grad
unscale → clip
FP32 reference
```

确认只有正确顺序的 pre-clip norm 与 FP32 参考一致。

### 练习 4：验证全局 Token Loss

构造两个 ranks，分别有 100 与 20 个 valid tokens，比较 rank mean average 与 global token mean 的差异，并验证 backward scaling。

### 练习 5：实现 Batch Incident Capsule

把 sample IDs、shard offsets、pack boundaries、valid tokens、LR、scale、rank、step、数据与代码版本写成不可变 JSON；不要记录原始敏感文本。

### 练习 6：实现 Transactional Checkpoint

使用临时 checkpoint ID、校验、`COMMITTED` marker 和 `LATEST-GOOD` 指针。模拟保存中途崩溃，确保旧恢复点不受影响。

### 练习 7：做分布式 Fault Injection

只在一个 rank 注入 NaN，验证全局 OR、共同丢弃、无 hang，并比较所有 ranks 的 committed step 与参数 checksum。

### 练习 8：Exact Resume 测试

保存后继续 10 steps，再恢复重跑，比较 batch IDs、LR、scale、optimizer state、参数与 loss；明确记录不能跨版本保证完全一致的部分。

---

## 47. 参考资料

- [PyTorch AMP Examples：Unscale、Clipping、Accumulation 与多 Optimizer](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch AMP Recipe：NaN/Inf 排查与 Scaler 保存恢复](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [PyTorch `clip_grad_norm_`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [PyTorch Autograd Anomaly Detection](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)
- [PyTorch Numerical Accuracy Notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch ShardedGradScaler](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/sharded_grad_scaler.py)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch Reproducibility Notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Spike No More：Stabilizing the Pre-training of Large Language Models](https://arxiv.org/abs/2312.16903)
- [OLMo：开放训练配置与可重建数据顺序](https://arxiv.org/abs/2402.00838)

---

## 今日总结

今天建立了一套从异常发现到安全恢复的完整框架：

1. loss spike 是症状，必须区分 finite/non-finite 与首次异常阶段；
2. 日志 loss 应按全局有效 token 归一化，并观察 rank spread；
3. 数据、mask、label 和空监督 batch 是第一批排查对象；
4. AMP 必须遵守 `scale → backward → unscale → inspect/clip → step → update`；
5. gradient clipping 限制有限更新，不修复 NaN、Inf 或损坏状态；
6. FSDP 的 norm 与 overflow 检测必须分片感知；
7. accumulation 中任一 microbatch 失败，默认共同丢弃整个窗口；
8. 所有相关 ranks 必须对 backward、skip、commit 与 rollback 做一致决策；
9. checkpoint 必须覆盖 optimizer、scheduler、scaler、RNG 和数据状态；
10. 只有通过校验并带 COMMITTED 标记的 checkpoint 才能成为 latest-good；
11. 单次安全 AMP skip 通常无需 rollback，step 后状态损坏则必须完整恢复；
12. incident capsule 把训练异常重新连接到昨天建立的数据 provenance。

最核心的一句话是：

> 稳定训练不是“永远不出现异常”，而是能准确找到第一个坏状态、阻止它被提交，并从一个可验证的一致状态继续前进。

## 明日衔接建议

从“如何发现并恢复异常”继续进入“如何系统评估一个训练 checkpoint”：学习 validation loss、下游任务、生成质量、calibration、污染敏感评测、统计置信区间，以及如何构建不被单一平均分误导的 evaluation harness。