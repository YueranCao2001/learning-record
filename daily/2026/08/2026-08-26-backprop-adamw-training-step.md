# 2026-08-26：稳定训练一步——反向传播、AdamW、梯度裁剪与 Warmup

## 今日目标

- 从 cross-entropy 的梯度 `p - y` 出发，追踪梯度如何经过 LM Head 和 Transformer。
- 理解 `loss.backward()`、`optimizer.step()` 与 `zero_grad()` 各自负责什么。
- 掌握 gradient accumulation 的缩放规则和有效 batch size。
- 理解 gradient norm、gradient clipping 以及正确的裁剪时机。
- 区分 Adam、AdamW 与普通 L2 正则化。
- 理解 learning-rate warmup 和衰减调度为什么重要。
- 组合出一个包含 AMP、累积、裁剪、AdamW 和 scheduler 的稳定训练 step。

## 1. 从昨天的 Loss 继续向后

昨天得到 token-level cross-entropy：

```text
tokens
→ Transformer
→ LM Head
→ logits
→ cross-entropy loss
```

今天从 loss 反向走：

```text
loss
→ logits gradient
→ LM Head gradient
→ final hidden states
→ Transformer blocks
→ embeddings
→ optimizer update
```

训练的核心循环不是“算出 loss 就结束”，而是：

```text
forward
→ loss
→ backward
→ optional gradient processing
→ optimizer step
→ scheduler step
→ clear gradients
```

## 2. Cross-Entropy 的起点：p - y

对单个位置，模型概率为 `p`，真实 one-hot 分布为 `y`：

```text
∂L / ∂z = p - y
```

其中 `z` 是 vocabulary logits。

假设有三个 token：

```text
p = [0.70, 0.20, 0.10]
y = [0.00, 1.00, 0.00]
```

那么：

```text
p - y = [0.70, -0.80, 0.10]
```

含义：

- 第一个错误 token 概率过高，梯度推动其 logit 降低。
- 真实 token 的梯度为负，梯度下降会推动其 logit 增大。
- 第三个 token 也会被轻微压低。
- 所有类别梯度之和为 0。

## 3. 梯度如何通过 LM Head

设最终隐藏状态为 `h`，LM Head 为：

```text
z = h W + b
```

其中：

```text
h: [D]
W: [D, V]
z: [V]
g = ∂L/∂z: [V]
```

梯度为：

```text
∂L/∂W = outer(h, g)
∂L/∂b = g
∂L/∂h = g W^T
```

因此 cross-entropy 不只训练 LM Head。`∂L/∂h` 会继续传回最后一层 Transformer，再逐层传到 attention、MLP、normalization 和 embedding。

如果输入 embedding 与 LM Head 权重共享，同一参数会从输入路径和输出路径同时收到梯度，自动求导会把这些贡献累加起来。

## 4. 计算图与链式法则

如果：

```text
a = f(x)
b = g(a)
L = h(b)
```

链式法则：

```text
∂L/∂x
= ∂L/∂b
  · ∂b/∂a
  · ∂a/∂x
```

深度学习框架在 forward 时记录计算关系，backward 时沿图反向应用局部导数。

重要区别：

- 参数 `parameter`：需要 optimizer 更新的叶子张量。
- 激活 `activation`：forward 中产生的中间结果。
- 梯度 `.grad`：累积到参数上的导数。
- optimizer state：例如 AdamW 的一阶、二阶矩估计。

`loss.backward()` 只计算并累积梯度，不会更新参数。

## 5. Transformer 中的梯度分流

残差连接：

```text
y = x + F(x)
```

反向传播：

```text
∂L/∂x
= ∂L/∂y
  + ∂L/∂y · ∂F/∂x
```

梯度既可以沿 identity 路径直接通过，也可以经过子层。这是深层网络更容易优化的重要原因之一。

在一个 decoder block 中，梯度会进入：

- attention 的 Q/K/V/O 投影。
- MLP 的上下投影与激活函数。
- normalization 的 scale 参数。
- residual stream。
- 更早位置的隐藏状态。
- token embedding 和位置相关参数。

Causal mask 会阻断不允许的前向依赖；反向传播只能沿前向计算图中真实存在的边传播。

## 6. backward 不会自动清空梯度

PyTorch 默认把新梯度累加到已有 `parameter.grad`：

```python
loss_1.backward()
loss_2.backward()

# parameter.grad 约等于两个 loss 的梯度之和
```

这既是 gradient accumulation 的基础，也是忘记清空梯度时的常见 bug。

典型顺序：

```python
optimizer.zero_grad(set_to_none=True)

loss = model(batch).loss
loss.backward()
optimizer.step()
```

`set_to_none=True` 让梯度引用变成 `None`，通常可以减少不必要的清零写入。下一次 backward 会重新创建梯度。

## 7. Gradient Accumulation

显存只能放下 micro-batch，但希望使用更大的有效 batch 时，可以连续执行多个 forward/backward，再更新一次参数。

```text
micro-batch 1 → backward
micro-batch 2 → backward
micro-batch 3 → backward
micro-batch 4 → backward
→ optimizer.step()
```

如果每个 micro-batch loss 已经是 mean reduction，应除以累积步数：

```python
loss = loss / accumulation_steps
loss.backward()
```

否则累积梯度会放大 `accumulation_steps` 倍，相当于意外改变 learning rate。

## 8. 有效 Batch Size 与有效 Token 数

样本长度一致时：

```text
effective batch size
= micro batch size
  × accumulation steps
  × data-parallel world size
```

语言模型更准确的量通常是每次更新的有效 token 数：

```text
tokens per update
= 每个设备的有效 token
  × accumulation steps
  × world size
```

如果各 micro-batch 的有效 token 数不同，简单对每个 mean loss 再除以固定 accumulation steps，并不严格等价于对全部有效 token 做一次总平均。

更严谨的做法是：

```text
累计 token loss 的总和
÷ 本次 update 的有效 token 总数
```

分布式训练还需要确保各 worker 的归一化规则一致。

## 9. 累积梯度何时近似大 Batch

在以下条件下，gradient accumulation 可以近似一次大 batch：

- 参数在累积期间不更新。
- 所有 micro-batch 的 loss 缩放正确。
- reduction 与有效 token 归一化一致。
- optimizer 和 scheduler 只在累积结束后 step。
- 随机层和 batch 相关运算没有改变数学语义。

它不一定逐位完全相同，因为 Dropout 随机数、浮点加法顺序、动态 padding 和分布式归约顺序都可能不同。

## 10. Gradient Norm

把所有参数梯度看成一个大向量 `g`，L2 global norm 为：

```text
||g||₂
= sqrt(Σ_parameters Σ_elements g_i²)
```

监控 gradient norm 可以帮助发现：

- 梯度爆炸。
- loss spike。
- 学习率过高。
- 异常 batch。
- 数值溢出。
- 长时间接近 0 的梯度。

梯度范数高不一定意味着错误，但突然比平时大几个数量级值得检查。

## 11. Gradient Clipping

Global norm clipping 在梯度过大时按相同比例缩小全部梯度：

```text
if ||g|| > max_norm:
    g ← g × max_norm / (||g|| + ε)
```

方向保持不变，只有长度被限制。

```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

返回值是裁剪前的总范数，可用于日志。

Gradient clipping 是稳定性保护，不是修复所有训练问题的万能开关。持续触发裁剪可能说明 learning rate、数据、初始化或数值精度存在问题。

## 12. Value Clipping 与 Norm Clipping

### Value clipping

```text
每个梯度元素限制到 [-c, c]
```

它会分别改变各元素，可能明显扭曲整体方向。

### Norm clipping

```text
整个梯度向量按同一比例缩放
```

它更好地保留方向，因此 Transformer 训练更常见的是 global norm clipping。

## 13. SGD、Momentum 与 Adam

最基本的 SGD：

```text
θ_t = θ_{t-1} - lr × g_t
```

Momentum 使用梯度的移动平均，使更新方向更平滑。

Adam 同时维护：

```text
m_t = β₁ m_{t-1} + (1 - β₁) g_t
v_t = β₂ v_{t-1} + (1 - β₂) g_t²
```

偏差修正：

```text
m_hat = m_t / (1 - β₁^t)
v_hat = v_t / (1 - β₂^t)
```

更新：

```text
θ_t
= θ_{t-1}
  - lr × m_hat / (sqrt(v_hat) + ε)
```

直觉：

- `m` 跟踪平均梯度方向。
- `v` 跟踪梯度平方尺度。
- 每个参数获得自适应步长。
- `ε` 防止分母过小并改善数值稳定性。

## 14. AdamW 的解耦 Weight Decay

AdamW 把 weight decay 与基于梯度矩的 Adam 更新分开：

```text
θ ← θ - lr × λ × θ
θ ← θ - AdamUpdate
```

其中 `λ` 是 weight decay 系数。

关键点：

- AdamW 的 decay 不累积到一阶、二阶矩中。
- 它不同于把 `λ||θ||²/2` 直接加入 loss 后交给 Adam。
- 在普通 SGD 的简单情形下，两者可能表现等价；在自适应 optimizer 中通常不等价。

PyTorch 的 `torch.optim.AdamW` 实现的正是这种 decoupled weight decay。

## 15. 哪些参数通常不做 Weight Decay

常见实践是：

- 对 Linear/Embedding 的二维权重使用 decay。
- 对 bias 不使用 decay。
- 对 LayerNorm/RMSNorm 的一维 scale 不使用 decay。

一种实用分组：

```python
decay_params = []
no_decay_params = []

for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue

    if parameter.ndim < 2 or name.endswith(".bias"):
        no_decay_params.append(parameter)
    else:
        decay_params.append(parameter)

optimizer = torch.optim.AdamW(
    [
        {
            "params": decay_params,
            "weight_decay": 0.1,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
)
```

这是一种启发式规则，不是所有模型都必须照搬。最终应以模型结构和训练配方为准。

## 16. Learning Rate 为什么关键

Learning rate 决定参数更新的整体尺度。

过大可能导致：

- loss 剧烈震荡。
- 梯度或激活出现 NaN/Inf。
- 模型跨过良好区域。
- 训练早期直接发散。

过小可能导致：

- 收敛过慢。
- 在有限预算内欠拟合。
- 更新小到接近浮点噪声。

AdamW 的自适应分母不会让 learning rate 失去意义；全局 learning rate 仍然是最重要的训练超参数之一。

## 17. Warmup

训练刚开始时：

- AdamW 的 moment estimate 还不稳定。
- 随机初始化产生的激活与梯度尺度仍在调整。
- 大 batch 或深模型对突然施加峰值 learning rate 更敏感。

Linear warmup 从较小 learning rate 增加到峰值：

```text
lr(step)
= peak_lr × step / warmup_steps
```

当 `step >= warmup_steps` 后进入正常衰减阶段。

Warmup 不是“前几步不学习”，而是逐步增加更新强度。

## 18. Warmup 后的衰减

常见策略：

- constant：warmup 后保持峰值。
- linear decay：线性下降到最小值。
- cosine decay：平滑地按余弦曲线下降。
- inverse square root：随步数平方根倒数下降。

Cosine 示例：

```text
progress
= (step - warmup_steps)
  / (total_steps - warmup_steps)

lr
= min_lr
  + 0.5 × (peak_lr - min_lr)
  × (1 + cos(π × progress))
```

Scheduler 的 step 单位必须明确：

- optimizer update。
- micro-batch。
- epoch。
- processed tokens。

有 gradient accumulation 时，通常应按 optimizer update 调度，而不是每个 micro-batch 调度。

## 19. Scheduler 与 Optimizer 的顺序

常见训练循环是：

```python
optimizer.step()
scheduler.step()
```

这样 scheduler 为下一次更新准备 learning rate。

但不同 scheduler 的定义可能不同，必须确认：

- 第 0 步 learning rate 是多少。
- warmup 是否包含第一个 optimizer step。
- 恢复 checkpoint 时 scheduler 当前 step 是否同步。
- 是否因为 accumulation 把 scheduler 多调用了几次。

典型 off-by-one 会让 warmup 少一步或多一步，虽然不一定立即报错。

## 20. Mixed Precision

Mixed precision 让部分算子使用 `float16` 或 `bfloat16`，并让需要更大动态范围的操作保留更高精度。

PyTorch 当前推荐使用：

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    outputs = model(**batch)
    loss = outputs.loss
```

不要在使用 autocast 时手工把整个模型统一调用 `half()`。

通常：

- BF16 动态范围接近 FP32，常不需要 loss scaling。
- FP16 更容易梯度下溢，通常配合 `torch.amp.GradScaler`。
- backward 应在 autocast 上下文之外调用。

硬件、算子和模型支持情况不同，不能假设所有设备都适合相同精度。

## 21. AMP 与梯度裁剪顺序

FP16 GradScaler 会先把 loss 乘以 scale，产生被放大的梯度。

在裁剪前必须先 unscale：

```python
scaler.scale(loss).backward()

scaler.unscale_(optimizer)

grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

scaler.step(optimizer)
scaler.update()
```

如果直接裁剪仍处于 scaled 状态的梯度，阈值将失去原本含义。

## 22. 一个完整的稳定训练 Step

下面的示例同时支持：

- gradient accumulation。
- BF16 或 FP16 autocast。
- FP16 gradient scaling。
- global norm clipping。
- AdamW update。
- 每个 optimizer update 调用一次 scheduler。

```python
import torch


device_type = "cuda"
use_amp = True
amp_dtype = torch.bfloat16
use_fp16_scaler = use_amp and amp_dtype == torch.float16

accumulation_steps = 4
max_grad_norm = 1.0

scaler = torch.amp.GradScaler(
    device_type,
    enabled=use_fp16_scaler,
)

optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(train_loader):
    with torch.autocast(
        device_type=device_type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        outputs = model(**batch)
        raw_loss = outputs.loss
        loss = raw_loss / accumulation_steps

    scaler.scale(loss).backward()

    should_update = (
        (micro_step + 1) % accumulation_steps == 0
    )

    if not should_update:
        continue

    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
        error_if_nonfinite=True,
    )

    scaler.step(optimizer)
    scaler.update()

    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    log_metrics(
        loss=float(raw_loss.detach()),
        grad_norm=float(grad_norm),
        learning_rate=scheduler.get_last_lr()[0],
    )
```

最后一个不满 `accumulation_steps` 的尾 batch 需要单独处理，否则它的梯度可能永远不会 step。

## 23. 上面代码仍需注意 Token 权重

`outputs.loss` 通常已经对当前 micro-batch 的有效 token 取平均。

如果不同 micro-batch 的有效 token 数差异很大，固定除以 `accumulation_steps` 会让每个 micro-batch 权重相同，而不是让每个 token 权重相同。

更严格的 token-weighted accumulation 可以：

1. 获取每个 micro-batch 的 token loss sum。
2. 对 loss sum 做 backward。
3. 累计本次 update 的有效 token 数。
4. 在分布式归约后按总有效 token 数归一化梯度。

具体实现取决于框架如何返回 loss，以及 DDP 是求和还是平均梯度。

## 24. 分布式训练中的 no_sync

Data Distributed Parallel 通常在每次 backward 后同步梯度。做 gradient accumulation 时，前几个 micro-batch 不必立刻同步。

概念上：

```python
with model.no_sync():
    # 非 update 边界
    loss.backward()

# update 边界正常 backward，触发同步
loss.backward()
optimizer.step()
```

这样可以减少通信次数。

但必须保证最后一个 micro-batch 执行同步，否则各 worker 会用不同梯度更新参数。

## 25. Optimizer State 的显存成本

AdamW 通常为每个参数维护：

- 参数本身。
- 梯度。
- 一阶矩 `m`。
- 二阶矩 `v`。
- 混合精度训练中可能还有 FP32 master weights。

因此 optimizer state 可能比模型权重本身占用更多训练显存。

这也是大模型训练需要 optimizer state sharding、ZeRO/FSDP 或 CPU offload 的原因之一。

KV Cache 是推理阶段的主要状态；AdamW moments 是训练阶段的主要持久状态。两者用途完全不同。

## 26. Checkpoint 应保存什么

为了真正恢复训练，应保存：

- model parameters。
- optimizer state。
- scheduler state。
- GradScaler state（如果使用）。
- global update step。
- 已处理 token 数或数据位置。
- 随机数状态。
- 数据采样器状态。
- 训练配置与 tokenizer 版本。

只保存模型权重可以继续微调，但不能严格恢复原训练轨迹。

## 27. 训练日志应该看什么

至少记录：

- train loss。
- validation loss。
- perplexity。
- learning rate。
- global gradient norm。
- clip 触发频率。
- tokens per second。
- 有效 token 数。
- AMP scale（FP16 时）。
- NaN/Inf 次数。
- GPU memory 与 step time。

单看 loss 很难区分“正常学习”“学习率过低”或“频繁裁剪后勉强稳定”。

## 28. NaN/Inf 排查顺序

当 loss 或梯度变成非有限值：

1. 记录第一个异常 step 和对应 batch。
2. 检查 input ids、labels、mask 与有效 token 数。
3. 检查 logits、loss、gradient norm 是否有限。
4. 暂时关闭 AMP，用 FP32 重现。
5. 降低 learning rate。
6. 确认裁剪发生在 unscale 之后。
7. 检查除零、`log(0)` 和错误 mask。
8. 检查恢复 checkpoint 后 optimizer/scheduler 是否匹配。
9. 不要只是跳过所有异常 batch 而不调查根因。

`error_if_nonfinite=True` 可以让 gradient clipping 在范数异常时尽早失败。

## 29. 常见错误

- 调用 `loss.backward()` 后期待参数自动更新。
- 忘记 `optimizer.zero_grad()`，无意中跨 update 累积梯度。
- 在每个 micro-batch 都调用 `optimizer.step()`，导致 accumulation 失效。
- 对 mean loss 累积时忘记除以 accumulation steps。
- micro-batch 有效 token 数不同，却误以为固定缩放严格等价。
- 在 accumulation 中每个 micro-batch 都调用 scheduler.step()。
- 最后不足一个完整 accumulation window 的梯度没有更新。
- 先裁剪 scaled FP16 梯度，再调用 `unscale_`。
- 在 `optimizer.step()` 之后才做 gradient clipping。
- 把 gradient clipping 当作解决错误数据的办法。
- AdamW 对所有 bias 和 normalization 参数统一做 decay。
- 把 AdamW 的 decoupled decay 与普通 L2 penalty 完全等同。
- checkpoint 只保存模型，丢失 optimizer 和 scheduler 状态。
- 恢复训练后 learning rate 从头 warmup。
- 使用 autocast 时又手工对整个模型调用 `half()`。
- backward 在 autocast context 内执行。
- 分布式 accumulation 的最后一步忘记同步梯度。
- 只记录裁剪后的 norm，无法观察原始梯度爆炸。

## 30. 概念检查

1. 为什么 cross-entropy 对 logits 的梯度是 `p - y`？
2. LM Head 如何把 vocabulary 梯度传回 hidden state？
3. `backward()` 与 `optimizer.step()` 的职责有什么区别？
4. 为什么 PyTorch 默认的梯度累积既有用又容易出错？
5. mean loss 做 gradient accumulation 时为什么要缩放？
6. 什么情况下有效 batch size 不如有效 token 数准确？
7. global norm clipping 如何保留梯度方向？
8. 使用 FP16 GradScaler 时为什么必须先 unscale 再 clipping？
9. Adam 的一阶矩和二阶矩分别表示什么？
10. AdamW 为什么称为 decoupled weight decay？
11. warmup 解决训练初期的什么问题？
12. 有 accumulation 时 scheduler 应按 micro-step 还是 update-step 调用？
13. 为什么恢复训练需要 optimizer 和 scheduler state？
14. KV Cache 与 AdamW optimizer state 有什么本质区别？

## 31. 今日练习

- [ ] 对三分类例子手算 `p - y`。
- [ ] 用矩阵形状推导 `∂L/∂W` 和 `∂L/∂h`。
- [ ] 连续调用两次 backward，观察 `parameter.grad` 是否相加。
- [ ] 比较 accumulation steps 为 1 和 4 时的梯度范数。
- [ ] 故意不除以 accumulation steps，观察更新尺度。
- [ ] 手工实现 global norm clipping，并与 PyTorch 对比。
- [ ] 记录裁剪前 norm 和裁剪后的实际 norm。
- [ ] 为 Adam 实现一阶矩、二阶矩和偏差修正。
- [ ] 比较 Adam 中 L2 penalty 与 AdamW decay 的一步更新。
- [ ] 画出 linear warmup + cosine decay 曲线。
- [ ] 检查 scheduler 是否只在 optimizer update 后调用。
- [ ] 使用 FP16 时验证 `unscale_ → clip → step` 的顺序。
- [ ] 保存完整 checkpoint，加载后确认 learning rate 和 global step 连续。
- [ ] 给一个 batch 注入异常 label，练习定位第一个非有限梯度。

## 32. 参考资料

- [PyTorch：AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch：clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [PyTorch：Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch：Automatic Mixed Precision Examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)

## 今日总结

一句话总结：**loss 通过 `p - y` 将误差信号传入 logits，再沿 LM Head 和 Transformer 反向传播；稳定训练则依赖正确的梯度归一化、累积、裁剪、AdamW 更新和 learning-rate 调度。**

完整 update 路径：

```text
forward under autocast
→ token-normalized loss
→ scaled backward（FP16）
→ accumulation boundary
→ unscale
→ global norm clipping
→ AdamW step
→ scaler update
→ scheduler step
→ zero gradients
```

需要牢记：

- backward 计算并累积梯度，optimizer 才更新参数。
- accumulation 必须保持正确的 loss/token 缩放。
- 裁剪应发生在 optimizer step 前；FP16 下还必须在 unscale 后。
- AdamW 将 weight decay 与 adaptive gradient update 解耦。
- warmup 和 scheduler 应按真实 optimizer update 计步。
- 完整 checkpoint 必须包含 optimizer、scheduler 和训练进度状态。

## 明日衔接建议

从“如何更新参数”继续到“如何节省训练显存”：学习 activation memory、mixed precision、gradient checkpointing、optimizer state sharding、ZeRO/FSDP，以及模型参数、梯度、激活和 optimizer state 各自占用多少显存。
