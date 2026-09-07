# 2026-09-07：Supervised Fine-Tuning——Chat Template、Assistant-only Loss、Packing 与 LoRA

## 今日目标

上一篇在[Checkpoint 评估](2026-09-04-checkpoint-evaluation-harness.md)中建立了冻结协议、逐题日志和配对比较。今天进入 post-training 的第一步：怎样把基础语言模型训练成能按角色、格式和任务要求回答的模型。

完成后应能回答：

1. SFT 与继续预训练的目标形式相近，为什么行为却不同？
2. 一条 `messages` 记录怎样变成 `input_ids`、`labels` 与 loss mask？
3. 为什么 chat template、special tokens 和 EOS 都属于训练协议？
4. assistant-only loss 到底监督哪些 token，怎样验证没有 mask 错位？
5. packing 提升了什么，又可能引入哪些边界错误？
6. 全参数微调、LoRA 与 QLoRA 分别更新什么、保存什么？
7. 怎样判断 instruction following 提升的同时，基础能力没有明显遗忘？

本文是学习笔记与实现检查清单。本次没有下载模型、启动 GPU 训练或报告真实 benchmark 成绩；配置数值是实验起点，不是通用最优值。

---

## 1. SFT 在训练什么

给定已经格式化并分词的序列 $`x_1,\ldots,x_T`$，以及真正参与监督的位置集合 `M`，SFT 的 token-level 目标可写成：

```math
\mathcal{L}_{\mathrm{SFT}}
=
-\frac{1}{|M|}
\sum_{t\in M}
\log p_\theta(x_t\mid x_{\lt t})
```

它仍是 next-token prediction。变化来自数据分布和监督位置：

- 继续预训练主要学习“自然文本接下来是什么”；
- instruction SFT 主要学习“看到这种角色与指令后，assistant 应如何回答”；
- assistant-only SFT 通常不要求模型复述 user/system 内容，但这些 token 仍作为条件上下文；
- 是否监督 assistant role token、正文、工具调用和 EOS，由模板与 mask 共同决定。

因此，“用了交叉熵”并不能完整描述一次 SFT；数据、模板、mask、截断和聚合口径缺一不可。

---

## 2. 一条样本应是一份可追踪的数据契约

推荐保留结构化对话，不要只留下不可逆的拼接字符串：

```json
{
  "id": "support-000042",
  "group_id": "source-document-17",
  "messages": [
    {"role": "system", "content": "Answer accurately and concisely."},
    {"role": "user", "content": "Why does padding need a loss mask?"},
    {"role": "assistant", "content": "Padding is batching structure, not a target."}
  ],
  "source": "approved-internal-example",
  "license": "recorded-separately",
  "language": "en",
  "quality_flags": [],
  "split": "train"
}
```

至少记录：

- stable sample ID 与来源版本；
- 可用于 group split 的 `group_id`；
- 原始角色、内容和顺序；
- 数据许可、隐私与使用范围；
- 质量过滤、去重和污染扫描版本；
- 训练/开发/测试的实际用途；
- 后续渲染所用 template 与 tokenizer revision。

只保存最终文本会让角色边界、模板版本和 loss mask 很难复原。

---

## 3. 先选定支持的 Dataset 形态

常见输入可分为：

1. **Language-modeling text**：一列 `text`，通常对全部有效 token 计算 loss。
2. **Prompt-completion**：结构化 `prompt` 与 `completion`；可只监督 completion。
3. **Conversational language modeling**：`messages` 包含完整对话。
4. **Conversational prompt-completion**：prompt 和 completion 本身都是消息列表。

“completion-only”与“assistant-only”不是同义词：

- completion-only 根据 prompt/completion 边界决定监督范围；
- assistant-only 根据消息角色和模板返回的 assistant mask 决定；
- conversational prompt-completion 可以同时使用两种约束，最终应检查实际 labels，而不是只看配置名。

不同 TRL/Transformers 版本的默认行为可能改变。本文核对的是 TRL `v1.12.0`；应锁定依赖版本，并把 resolved config 写入 manifest。[TRL SFTTrainer v1.12.0](https://huggingface.co/docs/trl/v1.12.0/sft_trainer)

---

## 4. 简化 Schema 的验证器

下面的函数只适用于“可选一个 system，随后 user/assistant 严格交替，并以 assistant 结束”的本项目教学 schema；它不是所有工具调用对话的通用规范。

```python
ALLOWED_ROLES = {"system", "user", "assistant"}

def validate_chat_example(example):
    if not isinstance(example, dict):
        raise TypeError("example must be a dict")
    for field in ("id", "group_id"):
        if (
            not isinstance(example.get(field), str)
            or not example[field].strip()
        ):
            raise ValueError(f"stable non-empty {field} required")

    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    roles = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {i} must be a dict")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unsupported role at message {i}: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"empty content at message {i}")
        roles.append(role)

    start = 1 if roles[0] == "system" else 0
    if "system" in roles[start:]:
        raise ValueError("system is allowed only once at the beginning")

    expected = "user"
    for i in range(start, len(roles)):
        if roles[i] != expected:
            raise ValueError(
                f"message {i}: expected {expected}, got {roles[i]}"
            )
        expected = "assistant" if expected == "user" else "user"

    if roles[-1] != "assistant":
        raise ValueError("training example must end with an assistant answer")
    return True
```

若加入 `tool`、多模态内容、并行工具调用或 observation，需要重新定义状态机，不能简单把新角色塞进此验证器。

---

## 5. Split 必须早于会泄漏信息的变换

随机按行切分很容易把近重复、同一用户、同一文档或同一模板的派生样本分到两侧。

更稳妥的顺序是：

```text
raw records
  → stable IDs / provenance
  → exact + near dedup groups
  → group-aware train/dev/test split
  → train-only statistics and filtering thresholds
  → chat rendering / tokenization / packing
```

关键约束：

- 生产数据必须先生成稳定的 `group_id`；缺失时直接失败，不能把缺失值当作互不相关的单例；
- 各 split 的 sample ID、source group 与规范化内容 hash 必须完全不相交；
- exact/near duplicate 以及 benchmark prompt/answer 都要做跨 split 污染扫描；
- test 不参与模板、过滤阈值、长度上限或 checkpoint 选择；
- 开发集用于调学习率、LoRA rank、数据配比和 early stopping；
- benchmark prompt/answer 不应混入 SFT 数据；
- 数据版本变化后重新运行 contamination audit。

---

## 6. Chat Template 不是展示层

Chat model 最终只看到 token 序列。chat template 负责把结构化消息映射为模型熟悉的控制协议：

```text
messages
  → role markers
  → message content
  → end-of-turn / EOS markers
  → optional assistant-generation prefix
  → token IDs
```

不同模型可能使用完全不同的 `<|user|>`、`[INST]`、`<|im_start|>` 或 end-of-turn token。即使正文相同，套错模板也会改变输入分布。

训练、验证和推理至少要绑定：

- tokenizer revision；
- `chat_template` 内容或 hash；
- role 与工具 schema；
- BOS、EOS、PAD、end-of-turn token ID；
- `add_generation_prompt` 与 `continue_final_message` 的取值；
- 模板是否能返回 assistant token mask。

[Transformers Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)

---

## 7. 训练时通常不添加 Generation Prompt

推理时，一段以 user 结束的对话常需要追加 assistant 起始标记，告诉模型从哪个角色继续：

```python
inference_ids = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

完整 SFT 样本已经包含 assistant 回答，通常应：

```python
training_ids = tokenizer.apply_chat_template(
    full_messages,
    tokenize=True,
    add_generation_prompt=False,
)
```

但“通常”不是硬编码规则：具体效果由模板决定。应对真实 tokenizer 做 golden test，直接检查头尾 token 与角色边界。

---

## 8. 避免重复添加 Special Tokens

最安全的路径是直接令 `apply_chat_template(..., tokenize=True)` 完成渲染与分词。

如果必须先渲染字符串再单独 tokenize：

```python
rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
encoded = tokenizer(
    rendered,
    add_special_tokens=False,
    truncation=False,
)
```

原因是 chat template 通常已经输出所需的 BOS/EOS 或 turn tokens；再次让 tokenizer 自动添加，可能产生重复或错位的 special tokens。

不要凭肉眼检查字符串后就认为 token 正确。保存一小组 golden fixtures，固定 `input_ids`、decoded token、role span 和模板 hash。

---

## 9. Labels 不应预先做 Causal Shift

常见 causal LM 接口令 `labels` 与 `input_ids` 对齐，模型 loss 内部再把 logits 与下一 token 标签错位比较：

```text
input_ids:  x0  x1  x2  x3
labels:     x0  x1  x2  x3
compares:  logit0→x1, logit1→x2, logit2→x3
```

数据 collator 的职责是复制并 mask labels，而不是先删除一位。首个 label 会在模型内部 shift 后自然不计分；只有 assistant-only、padding 或 packed segment 边界等策略才需要把相应位置改成 `-100`。若数据端先 shift、模型端又 shift，会少学一个 token 并造成整体错位。

统计有效 target 时，应检查模型实际 shift 后的 `labels[..., 1:] != -100`；不要简单用序列长度或 mask 中 1 的总数代替。

---

## 10. Assistant-only Loss 的含义

令 `m_t` 表示位置 `t` 是否属于已声明的 assistant 监督区域：

```math
\mathcal{L}
=
-\frac{
\sum_{t=1}^{T}
m_t\log p_\theta(x_t\mid x_{\lt t})
}{
\sum_{t=1}^{T}m_t
}
```

通常：

- system/user token 进入上下文，但 label 为 `-100`；
- assistant 正文 token 参与 loss；
- PAD 不参与 loss；
- 每个新 packed segment 的首 token 不能被上一段预测；
- assistant 的 end-of-turn/EOS 是否参与监督，要由明确策略决定。

最后一点很重要：若 EOS 被排除，模型可能学会回答内容却没有得到停止行为的直接监督；若 EOS 被包含，也必须确认它确实对应 assistant turn 的结束。

---

## 11. Mask 必须来自 Token 级边界

不要在 decoded 文本中搜索字符串 `"assistant"` 或分隔符来重建 mask：

- 用户内容可能包含同样字符串；
- BPE 会改变字符到 token 的映射；
- special token 可能被规范化或合并；
- 截断可能切断一个角色标记；
- 多轮对话有多个 assistant span。

TRL `v1.12.0` 的 `assistant_only_loss=True` 仅支持 conversational 数据，并依赖最终生效的 training template 用 `{% generation %}` / `{% endgeneration %}` 标出 assistant 生成区域，从而返回 token-level assistant mask。若 tokenizer 自带模板缺少标记，TRL 会先尝试为已知模型族换入随库提供的 training template；自动补丁后仍无法产生 mask，或某个样本的 mask 全 0，才必须报错。不能只检查最初请求的模板，也不能假设所有模型族都能自动修补。[TRL Chat Templates v1.12.0](https://huggingface.co/docs/trl/v1.12.0/chat_templates)

启用前必须断言：

1. mask 存在且长度等于 `input_ids`；
2. mask 不是全 0；
3. 所有 assistant 回合都被覆盖；
4. user/system 与 padding 不被覆盖；
5. EOS 策略符合预期；
6. 截断后仍至少有一个有效 shifted target。

---

## 12. 一个独立于框架的 Labels 构造器

```python
IGNORE_INDEX = -100

def build_labels(
    input_ids,
    assistant_mask,
    attention_mask=None,
    segment_starts=None,
):
    n = len(input_ids)
    if len(assistant_mask) != n:
        raise ValueError("assistant_mask length mismatch")

    if attention_mask is None:
        attention_mask = [1] * n
    else:
        attention_mask = list(attention_mask)
    segment_starts_was_supplied = segment_starts is not None
    if segment_starts is None:
        segment_starts = [0] * n
    else:
        segment_starts = list(segment_starts)
    if len(attention_mask) != n or len(segment_starts) != n:
        raise ValueError("mask length mismatch")

    for name, values in {
        "assistant_mask": assistant_mask,
        "attention_mask": attention_mask,
        "segment_starts": segment_starts,
    }.items():
        if any(value not in (0, 1) for value in values):
            raise ValueError(f"{name} must contain only 0/1")

    real_positions = [
        i for i, is_real in enumerate(attention_mask) if is_real
    ]
    if not real_positions:
        raise ValueError("no real tokens")
    if any(
        is_segment_start and not is_real
        for is_segment_start, is_real in zip(
            segment_starts, attention_mask
        )
    ):
        raise ValueError("segment start cannot be padding")

    first_real = real_positions[0]
    if segment_starts_was_supplied:
        if not segment_starts[first_real]:
            raise ValueError("first real token must start a segment")
    else:
        segment_starts[first_real] = 1

    labels = []
    for token, is_assistant, is_real, is_segment_start in zip(
        input_ids,
        assistant_mask,
        attention_mask,
        segment_starts,
    ):
        supervised = is_assistant and is_real and not is_segment_start
        labels.append(token if supervised else IGNORE_INDEX)

    segment_ranges = []
    start = real_positions[0]
    for i in real_positions[1:]:
        if segment_starts[i]:
            segment_ranges.append((start, i))
            start = i
    segment_ranges.append((start, real_positions[-1] + 1))

    for start, end in segment_ranges:
        if not any(
            labels[i] != IGNORE_INDEX
            for i in range(start + 1, end)
        ):
            raise ValueError(
                f"segment starting at {start} has no valid shifted "
                "assistant targets"
            )
    return labels

def count_shifted_targets(labels):
    return sum(label != IGNORE_INDEX for label in labels[1:])
```

`segment_starts` 用于 packed/padding-free 表示：每段首 token 没有本段前文，不能让它以另一条样本结尾作为预测条件。

---

## 13. 对 Mask 做可执行的边界测试

```python
def test_build_labels():
    # 两段长度分别为 3 和 2；第二段在位置 3 重新开始。
    input_ids = [10, 11, 12, 20, 21]
    assistant_mask = [0, 1, 1, 1, 1]
    attention_mask = [1, 1, 1, 1, 1]
    segment_starts = [1, 0, 0, 1, 0]

    labels = build_labels(
        input_ids,
        assistant_mask,
        attention_mask,
        segment_starts,
    )
    assert labels == [-100, 11, 12, -100, 21]
    assert count_shifted_targets(labels) == 3

def test_padding_is_never_supervised():
    labels = build_labels(
        input_ids=[10, 11, 0, 0],
        assistant_mask=[0, 1, 1, 1],
        attention_mask=[1, 1, 0, 0],
        segment_starts=[1, 0, 0, 0],
    )
    assert labels == [-100, 11, -100, -100]

    left_padded = build_labels(
        input_ids=[0, 0, 10, 11],
        assistant_mask=[0, 0, 1, 1],
        attention_mask=[0, 0, 1, 1],
    )
    assert left_padded == [-100, -100, -100, 11]

def test_zero_target_segment_is_rejected():
    try:
        build_labels(
            input_ids=[10, 11, 12, 20],
            assistant_mask=[0, 1, 1, 0],
            attention_mask=[1, 1, 1, 1],
            segment_starts=[1, 0, 0, 1],
        )
    except ValueError as exc:
        assert "segment starting at 3" in str(exc)
    else:
        raise AssertionError("zero-target segment was not rejected")

def test_eos_equal_to_pad_id_uses_position_mask():
    labels = build_labels(
        input_ids=[10, 2, 2],  # 位置 1 是真实 EOS，位置 2 是 PAD。
        assistant_mask=[0, 1, 1],
        attention_mask=[1, 1, 0],
        segment_starts=[1, 0, 0],
    )
    assert labels == [-100, 2, -100]

def test_invalid_masks_and_all_zero_assistant_fail():
    invalid_cases = [
        dict(input_ids=[10, 11], assistant_mask=[1]),
        dict(input_ids=[10, 11], assistant_mask=[0, 2]),
        dict(input_ids=[10, 11], assistant_mask=[0, 0]),
        dict(input_ids=[7], assistant_mask=[1]),
        dict(
            input_ids=[10, 11],
            assistant_mask=[0, 1],
            segment_starts=[0, 0],
        ),
    ]
    for kwargs in invalid_cases:
        try:
            build_labels(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid case was accepted: {kwargs}")
```

第一个反例刻意构造“一段有效、一段零目标”：若只检查整包 `labels`，第二段会被第一段静默掩盖。EOS 与 PAD 共用 token ID 的反例说明，是否监督必须依据位置级 `attention_mask`，不能按 token ID 一刀切。生产实现应在 packing 前逐原始样本验证一次，并在 packing 后逐 segment 再验证一次；同时仍要从真实模板输出核对 token span，不能把这个教学函数直接当作 tokenizer 集成。

---

## 14. 多轮对话需要监督多个 Assistant Span

对话：

```text
system → user₁ → assistant₁ → user₂ → assistant₂
```

assistant-only policy 通常监督 `assistant₁` 和 `assistant₂`，而不是只监督最后一个回答。这样第二轮 assistant 的条件上下文包含第一轮完整交互，但 user 内容仍不作为预测目标。

需要事先决定：

- 是否保留较早的 assistant 回合；
- tool call 是 assistant target、结构化 target，还是只作上下文；
- tool output 是否可信、是否参与监督；
- reasoning/hidden fields 是否允许进入数据；
- refusal、空答案和异常终止怎样处理；
- system prompt 变化是否成为新的数据版本。

角色合法不等于内容安全。来自网页、工具或用户的文本仍是数据，不能因为出现在 assistant 字段就自动变成高质量标签。

---

## 15. Truncation 最危险的是静默删除监督

在本文锁定的 TRL `v1.12.0` 中，默认 `truncation_mode="keep_start"`：保留序列开头、丢弃右侧超长部分。对“长 prompt + 短回答”而言，这可能恰好删除 assistant 答案。`packing=False` 路径会在截断后过滤 fully-masked 样本；packing 路径由 packer 处理截断，却没有同一层 post-truncation zero-target 过滤。因此不能把默认过滤当成完整数据质量保证，仍要在 packing 前逐样本、packing 后逐 segment 校验。

每个 split 至少统计：

- 渲染前字符/消息长度；
- tokenized length；
- 截断前、截断后 supervised token 数；
- 被截断的角色与回合；
- zero-target drop 数；
- 长度分位数与超限来源；
- EOS/turn-end 是否保留。

不要只报告“截断率”。同样截掉 20%，截 prompt 与截答案的后果不同。

可选策略包括过滤、提高 context、压缩 system、按完整消息边界裁剪较早回合，或把长任务单独建模；策略应在 dev 上选择，并写入数据版本。

---

## 16. Packing 解决的是 Padding 浪费

若一个 batch 的实际 padded length 为 `C`，第 `i` 个样本长度为 `L_i`，未 packing 时 padding 浪费近似为：

```math
\sum_i(C-L_i)
```

这里的 `C` 通常是该 batch 最长样本的长度，也可能再按 `pad_to_multiple_of` 上取整；它不一定等于配置中的 `max_length`。packing 把多个短样本装入固定长度块，提高有效 token 比例和吞吐。它不会自动增加独立训练信息，也不会修复低质量样本。

记录：

- `packing=True/False`；
- packing strategy 与实现版本；
- `max_length`；
- padding-free 是否启用；
- 每个 pack 的 segment IDs、position IDs 与 supervised counts；
- 相对声明的截断/切分策略，每个 stable token coordinate 的保留次数；
- 丢弃、切分或跨 epoch carry 的规则。

---

## 17. EOS 不是 Attention 隔离墙

简单拼成：

```text
example A + EOS + example B
```

只加入 EOS，并不保证 B 不能 attention 到 A。packing 与样本间 attention isolation 是两个问题。

具体实现可能：

- 允许跨样本因果 attention，把 EOS 当软边界；
- 使用 block-diagonal/varlen attention 隔离 segment；
- 采用 padding-free flatten，并通过重置的 `position_ids` 与支持的 attention backend 表示边界；
- 把长样本切开后继续到下一个 block。

必须阅读所锁定版本的实现并做等价性测试。不要假设普通 2D attention mask、位置重置或 EOS 中任意一个单独就足以隔离。

本文核对的 TRL `v1.12.0` 中，`bfd` 不切开能放入 block 的样本，但会截断超出 `max_length` 的 overflow；`bfd_split` 会切分超长样本以保留 token；`wrapped` 可能在任意边界切分。`bfd` 与 `bfd_split` 都会自动启用 padding-free，但 packing strategy 只定义装箱/切分方式，不等于 attention 已经隔离。

对标准 Transformer self-attention，重置 `position_ids` 的 padding-free 路径还要求兼容且经过验证的 FlashAttention 2/3。对含 gated delta network、linear attention、causal convolution 等混合层的架构，仅重置位置通常不够；例如相关实现可能还需要预计算 boundary kwargs 或 `seq_idx`。若框架默认路径没有传递这些边界信息，应禁用 packing 或采用架构专用且通过扰动测试的路径。换模型架构、attention backend 或版本后都要重新验证。[Transformers Padding-free](https://huggingface.co/docs/transformers/padding_free)

---

## 18. Packing 的等价性测试

在 `model.eval()`、关闭 dropout、固定 dtype/backend 的条件下：

1. 分别对样本 A、B 前向；
2. 用目标 packing 路径对 A、B 前向；
3. 按 stable token coordinates 对齐 supervised logits；
4. 只排除每段首 token 作为 target；仍保留该位置的 logit 去预测本段第二个 token；
5. 比较逐 token NLL 与目标覆盖；
6. 核对总 NLL sum 和 scored target count；
7. 固定 B，把前置样本 A 换成等长且边界元数据相同、只改变 token 内容的 A′；若承诺隔离，B 的所有监督位置 logits/NLL 必须保持在容差内不变。

若实现承诺 segment 隔离，隔离后的逐 token 结果应在声明的数值容差内一致。只比较最终 mean loss 可能让漏 token 与重复 token 相互抵消。coverage 也要按策略判断：`bfd` 明确断言预期 overflow 丢弃量；`bfd_split` 的 input coordinates 应恰好出现一次，但 supervised target coordinates 应等于原目标集合减去每个新 split chunk 的首 token，因为人工切分边界没有可用的本段前文。

训练时 batch 组成与浮点归约仍可能造成微小数值差异；测试目标是抓边界/口径错误，不是承诺 bitwise 相同。

---

## 19. Token 平均会让长回答权重更高

标准 token mean：

```math
\mathcal{L}_{\mathrm{token}}
=
\frac{\sum_i S_i}{\sum_i T_i}
```

其中 `S_i` 是样本 `i` 的 assistant NLL sum，`T_i` 是有效 target 数。长回答自然贡献更多梯度。

样本等权目标则是：

```math
\mathcal{L}_{\mathrm{example}}
=
\frac{1}{N}
\sum_i\frac{S_i}{T_i}
```

两者优化目标不同。大多数语言模型训练采用 token-level 口径；若为了领域、任务或样本公平引入权重，应明确 estimand、归一化和分布式聚合方式。

重复样本、过采样小域或 packing 策略都会改变实际 token exposure。继续记录 observed mix，而不只记录配置权重。

---

## 20. 一个带 LoRA 的 SFT 配置骨架

以下依赖本文核对的 TRL `v1.12.0` 与配套 PEFT 接口，使用前必须锁定版本并检查真实模型模块名；本次未运行：

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

sft_args = SFTConfig(
    output_dir="outputs/sft-example",
    max_length=4096,
    assistant_only_loss=True,
    loss_type="nll",  # 先用兼容面更清楚的基线
    packing=False,  # 先用无 packing 基线通过 mask 测试
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    bf16=True,
    seed=42,
    data_seed=42,
)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
```

这里的学习率、rank、target modules 与 batch 只是待验证假设。`assistant_only_loss=True` 只用于 conversational 数据；若最终生效的 training template 经已知模型族自动补丁后仍不返回有效 assistant mask，应失败或被明确拒绝，不能静默退化为 full-sequence loss。

在本文核对的版本中，`loss_type=None` 且未启用 Liger 时默认解析为 `chunked_nll`；启用 `use_liger_kernel=True` 时，这个未显式设置的默认才会解析为 `nll`。若同时显式指定 `loss_type="chunked_nll"` 和 Liger，Trainer 会报错，而不是自动替换。chunked NLL 与标准 NLL 数学目标一致，但会跳过 ignored-label token 的完整 LM-head 投影并分块计算交叉熵，以降低峰值显存；若 `lm_head` 本身被 PEFT adapter 包装，该版本也会拒绝。它与具体模型、adapter、LM head 和 kernel 的兼容性必须按锁定 release source 确认。因此教学骨架先显式使用 `nll`，等基线通过后再对 `chunked_nll` 做 loss/gradient 等价与峰值显存测试。[TRL v1.12.0 SFTConfig source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/sft_config.py#L311-L313) [TRL v1.12.0 SFTTrainer source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/sft_trainer.py#L1247-L1248)

---

## 21. 全参数微调更新什么

Full fine-tuning 原则上令全部模型参数参与梯度；若只解冻选定层，应明确称为 partial fine-tuning：

- 表达能力最大，适合数据与算力充足、任务变化较大的场景；
- optimizer states、gradients 与可训练权重占用大；
- 每个任务通常需要一份完整 checkpoint；
- 对错误数据、过高学习率和分布漂移更敏感；
- 不能因为“更新得多”就断言效果一定优于 PEFT。

需要明确 embedding、LM head、norm、bias 是否全部 trainable，以及 tied weights 是否保持一致。

---

## 22. LoRA 把更新限制在低秩子空间

对冻结权重 $`W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}`$，LoRA 学习：

```math
h
=
W_0x
+
sBAx
```

其中：

```math
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r},
\qquad
r\ll\min(d_{\mathrm{in}},d_{\mathrm{out}})
```

经典 LoRA 常用 $`s=\alpha/r`$；PEFT 的 `use_rslora=True` 使用 $`s=\alpha/\sqrt r`$。两者不能只靠相同 `alpha` 横比。[LoRA 论文](https://arxiv.org/abs/2106.09685)

单个 dense layer 的 LoRA 参数量（忽略 bias）为：

```math
r(d_{\mathrm{in}}+d_{\mathrm{out}})
```

而原矩阵有 $`d_{\mathrm{in}}d_{\mathrm{out}}`$ 个参数。

---

## 23. 参数量小不等于显存只剩同样比例

例如 $`d_{\mathrm{in}}=d_{\mathrm{out}}=4096`$、`r=16`：

```text
dense matrix: 4096 × 4096 = 16,777,216 parameters
LoRA A+B:    16 × (4096 + 4096) = 131,072 parameters
ratio:       0.78125%
```

这只比较该层的权重参数。训练显存还包括：

- 冻结 base weights；
- activations；
- attention 中间量；
- LoRA gradients 与 optimizer states；
- logits/loss；
- temporary buffers、通信和 allocator 碎片。

LoRA 主要节省可训练参数相关状态与 checkpoint 体积；长 context 的 activation 仍可能主导显存。

---

## 24. Target Modules 决定 LoRA 真正改了哪里

`target_modules=["q_proj", "v_proj"]` 不是跨模型通用答案：

- 架构命名可能不同；
- GQA/MQA 中 Q、K、V 维度可能不同；
- MoE 还涉及 experts、router 与 shared layers；
- `"all-linear"` 会覆盖更多线性层，但容量、显存和风险也随之变化；
- embedding 或 LM head 若需要训练，可能要用 `modules_to_save` 或显式策略。

启动训练前应输出：

```python
def trainable_parameter_report(model):
    total = 0
    trainable = 0
    names = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            names.append((name, tuple(parameter.shape), count))
    if trainable == 0:
        raise ValueError("no trainable parameters")
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / total,
        "tensors": names,
    }
```

把完整 trainable tensor 清单保存到 manifest；仅打印一个百分比不足以发现错误匹配。

[PEFT LoRA Reference](https://huggingface.co/docs/peft/main/en/package_reference/lora)

---

## 25. Rank、Alpha、Dropout 与初始化

- `r` 控制低秩更新容量，不保证越大越好；
- `lora_alpha` 与 scaling rule 共同决定更新尺度；
- `lora_dropout` 只作用于 adapter 路径，是否有益需在 dev 验证；
- 默认初始化通常让初始 adapter 接近 no-op，训练开始时应与 base 输出一致；
- bias、DoRA、rsLoRA、LoftQ 等选项会改变实验，不能统称为“同一个 LoRA”。

最小初始化测试：

1. 插入 adapter 前缓存 base logits，或插入后用 `disable_adapter()` 得到基线；`get_peft_model()` 可能原地修改传入模型，不能把两个变量名误当成两个独立模型；
2. `eval()`、固定 dtype/backend；
3. 比较 logits 的最大/均方差；
4. 输出所有 trainable tensors；
5. 做一次 backward，确认只有预期参数得到梯度。

---

## 26. Adapter Checkpoint 不是完整模型

PEFT checkpoint 通常只含 adapter weights 与配置，不含 base model。恢复和部署必须绑定：

- exact base model repository + revision/hash；
- tokenizer files 与 special-token map；
- chat template；
- adapter config、adapter weights 与名称；
- 新增词表及 embedding/LM head 的保存策略；
- dtype、quantization config 与加载库版本；
- merge 状态。

继续训练已有 adapter 时应以 `PeftModel.from_pretrained(..., is_trainable=True)` 等受支持路径加载；默认推理加载通常会冻结 adapter。精确续训还必须恢复 Trainer 的 optimizer、scheduler 与 RNG 状态。

`merge_and_unload()` 会返回合并后的普通模型，必须接住返回值；之后不能再 unmerge、禁用或切换该 adapter。某些 PEFT 方法与量化配置不支持合并，QLoRA 也不能无条件承诺可直接合并。合并前后仍应做 logits 与生成回归测试；生成比较要使用 greedy/deterministic decoding，或恢复完全相同的 RNG/generator 与 generation config。还应保留原始 base + adapter artifact，便于审计、切换和复现。[PEFT Checkpoint Format](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)

---

## 27. QLoRA 更新的仍是 Adapter

QLoRA 的核心不是“直接训练 4-bit base weights”，而是：

1. 以 4-bit 形式加载冻结 base；
2. 4-bit 主要是冻结基座的存储形式；使用时将权重反量化到计算 dtype（原论文通常为 BF16）做矩阵乘，梯度穿过这些运算流向 LoRA，但不生成或更新 base weight gradient；
3. 更新更高精度的 LoRA 参数；
4. 采用适合训练的 NF4；double quantization 与 paged optimizer 可进一步节省量化常数或缓解内存峰值，但不是每次运行都必然启用。

Transformers/bitsandbytes 的 8-bit/4-bit 训练路径只支持训练额外参数。常见准备步骤还包括 `prepare_model_for_kbit_training()`。具体支持依赖硬件、bitsandbytes、Transformers 和 PEFT 版本。[PEFT Quantization Guide](https://huggingface.co/docs/peft/main/developer_guides/quantization)

前面的 `q_proj/v_proj` 配置是普通 LoRA 教学基线。若复现 PEFT 文档中的 QLoRA-style 覆盖，应显式验证 `target_modules="all-linear"`；对 Transformers `PreTrainedModel`，该设置会排除输出层，额外 trainable modules 仍需单独声明。

量化降低 base 权重存储，不消除 activations、临时反量化计算或 adapter optimizer states；质量也必须与 BF16 LoRA/full FT 基线实测比较。[QLoRA 论文](https://arxiv.org/abs/2305.14314)

---

## 28. SFT 的 Validation Loss 也要用相同 Mask

训练与开发 loss 必须共享：

- 相同 chat template 与 tokenizer；
- 相同 assistant/completion mask 定义；
- 相同 EOS、截断策略、target 选择与样本边界语义；
- 先跨 batch、device 与 rank 汇总 NLL sum 和 valid target count，最后只做一次除法，不能平均各 batch mean；
- 明确 `eval_packing` 是否跟随训练配置；物理布局可以不同，但同一样本的 packed/unpacked 表示必须先通过 isolation、target-coordinate coverage 与逐 token NLL 等价测试。

若训练只监督 assistant，而验证对整段 user + assistant 计算 loss，曲线没有可比意义。

同时报告：

```text
requested examples
rendered examples
dropped zero-target examples
truncated examples
assistant target tokens
total non-pad tokens
target-token ratio
NLL sum / target count
```

---

## 29. 评估 SFT 不能只看 SFT Loss

至少比较 base 与每个候选 checkpoint：

| 维度 | 问题 | 示例 |
|---|---|---|
| 目标行为 | 是否更会按指令回答？ | instruction following、格式遵循、目标任务 |
| 语言建模 | 原有文本建模是否退化？ | 冻结 held-out NLL/PPL |
| 知识/推理 | 通用能力是否保留？ | 冻结 core benchmark |
| 多轮行为 | 是否正确使用历史与角色？ | multi-turn regression set |
| 停止行为 | 是否按 EOS/turn-end 结束？ | 长度、截断、重复 |
| 安全与诚实 | 风险行为是否变化？ | 预声明风险集 |
| 成本 | 训练/部署是否满足预算？ | memory、throughput、latency |

使用上一篇建立的 evaluation harness，保留相同 sample IDs、prompt/rendering revision 和 paired per-sample results。

---

## 30. 用能力矩阵观察遗忘

只看 aggregate 可能掩盖 SFT 目标提升和基础能力下降。建议维护：

```text
                  base   checkpoint-A   checkpoint-B
instruction       ...        ...            ...
domain task       ...        ...            ...
held-out NLL      ...        ...            ...
general QA        ...        ...            ...
reasoning         ...        ...            ...
multilingual      ...        ...            ...
safety            ...        ...            ...
latency/memory    ...        ...            ...
```

checkpoint 选择应预先声明：

- 每项指标的改善方向与主目标；
- 每项允许的回归阈值；
- paired delta、置信区间与放行规则；
- 成本与风险门槛；
- 可反复用于 checkpoint 选择的 retention-dev；
- 只在全部选择冻结后使用一次的 sealed retention-test。

若反复查看所谓 final test 再调 checkpoint，它就已经退化为开发集；必须重新准备未触碰的 sealed test。

LoRA 参数少不等于不会遗忘；它仍可显著改变模型输出。全参数更新多也不等于必然遗忘，结论必须来自同协议实测。

---

## 31. 可恢复训练与 Manifest

除上一篇的 checkpoint 字段外，SFT 还应保存：

```yaml
base_model_revision: immutable-hash
tokenizer_revision: immutable-hash
chat_template_hash: sha256:...
special_tokens_hash: sha256:...
dataset_snapshot: immutable-id
dedup_and_filter_revision: git-hash
split_manifest_hash: sha256:...
assistant_mask_policy: assistant-content-plus-turn-end
truncation_policy: keep-start-drop-right-with-zero-target-rejection
max_length: 4096
packing: false
packing_strategy: null
eval_packing: false
framework_versions:
  transformers: pinned
  trl: pinned
  peft: pinned
adapter:
  method: lora
  r: 16
  alpha: 32
  target_modules: [q_proj, v_proj]
optimizer_and_scheduler: hashed-config
global_step: <checkpoint-actual-value>
training_tokens: <checkpoint-actual-assistant-target-tokens-seen>
training_token_count_policy: non-ignored-shifted-targets-across-all-ranks
world_size_and_parallel_topology: hashed-config
determinism_settings: hashed-config
random_states_per_rank: stored
```

真正 resume 还需要 optimizer、scheduler、scaler、sampler/dataloader、每 rank RNG、未消费 sample IDs 与数据游标状态。只加载 adapter weights 是 warm start，不等同于精确续训。

建议只在 optimizer-step 边界保存；若允许在梯度累积中途保存，还必须保存尚未 step 的梯度、micro-step 与 loss-scaling 状态。端到端测试应比较“连续训练 N 步”和“训练 K 步 → 新进程加载 → 再训练 N−K 步”的完整 sample-ID 序列、LR/scheduler、optimizer slots、scaler、每 rank RNG、下一步 loss/grad norm 与最终权重。

“精确恢复”还应限定在相同依赖、硬件、world size、并行拓扑和确定性设置下。跨库版本或平台时通常只能声明数值容差，不能承诺 bitwise 一致。[PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

---

## 32. 常见错误

### 训练与推理使用不同 Template

角色 token 或 turn-end 不一致会造成分布偏移。

### 先渲染，再重复添加 BOS/EOS

special tokens 重复，模型看到的格式与预期不同。

### 把 Labels 在数据端提前 Shift

模型内部再 shift 后产生错位。

### `assistant_only_loss=True`，但 Template 不返回 Mask

配置名存在不代表监督范围正确；必须检查 labels。

### Mask 掉 Assistant EOS

内容能学习，但停止行为的监督可能丢失。

### 截断后留下零 Target 样本

batch 可能产生 NaN、零 loss，或静默浪费训练。

### 认为 EOS 自动隔离 Packed Samples

EOS 是 token，不是通用 attention firewall。

### 只比较 Trainable Parameter 百分比

忽略 activations、base weights、临时 buffer 和上下文长度。

### 保存 Adapter，却没固定 Base Revision

artifact 无法独立复现。

### 只看 SFT Dev Loss

不能说明指令能力、风险与基础能力都改善。

---

## 33. 最小测试清单

1. 单样本 schema 拒绝空消息、未知角色与非法角色顺序，dataset-level 检查拒绝重复 ID；
2. `group_id` 缺失直接失败，同一 group 绝不跨 split；
3. 各 split 的 sample IDs、source groups、规范化内容 hash 完全不相交；
4. exact/near duplicate 与 benchmark prompt/answer 污染扫描通过；
5. golden chat 样本的 rendered text 与 token IDs 固定；
6. BOS/EOS/turn-end 不重复；
7. 训练样本不追加多余 generation prompt；
8. assistant/attention/segment masks 长度正确且只含 0/1，非法输入必须报错；
9. 全零 assistant mask 必须拒绝；
10. user/system/PAD labels 全为 `-100`；
11. EOS 与 PAD 共用 token ID 时，只 mask 真实 padding 位置；
12. 每个 assistant 回合都至少有一个 target；
13. EOS 是否监督与声明策略一致；
14. shifted target count 与逐 token coordinates 一致；
15. 真实模型返回的 loss 与显式 `F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)` 在容差内一致，证明类别维正确且没有重复 shift；
16. 截断后 zero-target 样本被显式拒绝；
17. packing 前逐样本、packing 后逐 segment 校验 target count；
18. “一个有效段 + 一个零目标段”必须失败或被显式 drop；
19. 相对声明的策略，`bfd` 的 overflow 丢弃量符合预期；`bfd_split` 的 input coordinates 恰好一次，target coordinates 明确扣除新 chunk 首 token；
20. 每个 packed segment 首 token label 为 `-100`；
21. 若承诺隔离，packed/unpacked 逐 token NLL 在容差内一致；
22. 固定 B、把前置 A 换成等长且边界不变的 A′ 后，B 的监督 logits/NLL 不变；
23. validation 先跨 batch/device/rank 汇总 NLL sum 与 target count，再相除；
24. LoRA trainable tensor 名称与形状完全匹配预期；
25. adapter 初始化前后 logits 在声明容差内一致；
26. backward 后冻结 base 无梯度、adapter 有有限梯度；
27. save/load 后固定输入 logits 一致；生成比较使用 greedy/deterministic decoding，或恢复相同 RNG/generator 与 generation config；
28. merge 前后结果在声明容差内一致；
29. 连续训练与新进程 resume 的 sample IDs、状态、下一步数值和最终权重符合声明的确定性标准；
30. base/SFT 使用同一冻结 evaluation harness 做 paired comparison；
31. checkpoint 选择只使用 target-dev、retention-dev、风险与成本等开发指标，绝不查看 sealed retention-test。

---

## 34. 今日练习

### 已完成的教学验证

本文中的纯 Python mask 示例应满足：

- 两段首 token 均被 mask；
- PAD 即使误标为 assistant 也不会参与监督；
- EOS 与 PAD 共用 ID 时仍能按位置正确区分；
- 非法 mask、长度不一致与全零 assistant mask 会被拒绝；
- “有效段 + 零目标段”的 pack 会被逐段校验拒绝；
- shifted target count 为实际参与 next-token loss 的数量；
- 4096×4096、rank 16 的单层 LoRA 参数比为 0.78125%。

这些是合成不变量，不是模型训练结果。

### 后续动手任务

1. 选择一个小型 causal LM，固定 model/tokenizer revision；
2. 保存三条真实 chat template golden fixtures；
3. 可视化每个 token 的 role、ID、decoded form 与 label；
4. 统计 train/dev 的 target-token ratio 与截断分位数；
5. 先用 `packing=False` 跑一个 overfit-one-batch 测试；
6. 再启用 packing，完成边界和 NLL 等价性检查；
7. 对同一数据比较 full FT、LoRA 与 QLoRA 的显存、吞吐和 scorecard；
8. 保存并恢复中途 checkpoint，验证精确 resume；
9. 用 frozen base regression suite 检查遗忘；
10. 所有结论都绑定 manifest、逐题结果与 commit。

---

## 35. 参考资料

- [Transformers：Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating)
- [Transformers：Writing Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating_writing)
- [TRL v1.12.0：SFTTrainer](https://huggingface.co/docs/trl/v1.12.0/sft_trainer)
- [TRL v1.12.0：Chat Templates](https://huggingface.co/docs/trl/v1.12.0/chat_templates)
- [TRL v1.12.0：Dataset Formats](https://huggingface.co/docs/trl/v1.12.0/dataset_formats)
- [TRL v1.12.0：SFTConfig Source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/sft_config.py)
- [TRL v1.12.0：SFTTrainer Source](https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/sft_trainer.py)
- [Transformers：Padding-free Transformers](https://huggingface.co/docs/transformers/padding_free)
- [Transformers：Causal LM Loss Source](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py)
- [PEFT：LoRA Reference](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [PEFT：LoRA Conceptual Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [PEFT：Checkpoint Format](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)
- [PEFT：Quantization Guide](https://huggingface.co/docs/peft/main/developer_guides/quantization)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PyTorch：Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Deduplicating Training Data Makes Language Models Better](https://aclanthology.org/2022.acl-long.577/)

---

## 今日总结

今天把“准备对话数据”推进成了一份可验证的 SFT 训练协议：

1. SFT 仍是 next-token prediction，行为差异来自数据分布和监督 mask；
2. 结构化 messages、provenance 与 group split 应先于不可逆渲染；
3. chat template、special tokens、generation prompt 与 EOS 都属于模型接口；
4. assistant-only loss 必须验证真实 token mask，不能只相信配置名；
5. labels 与 input 对齐，由 causal LM 内部 shift；
6. truncation 要统计丢失的 assistant targets，而不只看总长度；
7. packing 提高 token 利用率，但边界、attention 与段首 label 要单独验证；
8. LoRA 学习低秩更新，参数少并不等于 activation 或全部显存同比减少；
9. QLoRA 冻结量化 base，训练的是额外 adapter 参数；
10. adapter artifact 必须绑定 base、tokenizer、template 与 merge 状态；
11. SFT checkpoint 应同时评估目标行为、基础能力、风险与成本；
12. 可复现性来自 manifest、golden fixtures、边界测试和逐题证据。

最核心的一句话是：

> SFT 不是把对话 JSON 扔进 Trainer，而是精确定义“模型看见哪些 token、预测哪些 token、样本怎样隔离、权重怎样保存”，并用测试证明实现与定义一致。

## 下次衔接建议

在 SFT 建立行为起点后，继续学习 preference alignment：pairwise preference 数据、Bradley–Terry 视角、DPO 目标、reference model、chosen/rejected 长度偏差，以及怎样沿用今天的模板、mask 与评测协议比较 SFT 和 preference-tuned checkpoints。
