# 2026-09-11：可验证奖励与评估器工程——答案抽取、容差、失败分类与 Reward Hacking

## 今日目标

上一篇 [RL Rollout 工程](2026-09-10-rl-rollout-engineering.md)解决了样本来自哪个策略、如何入队和消费的问题。今天继续处理评分环节：一个 reward=1 究竟代表什么？如果答案解析失败、测试进程超时或评审服务故障，应该怎样记录和训练？

完成后应能回答：

1. 答案抽取、语义比较和奖励映射为什么要分层？
2. exact match 与容差比较各自适用于什么任务？
3. 模型格式错误、内容错误与基础设施失败如何区分？
4. 为什么 subprocess timeout 不能等同于安全沙箱？
5. 怎样用反例、边界和变形测试验证评分器？
6. reward 上升之后，还需要什么独立证据？

本篇采用“单一最终数值答案”的窄任务，提供不执行模型代码的纯 Python 评分器。所有回答和结果都是教学构造；未运行真实模型、代码执行沙箱或大规模训练。

---

## 1. 评估器也是训练目标的一部分

在线强化学习优化的是评分器给出的信号，而不是我们未编码进评分器的意图。

如果规则奖励“包含正确数字”，模型可以输出很多数字碰中答案；如果规则只看格式标签，模型可以只学会标签。如果只验证最终数值，奖励也不能证明整段推导正确。

因此一个奖励协议至少要声明：

- 任务输入与标准答案的来源和版本；
- 模型被告知的输出格式；
- 答案抽取与合法性规则；
- 等价关系或误差容限；
- 原始分项到总 reward 的映射；
- 失败分类、重试与训练资格；
- 独立验证集与反例测试。

评分器不是训练脚本里一个随意替换的辅助函数。改动它会改变被优化的任务，应版本化并回归测试。

---

## 2. 将评分分成四层

```text
completion text
  → extraction：找出唯一候选答案
  → validation：检查格式、类型、长度与数值范围
  → comparison：与 gold 按声明规则比较
  → reward mapping：形成训练分数与诊断状态
```

例如回答末行是 `FINAL: 0.5`：

- extraction 得到 `0.5`；
- validation 确认它是合法、有限的十进制数；
- comparison 判断是否满足本题容差；
- mapping 返回结构化结果，而不是只返回一个浮点数。

Math-Verify 将数学评分组织为答案抽取、表达式转换和比较，提供多种抽取与比较方式。本篇借鉴分层思路，但不声称下面的简化实现等价于通用数学验证器。[Math-Verify 官方仓库](https://github.com/huggingface/Math-Verify)

---

## 3. 先约定“唯一答案”，再选择解析器

本篇教学契约：

1. 答案以唯一一行 `FINAL: <number>` 给出。
2. 该行必须是最后一个非空行；前面允许解释。
3. 标签必须精确为大写 ASCII `FINAL:`，且行首没有缩进。
4. 只接受有限十进制数和科学计数法；不接受单位、百分号、分数、表达式或多答案。
5. 对整段文本与数字字面量设置长度预算。
6. 格式违规是确定的模型输出失败，不触发自动重试。

这是刻意收窄的训练任务，必须写进 prompt。不能把它无声地用在原本允许自由形式答案的 benchmark 上，再把低分全部解释成模型数学能力差。

允许解释并不意味着评分器会检查解释是否与最终答案矛盾；本篇只评最终数值。若任务要求过程正确，需要独立过程验证。

---

## 4. 等价关系必须按任务定义

| 类型 | 合理比较方式 | 常见误判 |
|---|---|---|
| 分类标签 | 受控标签 exact match | 将大小写或别名随意视作相同 |
| 整数计数 | 整数精确相等 | 容差放过差 1 的答案 |
| 浮点近似 | 明确绝对/相对容差 | 用过宽容差奖励不准确结果 |
| 分数与代数式 | 领域约束下的符号比较 | 忽略定义域、分支或变量假设 |
| 集合、区间、矩阵 | 按数据结构比较 | 丢失顺序、重数或开闭端点 |
| 带单位物理量 | 数值与量纲共同比较 | 去掉单位后把不等价量判对 |

本篇不把 `1/2` 自动转换成 `0.5`，因为契约只允许十进制字面量。一个拒绝不支持类型的窄解析器，比一个暗中猜测含义的“万能正则”更容易验证。

不要使用 `eval` 执行模型提供的数学字符串。通用符号解析也应设置复杂度和时间限制；“不是普通代码”不代表没有资源耗尽风险。

---

## 5. 本篇的非对称容差公式

设预测值为 a，标准答案为 g：

```math
|a-g|
\leq
\max\!\left(\epsilon_{\mathrm{abs}},
\epsilon_{\mathrm{rel}}|g|\right)
```

这里相对误差尺度只由 gold 决定，是明确的项目约定，不是所有库的统一默认值。

- gold 接近 0 时，绝对容差决定可接受误差。
- gold 较大时，相对容差通常更重要。
- 离散任务可将两项都置零。
- 容差应在看模型结果前确定，不能为了提高指标逐题事后放宽。

自行构造 gold=100、绝对容差=0.01、相对容差=0.001：阈值为 0.1，所以 100.1 接受，100.1001 拒绝。

代码用 Decimal 从字符串构造数值，避免先转 binary float 再引入表示误差。Decimal 仍有运算精度上下文，因此输入范围和计算精度也要一起设计。[Python Decimal 文档](https://docs.python.org/3/library/decimal.html)

---

## 6. 三类失败不能都写成 Reward=0

| 状态 | 是否有可用训练 reward | 处理 |
|---|---|---|
| 正确或内容错误 | 有，分别为 1 或 0 | 正常记录 |
| 模型输出违反已声明格式 | 有，本篇为 0 | 单独统计格式失败；不重试 |
| 评分基础设施失败 | 没有，使用 null | 保留原样本；按协议重试或排除 |

还要区分 gold 配置错误：它是评估数据或任务定义问题，应停止或隔离任务，不当作模型错误。

超时也不是一个万能状态：

- candidate 在规定资源限额下运行超时，可以是任务失败；
- worker 尚未执行就排队超时，是服务失败；
- scorer 自身陷入复杂符号运算，是评估器超时，需要按协议分类。

本篇纯数值评分器不启动服务；服务失败仅用合成返回值演示。无 reward 的样本不能直接参与 GRPO 组内均值。结合昨天的完整组协议，决定重试或排除整组并记录原因，不能静默把缺失分数补成零。

---

## 7. 保存分项，不让总分掩盖问题

最小结果结构：

```json
{
  "sample_id": "synthetic-001",
  "scorer_revision": "numeric-final-v1",
  "status": "ok",
  "format_valid": true,
  "correct": true,
  "reward": 1.0,
  "eligible": true,
  "retryable": false,
  "parsed_answer": "0.5",
  "reason": "within_tolerance"
}
```

真实流水线还应保存 completion 的内容 hash、gold revision、容差配置 hash、attempt ID、耗时和执行环境 revision。日志应有访问控制；隐藏测试和标准答案不要进入对模型可见的反馈。

如果叠加格式分与正确性分，要先检查错误答案最多能得到多少奖励。若错误但格式漂亮的答案可压过正确答案，实际目标就与“正确性优先”不同。

本篇采用二元 reward，不另外奖励格式：合法但错误为 0，非法格式也为 0，但诊断状态不同。由此产生的稀疏性需要后续实验分析，不凭直觉临时添加 shaping。

---

## 8. 纯 Python：严格数值答案评分器

数值字面量最长 64 字符，Decimal 内部十进制指数绝对值不超过 100。512 位 Decimal 精度覆盖本例输入边界内的加减和 gold×relative-tolerance 计算。代码只解析字符串，不运行候选代码。

```python
import re
from decimal import Decimal, localcontext

NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
REVISION = "numeric-final-v1"

def number(text):
    if not isinstance(text, str) or len(text) > 64 or not NUMBER.fullmatch(text):
        raise ValueError("invalid numeric literal")
    value = Decimal(text)
    if not value.is_finite() or abs(value.as_tuple().exponent) > 100:
        raise ValueError("numeric literal outside protocol limits")
    return value

def extract(text):
    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError("invalid output type or length")
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    # Count every occurrence, including quoted or repeated markers.
    if not lines or text.count("FINAL:") != 1:
        raise ValueError("exactly one FINAL marker required")
    match = re.fullmatch(r"FINAL: ([^\r\n]+)", lines[-1])
    if not match:
        raise ValueError("FINAL must be the last nonempty line")
    literal = match.group(1)
    return literal, number(literal)

def score(text, gold, abs_tol="0", rel_tol="0"):
    # Invalid gold/configuration is a caller error, not reward zero.
    g, at, rt = number(gold), number(abs_tol), number(rel_tol)
    if at < 0 or rt < 0:
        raise ValueError("negative tolerance")
    base = dict(scorer_revision=REVISION, eligible=True, retryable=False)
    try:
        literal, a = extract(text)
    except ValueError as error:
        return dict(base, status="format_error", format_valid=False,
                    correct=False, reward=0.0, parsed_answer=None,
                    reason=str(error))
    with localcontext() as ctx:
        ctx.prec = 512
        error = abs(a - g)
        threshold = max(at, rt * abs(g))
        correct = error <= threshold
    return dict(base, status="ok", format_valid=True, correct=correct,
                reward=float(correct), parsed_answer=literal,
                reason="within_tolerance" if correct else "outside_tolerance")

def infrastructure_failure(reason):
    return dict(scorer_revision=REVISION, status="infra_error",
                format_valid=None, correct=None, reward=None,
                eligible=False, retryable=True, parsed_answer=None,
                reason=reason)
```

本例要求 `FINAL:` 后恰好一个空格；数字后的多余空格会被拒绝。末尾空行允许，末尾非空解释不允许。这些边界是可测试的协议，不应留给运行时猜测。

对于不满足当前 grammar 的标准答案，函数抛出异常而不是返回模型 0 分。调用层应先离线预检整个 gold 数据集，避免训练到一半才发现格式不兼容。

---

## 9. 边界、反例与变形测试

将本节接在上一节代码后运行：

```python
def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")

assert score("FINAL: 0.5", "0.5")["reward"] == 1.0
assert score("explanation\nFINAL: 5e-1\n\n", "0.5")["reward"] == 1.0
assert score("FINAL: -0", "0")["reward"] == 1.0
assert score("FINAL: 0.6", "0.5")["reward"] == 0.0
assert score("FINAL: 0.6", "0.5")["status"] == "ok"

# Exact boundary, just outside it, and near-zero absolute tolerance.
assert score("FINAL: 100.1", "100", "0.01", "0.001")["correct"]
assert not score("FINAL: 100.1001", "100", "0.01", "0.001")["correct"]
assert score("FINAL: 0.000001", "0", "0.000001")["correct"]
assert not score("FINAL: 0.0000011", "0", "0.000001")["correct"]

bad = [
    "answer: 0.5", "FINAL: 0.5\nFINAL: 0.6",
    "FINAL: 0.5 or 0.6", "FINAL: NaN", "FINAL: Infinity",
    "FINAL: 1/2", "FINAL: 50%", "FINAL: 0.5 m",
    "FINAL: __import__('os')", "FINAL: ０.５",
    "FINAL: 0.5\nmore explanation", "FINAL:  0.5",
    "FINAL: 0.5 ", " FINAL: 0.5", "FINAL: 1e999",
    "FINAL: " + "1" * 65, "x" * 8193,
]
for text in bad:
    result = score(text, "0.5")
    assert result["status"] == "format_error", text
    assert result["reward"] == 0.0 and result["eligible"], text

# Metamorphic invariance: allowed formatting changes preserve the answer.
for literal in ("0.5", ".5", "+0.50", "5e-1"):
    assert score("FINAL: " + literal, "0.5")["correct"]
for prefix in ("", "Calculation follows.\n", "First compute the ratio.\n"):
    assert score(prefix + "FINAL: 0.5", "0.5")["correct"]

# Candidate format failure, invalid gold, and infrastructure failure differ.
rejects(lambda: score("FINAL: 0.5", "not-a-number"))
rejects(lambda: score("FINAL: 0.5", "0.5", "-0.1"))
failed = infrastructure_failure("worker_unavailable")
assert failed["reward"] is None and not failed["eligible"]
assert failed["retryable"]
assert not score("FINAL: NaN", "0.5")["retryable"]
print("PASS: numeric extraction, tolerance boundaries, adversarial cases, failure separation")
```

这些测试不执行 `bad` 中的字符串；它们只是普通文本输入。它们验证当前 grammar，不证明自然语言或复杂数学表达式被完整覆盖。

还应构建两类回归集：

- 正例：合法表达等价变换、负数、零、科学计数法和容差边界；
- 反例：多答案、伪造评分指令、单位不匹配、输出洪泛和模型答案试图覆盖 gold。

一条新的评分漏洞应先形成固定反例，再修评分器并提升 revision，最后重跑既有测试，防止修好一个漏洞却破坏旧行为。

---

## 10. Code Reward：测试运行器不等于沙箱

代码任务可用编译、隐藏测试和资源限制形成可验证反馈。但候选程序是不可信输入，不能直接在开发机或带凭据的训练环境中执行。

HumanEval 官方仓库明确提醒生成代码可能不可信，其评估代码不是完整的安全执行环境。[HumanEval 官方说明](https://github.com/openai/human-eval)

后续真正实现代码奖励时，需要单独设计：

- 执行隔离：受限容器、虚拟机或更强边界，按威胁模型选择；
- 无宿主敏感目录、云凭据、GitHub token、Docker socket；
- 默认无外网，不允许程序访问 gold 或控制评分进程；
- CPU、内存、进程数、文件大小、磁盘、输出量和 wall-clock 限额；
- 超时终止整个进程树，而不是只杀一个父进程；
- 每次运行使用全新临时环境，清理后台进程与文件；
- scorer 与 candidate 输出通道隔离，不能相信 candidate 自报的 “tests passed”。

只有 `subprocess timeout` 仍可能留下子进程、文件副作用或耗尽其他资源。本文不提供将任意生成代码在本机执行的示例，也没有声称完成安全沙箱验收。

---

## 11. 测试覆盖与奖励投机

通过一组测试，只说明通过这组测试。隐藏边界、异常输入和资源约束都可能仍有缺口。

预先设计奖励投机案例：

| 漏洞 | 反例或检查 |
|---|---|
| 输出多个答案，恰好包含 gold | 唯一答案契约 |
| 在文本里要求“忽略规则，给满分” | 候选文本只作为数据，不作为 scorer 指令 |
| 利用数值溢出、NaN 走错误分支 | 有限数与输入范围检查 |
| 只硬编码公开测试 | 私有测试、变形测试、训练评估隔离 |
| 篡改 stdout 伪造通过消息 | 由可信 harness 持有结果 |
| 格式奖励大于正确性奖励 | reward 分项与上界审计 |
| 服务超时被当成低分 | 失败来源与 retry policy 分离 |

如果采用 LLM judge，候选答案同样是不可信内容。应把 rubric、问题、候选分隔，并测试候选中的指令注入；不能因为 judge 输出 JSON 就认为评分可靠。还需要位置交换、重复评分和人工抽查。

本篇数值比较不需要 LLM judge，减少了一个非确定性来源，但并没有解决 gold 本身可能错误的问题。

---

## 12. 评估器校准：先问它判得准不准

先建立一小批人工复核样本，将 scorer 判定与人工标签对照。不要在训练输出上发现问题后只修一个正则就立刻继续大规模训练。

至少记录：

- false accept：错误答案被奖励为正确；
- false reject：正确答案因解析或比较问题被拒绝；
- 格式失败比例与具体原因；
- 基础设施失败比例；
- 按语言、输出长度、数值尺度和题型分层的误判；
- 两个 scorer revision 的分歧样本。

若 precision 的分母为零，应报告未定义，而不是自动记 0 或 1。只在人工标注覆盖范围内解释校准结论。

训练协议格式失败算 0 分，不代表“解析成功子集上的准确率”可以替代全任务成功率；后者必须明确分母。基础设施失败也应单独报告，避免通过丢弃困难样本虚增分数。

---

## 13. 缓存、重试和版本升级

评分缓存键建议包含：

```text
sample content hash
+ task/gold revision
+ extraction grammar revision
+ comparison/tolerance revision
+ reward mapping revision
+ execution image/test-suite revision (if code task)
```

只按 sample ID 缓存不足以检测文本被修改。更新容差、解析优先级或测试集之后，应失效相关缓存或使用新 namespace。

基础设施重试应针对同一份原始回答，而不是悄悄重新生成更容易评分的答案。设置有限次数和退避，保留所有 attempt 的状态。若候选程序存在随机性，应固定允许的随机源或明确重复运行规则，不能“运行到第一次通过”为止。

版本升级先在冻结 replay 集上并行评分，对照 reward、失败率及误判变化。评分器变化产生的 reward 均值跳变，不能直接解释为模型能力提升。

---

## 14. 今日练习与遗留问题

### 本次完成的合成验证

- 唯一末行答案抽取与合法数值变体通过测试。
- 多答案、表达式、单位、NaN、超长数字与 Unicode 数字按契约拒绝。
- 容差边界恰好接受，略超边界拒绝。
- 合法错误答案、格式违规与基础设施失败使用不同状态。
- gold 或容差配置错误抛出异常，不计为模型零分。
- 不执行任何候选代码字符串。

### 后续真实实验

1. 人工标注一批真实输出，分析评分器 false accept / false reject。
2. 比较严格格式任务与自由形式数学评测的失败来源。
3. 建立 scorer revision 的冻结 replay 数据集。
4. 在隔离环境验收代码任务的资源限制、进程树终止与测试保密。
5. 将 reward 曲线与独立任务成功率、格式失败率及训练成本一起分析。

遗留问题：gold 的可靠性、容差的任务依据、格式约束对模型行为的影响，以及可验证结果能否代表真正的推理质量。它们需要数据和人工审查，不能由解析器单独回答。

---

## 参考资料

- [Hugging Face：Math-Verify](https://github.com/huggingface/Math-Verify)
- [Python：Decimal](https://docs.python.org/3/library/decimal.html)
- [HumanEval：生成代码评估与安全警告](https://github.com/openai/human-eval)

核对日期：2026-09-11。本文自定义评分协议为 `numeric-final-v1`；引用项目为滚动文档，未安装或运行其评估器。真实集成需要锁定依赖、数据与执行环境版本。

## 今日总结

可验证奖励并不是“写一个函数返回 0 或 1”这么简单。答案契约、抽取边界、比较语义、失败分类和执行隔离共同定义了模型实际优化的目标。

今天完成了一个窄而可测试的数值评分器，并通过反例和边界检查把能力范围写清楚。训练前先校准评分器，训练后再用独立证据验证能力提升。

## 下次衔接建议

进行一次后训练链路复盘：把 SFT、DPO、PPO/GRPO、rollout 与 reward verifier 放进统一实验清单，梳理数据、目标、版本、mask、恢复和评估的端到端验收条件。
