# 2026-09-25：结构化输出与约束解码——Schema、语法状态和 Token Mask

## 今日目标

上一篇 [投机解码](2026-09-24-speculative-decoding-rejection-sampling.md)讨论了采样分布与验证。今天进一步限制输出形式：如何让每一步生成都保持在指定语言的合法前缀中，同时承认格式正确不等于内容正确？

完成后应能回答：

1. JSON 可解析、Schema 合格和业务正确有什么区别？
2. 为什么一个未闭合的 JSON 前缀仍可能合法？
3. Token mask 如何限制下一步采样？
4. 为什么约束解码与 top-p、投机验证的顺序需要明确？
5. 截断、空候选集和工具调用权限应怎样处理？

本篇只运行有限输出集合上的 Python 教学实现，没有接入模型或通用语法引擎，没有实现完整 JSON Schema 编译器。

---

## 1. 三层验收

| 层级 | 检查内容 | 不能证明什么 |
| --- | --- | --- |
| JSON 解析 | 文本符合 JSON 语法 | 字段、类型与业务含义正确 |
| Schema 验证 | 类型、必填字段、枚举及其他约束 | 内容与事实一致 |
| 业务验证 | ID 存在、数值关系、权限与任务条件 | 外部操作必然执行成功 |

例如 {"answer":4} 可以格式合格，但如果问题是“1+1 等于多少”，内容仍然错误。

对于工具参数，Schema 合格也不构成执行授权。真正调用前仍需检查用户意图、权限、资源归属和操作风险。

## 2. Schema 的基本字段

下面是一个仅允许 answer 字段、值为 2 或 4 的教学 Schema：

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "answer": {"type": "integer", "enum": [2, 4]}
  },
  "required": ["answer"],
  "additionalProperties": false
}
```

properties 描述字段出现时的验证规则，不自动要求它出现；required 指定必填；additionalProperties:false 禁止这里未声明的其他字段。参见 [JSON Schema 对象文档](https://json-schema.org/understanding-json-schema/reference/object)。

服务框架可能只支持某个 Schema 子集。必须记录支持范围和版本；不能静默忽略不支持的约束后仍声称完整保证。

## 3. 合法前缀不等于完整结果

生成到 {"answer": 时，完整 JSON 解析会失败，但它仍可以补成合法结果。因此不能把“当前文本必须已经能解析”当作逐 token 判定条件。

约束器需要区分：

- 可继续：当前前缀仍存在合法补全。
- 可结束：当前已经是完整合法结果。
- 不可行：不存在合法补全。

EOS 通常只能在可结束状态允许。某个前缀也可能既可结束又可继续，取决于语法；今天的有限语言不包含这种歧义。

## 4. Token 不是字符

一个 token 可能包含多个字符，例如完整字段名或多个标点；实际 tokenizer 还涉及字节、Unicode、转义和特殊 token。

候选检查必须看整个 token 对语法状态的影响，不能只检查其第一个字符。一个 token 的开头合法，但后半部分多出逗号或括号，仍应拒绝。

[XGrammar 的约束解码说明](https://github.com/mlc-ai/xgrammar/blob/main/docs/start/constrained_decoding.md)描述了根据生成状态更新匹配器并构造下一步 token mask 的流程。本文用有限字符串集合模拟这一思想，不调用该库。

## 5. Mask 与重新归一化

设当前合法 token 集合为 A(h)，原分布为 p：

```math
p_A(x\mid h)=
\frac{p(x\mid h)\mathbf{1}[x\in A(h)]}
{\sum_{y\in A(h)}p(y\mid h)}
```

分母必须大于零。logits 实现通常把禁止位置设为负无穷，再进行归一化；如果全部被屏蔽，应显式报错或按已定义的恢复协议处理，而不是偷偷取消约束。

局部 mask 得到的生成分布一般不等于“原模型在完整输出满足约束条件下的全局条件分布”。后者还涉及不同前缀未来合法补全的概率，不能混称。

## 6. 与采样和投机解码的关系

先 top-k 再 mask，可能只剩非法候选；先 mask 再 top-k 则会在合法集合中选择。操作顺序会改变分布，应作为解码协议记录。

对于[昨天的投机校正](2026-09-24-speculative-decoding-rejection-sampling.md)，p 必须是当前语法状态下真正想实现的目标分布，q 必须是实际草稿分布。每个草稿位置需要对应的语法状态。

首次拒绝时，不仅 KV 状态需要回滚，语法状态也需要恢复。后续候选不能沿错误分支继续验证。

## 7. 有限语言教学实现

为避免把玩具解析器伪装成通用实现，今天只允许两种固定序列化文本：

- {"answer":2}
- {"answer":4}

这比上面的 Schema 更严格：不允许额外空白、不同数字写法等其他合法序列化形式。前缀检查通过枚举所有合法完整文本实现，只适合有限小集合。

```python
import json
import math

LANGUAGE = ('{"answer":2}', '{"answer":4}')
VOCAB = {
    0: '{"answer":',
    1: '2',
    2: '4',
    3: '}',
    4: '2}',
    5: '4}',
    6: ',',
    7: '{"answer":2}garbage',
}
EOS = 8


def allowed(prefix):
    if not any(text.startswith(prefix) for text in LANGUAGE):
        raise ValueError("invalid prefix")
    result = {
        token for token, piece in VOCAB.items()
        if any(text.startswith(prefix + piece) for text in LANGUAGE)
    }
    if prefix in LANGUAGE:
        result.add(EOS)
    return result


def consume(prefix, token):
    if token not in allowed(prefix):
        raise ValueError("illegal token or premature EOS")
    return (prefix, True) if token == EOS else (prefix + VOCAB[token], False)


def mask_probabilities(weights, legal):
    if not legal or not legal <= weights.keys():
        raise ValueError("invalid allowed set")
    if not all(math.isfinite(x) and x >= 0 for x in weights.values()):
        raise ValueError("invalid weights")
    total = sum(weights[token] for token in legal)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("no positive legal probability mass")
    return {token: (value / total if token in legal else 0.0)
            for token, value in weights.items()}


def no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate key")
        obj[key] = value
    return obj


def validate_answer(text):
    # A narrow application validator, not a general JSON Schema engine.
    obj = json.loads(text, object_pairs_hook=no_duplicate_keys)
    if type(obj) is not dict or set(obj) != {"answer"}:
        raise ValueError("wrong fields")
    if type(obj["answer"]) is not int or obj["answer"] not in (2, 4):
        raise ValueError("wrong answer type or enum")
    return obj
```

此应用校验器故意只接受 Python int；JSON Schema 的 integer 语义可以接受数学上无小数部分的数值表示，例如 2.0。因此它只是较严格的业务子集验证，不能称为完整实现该 Schema。

## 8. 完整路径、EOS 与业务反例

以下代码接在上一段之后运行。

```python
def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


assert allowed("") == {0}
assert allowed('{"answer":') == {1, 2, 4, 5}
assert allowed('{"answer":2') == {3}
assert allowed('{"answer":2}') == {EOS}
assert 7 not in allowed("")  # Valid beginning, invalid trailing content.

prefix, ended = "", False
for token in [0, 4, EOS]:
    prefix, ended = consume(prefix, token)
assert ended and prefix == '{"answer":2}'
assert validate_answer(prefix) == {"answer": 2}

weights = {token: 1.0 for token in range(9)}
weights[6] = 1000.0  # Strong preference for an illegal comma.
masked = mask_probabilities(weights, allowed('{"answer":'))
assert masked[6] == 0
assert math.isclose(sum(masked.values()), 1.0)
assert all(masked[token] == 0.25 for token in (1, 2, 4, 5))

reject(lambda: consume("", EOS))
reject(lambda: consume('{"answer":', 6))
reject(lambda: allowed('{"wrong":'))
reject(lambda: mask_probabilities({t: 0.0 for t in range(9)}, {1, 2}))
reject(lambda: validate_answer('{"answer":2,"answer":4}'))
reject(lambda: validate_answer('{"answer":true}'))
reject(lambda: validate_answer('{"answer":2,"extra":0}'))
reject(lambda: validate_answer('{"answer":'))

# Format valid, but wrong for the concrete question "1+1 = ?".
wrong = validate_answer('{"answer":4}')
assert wrong["answer"] != 1 + 1

# Exhaust all toy token paths; every EOS-terminated path is in LANGUAGE.
finished = set()
def walk(prefix):
    for token in allowed(prefix):
        next_prefix, ended = consume(prefix, token)
        if ended:
            finished.add(next_prefix)
        else:
            assert len(next_prefix) > len(prefix)
            walk(next_prefix)
walk("")
assert finished == set(LANGUAGE)
print(f"complete_outputs={len(finished)}")
print("illegal high-probability comma masked; premature EOS rejected")
print("format-valid wrong answer detected by separate business check")
print("All constrained-output checks passed.")
```

词表提供多种切分同一文本的路径；代码检查每种可达路径，不意味着真实 tokenizer 会选择这些切分。真实系统必须使用其实际词表、解码行为和特殊 token 约定。

## 9. 截断、拒绝与空候选集

约束只能保证每一步可继续，不保证有限 token 预算内必然完成。

请求可能因为长度限制、超时、取消、拒绝回答或内部错误终止。应将 incomplete、refused、invalid 与 completed 分开，而不是尝试自动补括号后冒充成功输出。

如果合法候选集为空，可能是语法、tokenizer、采样过滤或状态同步出了问题。保留诊断信息，执行有限次数的明确重试策略，并把重试成本纳入延迟和成功率。

## 10. 约束字符串仍然是不可信数据

即使工具参数结构合法，字符串字段也可能包含路径穿越、SQL 片段或提示注入文本。

接收方仍应进行权限验证、参数化调用、路径范围检查和业务校验。不能因为值位于合法 JSON 中，就把它提升成系统指令或直接执行的代码。

日志还应避免泄露密钥和个人信息。格式保证解决的是接口稳定性，不是所有安全问题。

## 11. 如何验收真实系统

在固定模型、Schema、tokenizer 和解码配置下分别报告：

- 完整输出率、JSON 解析率与 Schema 通过率。
- 业务正确率和任务完成率。
- 拒绝、截断、超时与重试占比。
- 首次语法编译开销、缓存后的开销和生成延迟。
- 嵌套、Unicode、转义、枚举、长字段和边界条件。

成功率分母要包含全部请求，不仅是拿到完整 JSON 的请求。将编译缓存键与 Schema、tokenizer 和相关版本绑定，避免复用不兼容的 token mask。

## 12. 今日练习结果与边界

已在本地执行两个 Python 代码块，全部断言通过。穷举得到 2 种完整输出；非法高概率逗号被屏蔽，提前 EOS 被拒绝，独立业务检查识别出格式合法的错误答案。

终端确认：`All constrained-output checks passed.` 已检查 UTF-8、JSON 示例、代码围栏、公式结构和本地链接；未进行 GitHub 页面视觉渲染验收。

示例覆盖多字符 token、非法后缀、提前 EOS、空概率质量、重复字段、布尔值冒充整数，以及格式合法但业务错误。

没有实现任意嵌套 JSON、完整 Schema、增量字节解析或模型采样；有限语言的穷举测试不能替代通用语法引擎验收。

## 参考资料

- [JSON Schema：Object](https://json-schema.org/understanding-json-schema/reference/object)。
- [XGrammar：Constrained Decoding](https://github.com/mlc-ai/xgrammar/blob/main/docs/start/constrained_decoding.md)。
- [前篇：投机解码](2026-09-24-speculative-decoding-rejection-sampling.md)。

## 今日总结

1. 格式、Schema 和业务正确性是不同验收层。
2. 逐步生成需要识别可完成的前缀，而不只是完整 JSON。
3. Token mask 必须匹配实际 tokenizer 和状态。
4. 采样顺序、EOS 与截断处理都属于协议。
5. 结构化输出不能替代权限验证或事实核查。

## 下次衔接建议

继续学习工具调用的可靠执行：区分模型提出操作与系统授权执行，理解参数校验、幂等键、重试和执行结果回传。
