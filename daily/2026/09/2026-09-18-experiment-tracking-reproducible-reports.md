# 2026-09-18：实验追踪与可复现报告——Run Manifest、内容指纹与结果重算

## 今日目标

上一篇 [错误分析与消融实验](2026-09-17-error-analysis-ablation-experiments.md)提出了可检验的改进假设。今天补齐实验的记录层：一条“分数提高”的结论，怎样追溯到确切的数据、配置、模型和评分结果？

完成后应能回答：

1. run_id、配置指纹和产物哈希分别标识什么？
2. 为什么只保存 seed、最终分数和 Git commit 不够？
3. 如何发现产物被修改、结果漏题或重复？
4. 如何从逐题结果重算指标，而不是相信手填汇总？
5. 重新评分、重新生成和重新训练有什么区别？

本篇提供纯 Python 内存示例，没有启动真实训练、下载模型或搭建实验追踪服务。所有示例数据和模型字节都是人工构造的。

---

## 1. 可追溯不等于逐位可复现

先区分三个目标：

- 可追溯：知道结果来自哪些输入、代码与处理步骤。
- 可重新计算：保存逐题产物后，可以重新计算同一指标。
- 可重新运行：重新训练或生成，得到相同或在预先定义容差内一致的结果。

第一项是后两项的重要基础，但不能自动保证后两项。

PyTorch 官方说明，即使使用相同 seed，也不保证跨版本、平台或 CPU/GPU 得到完全相同结果；确定性设置还可能带来性能代价。参见 [PyTorch：Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。

因此报告应写清“重算了指标”还是“重新运行了模型”，不要统一称为“已复现”。

## 2. 三种标识不要混为一谈

| 标识 | 回答的问题 | 注意事项 |
| --- | --- | --- |
| run_id | 哪一次实际执行？ | 相同配置重跑也需要不同执行标识 |
| config_digest | 哪一份解析后的配置？ | 同配置不保证同输出 |
| artifact_digest | 哪一份确切字节内容？ | 哈希不证明内容正确或来源可信 |

时间戳可以记录执行时间，但不适合作为唯一的内容版本。文件名叫 latest 或 best，也不能代替不可变的产物标识。

SHA-256 可用于检测相对于已知摘要的字节变化；如果产物和摘要都被一起替换，单纯哈希检查无法验证真实性。可信发布还需要访问控制、可信清单或签名等机制。

## 3. Run Manifest 应保存什么

manifest 是一次执行的结构化清单。建议分成：

- 身份：schema_version、run_id、experiment_id、父运行、重试次数。
- 输入：数据快照、split manifest、模型与 tokenizer 的固定版本和摘要。
- 代码：Git commit、未提交修改的补丁或源码快照、执行入口。
- 配置：默认值与命令行覆盖合并之后的实际配置。
- 环境：Python、关键依赖、驱动、设备、精度、分布式规模。
- 随机性：各随机数源的 seed、worker 设置、确定性选项。
- 输出：模型、原始响应、逐题分数、日志的地址、大小和摘要。
- 状态：开始和结束时间、成功或失败、错误类别、资源消耗。

记录 dirty=true 只是在承认有未提交修改，不足以恢复执行代码；还需要保存相应内容。只记录依赖包名，不记录版本或锁文件，也容易产生漂移。

这是一份设计清单，后文代码只实现其中的最小子集。

## 4. 配置指纹必须建立在实际配置上

假设命令行没有显式指定 temperature，但程序默认值后来改变。两次命令文本相同，执行配置却不同。

应先完成默认值展开、覆盖合并与类型检查，再计算摘要。不要将密码、访问令牌或带签名的下载链接写进配置日志；优先使用允许记录字段的白名单。

本文使用键排序、固定分隔符、UTF-8 和拒绝 NaN 的 JSON 编码。相关参数见 [Python json 文档](https://docs.python.org/3/library/json.html)。

注意：这只是本文约定的 Python 序列化规则，不是通用的跨语言规范化标准。它不统一 1 与 1.0，也不自动统一 Unicode 的不同表示；列表顺序有意义，不应随便排序。

## 5. 数据版本要包含内容与成员关系

只保存“用了 train.jsonl”不足以复现：

- 文件内容可能被覆盖。
- 同一题目可能被移入不同 split。
- 参考答案、来源组或清洗规则可能改变。
- 流式数据源可能已经更新。

保存内容摘要之外，还应记录逐题 ID、来源组、split 成员清单和预处理版本。训练采样顺序与恢复游标属于执行状态，不能简单用数据集合摘要代替。

大数据不必全部放进 Git，但必须有可访问、权限合适且保留期限明确的快照地址。摘要不是备份，只有摘要无法恢复文件内容。

## 6. 把生成、评分和汇总分开

建议形成以下关系：

1. 生成记录引用模型、输入集、模板和解码配置。
2. 评分记录引用原始响应、参考答案与评分器版本。
3. 汇总记录引用逐题分数、指标定义与统计程序。
4. 报告引用汇总产物及本次选择过程。

评分器修复后，可以对相同原始响应建立新的评分运行；不要覆盖旧分数。两个模型必须使用可比较的评分协议。

更换 prompt 或重新生成响应则产生新的生成运行。只有重算指标时，不应声称进行了新的推理或训练。

## 7. 成功状态要有验收条件

训练程序退出码为零，不代表所有产物已写完。评估程序生成一个平均分，也不代表所有题都已覆盖。

建议按阶段验收：

- 输出文件存在，大小和摘要与清单一致。
- 逐题 ID 与预期集合完全一致。
- 无重复、无未知题目、无缺失结果。
- 来源组、评分版本和运行关系正确。
- 汇总可从逐题分数重新计算。
- 验收后再发布 completed 状态或完成标记。

实际文件系统与对象存储的发布语义不同，应按存储系统设计临时写入和最终发布流程。后文示例不实现事务、并发或崩溃恢复。

## 8. 最小实现：稳定编码、产物校验与指标重算

下面只使用标准库，所有产物放在内存字典中。指标明确为每题等权的二元成功率，不是来源组等权平均。

```python
import hashlib
import json
from copy import deepcopy


def check_json_value(value):
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) is list:
        for item in value:
            check_json_value(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            check_json_value(item)
        return
    raise ValueError("only JSON values with string object keys are allowed")


def encode(value):
    check_json_value(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def descriptor(blob):
    return {"sha256": digest(blob), "bytes": len(blob)}


def verify_artifacts(manifest, blobs):
    for name, expected in manifest["artifacts"].items():
        if name not in blobs:
            raise ValueError("missing artifact")
        if descriptor(blobs[name]) != expected:
            raise ValueError("artifact integrity mismatch")


def recompute(rows, expected_groups, run_id, scorer_version):
    if not expected_groups:
        raise ValueError("empty expected set")
    seen = set()
    passed = 0
    for row in rows:
        pid = row["prompt_id"]
        if pid in seen or pid not in expected_groups:
            raise ValueError("duplicate or unknown prompt_id")
        if row["source_group"] != expected_groups[pid]:
            raise ValueError("source_group mismatch")
        if row["run_id"] != run_id or row["scorer_version"] != scorer_version:
            raise ValueError("run or scorer mismatch")
        if type(row["ok"]) is not bool:
            raise ValueError("ok must be bool")
        seen.add(pid)
        passed += row["ok"]
    if seen != set(expected_groups):
        raise ValueError("incomplete evaluation")
    return {"n": len(seen), "passed": passed,
            "micro_success_rate": passed / len(seen)}


config = {"seed": 7, "decode": {"temperature": 0.0, "max_new_tokens": 64}}
expected_groups = {"p0": "g0", "p1": "g0", "p2": "g1", "p3": "g2"}
rows = [
    {"prompt_id": pid, "source_group": group, "run_id": "demo-eval-001",
     "scorer_version": "exact-v1", "ok": i != 2}
    for i, (pid, group) in enumerate(expected_groups.items())
]
blobs = {
    "config.json": encode(config),
    "split.json": encode(expected_groups),
    "scores.json": encode(rows),
    "model.bin": b"synthetic-model-placeholder",
}
manifest = {
    "schema_version": 1,
    "run_id": "demo-eval-001",
    "scorer_version": "exact-v1",
    "config_digest": digest(blobs["config.json"]),
    "artifacts": {name: descriptor(blob) for name, blob in blobs.items()},
}
verify_artifacts(manifest, blobs)
# Consume the verified bytes, rather than an unrelated in-memory result.
verified_rows = json.loads(blobs["scores.json"])
verified_groups = json.loads(blobs["split.json"])
summary = recompute(
    verified_rows, verified_groups, manifest["run_id"], manifest["scorer_version"]
)
assert summary == {"n": 4, "passed": 3, "micro_success_rate": 0.75}
```

此处 manifest 的内容在内存中被当作可信输入。真实系统还需校验清单 schema、摘要字段格式、配置关联、访问权限，以及读取期间的并发修改；不能把这一小段代码称为完整安全校验器。

## 9. 测试：相同配置、篡改、漏题与版本混用

以下代码接在上一段之后执行。

```python
def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


reordered = {"decode": {"max_new_tokens": 64, "temperature": 0.0}, "seed": 7}
assert encode(config) == encode(reordered)
changed = deepcopy(config)
changed["seed"] = 8
assert digest(encode(config)) != digest(encode(changed))
assert encode(["a", "b"]) != encode(["b", "a"])
assert json.loads(encode({"note": "学习记录"}))["note"] == "学习记录"
reject(lambda: encode({"loss": float("nan")}))
reject(lambda: encode({"loss": float("inf")}))
reject(lambda: encode({1: "non-string key"}))

tampered = dict(blobs)
tampered["model.bin"] = b"Synthetic-model-placeholder"  # Same byte length.
reject(lambda: verify_artifacts(manifest, tampered))
missing = dict(blobs)
del missing["scores.json"]
reject(lambda: verify_artifacts(manifest, missing))

def score(test_rows):
    return recompute(test_rows, expected_groups, "demo-eval-001", "exact-v1")

assert score(rows[::-1]) == summary
reject(lambda: score(rows[:-1]))
reject(lambda: score(rows + [rows[0]]))
reject(lambda: score([dict(rows[0], prompt_id="unknown")] + rows[1:]))
reject(lambda: score([dict(rows[0], scorer_version="exact-v2")] + rows[1:]))
reject(lambda: score([dict(rows[0], source_group="wrong")] + rows[1:]))
reject(lambda: score([dict(rows[0], run_id="other-run")] + rows[1:]))
reject(lambda: score([dict(rows[0], ok=1)] + rows[1:]))

print(f"verified_artifacts={len(blobs)}")
print(f"n={summary['n']}, passed={summary['passed']}, "
      f"micro_success_rate={summary['micro_success_rate']:.2f}")
print("All manifest and recomputation checks passed.")
```

改变列表顺序会改变字节摘要，但逐题平均数可以保持不变。两种结果分别回答“是不是同一份字节”和“统计量是否一样”，不矛盾。

## 10. 失败运行也要保留

如果只保留成功运行，后续报告可能低估搜索成本，也看不到不稳定性。

失败记录至少保留：

- 失败阶段、异常类别和退出状态。
- 最后一个有效 checkpoint 的引用。
- 已消耗时间、token 数和资源预算。
- 是否重试，以及新运行的父子关系。
- 哪些产物只是部分完成，不能进入最终比较。

重试不应悄悄覆盖上一次运行；恢复训练也应记录恢复点与状态范围。保存最终权重不足以完整恢复优化器、采样器和随机数状态，可回顾 [小模型实验设计](2026-09-15-small-model-experiment-design.md)。

## 11. 一份可审查报告的最小内容

报告不是只有“候选 75%，baseline 70%”。至少写清：

1. 比较的问题、两边运行标识和实际预算。
2. 训练及评估数据快照、是否参与模型选择。
3. 指标、权重、分母、失败计分规则和评分版本。
4. 点估计、来源组数量、不确定性方法及其局限。
5. 预定切片、修复与新增退化、探索性发现。
6. 已试配置数量、失败运行及选择规则。
7. 原始结果、汇总程序与可访问的产物引用。
8. 实际执行了哪些复现检查，哪些仍未验证。

[统计解释笔记](2026-09-16-paired-evaluation-group-bootstrap.md)中的置信区间与[消融笔记](2026-09-17-error-analysis-ablation-experiments.md)中的四配置对照，都应能追溯到相应逐题记录，而不是复制一个无来源的数字。

## 12. 隐私、保留期限与发布范围

可复现不意味着公开所有数据。日志可能包含用户输入、个人信息、内部文档或凭据，应按最小必要原则保存，并设置权限与保留期限。

公开报告可以提供脱敏示例、数据版本描述和受控访问方式。哈希不等于匿名化，尤其不能把短密码、邮箱等敏感字段的哈希当作安全公开形式。

本文只在学习仓库中记录合成数据，不上传真实模型、私密数据集或环境凭据。

## 13. 今日练习结果与边界

已在本地执行上述两个 Python 代码块，全部断言通过。实际输出：

```text
verified_artifacts=4
n=4, passed=3, micro_success_rate=0.75
All manifest and recomputation checks passed.
```

已检查 UTF-8 编码、Markdown 围栏和本地链接；未进行 GitHub 页面视觉渲染验收。

本篇测试覆盖配置键重排、有效配置变化、列表顺序、中文往返、非有限值拒绝、产物修改与缺失，以及评估记录的重复、遗漏和版本混用。

这些检查验证的是教学代码的局部性质。没有验证跨语言规范化、GPU 确定性、断电恢复、并发发布或真实产物存储的可靠性。

## 参考资料

- [Python：json — JSON encoder and decoder](https://docs.python.org/3/library/json.html)：编码参数及数据类型约束。
- [PyTorch：Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)：随机性控制与跨环境复现边界。
- [前篇：错误分析与消融实验](2026-09-17-error-analysis-ablation-experiments.md)。

## 今日总结

1. run_id 标识执行，配置指纹标识配置，产物摘要标识字节内容。
2. 保存实际生效的配置，而不只是启动命令。
3. 汇总必须能从完整逐题记录重算，缺失不能静默忽略。
4. 重新评分、重新生成与重新训练应有独立可追溯记录。
5. 哈希不能代替备份、真实性验证或可复现性实验。

## 下次衔接建议

继续学习模型推理部署前的验收：把质量、延迟、吞吐、显存和失败率放进统一测试协议，区分离线指标通过与服务实际可用。
