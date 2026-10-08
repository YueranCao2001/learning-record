# 2026年10月8日 RAG 工程端到端复盘与统一验收

## 今日目标

从 [检索证据与引用](../09/2026-09-27-rag-retrieval-evidence-citations.md)到昨天的 [流量控制与过载保护](2026-10-07-rag-admission-overload-protection.md)，各环节已经有独立练习。今天把它们串成一套候选版本验收协议，检查组件连接后是否仍满足原来的约束。

核心结论是：验收不能只看一个平均分，也不能把几段单元测试通过当作真实服务已经可发布。每项结论都要对应同一候选配置、明确的测试范围和可核验的结果。

练习用人工结果实现严格的验收汇总器；它判断报告是否满足约定，不替代检索、模型或安全测试本身。

## 1 固定一次验收的对象

先保存候选版本清单：应用代码、语料快照、索引配置、嵌入模型、生成模型、提示、权限策略、缓存策略，以及评估数据和评估器版本。

将这些配置绑定到一个不可变的候选标识。不能把候选 A 的权限报告、候选 B 的质量分数和候选 C 的压测结果拼成“一次验收通过”。

测试运行标识与候选标识不同：前者识别一次执行，后者识别被测配置。重跑可以产生新的运行记录，但不能覆盖失败历史让报告只剩成功样本。

## 2 请求路径与旁路一起验收

正常路径包括身份与范围检查、准入、缓存复核、检索、证据组织、生成和结果返回。但缓存命中可能绕过检索，降级可能跳过重排，多跳可能增加请求，重试可能重复操作。

验收必须覆盖这些旁路。例如权限检查只出现在正常检索路径中，缓存路径就可能漏掉撤权；总截止时间只传给生成器，排队和退避就可能不受约束。

[Google SRE 的可靠性测试章节](https://sre.google/sre-book/testing-reliability/)区分单元、集成与系统测试，并强调测试通过不等于可靠性的完整证明。对 RAG，应同时检查组件规则和跨组件的数据、权限及时间边界。

## 3 统一验收清单

下表是这一轮练习的检查集合。每个检查在真实项目中还需展开成多个带输入、预期和证据的案例。

| 检查 ID | 核心要求 | 代表性失败 |
| --- | --- | --- |
| snapshot | 一次请求的证据使用一致快照 | 检索旧块却读取新正文 |
| evidence | 关键事实与限制条件完整 | 压缩后丢失例外 |
| citation | 引用可追溯且支持主张 | 引用存在但不支持答案 |
| authorization | 所有返回路径遵守当前权限 | 缓存命中泄露撤权内容 |
| cache | 主体、版本与依赖变化使旧结果失效 | 新语料出现却复用旧结论 |
| deadline | 排队、退避和调用共享截止时间 | 每次重试重置预算 |
| overload | 队列与执行容量有界 | 取消未完成就提前释放槽位 |
| recovery | 恢复不破坏版本与权限限制 | 回滚恢复已禁止访问的内容 |

这些检查并非可以互相加权抵消的八个得分项。例外条件错误、越权或预算失控都要单独处理。

## 4 状态不只有通过与失败

采用 pass、fail、not_run 三种结果。缺少报告仍是未测，而不是默认成功；报告中的未知状态或重复 ID 视为无效输入。

不适用也不能随意替代未测。若服务没有答案缓存，可在测试前批准一个不同的检查清单，并留下适用范围；不能测试后删掉失败项来让结果通过。

真实报告需要保存测试步骤、时间、环境、输入指纹、观察结果与证据位置。代码中的证据字符串只是占位引用，非空不等于证据真实存在。

## 5 最小验收汇总器

所有必需项通过时输出 eligible，表示满足这份教学清单；存在失败输出 blocked，只有缺测而没有已知失败时输出 incomplete。

```python
from dataclasses import dataclass, replace


REQUIRED = frozenset({
    "snapshot", "evidence", "citation", "authorization",
    "cache", "deadline", "overload", "recovery",
})


@dataclass(frozen=True)
class Result:
    check: str
    candidate: str
    status: str
    evidence_ref: str


def assess(candidate, rows):
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("candidate required")
    indexed = {}
    for row in rows:
        if row.check not in REQUIRED or row.check in indexed:
            raise ValueError("unknown or duplicate check")
        if row.candidate != candidate:
            raise ValueError("mixed candidate reports")
        if row.status not in {"pass", "fail", "not_run"}:
            raise ValueError("invalid result status")
        if row.status != "not_run" and not row.evidence_ref.strip():
            raise ValueError("executed check needs evidence reference")
        indexed[row.check] = row
    failed = sorted(k for k, r in indexed.items() if r.status == "fail")
    untested = sorted(
        k for k in REQUIRED
        if k not in indexed or indexed[k].status == "not_run"
    )
    decision = "blocked" if failed else "incomplete" if untested else "eligible"
    return {"decision": decision, "failed": failed, "untested": untested}
```

该函数只检查输入结果的结构与组合，不加载 evidence_ref，不运行实际测试，也不证明报告内容可信。真实流水线必须限制结果写入者，验证制品和被测配置的关联，防止候选标识只是一段随意填写的文字。

eligible 也不是自动上线授权；部署审批、变更窗口和回退准备仍属于独立流程。

## 6 合成验收案例

以下代码接在上一段之后执行。全部状态是人为构造的测试输入。

```python
candidate = "rag-demo-v1"
good = [
    Result(k, candidate, "pass", f"fixture://{k}")
    for k in sorted(REQUIRED)
]
assert assess(candidate, good) == {
    "decision": "eligible", "failed": [], "untested": [],
}
assert assess(candidate, good[::-1]) == assess(candidate, good)

bad = [
    replace(r, status="fail") if r.check == "authorization" else r
    for r in good
]
assert assess(candidate, bad)["decision"] == "blocked"
assert assess(candidate, bad)["failed"] == ["authorization"]

missing = [r for r in good if r.check != "recovery"]
assert assess(candidate, missing)["decision"] == "incomplete"
assert assess(candidate, missing)["untested"] == ["recovery"]

pending = [
    replace(r, status="not_run", evidence_ref="") if r.check == "cache" else r
    for r in good
]
assert assess(candidate, pending)["untested"] == ["cache"]
both = [r for r in bad if r.check != "recovery"]
assert assess(candidate, both) == {
    "decision": "blocked",
    "failed": ["authorization"],
    "untested": ["recovery"],
}
assert assess(candidate, [])["untested"] == sorted(REQUIRED)


def reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected rejection")


reject(lambda: assess(candidate, good + [good[0]]))
reject(lambda: assess(candidate, [replace(good[0], candidate="other")]))
reject(lambda: assess(candidate, [replace(good[0], status="skipped")]))
reject(lambda: assess(candidate, [replace(good[0], evidence_ref=" ")]))
reject(lambda: assess(candidate, [replace(good[0], check="unknown")]))
reject(lambda: assess("", good))

print("complete=eligible; authorization failure=blocked")
print("missing recovery=incomplete; failure plus missing=blocked")
print("mixed candidates and duplicate reports rejected")
print("All acceptance gate checks passed.")
```

七项通过不能抵消一项权限失败；失败与缺测并存时，两者都保留在报告中。这里的 eligible 用例只是测试汇总规则，不表示任何真实 RAG 服务已经完成八项验收。

## 7 跨组件故障场景

优先补充以下集成场景，每个场景同时检查用户可见结果和实际执行轨迹：

- 先生成并缓存答案，再撤销用户权限，确认缓存不能绕过当前授权。
- 检索过程中发布新索引，确认原文回查和缓存写入仍绑定请求快照。
- 压缩包含一般规则与例外的证据，确认答案没有扩大适用范围。
- 队列等待消耗大部分预算，确认出队后不会重置截止时间。
- 下游超时且客户端取消，确认运行槽位在工作真正停止后释放。
- 回滚到旧索引时保留当前禁用规则，防止旧版本重新暴露敏感内容。

这些场景不能只检查返回文本。需要核对文档访问、工具调用、版本、计时和资源释放事件。重放涉及外部写操作时使用隔离环境，避免测试产生真实副作用。

## 8 质量门槛与性能门槛

复用 [9 月 30 日的质量指标](../09/2026-09-30-rag-evaluation-regression-tests.md)，分别报告答案正确、引用支持、回答覆盖和合理拒答，而不是将它们压成一个无法解释的分数。

性能评估固定流量构成、输入长度范围与硬件配置，同时保留超时、取消和过载拒绝。只统计成功请求的延迟可能掩盖可用性退化。

发布门槛应在实验前约定，并考虑样本规模与统计不确定性。当前八项结构清单不包含通用准确率或 p95 阈值，因为这些数值必须由具体业务目标确定。

## 9 失败后如何定位与重测

先保存失败案例和运行环境，再沿着 [追踪信息](2026-10-05-rag-observability-trace-diagnostics.md)判断是输入、版本、权限、证据、生成还是资源控制问题。

修复后既重跑失败案例，也重跑受影响路径。例如修复缓存键后，要检查正常命中、跨主体隔离、版本变化和权限撤销，不只验证原来的一个反例。

定期补充独立测试样本。长期围绕固定回归集调参后，它更适合防止已知问题复发，不应继续被视为完全未见数据。

## 10 发布与恢复准备

发布报告绑定实际待发布配置，保存检查清单版本、每项状态、证据引用、遗留风险和回退方案。部署前核对配置，避免测试的是一个版本，上线的是另一个版本。

小流量观察仍要关注逐类失败、敏感数据事件、延迟和合格完成量。未观察到错误不表示所有路径都已覆盖。

回退不仅是切换版本号，还要检查缓存、在途请求与当前权限。已向外部发送的内容或已执行的副作用不会因为回退自动消失。

## 11 今日练习结果

两个 Python 代码块已在本地依次执行，全部断言通过。实际输出：

```text
complete=eligible; authorization failure=blocked
missing recovery=incomplete; failure plus missing=blocked
mixed candidates and duplicate reports rejected
All acceptance gate checks passed.
```

验收汇总器验证报告规则。真实端到端验收还需要将各检查接到实际服务和独立评估数据，不能从合成结果推断生产安全或质量达标。

## 参考资料与复盘入口

- [Google SRE Testing for Reliability](https://sre.google/sre-book/testing-reliability/)。
- [证据组织与引用核验](../09/2026-09-27-rag-retrieval-evidence-citations.md)。
- [长上下文与上下文压缩](2026-10-01-long-context-evidence-compression.md)。
- [提示注入与信任边界](2026-10-02-rag-prompt-injection-trust-boundaries.md)。
- [缓存与失效策略](2026-10-03-rag-cache-invalidation.md)。
- [增量索引与版本发布](2026-10-04-rag-incremental-index-publication.md)。
- [超时预算与降级策略](2026-10-06-rag-deadlines-retries-fallbacks.md)。
- [流量控制与过载保护](2026-10-07-rag-admission-overload-protection.md)。

## 今日总结

1. 验收对象是同一候选配置，而不是拼接各版本的最好成绩。
2. 正常、缓存、降级、重试和恢复路径都需要覆盖。
3. 未测不能默认为通过，关键失败不能被平均分抵消。
4. 规则测试、实际集成测试和模型质量评估各有职责。
5. 发布与回退都要同时考虑版本、权限、缓存和在途请求。

## 下次衔接建议

将验收清单落到一个最小离线 RAG 实验：使用固定小语料、确定性检索基线和问题集，先完成证据召回与引用检查，再逐步接入生成模型。
