# 2026年11月2日 中文信息抽取微调：训练预算对照与拟合回归

## 今日目标

承接 [昨天的拟合诊断](2026-11-01-sft-training-fit-diagnostic.md)，检验较少优化预算是否限制了那两条训练样例的拟合。固定数据、混合顺序、初始adapter和学习率，比较12步与36步两个预算端点，同时检查新6条训练样例、新12条dev和45条旧回归。

今天实际执行36次优化器更新、189次GPU生成、12次目标前向。36步模型的新训练样例从4/6变为6/6，新dev从8/12变为11/12，旧回归从42/45变为44/45；在当前题集未新增退化。结果支持本次预算假设，但不是最佳步数或通用泛化结论。

## 核心概念

- 重复相同混合序列以增加总预算，旧、新训练样例都增加曝光。它检验的是整套优化预算，不是仅增加修正样例曝光的效果。
- 不从旧12步权重重建优化器再追加，而是从相同初始adapter运行完整36步，周期之间保持同一个AdamW状态，避免把优化器重置混入对照。
- 拟合通过与新表达可靠性分别验收。训练样例6/6不能代替dev/回归结果，也不与dev合并成一个质量准确率。
- 只测12和36两个端点，不能得出24步已足够、质量单调增长或36步最优；停止条件应先于结果固定。

## 示例与练习

### 1. 预先固定预算和停止条件

[optimization-budget-v1.json](../../../practice/sft-extraction/data/optimization-budget-v1.json) 在执行前固定预算、评估范围和停止条件。原12条混合序列为旧6条各一次，再接新6条各一次；36步将该序列完整重复三次，不重新排列或按失败题选样。

| 端点 | 权重及执行方式 | 每条旧/新训练样例曝光 | 监督token |
| --- | --- | --- | --- |
| 12步 | 复用10月30日repair adapter，只读复现 | 各1次 | 278 |
| 36步 | 从相同原候选adapter新建AdamW，连续36步 | 各3次 | 834 |

36步轨迹包含今天重新执行的前12步，今天新执行的更新数为36。没有在训练中看分数后早停、追加周期或修改标签。

基座仍为 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`，初始adapter为HF commit `e3b24d1115c21cf69feaa5e6e9cf85a98f502c84`。LoRA rank=8/alpha=16/dropout=0、q_proj/v_proj，BF16基座、batch=1，无累积/packing；AdamW lr=0.00002、weight_decay=0、常量LR、裁剪1.0、seed=42，均沿用原配置。

对昨天两条未通过样例重新核对，原文中“周日上午09:40”和“雾松教室”仍为明确连续片段，保持原标签；这项复核不替代独立人工标注审查。

### 2. 复用冻结训练函数与验证参考端点

[budget_experiment.py](../../../practice/sft-extraction/budget_experiment.py) 复用10月30日的 `train_arm()`、原生成与保存重载函数，并复用昨天的teacher-forced诊断，没有再创建另一套训练循环。数据快照、提示、旧源码和历史权重保持不变。

```powershell
# 从 practice/sft-extraction 执行。复现须换新run名，不覆盖旧证据。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 budget_experiment.py --run 2026-11-02-optimization-budget-v1
```

先核对旧12步adapter文件SHA，再对6条训练样例+12条新dev+45条旧回归生成63条输出，全部逐token复现10月30日及11月1日记录。参考权重前后哈希不变。

36步模型的初始adapter指纹与原12步实验相同，前12条训练记录逐项一致，包括样例、loss、梯度范数、LR和目标token数。没有保存并直接比较36步轨迹在step 12时的参数张量，因此不把日志相同扩大为一次新的中途参数一致性证明。

36步loss和梯度均有限，冻结基座哈希不变。AdamW初始状态为空，最终参数步数均为36，周期之间未重建优化器；112个adapter张量发生改变。沿用RTX 5070和既有训练环境，无新增依赖。

### 3. 拟合、dev和回归分别记录

[review_budget.py](../../../practice/sft-extraction/review_budget.py) 复用现有严格评分、字段错误和配对变化函数，从原始输出重算 [完整汇总](../../../practice/sft-extraction/results/2026-11-02-optimization-budget-v1/review.json)。

| 端点 | 新6条训练样例 | 新12条dev | 45条历史回归 |
| --- | --- | --- | --- |
| 12步 | 4/6 | 8/12 | 42/45 |
| 36步 | 6/6 | 11/12 | 44/45 |

两端点的全部63条输出都通过JSON及字段类型检查，错误差异来自抽取内容。训练样例分数单列，不与后两部分合并报告为质量准确率。

新6条训练样例的目标前向结果为：

| 端点 | 目标token为首选 | token加权平均NLL |
| --- | --- | --- |
| 12步 | 140/142 | 0.027722 |
| 36步 | 142/142 | 0.005093 |

两组各6次目标前向，共12次；均在eval/inference模式下执行，不额外训练。每组固定142个JSON+EOS监督位置，prompt不计入。36步的自由生成也与这6条规范目标token一致，而不是仅平均loss下降。

### 4. 两条训练错误已修复

36步模型对repair-train-03输出：

```json
{"event":"古琴赏析课","time":"周日上午09:40","location":null}
```

它保留了12步时丢失的星期与上午限定。repair-train-06的明确地点也恢复：

```json
{"event":"版画拓印体验课","time":null,"location":"雾松教室"}
```

这支持“在当前初始化、学习率与混合序列下，增加预算能修复这两条拟合不足”的判断。旧、新训练样例同时重复，只有一个seed，也没有中间预算点评估，不能唯一归因于某条样例或认定达到最佳预算。

### 5. 新dev改善与仍未解决的问题

| 分组 | 相对12步新增修复 | 新增退化 |
| --- | --- | --- |
| 新训练样例 | repair-train-03、repair-train-06 | 无 |
| 新dev | repair-dev-02-missing-time、repair-dev-03-missing-time、repair-dev-03-missing-location | 无 |
| 旧回归 | audit-03、pair-01-missing | 无 |

新dev中，“白帆艺坊”“海棠实验间”的明确地点恢复，完整时间中的“上午”不再丢失。旧audit-03的“潮汐教室”和pair-01-missing的“流萤教室”也恢复。

仍有两条未提供地点的输入被错误填值：

- 新dev的repair-dev-01-missing-location仍输出location为“集合处”，正确值应为null。
- 旧confirm-03-missing-location仍输出“相聚之处还没有定下”，正确值应为null。

当前新训练样例拟合通过，部分未定说明的语义抽取仍不可靠。不能因为6/6就忽略剩余错误，也不能因本轮可见题集没有退化就声称更广泛能力安全。

### 6. 工程验证、计数与同步

189次生成分别为：12步参考63次、36步训练后63次、36步保存重载63次；12次目标前向另计。36步adapter重建基座后加载，张量与逐题生成token均与保存前一致。这里验证推理重载，没有进行本轮精确续训。

[训练日志](../../../practice/sft-extraction/results/2026-11-02-optimization-budget-v1/extended/training.json)、[前缀比较](../../../practice/sft-extraction/results/2026-11-02-optimization-budget-v1/prefix-comparison.json) 与 [manifest](../../../practice/sft-extraction/results/2026-11-02-optimization-budget-v1/manifest.json) 已保存。118项微调测试、41项原RAG测试通过，覆盖预算重复、历史数据、36步日志、目标token、参考复现、NLL重算和重载输出。

HF仅追加16个JSON报告，commit [a7a2d311](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/a7a2d311bac0163fa5c6cbab629b16599dd93c64/evaluations/2026-11-02-optimization-budget-v1) 下载哈希通过。[同步证据](../../../practice/sft-extraction/results/2026-11-02-optimization-budget-v1/hf-audit-upload.json) 确认仓库仍私有、根adapter及main不变。新adapter保存在本地被忽略的 `artifacts/2026-11-02-optimization-budget-v1/extended/`，没有替换已发布模型或默认提示。

## 遇到的问题

昨天未完全拟合时，不能把全部失败归为迁移不足；今天增加预算后拟合达到6/6，且dev/回归有改善，说明较少预算确实限制了本次结果。但剩余未定说明误抽取没有消失，需要区分训练拟合与语义覆盖边界。

已有所有题目和标签都可见，曾用于提出修正与预算假设。它们保持开放诊断身份；来源分组与同一助手编写的模板族名称不能证明独立性，人工审查仍为pending。

## 今日总结

完成一次预先固定的12/36步预算对照。今天实际36次更新、189次生成及12次目标前向；新训练样例6/6、新dev11/12、旧回归44/45，当前未见新增退化。训练拟合与部分诊断结果改善，但仍有两处未定语义错误，不推广为通用修复或默认模型。

## 下一步

- 冻结36步结果，不立即继续加步数。围绕两条剩余的缺失语义误抽取，整理标注复核与来源覆盖清单，先确定需要检验的新表达边界，再决定是否改训练内容。
- 阶段复盘中分别报告预算收益、未修复项与独立性限制；推进独立人工复核或不同来源的小型评估，不将当前可见题集重新包装成独立测试。
