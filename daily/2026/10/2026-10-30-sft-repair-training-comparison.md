# 2026年10月30日 中文信息抽取微调：修正数据的双组训练与逐题回归

## 今日目标

执行 [昨天固定的数据修正方案](2026-10-29-sft-repair-data-and-frozen-baseline.md)：两组从同一候选adapter权重开始，分别新建优化器，各训练12步，对比重复旧样例与替换部分训练数据的结果。今天实际完成24次优化器更新、285次GPU生成，包括训练前基线复现和两组保存重载验证。

控制组为新dev9/12、旧回归41/45；修正组为8/12、42/45。两组总数都是50/57，收益却落在不同题目上。修正组没有在新dev上超过控制组，今天不推广任一组。

## 核心概念

- 控制组用于区分“额外训练本身的变化”和“替换部分训练数据的变化”。若只与原基线比较，就会把两组共同修好的题也算成新数据的独有贡献。
- 两组分别加载相同adapter，再创建空状态的AdamW；不恢复以前的optimizer、scheduler或RNG轨迹，因此是新的训练分叉，不称精确续训。
- 匹配12步并不匹配监督token、样例曝光或训练顺序。本次沿用272与278个监督token的差异，解释范围是整套数据替换方案，不是某一条样例的独立因果效果。
- 总分受题集构成影响。45条旧回归比12条新dev多，不能用相同总数掩盖两部分的取舍；逐题修复/退化与字段指标继续分别记录。

## 示例与练习

### 1. 固定训练配置与完整执行预算

[repair-execution-v1.json](../../../practice/sft-extraction/data/repair-execution-v1.json) 在运行前记录今天的执行顺序和预算；训练配置仍使用昨天的 [repair-protocol-v1.json](../../../practice/sft-extraction/data/repair-protocol-v1.json)，没有改提示、标签、学习率或步数。新run中的 `source-protocol.json` 是准备阶段原样快照，其中的0更新描述昨天；今天实际更新以训练日志和review为准。

固定基座 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`，初始adapter为HF commit `e3b24d1115c21cf69feaa5e6e9cf85a98f502c84`。每组加载前验证下载文件哈希，使用RTX 5070、PyTorch 2.7.1+cu128、Transformers 4.57.1、PEFT 0.17.1，无新增依赖。

配置为BF16基座、LoRA rank=8/alpha=16/dropout=0、q_proj/v_proj、batch=1，无累积/packing；AdamW lr=0.00002、weight_decay=0、foreach/fused关闭，常量学习率、裁剪1.0，每组训练前seed=42。

| 组别 | 前6步 | 后6步 | 实际更新 | 监督token |
| --- | --- | --- | --- | --- |
| control | 旧6条各一次 | 旧6条再各一次 | 12 | 272 |
| repair | 旧6条各一次 | 新6条各一次 | 12 | 278 |

全部12条可训练记录的模板和监督fixture与昨天一致；新12条dev和旧45条回归均未进入训练采样。

生成预算为：原候选基线57次、control训练后57次、control重载57次、repair训练后57次、repair重载57次，合计285次。每次沿用原提示、非思考模板、贪心生成和 `max_new_tokens=128`，没有看到结果后调参重跑。

### 2. 复用已有工程函数并记录训练过程

新增 [repair_experiment.py](../../../practice/sft-extraction/repair_experiment.py) 负责加载已有adapter并执行上述分叉；继续复用原模型加载、fixture、生成、基座哈希，以及 `boundary_experiment.reload_arm()` 保存重载检查，不改写历史实验源码。

```powershell
# 从 practice/sft-extraction 执行。已有run不可覆盖，复现须换新名称。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 repair_experiment.py --run 2026-10-30-repair-comparison-v1
```

两组训练前optimizer state均为空，初始adapter和冻结基座指纹相同。两组前6步的样例、loss、梯度范数、LR和监督token记录完全一致，证明实际共同前缀一致；第7步开始才按协议分叉。

全部24步loss/梯度有限，基座梯度保持为空，基座参数哈希在训练前后相同。每组1,146,880个可训练参数、112个adapter张量发生改变，最终AdamW参数步数均为12。

[控制组日志](../../../practice/sft-extraction/results/2026-10-30-repair-comparison-v1/control/training.json) 与 [修正组日志](../../../practice/sft-extraction/results/2026-10-30-repair-comparison-v1/repair/training.json) 保留原始数值。修正组新样例有较高的更新前loss和裁剪前梯度范数，但不同输入的loss不能直接作为两组质量排名，也不能据此判定训练目标最终是否学会。

### 3. 新dev与历史回归分开比较

[review_repair.py](../../../practice/sft-extraction/review_repair.py) 复用原严格评分与字段分析函数，从原始输出重新汇总。[完整结果](../../../practice/sft-extraction/results/2026-10-30-repair-comparison-v1/review.json) 同时保存baseline到两组、以及control到repair的逐题配对变化。

| 模型状态 | 新dev整条匹配 | 旧回归整条匹配 | 合计 | JSON与字段契约通过 |
| --- | --- | --- | --- | --- |
| 原候选基线 | 8/12 | 40/45 | 48/57 | 57/57 |
| control | 9/12 | 41/45 | 50/57 | 57/57 |
| repair | 8/12 | 42/45 | 50/57 | 57/57 |

所有输出均通过格式与字段类型检查，当前差异来自内容值，而不是围栏或额外解释。新dev中的event均为12/12、time均为11/12；location依次为9/12、10/12、9/12。

新dev按提供状态比较，每列分母为3：

| 模型状态 | 完整 | 缺时间 | 缺地点 | 两者都缺 |
| --- | --- | --- | --- | --- |
| 原候选 | 3/3 | 1/3 | 1/3 | 3/3 |
| control | 3/3 | 1/3 | 2/3 | 3/3 |
| repair | 3/3 | 1/3 | 1/3 | 3/3 |

### 4. 区分共同修复、修正组独有收益和相对退化

| 对照 | 新dev变化 | 旧回归变化 |
| --- | --- | --- |
| 原候选→control | 修复repair-dev-01-missing-location，无新增退化 | 修复pair-02-missing，无新增退化 |
| 原候选→repair | 没有整条修复或新增退化 | 修复pair-02-missing与confirm-04-missing-time，无新增退化 |
| control→repair | repair-dev-01-missing-location相对退化 | confirm-04-missing-time多修复一题 |

pair-02-missing的明确地点“溪石工坊”在两组中都恢复，不能全部归功于新训练样例。修正组额外修好报名卡缺时间时的地点“榛果工作间”，但新dev上仍保留原来四条失败。

新dev中，参与者消息只有“集合处等待后续消息”，正确location是null。两组实际输出分别为：

```json
{"event":"香囊缝制课","time":"12月7日16:30","location":null}
```

```json
{"event":"香囊缝制课","time":"12月7日16:30","location":"集合处"}
```

第一份为control，第二份为repair。后者相对control更差，但相对原候选是旧错误未修好，不能混称为“修正组相对基线新增了错误”。

仍未解决的错误包括：旧audit-03遗漏“潮汐教室”、pair-01-missing遗漏“流萤教室”，以及把“相聚之处还没有定下”当地点。新dev中“白帆艺坊”“海棠实验间”的遗漏和时间片段丢失也仍在。六条修正样例没有形成通用修复。

### 5. 保存重载与发布范围

今天重新生成的57条基线全部逐token复现昨天结果。两组训练后均保存adapter、重建基座再加载，逐张量相等且57题生成token全部一致，评分也可重算。这是推理重载验证，没有保存或测试本轮训练的精确续训状态。

两组权重保存在被Git忽略的本地 `artifacts/2026-10-30-repair-comparison-v1/control/` 与 `repair/`，文件指纹记录于各组summary。它们没有替换已发布候选，也没有上传为HF新权重。

HF只追加16个训练日志与诊断JSON，commit [efb96781](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/efb967816bbd663868e908913c971a8d3ae55da2/evaluations/2026-10-30-repair-comparison-v1)。[回读证据](../../../practice/sft-extraction/results/2026-10-30-repair-comparison-v1/hf-audit-upload.json) 确认全部文档哈希一致、私有状态及根adapter/main不变。`tested_adapter_revision=null` 区分本地新训练分叉与HF原参考权重，参考仍是e3b24d11。

[manifest](../../../practice/sft-extraction/results/2026-10-30-repair-comparison-v1/manifest.json) 绑定代码、数据、执行协议和准备阶段指纹。103项微调测试、41项原RAG测试通过；新增测试覆盖同总分下的修复/退化、原始评分重算、24步训练身份与预算、初始化、基座冻结、基线及重载输出、远程报告哈希。

## 遇到的问题

若只看原候选48/57到修正组50/57，会忽略控制组也达到50/57；若只看旧回归，修正组更好，又会忽略新dev少修复了一题。今天的结论必须保留这两层比较。

本次单seed、少量合成数据、所有dev标签可见，且替换数据同时改变了内容、曝光与末尾顺序；不能唯一解释哪条训练样例导致哪项变化，也不能称泛化或通用能力保留验收。独立人工审查仍为pending。

## 今日总结

完成一次真实双组LoRA训练与保存重载：24步更新、285次生成，工程检查全部通过。修正组改善部分旧地点遗漏，新dev却未优于重复旧数据的控制组；两组总分相同且收益不同，因此保留结果、不推广权重或默认提示。

## 下一步

- 对两组本地adapter做只读的训练集拟合检查：固定新6条样例，记录逐题生成与训练后teacher-forced loss，区分训练目标本身尚未稳定拟合与迁移到新表达不足。检查前固定协议，不改权重，不据更新前loss直接下结论。
- 结合逐题差异再提出一项可区分的假设，不立即扩大训练轮数或继续对同一dev调参；独立人工标注审查和更广泛能力评估继续保留。
