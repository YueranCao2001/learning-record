# 2026年11月1日 中文信息抽取微调：训练样例拟合与逐token诊断

## 今日目标

承接 [10月30日双组训练](../10/2026-10-30-sft-repair-training-comparison.md) 的待办，对同一批新6条训练样例做只读检查：比较原候选、control与repair的自由生成和teacher-forced目标loss，先确认训练样例本身是否拟合，再讨论迁移到新表达的问题。

今天实际完成18次GPU生成、18次带正确目标的前向计算、0次训练更新。原候选和control整条正确均为3/6，repair为4/6。修正组仍有两条训练样例错误，因此当前失败不能全部归于新dev迁移不足。

## 核心概念

- 自由生成把自己生成的token继续作为前缀；teacher forcing使用正确目标前缀来计算下一token概率。分歧之后，两者经历的前缀不同，不能把带正确前缀的逐token成绩当作整条生成正确率。
- 沿用原assistant-only监督：prompt标签为-100，目标JSON和EOS计入loss；next-token预测需要将logits与后一位置的label对齐。[Qwen3接口说明](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3)
- loss的均值取决于有效监督位置。忽略位置不参与平均，不能把不同长度样例的loss简单平均后称为全部token的平均loss。[PyTorch 2.7 CrossEntropyLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html)
- 这些是已参与数据修正设计的训练样例。拟合检查用于诊断训练行为，不是新增独立评估，也不代表泛化或通用能力提高。

## 示例与练习

### 1. 固定输入、权重和运行方式

[training-fit-protocol-v1.json](../../../practice/sft-extraction/data/training-fit-protocol-v1.json) 在运行前固定6个ID、三个模型状态及18次生成+18次目标前向的预算。输入、原提示、标签与10月29日训练fixture逐项相同。

| 模型状态 | 权重来源 | 新6条在10月30日训练中的曝光 |
| --- | --- | --- |
| baseline | 原候选HF commit e3b24d11 | 0次 |
| control | 本地control adapter | 每条0次 |
| repair | 本地repair adapter | 每条1次 |

基座固定 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。两组本地adapter先核对10月30日summary记录的文件SHA-256，原候选核对已下载快照。使用既有RTX 5070、PyTorch 2.7.1+cu128、Transformers 4.57.1、PEFT 0.17.1，无新增依赖。

[evaluate_fit.py](../../../practice/sft-extraction/evaluate_fit.py) 复用已有模型加载、fixture、生成与权重哈希函数。三组均为 `eval()`、`inference_mode()`，全部参数不启用梯度，不创建optimizer。原生成函数只选择split=dev，因此仅建立临时推理视图；原训练清单中的split仍为train，没有改写数据角色。

```powershell
# 从 practice/sft-extraction 执行；新run不可覆盖旧证据。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 evaluate_fit.py --run 2026-11-01-training-fit-v1
```

### 2. 核验loss对齐与统计口径

[fit_metrics.py](../../../practice/sft-extraction/fit_metrics.py) 将前一位置logits与后一位置label对应，仅保留label不为-100的目标位置；用FP32 log-softmax取得正确token概率。每组6条合计142个监督token，包含JSON与EOS，不包含prompt。

逐条重算的目标平均负对数概率与模型返回loss比较，最大绝对差约2.15×10⁻⁸，小于预先固定的10⁻⁵容差。本地安装的 `ForCausalLMLoss` 源码也保存为 [运行快照](../../../practice/sft-extraction/results/2026-11-01-training-fit-v1/causal-loss-source.txt)，便于检查版本与移位行为。

主表使用全部目标负对数概率之和除以142，另保留六条样例均值的平均值。两者口径分开，不将长度不同的样例当作相同token权重。

### 3. 整条生成与逐token结果

[诊断汇总](../../../practice/sft-extraction/results/2026-11-01-training-fit-v1/review.json) 从原始输出和逐token记录重新计算：

| 模型状态 | 整条JSON正确 | 正确前缀下目标token为首选 | token加权平均NLL |
| --- | --- | --- | --- |
| baseline | 3/6 | 138/142 | 0.053350 |
| control | 3/6 | 139/142 | 0.044021 |
| repair | 4/6 | 140/142 | 0.027722 |

repair的loss较低、目标token首选更多，确实表现出这6条上的部分拟合变化；但140/142不能替代4/6。少数关键位置的分歧仍会让时间或地点错误。

| 样例 | baseline整条 | control整条 | repair整条 | repair目标平均loss |
| --- | --- | --- | --- | --- |
| repair-train-01 | 错 | 错 | 对 | 0.024828 |
| repair-train-02 | 对 | 对 | 对 | 0.010811 |
| repair-train-03 | 错 | 错 | 错 | 0.080922 |
| repair-train-04 | 对 | 对 | 对 | 0.003505 |
| repair-train-05 | 对 | 对 | 对 | 0.001126 |
| repair-train-06 | 错 | 错 | 错 | 0.054105 |

修正组相对另两组只新增repair-train-01这一条整条修复：时间缺失时，明确地点“鹿鸣工坊”被正确保留。没有新增整条退化。

### 4. 两条训练样例仍未拟合

repair-train-03应提取完整时间“周日上午09:40”，三组都只输出“09:40”。原候选还将location写为“未决定”；control与repair将其改成null，内容有部分修正，但时间仍错，整条分数仍不通过。

repair-train-06明确写有“参加地址：雾松教室”，三组自由生成均为：

```json
{"event":"版画拓印体验课","time":null,"location":null}
```

正确location应为“雾松教室”。这是repair已经实际训练过一次的同一个输入，不能把该失败解释为只有陌生表达才出现的问题。

### 5. 首个token分歧比平均loss更具体

保存的 [repair逐token记录](../../../practice/sft-extraction/results/2026-11-01-training-fit-v1/repair-teacher.json) 将生成与正确目标的首个不同位置对齐。在repair的两条错误中，正确前缀下的观察如下：

| 样例 | 正确目标首分歧token | 模型首选token | 正确token概率 | 首选概率 |
| --- | --- | --- | --- | --- |
| repair-train-03 | `周` | `0` | 0.2873 | 0.6892 |
| repair-train-06 | `":"` | `":` | 0.3485 | 0.6512 |

第一处对应完整星期/时段限定与直接生成时刻的分歧。第二处首先出现在JSON值是否打开字符串引号附近，随后自由生成进入null分支，尚未开始生成“雾松教室”的文字。

表中概率来自正确前缀的整段前向；自由生成记录独立保留，未把分歧后的错误前缀混入teacher-forced统计。观察能定位本次输出的分歧，不能据此证明某个内部语义机制，或保证增加步数必然修好。

### 6. 权重、测试与同步

三组模型在诊断前后完整参数哈希一致。今天没有追加训练，没有重跑新dev或旧回归，也没有把今天的训练样例成绩当作10月30日dev的改进。

[manifest](../../../practice/sft-extraction/results/2026-11-01-training-fit-v1/manifest.json) 绑定脚本、原数据、权重文件身份和协议；原始 [生成结果](../../../practice/sft-extraction/results/2026-11-01-training-fit-v1/repair-generation.json)、目标概率、mask/fixture均保存。110项微调测试、41项原RAG测试通过；新测试覆盖移位、prompt忽略、EOS监督、前缀提前终止、NLL重算、原fixture一致和权重不变。

HF仅同步13个JSON报告，commit [e1e0f0af](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/e1e0f0afd6e6dfaea8fccf0f233c04931a41e343/evaluations/2026-11-01-training-fit-v1) 的回读哈希通过。[同步证据](../../../practice/sft-extraction/results/2026-11-01-training-fit-v1/hf-audit-upload.json) 确认私有状态、根adapter和main不变；完整训练输入、fixture及模型权重不上传HF，参考模型仍为e3b24d11，两个本地训练分叉没有HF发布revision。

## 遇到的问题

10月30日看到新dev没有提升时，还不能区分“训练目标已经学会但没有迁移”和“训练目标本身也未完全拟合”。今天的结果排除了在这6条上已经全部正确这一解释，但尚不能唯一认定是预算、学习率、数据冲突或模型能力造成。

较低平均loss也会掩盖关键字段的token选择错误。今天同时记录整条生成、逐token首选和NLL，不用其中一个指标代替其他指标；这6条又是训练设计样例，没有独立测试含义。

## 今日总结

完成18次生成与18次目标前向的只读诊断。repair将新训练样例整条正确数从3提高到4，但时间限定和明确地点仍各有一处错误。当前既有未完全拟合，也有此前dev上未改善的事实；需要进一步区分原因，不能直接把问题全部归于迁移。

## 下一步

- 将训练曝光/优化预算不足列为待检验假设，先固定追加预算、数据顺序、学习率和停止条件，再同时检查训练拟合、新dev及旧回归；不为把训练集刷对而忽视退化。
- 先复核这两条目标边界与模板，保留其他可能原因；独立人工标注审查、更广泛能力检查和来源独立性仍待完成，不推广当前权重。
