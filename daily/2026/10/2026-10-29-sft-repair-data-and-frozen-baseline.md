# 2026年10月29日 中文信息抽取微调：修正数据分组与训练前基线

## 今日目标

接着 [昨天的确认实验](2026-10-28-sft-missing-field-confirmation.md)，停止围绕同一批可见题目继续改提示，整理拟用于下一轮训练的数据修正方案。今天新增6条拟训练样例、12条新dev，核查与历史诊断的来源隔离，固定两组采样顺序，并实际冻结训练前基线。

完成57次GPU生成、0次优化器更新。当前候选在新dev上为8/12、历史回归为40/45；这是训练前状态，尚无修正后的模型或质量增益。

## 核心概念

- 错误类型可以指导新数据设计，原失败题仍保留为诊断回归。把旧dev改名、改split后放入训练，不能再称其为未训练过的评估来源。
- 来源ID、原文重复和人工标记的模板族是不同层次的检查。ID和模板族名称不重叠，只能说明元数据约束通过，不能证明语义独立或消除了模板相似。
- 从adapter权重开始、为两组分别创建新优化器，是新的训练分叉；它不恢复旧optimizer/RNG轨迹，不能叫作 [已经验证过的精确续训](2026-10-26-sft-exact-training-resume.md)。
- 相同步数并不等于相同监督token预算。今天先用实际tokenizer核算，不等训练结束后才发现两组曝光、长度与顺序同时变化。

## 示例与练习

### 1. 根据三个错误方向编写六条新训练样例

[repair-data-v1.json](../../../practice/sft-extraction/data/repair-data-v1.json) 保存全部新增输入和标签。样例仍由助手编写，均为虚构活动；未更改既有标注规则，人工复核保持pending。

| 新来源组 | 拟训练样例 | 主要检查 |
| --- | --- | --- |
| notice-36 | 草木染入门课，完整/缺时间各一条 | 时间缺失时保留明确地点；完整时间片段 |
| notice-37 | 古琴赏析课，缺地点/两者都缺各一条 | “还没有决定”应对应null |
| notice-38 | 带“报名提醒”标题的版画拓印体验课，完整/缺时间各一条 | 活动名不取标题；时间缺失不丢明确地址 |

例如拟训练输入：

```text
本次开设的是古琴赏析课，定于周日上午09:40举行。在哪里上课还没有决定，报名仍然开放。
```

对应目标：

```json
{"event":"古琴赏析课","time":"周日上午09:40","location":null}
```

这里不是把“在哪里上课还没有决定”作为地点值，目标依然遵循 [标注规则](../../../practice/sft-extraction/annotation-rules-v1.md)。六条拟训练样例为2条完整、2条缺时间、1条缺地点、1条两者都缺，未声称训练分布完全均衡。

### 2. 新dev与旧回归的角色保持分开

新增dev使用另三个来源notice-39至notice-41：香囊缝制课的参与者消息、皮影制作课的课程目录、珐琅配色体验课的消息摘要。每组都覆盖完整、缺时间、缺地点、两者都缺四种状态，共12条，每种状态3条。

[repair_data.py](../../../practice/sft-extraction/repair_data.py) 按首次出现ID整理历史数据，并比较重复ID的原文、来源、split和标签，避免把同一条回放重复计数。最终清单为：

| 角色 | 样例数 | 允许进入后续采样计划 |
| --- | --- | --- |
| 旧train | 6 | 是 |
| 新拟训练样例 | 6 | 是 |
| 新dev | 12 | 否 |
| 历史诊断回归 | 45 | 否 |

共69条记录、41个来源组。新train来源为notice-36/37/38，新dev来源为notice-39/40/41；历史原文和标签不变。历史回归包含旧边界dev、审计、配对与确认题，已经是开放诊断，不能因本次重新整理而恢复“独立测试”的身份。

新train与新dev采用不同的人工模板族名称，实际表述仍可能与旧数据相似；没有进行独立语义去重或人类审查。每种新dev表达也只有一个活动来源。

### 3. 七类错误注入验证拒绝逻辑

以下操作均作用于内存中的临时数据/协议副本，原文件不改，不创建模型或优化器：

| 注入错误 | 实际拒绝原因 |
| --- | --- |
| 将历史dev改为train | 历史记录被修改 |
| 新train复用历史来源组 | 来源组被历史数据保留 |
| 新dev复用新train来源组 | 来源组跨split |
| 新dev复用新train模板族名 | 新模板族名跨split |
| 新dev复制新train原文 | 重复文本 |
| 标签写入原文不存在的场地 | 非原文片段标签 |
| 将dev ID放入修正组采样顺序 | 非train记录进入计划 |

[七类拒绝记录](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/rejections.json) 均为拒绝、0更新。[分组审计](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/split-audit.json) 同时保留 `semantic_independence_proven=false`。这些断言防止已定义的错分方式，不是对数据语义质量的自动认证。

### 4. 固定下一轮对照方案并核算实际token

[repair-protocol-v1.json](../../../practice/sft-extraction/data/repair-protocol-v1.json) 在本次基线生成前保存。后续两组都拟从同一个已发布候选adapter开始，分别重建AdamW状态；保持原提示，不追加昨天的规则。

| 计划组别 | 前6步 | 后6步 | 完整训练序列token合计 | 监督token合计 |
| --- | --- | --- | --- | --- |
| control | 旧6条各一次 | 旧6条再各一次 | 1202 | 272 |
| repair | 旧6条各一次 | 新6条各一次 | 1293 | 278 |

完整序列计入prompt、目标和结束标记；监督计数仅含目标JSON与EOS。两组各12步，但token数、样例曝光、内容及末尾顺序并不等价，因此只能比较这套数据替换方案，不能把差异唯一归因于某条语义规则。

计划使用batch=1、无累积/packing、BF16、原LoRA rank=8/alpha=16/dropout=0、q_proj/v_proj；新AdamW lr=0.00002、weight_decay=0、常量LR、裁剪1.0、seed=42。学习率是待验证的固定试验值，不声称最优。**两组训练今天均未执行**，协议状态为 `prepared-not-trained`。

实际调用已有 `fixture()` 检查12条可训练记录的非思考模板、assistant-only mask、JSON目标、EOS与512长度限制，最长132 tokens，没有截断。[逐token监督记录](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/train-token-fixtures.json) 和 [逐步预算](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/token-budget.json) 已保存；这不等于执行了反向传播。

### 5. 真实GPU训练前基线

使用固定基座 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`，加载原候选adapter commit `e3b24d1115c21cf69feaa5e6e9cf85a98f502c84`，加载前核对下载文件哈希。沿用RTX 5070、PyTorch 2.7.1+cu128、Transformers 4.57.1与PEFT 0.17.1，无新依赖。

```powershell
# 从 practice/sft-extraction 执行。复现须换新run名。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 prepare_repair.py --run 2026-10-29-repair-preparation-v1
```

该入口先验证数据和token，再用原生成函数执行新12题+旧45题，一共57次贪心生成，`max_new_tokens=128`，没有创建optimizer。生成前后模型参数哈希一致。

| 分组 | 整条匹配 | JSON与字段契约通过 | 备注 |
| --- | --- | --- | --- |
| 新dev | 8/12 | 12/12 | 四条内容错误保留 |
| 历史回归 | 40/45 | 45/45 | 45条逐token重现原记录 |

新dev的四条失败为：

- 参与者消息缺地点时，候选将“集合处”当作地点；正确值为null。
- 课程目录缺时间时，明确地点“白帆艺坊”被遗漏成null。
- 消息摘要缺时间时，明确地点“海棠实验间”被遗漏成null。
- 消息摘要缺地点时，时间“12月9日上午08:50”被写成“12月9日08:50”，丢掉“上午”。

[基线原始输出](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/baseline.json) 与 [汇总](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/review.json) 已冻结；没有在看到这些结果后修改数据、标签或计划。这12题已经开放，后续也不能称封存测试。

### 6. 验证与同步

[manifest](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/manifest.json) 绑定数据快照、协议、源码与基座/adapter身份。95项微调测试、41项原RAG测试通过；新测试重算分组、拒绝、预算、mask、评分与回放，同时检查保存文件指纹及发布范围。

HF仅追加9个准备与基线JSON，commit [84e1d802](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/84e1d80271bc70d18de747bf088059530f7ee11f/evaluations/2026-10-29-repair-preparation-v1) 下载哈希通过。[同步证据](../../../practice/sft-extraction/results/2026-10-29-repair-preparation-v1/hf-audit-upload.json) 确认仓库仍私有、根adapter和main不变；完整数据及训练token记录不上传HF，代码与合成数据随GitHub学习仓库保存。

## 遇到的问题

以前只说按来源组切分，不能阻止把已见诊断题重新标成train。今天将历史内容固定、诊断采样禁入和新增来源检查串起来，补上了可执行拒绝，但仍不能自动判断两个不同模板名是否实际相似。

两组步数一致而token不一致，是实际核算结果；不为制造等预算表象修改样例长度或监督目标。新dev基线仍有明确遗漏与原文片段边界错误，说明计划数据是否有效需要后续训练对照，不能从标签检查通过推导出来。

## 今日总结

完成6条新拟训练样例、12条新dev、45条历史回归清单、七类错误拒绝、真实token预算和57题GPU训练前基线。今天的实际进展是可追溯的数据准备与对照起点，训练步骤和质量改进均未发生。

## 下一步

- 以这份已固定协议实现一次双组权重初始化训练，复用现有模型、模板与评分函数，在新run中保存两组相同初始adapter、训练日志、冻结基座和重载结果。执行前明确完整生成/重载预算；不覆盖本次准备报告。
- 逐题比较新dev与历史回归的修复和退化，不因某个总分变好就替换原权重。人工标注审查继续pending，也不将新dev或历史诊断重新并入训练。
