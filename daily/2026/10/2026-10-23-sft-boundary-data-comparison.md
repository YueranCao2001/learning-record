# 2026年10月23日 中文信息抽取微调：边界样例的固定预算对照

## 今日目标

接着 [昨天的首次 LoRA 实验](2026-10-22-sft-extraction-contract-and-preflight.md)，处理两类具体错误：时间在句首时误当活动名，以及把“欢迎参加”一并提取进活动名。今天不增加训练轮数，先固定样例替换方案、评估题目和停止条件，再比较真实 GPU 训练结果。

今天的问题是：相同六步预算下，将旧样例的第二轮重复替换为三条边界样例，生成会怎样变化？不是验证“增加数据必然提升”，也不将候选直接推广到默认版本。

## 核心概念

- **对照组保留旧方案。** 两组都从同一基座和相同初始 adapter 开始，而非在昨天的 adapter 上继续训练；避免把继续训练与数据修改混在一起。
- **步骤相同不等于 token 预算相同。** 新旧样例长度、曝光次数和最后出现的样例一起改变；结果只能归属于本次有序样例替换方案，不能唯一归因于某一种机制。
- **已知回归题与新增诊断题分开报告。** 新增题也根据已知错误设计，标签公开、句式有重叠，不是封存测试集。9/9 不代表真实业务准确率。
- **模型可同步，不代表应替换默认版本。** 本次用独立 HF 实验分支保存候选，保留旧 `main` 和历史 commit。

## 示例与练习

### 1. 固定数据、分组与协议

保持 [pilot-v1.json](../../../practice/sft-extraction/data/pilot-v1.json) 六条原记录及 instruction 不变，在 [新增样例](../../../practice/sft-extraction/data/boundary-additions-v2.json) 中加入3条 train 和6条 dev。合成数据均为助手编写的虚构活动通知，无用户私人数据。

训练新增例子之一：

```text
欢迎参加音乐工作坊。时间和地点稍后公布。
```

其预期目标由助手编写：

```json
{"event":"音乐工作坊","time":null,"location":null}
```

另两条分别覆盖“周五18:30在某地点举办活动”和“欢迎参加活动、提供时间、地点待通知”。新增 dev 使用不同来源通知及活动名，包括时间在句首、欢迎语、报名说明和地点在句首。共6条可用 train、9条可见 dev；来源组不跨 split，但没有把相同模板视为独立分布，也没有独立人工审查或封存真实测试集。

[预先固定的协议](../../../practice/sft-extraction/data/boundary-protocol-v2.json) 在模型调用前写入。只执行基座生成一次、两组各六步，不看到结果后追加训练或修改标签。

| 步骤 | 对照组 | 候选组 |
| --- | --- | --- |
| 1—3 | 旧 train-01、02、03 | 同左 |
| 4—6 | 再访问旧 train-01、02、03 | 新 boundary-train-01、02、03 |

### 2. 实际环境与运行

沿用昨天的独立 `.venv-sft`、RTX 5070、PyTorch 2.7.1+cu128、Transformers 4.57.1、PEFT 0.17.1；`pip check` 无冲突，没有安装新依赖。

基座固定为 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。两组均 batch=1、seed=42、AdamW、lr=0.0002、weight_decay=0、梯度裁剪1.0；LoRA rank=8、alpha=16、dropout=0，只训练 q_proj/v_proj。保持 BF16、非思考模板、assistant completion+EOS 监督、packing=False、最长512 tokens、不截断；生成使用 greedy、最多128个新 tokens。

[对照入口](../../../practice/sft-extraction/boundary_experiment.py) 复用昨天的模型加载、模板、mask 和生成函数，旧脚本及历史结果保留。数据验证提取为共享函数；没有修改评分语义。[manifest](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/manifest.json) 绑定数据、协议及代码指纹，完整合成数据快照也已保留。

```powershell
# 从 practice/sft-extraction 执行。此次运行已完成，复现须换新 run 名。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 boundary_experiment.py --run 2026-10-23-boundary-v2
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 review_boundary.py --run 2026-10-23-boundary-v2
```

两组各六步的 loss 和梯度范数均有限；前3步 loss 完全相同，对照组完整六步 loss 也重现了昨天的记录。两组初始 adapter SHA-256 相同，冻结基座参数 SHA-256 前后不变，各112个 adapter 张量更新。两组均完成本地保存重载，adapter 张量和生成 token 序列一致。[对照摘要](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/control/summary.json)、[候选摘要](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/candidate/summary.json) 保留完整证据。

监督 target tokens 总数为对照146、候选136，说明匹配的是 optimizer steps，不是 token 数或训练算力。两组都没有验证精确续训。

### 3. 逐题结果与错误修正

基座结果在第一次训练更新前保存，未修改 instruction、标签、生成配置或评分规则。[重算审查](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/review.json) 支持下面的整条精确匹配计数：

| 可见诊断子集 | 基座 | 对照组 | 边界候选 |
| --- | --- | --- | --- |
| 昨天3条回归题 | 2/3 | 1/3 | 3/3 |
| 新增6条诊断题 | 3/6 | 2/6 | 6/6 |
| 合计 | 5/9 | 3/9 | 9/9 |

对照组在旧3题的生成 token 与昨天完全相同，已知错误被重现，而非换了题后忽略旧退化。

| 已知问题 | 对照组输出 | 候选组输出 |
| --- | --- | --- |
| dev-02 活动名 | `周六15:00` | `手工沙龙` |
| dev-03 活动名 | `欢迎参加写作工坊` | `写作工坊` |

基座在新增“周日10:30…”题只提取了 `10:30`，在地点先出现的题遗漏了时间；候选在本次诊断中完整提取。所有三组的 JSON 与字段契约均通过，但事实分数不同；逐字段正确数为基座22/27、对照21/27、候选27/27。

原始输出：[基座](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/baseline.json)、[对照](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/control/after.json)、[候选](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/candidate/after.json)。当前证据支持“本次定向样例替换在所选题上修复边界错误”，不支持模型已普遍可靠，也不能排除模板重叠和样例顺序的作用。

### 4. HF 实验分支同步

使用私有仓库的 `boundary-v2-20261023` 分支保存候选，不覆盖 `main`。上传14个白名单文件，包括 adapter、必要 tokenizer/config、许可、模型卡和诊断评估，不上传训练文本、Token、基座权重或 optimizer 状态。模型卡明确了9/9的诊断性质和预算差异。

本次远程 commit：[e3b24d11](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/e3b24d1115c21cf69feaa5e6e9cf85a98f502c84)。按完整 commit 下载后，14个文件哈希一致，加载下载的 adapter 后，9条 greedy token 序列与本地候选完全一致。`main` 在发布前后均为昨天的 `def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f`，仓库仍为私有。[同步证据](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/candidate/upload.json) 和 [远程回读输出](../../../practice/sft-extraction/results/2026-10-23-boundary-v2/candidate/remote-reloaded.json) 已保存。

```powershell
& 'E:\Evren\Evren_work\.venv-hf\Scripts\python.exe' -X utf8 publish.py --run 2026-10-23-boundary-v2 --arm candidate --revision boundary-v2-20261023 --model-card boundary-model-card.md --execute
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 verify_remote.py --run 2026-10-23-boundary-v2 --arm candidate
```

## 遇到的问题

昨天的 loss 降低没有阻止生成退化；今天保留原对照，证明不能把 JSON 格式收敛当作事实抽取改善。对照组甚至在一条题中生成字符串 `"null"`，它与 JSON `null` 不同，字段结构通过仍会被事实匹配判错。

新增样例覆盖了旧训练集中欠缺的边界，但同时改变了曝光次数、最后三步内容和 target token 总数。因此不能写成“已证明某个单独样例或位置导致修复”。这些句式来自对旧错误的观察，不能再把相关 dev 分数包装成未见过的独立测试结果。

运行仍有 `torch_dtype` 弃用提示、greedy 下采样配置被忽略提示和 Windows 缓存 symlink 提示，未导致实验失败；没有为消除提示更换环境、模板或系统设置。

## 今日总结

完成了旧错误复现、新增合成样例及固定预算协议、两组各六步真实 GPU 训练、分组评分、本地重载，以及私有 HF 实验分支上传和回读验证。23项微调项目本地测试通过，包括数据分组、预算与评估隔离、指纹、评分重算、mask 及保存的同步证据检查；单元测试不代替实际模型和网络验证。

本次候选在9条可见诊断题上全部匹配，但仍仅作为实验分支保留。没有额外追加步骤、挑选更好 checkpoint 或宣称生产可用。

## 下一步

- 先写清活动名与时间原文边界的标注规则，再准备没有参与本轮问题设计的独立人工审查评估集；现有9条继续作为公开回归集，不再称独立测试。
- 固定新评估协议后，检查更广泛句式和基础能力保留；需要归因时再做匹配 token 预算、调整样例顺序或多 seed 的对照，一次回答一个问题。
- 精确训练续跑和 optimizer/RNG 状态恢复仍是未完成项，今天的保存加载只验收推理一致性。
