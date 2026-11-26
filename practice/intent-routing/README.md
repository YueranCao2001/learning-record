# 客服意图路由微调计划

计划周期为 2026 年 11 月 30 日至 12 月 4 日。沿用本机单卡训练和日期笔记流程，新项目用公开的英语客服查询做一次可复现的 QLoRA 对照。要回答的问题是，在相同的提示和解码条件下，adapter 能否提升银行与信用卡意图识别，同时保留对范围外问题的拒识能力。周五交付可从 Hugging Face 下载重载的私有实验版本，以及 GitHub 上指向同一版本的代码、配置和评估记录。

## 任务与数据

输入是一条用户查询，输出是只含 `intent` 键的 JSON。目标范围包括查询余额、查询账单、转账、报告欺诈等 11 个意图。范围外返回 `out_of_scope`。例如，`I lost my card yesterday.` 对应 `{"intent":"report_lost_card"}`。

意图词表固定为 `balance`、`bill_balance`、`bill_due`、`pay_bill`、`transfer`、`transactions`、`report_fraud`、`report_lost_card`、`card_declined`、`credit_limit`、`credit_limit_change`。原始 `oos` 映射为 `out_of_scope`。数据来自 CLINC150 的作者仓库，采用原始全量 JSON 的官方拆分。数据许可为 CC BY 3.0，数据清单记录作者、许可证和本项目的筛选与标签映射。[论文](https://aclanthology.org/D19-1131/) · [固定版本的原始数据](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json) · [数据许可](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE)

| 拆分 | 目标条数 | 用途 |
| --- | ---: | --- |
| 训练 | 650 | 各意图抽 50 条，另用 100 条原始 OOS |
| 验证 | 320 | 各意图 20 条，另用 100 条原始 OOS，记录基座和候选 |
| 固定测试 | 530 | 各意图 30 条，另从官方 OOS 测试抽 200 条 |
| 同域拒识测试 | 40 | 从 `freeze_account`、`pin_change`、`new_card`、`damaged_card` 的测试拆分各抽 10 条，按本项目范围标为 `out_of_scope`，单独报告 |

周一用固定种子生成样本 ID 清单，检查去空格、大小写归一后的跨集合重复和标签冲突。发现冲突时保留排除记录，使用同一官方拆分的可用样本补齐，最终以清单中的实际分母为准。周三只用验证集决定候选与配置，周四在冻结候选后运行固定测试。原始 OOS 衡量数据集已有的范围外查询，同域拒识衡量本项目特有的业务边界。

## 模型与训练配置

基座选 `Qwen/Qwen3-4B-Instruct-2507`，固定 HF revision `cdbee75f17c01a7cc42f958dc650907174af0554`。该版本直接输出答案，适合检验短 JSON 序列。预计使用 4-bit NF4 基座、BF16 计算、LoRA `r=8`、`alpha=16`、梯度检查点和助手回复监督。起跑参数为每步一条、梯度累积 8 次，序列上限 384 token，学习率 `1e-4`，计划训练一轮。批量、截断和显存预算以周一模板检查及周二实测确认的配置为准。[固定基座](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) · [PEFT 量化训练](https://huggingface.co/docs/peft/developer_guides/quantization) · [bitsandbytes 安装与 CUDA 支持](https://huggingface.co/docs/bitsandbytes/installation)

预查已确认 RTX 5070 有约 12 GiB 显存，现有训练环境的 PyTorch 2.7.1+cu128 能识别 CUDA 和 BF16。`bitsandbytes` 尚未在原训练环境安装，周一在新项目的独立环境验证 Windows 量化加载与基础前向。周二用真实样例验证有限 loss、梯度和 adapter 更新、峰值显存、保存与重载。若常用提示都能在更短的上限内完整保留，可先缩短训练序列再复验。若 4B 仍因显存或后端兼容性走不通，则用 `Qwen/Qwen3-1.7B` 做 BF16 LoRA，关闭思考输出并重新冻结基线和版本名。[备选基座](https://huggingface.co/Qwen/Qwen3-1.7B)

## 五天的验收

| 日期 | 当天唯一目标 | 完成条件 |
| --- | --- | --- |
| 11 月 30 日，周一 | 冻结数据和评估协议 | 生成样本清单与标签规则，核验跨拆分重复、输出校验和模板长度。确认量化前向可运行，归档验证集基线和逐条判分。 |
| 12 月 1 日，周二 | 跑通可重载的一步训练 | 确认监督仅覆盖助手答案，loss 与梯度有限，adapter 参数更新。记录峰值显存，从新进程加载保存的 adapter，用固定探针生成可校验的 JSON。 |
| 12 月 2 日，周三 | 形成一个候选 adapter | 按锁定配置完成一次固定预算训练，记录步数、监督 token 和显存。在同一验证集对比基座与候选，冻结用于周四的 adapter。 |
| 12 月 3 日，周四 | 完成固定测试对照 | 同一基座修订、精度、提示、解码和判分器下评估两组，唯一模型侧变量为 adapter。归档逐题预测、修复与退化、易混意图及拒识样例。 |
| 12 月 4 日，周五 | 完成双平台发布和回读 | Hugging Face 上传私有实验 adapter 与模型卡。从指定 HF commit 重新下载，核验文件 SHA-256 并用固定基座复跑探针。GitHub 补上两个平台的版本指针和阶段结论。 |

## 如何判定结果

- 结构合法率以全部提交为分母。必须能解析成 JSON 对象、仅含 `intent` 键，且值在词表内。结构错误单独计数，并计入任务错误。
- 对 11 个范围内意图报告宏平均 F1 和每类样本数。对官方 OOS 报告召回率，对范围内样本报告误拒率，同域拒识 40 条单独列分子和分母。报告保留同题从错到对和从对到错的计数。
- 首版工程目标设为固定测试结构合法率至少 98%、范围内宏平均 F1 比基座提升至少 2 个百分点、官方 OOS 召回与范围内误拒率的退化各不超过 2 个百分点。同域拒识用于定位边界。指标照实记录，达到目标和完成实验发布分别验收。质量目标未达到的候选标注为实验版本，保留基座作为对照。

## GitHub 与 Hugging Face 同步

- **GitHub** 使用现有私有 `learning-record`。`practice/intent-routing/` 收录复现脚本、配置、数据来源及样本索引、评估摘要和发布记录。日记写到对应的 `daily/2026/11/` 与 `daily/2026/12/`，沿用六个栏目。逐条评估用样本 ID、真实标签、模型预测和判分原因关联源数据。
- **Hugging Face** 拟建独立私有模型仓库 `YueranCao2001/qwen3-4b-clinc-intent-qlora`，延续已有私有发布方式。首版上传 `adapter_model.safetensors`、`adapter_config.json`、模型卡和评估摘要。模型卡写明基座 revision、数据出处、训练环境、测试分母和适用范围。已经核验本机 HF 账号和旧私有仓库可读，新仓写权限在本项目首次上传时核验。[HF 上传指南](https://huggingface.co/docs/hub/models-uploading) · [模型卡说明](https://huggingface.co/docs/hub/model-cards)
- 发布时先固定 GitHub 的训练与评估代码 commit。HF 模型卡链接该 commit。回读成功后，在 GitHub 发布记录写入 Hugging Face 仓库名、commit SHA、文件 SHA-256 和回读结果。这样可以从任一平台定位复现所需的同一版本。
