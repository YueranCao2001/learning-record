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

11 月 27 日用固定版 `data_full.json` 做了抽样预查。原文件 SHA-256 为 `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`。对候选样本做 Unicode NFKC、忽略大小写和折叠空白后，未发现重复或标签冲突。四组试抽数量分别为 650、320、530、40，均达到上表目标。本次证据来自数据文件的离线统计。

周一按相同口径生成正式清单。样本 ID 用 `原始拆分:零起始行号`，各类内按 `SHA-256(20261130:样本ID)` 排序后截取目标数量，方便重建完全相同的抽样。存在重复时按固定测试、同域拒识、验证、训练的优先级保留，同组重复保留首条，标签冲突排除并记录。正式指标以清单中的实际分母为准。原始 OOS 衡量数据集已有的范围外查询，同域拒识衡量本项目特有的业务边界。

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

### 每天的笔记证据

日期笔记沿用六栏模板。开头写当天的验收状态，练习部分记录运行编号、固定条件、命令与退出码并引用可回查的产物，结尾交接下一天要继承的版本或文件指纹。表里的文件名用于约定归档入口，日期笔记在当天实做后填写。[每日模板](../../templates/daily-template.md)

| 日期与拟用标题 | 应留的证据 | 写入下一步的条件 |
| --- | --- | --- |
| 11 月 30 日 · 数据契约与基座 | `data_manifest.json` 记录数据版本、SHA-256、逐类计数、样本 ID、去重规则和排除数。`baseline-dev.jsonl` 记录基座、固定提示、判分器版本与验证集逐条结果。 | 清单可重建，验证集基线已归档，量化前向通过后进入一步训练。 |
| 12 月 1 日 · 一步训练与重载 | `smoke.json` 保存包版本、GPU 和精度、实际序列长度、助手监督 token 数、loss、梯度、adapter 参数变化、显存峰值及新进程重载结果。 | 监督位置正确，loss 与梯度为有限值，adapter 有更新并能重载后确认正式配置。 |
| 12 月 2 日 · 固定预算训练 | `train_summary.json` 保存初始基座、配置指纹、实际 optimizer 步数、监督 token、耗时、显存与 adapter 哈希。用同一验证集和同一判分器比较基座与候选。 | 候选、样本清单、提示、解码及评分规则冻结后开启两组测试。 |
| 12 月 3 日 · 固定测试评估 | `eval_summary.json` 记各指标的分母和变化。逐题 JSONL 留样本 ID、真实标签、两组输出状态和判分原因，再补易混意图、修复及退化各一例。 | 按冻结阈值记录达标情况，留好供 Hugging Face 模型卡使用的摘要。 |
| 12 月 4 日 · 双平台回读 | `release.json` 记录实际训练代码的 GitHub commit、HF repo 与 commit、文件 SHA-256 及下载重载结果。 | 从锁定版本重载成功后写发布状态，并让两个平台能定位到同一次实验。 |

如果某天的验收条件没有满足，笔记记录实际失败和定位证据。下一次更新继续同一个目标。固定测试和同域拒识测试都留到候选、提示、解码及判分器冻结后。看过测试预测后产生的新方案，需要新的锁定确认集才能给出确认结果。

## 如何判定结果

11 月 28 日在本机用 39 条手写回复通过逐条判分预检。周一可复用此入口生成验证集逐题结果。[预检笔记](../../daily/2026/11/2026-11-28-intent-routing-judge-preflight.md) · [判分代码](judge.py) · [合成样例](test_judge.py)

- 结构合法率以全部提交为分母。仅去除首尾空白，整个输出须解析为 JSON 对象、恰好出现一次 `intent` 键，值为词表内的字符串。重复键、代码围栏和额外说明都计为结构错误，同时计入任务错误。
- 宏平均 F1 在固定测试的范围内样本上计算，固定 11 类等权平均。误判为 `out_of_scope` 或格式错误时，给真实意图记一次漏判。官方 OOS 召回率与同域拒识通过率分别以各自全部题目为分母。范围内样本分别报告只误判为 `out_of_scope` 的比例，以及误拒或格式错误的合计比例。保留两组同题从错到对和从对到错的计数。
- 首版工程目标设为固定测试结构合法率至少 98%、范围内宏平均 F1 比基座提升至少 2 个百分点、官方 OOS 召回与范围内误拒率的退化各不超过 2 个百分点。同域拒识用于定位边界。指标照实记录，达到目标和完成实验发布分别验收。质量目标未达到的候选标注为实验版本，保留基座作为对照。
- 这次使用公开的英语单意图数据。报告把基座与候选的同题差异限定在本次评估协议内。模型在其他语言、多意图输入或实际工单上的表现需要对应数据验证。[数据集说明](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/README.md)

## GitHub 与 Hugging Face 同步

- **GitHub** 使用现有私有 `learning-record`。`practice/intent-routing/` 收录复现脚本、配置、数据来源及样本索引、评估摘要和发布记录。日记写到对应的 `daily/2026/11/` 与 `daily/2026/12/`，沿用六个栏目。逐条评估用样本 ID、真实标签、模型预测和判分原因关联源数据。
- **Hugging Face** 拟建独立私有模型仓库 `YueranCao2001/qwen3-4b-clinc-intent-qlora`，延续已有私有发布方式。首版上传 `adapter_model.safetensors`、`adapter_config.json`、模型卡和评估摘要。模型卡写明基座 revision、数据出处、训练环境、测试分母和适用范围。已经核验本机 HF 账号和旧私有仓库可读，新仓写权限在本项目首次上传时核验。[HF 上传指南](https://huggingface.co/docs/hub/models-uploading) · [模型卡说明](https://huggingface.co/docs/hub/model-cards)
- 发布时先固定 GitHub 的训练与评估代码 commit。HF 模型卡链接该 commit。回读成功后，在 GitHub 发布记录写入 Hugging Face 仓库名、commit SHA、文件 SHA-256 和回读结果。这样可以从任一平台定位复现所需的同一版本。
