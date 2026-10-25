# 中文信息抽取微调练习

起点：`Qwen/Qwen3-0.6B` + LoRA。目标是从虚构活动通知提取 `event`、`time`、`location`，缺失值为 JSON `null`。不推断年份、具体日期或地点。

## 当前状态

2026-10-25 完成 [地点遗漏最小对照](../../daily/2026/10/2026-10-25-sft-location-minimal-pairs.md)：一个来源组上的六种固定输入、基座与候选共12次生成、0步训练。补入明确时间或改用“地点是”时，候选地点恢复；时间缺失措辞、顺序或开头缩短未修复。原题 token 重现昨天结果，不能据此宣称内部原因或通用修复。[逐条件记录](results/2026-10-25-location-minimal-v1/review.json)。42项本地测试通过。

HF 只追加7个诊断文件（commit `5f582e5a3da1d7a86c4116bb0512b9613f0d2ead`），下载哈希、权重不变及 main 不变验证通过。[同步证据](results/2026-10-25-location-minimal-v1/hf-audit-upload.json)。下一工程项优先精确续训对照，地点问题和独立复核待办保留。

### 前一轮开放审计

2026-10-24 完成 [冻结标注规则与开放审计](../../daily/2026/10/2026-10-24-sft-annotation-rules-open-audit.md)：使用固定 `e3b24d11` 候选，0步训练、52次生成。旧9题仍为基座5/9、候选9/9；新12条开放审计为6/12、11/12，保留一处候选遗漏明确地点的退化。5条基础探针严格通过为2/5、3/5，其中有内容正确而格式未遵循的失败。人工复核仍为 pending，不将结果称独立测试准确率。

今天仅将审计文档同步到 HF 实验分支的 `evaluations/2026-10-24-open-audit-v1/`（commit `4890a0afc0d0eb3f9561d34f7220e5b0ffeaf6ad`）；权重和 main 未改变。[逐题审查](results/2026-10-24-open-audit-v1/review.json)、[同步证据](results/2026-10-24-open-audit-v1/hf-audit-upload.json)。34项本地测试通过。下一项先做明确地点遗漏的最小对照，不自动追加训练。

### 前一轮数据对照

2026-10-23 完成 [边界样例固定预算对照](../../daily/2026/10/2026-10-23-sft-boundary-data-comparison.md)：两组从相同基座和初始 adapter 开始，各六步；对照重复旧3条 train，候选将后三步换为3条新边界样例。旧3题+新6题的整条匹配为基座5/9、对照3/9、候选9/9。候选只发布到私有分支 `boundary-v2-20261023`，远程 commit `e3b24d1115c21cf69feaa5e6e9cf85a98f502c84` 下载/加载通过，`main` 不变。23项本地测试通过。

这些是根据已知失败设计的可见合成题，模板有重叠、没有封存测试集；9/9不是泛化准确率。两组匹配步骤而非 token 总数（146 vs 136），样例曝光与顺序亦改变；候选不直接推广。[比较证据](results/2026-10-23-boundary-v2/review.json)、[同步证据](results/2026-10-23-boundary-v2/candidate/upload.json)。

### 历史起点

2026-10-22 完成六条合成样例、GPU 六步 LoRA 训练、本地保存加载，以及 HF 上传与固定 commit 下载加载验证。3 train + 3 dev 仅用于检查训练管线，不足以证明泛化。所有 dev 标签已可见，不是封存测试集。组 ID 检查不等于语义去重完成。

私有模型仓库：[qwen3-0.6b-zh-extraction-lora](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora)，验证 commit 为 `def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f`。不要向聊天发送 Token。

逐题整条匹配从基座的2/3降为 adapter 的1/3；虽然训练 loss 降低、输出格式紧凑了，但事实抽取有退化，不推广替代基座。证据保存在 [本次运行](results/2026-10-22-lora-smoke-v1/summary.json) 和 [上传验证](results/2026-10-22-lora-smoke-v1/upload.json)。

## 离线检查

地点最小对照入口为 `evaluate_location.py --run 新名称`，输入见 [location-minimal-v1.json](data/location-minimal-v1.json)，协议见 [location-minimal-protocol-v1.json](data/location-minimal-protocol-v1.json)。复用既有生成、评分和参数哈希函数，不创建 optimizer。文档发布使用 `publish_audit.py --kind minimal --expected-head 已审查分支头`，不更换被测模型 revision；模型 commit 和仅文档 commit 分开记录。

本次开放审计入口为 `audit_evaluation.py --run 新名称`，协议见 [audit-protocol-v1.json](data/audit-protocol-v1.json)，标注规则见 [annotation-rules-v1.md](annotation-rules-v1.md)。运行需要既有本地基座和已校验的 pinned adapter 下载副本；不会创建 optimizer、更新权重或重试调分。`publish_audit.py --run 名称 --execute` 只追加审核过的审计文件，拒绝在原分支已被其他提交改变时继续写入。

以下命令从本目录执行，用实际可用 Python 替换 `python`：

```powershell
python -X utf8 preflight.py
python -X utf8 contract.py
python -X utf8 -m unittest -v
```

`preflight.py` 不读取凭据，只输出当前解释器和训练包、GPU 可见性。报告中的 `null` CUDA 表示未检验，而不是 GPU 不可用。`contract.py` 检查样例身份、按组隔离、原文片段标签；`score()` 区分 JSON 解析、字段契约、逐字段和整条精确匹配。不允许代码围栏或额外解释，按解析后的值比较，不比较键顺序。该评分器仅适用于当前简单协议，不能代替生产级 JSON 安全解析或语义标注审查。

## 后续验收顺序

以下为阶段清单：第1—4项的 GPU、模板、六步训练及推理重载已完成，第6项也完成了私有上传及下载加载。10月23日进一步完成了边界样例对照；接下来优先明确标注边界和独立审查评估，不自动增加训练轮数。精确续训仍未验证。

1. 建独立虚拟环境，不修改应用自带 Python；安装兼容 RTX 5070 的训练依赖，执行 GPU 张量及反向传播检查，记录版本。
2. 固定基座/tokenizer revision，审核非思考模式模板、EOS、assistant 监督 mask 和截断；先 `packing=False`。
3. 在训练前保存固定 dev 的原始生成结果；执行一步 LoRA，验证有限 loss/梯度、adapter 更新和基座冻结。
4. 保存 adapter 并重新加载，检查固定输入输出。推理加载通过不等于精确续训通过；续训还需 optimizer、scheduler、随机状态等。
5. 补充按来源组隔离的数据后再训练和比较；不为了展示提升重复调整 dev 标签。
6. 本地认证后创建私有 HF 模型仓库，只上传审核后的 adapter、必要配置/tokenizer、模型卡和逐题评估摘要，排除数据、凭据及中间状态。检查远程 commit 和下载加载后再称同步成功。GitHub 只存代码、诊断数据与笔记。

## 本次真实训练的复现入口

下方默认命令对应10月22日的六步 smoke test；10月23日对照实验的入口如下。已有 run 不能覆盖，重新训练须换名称。原 smoke 脚本保留为冻结参考；新入口复用模型/模板/评分函数，而不修改历史证据。

```powershell
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 boundary_experiment.py --run new-comparison-run
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 review_boundary.py --run new-comparison-run
# 先审查产物和对应模型卡，再显式上传；保持独立实验分支。
& 'E:\Evren\Evren_work\.venv-hf\Scripts\python.exe' -X utf8 publish.py --run new-comparison-run --arm candidate --revision boundary-v2-20261023 --model-card boundary-model-card.md --execute
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 verify_remote.py --run new-comparison-run --arm candidate
```

新数据由不变的 `pilot-v1.json` 和 [boundary-additions-v2.json](data/boundary-additions-v2.json) 组成；[protocol](data/boundary-protocol-v2.json) 在执行前固定顺序、预算和停止条件。生成完整数据快照、源码指纹、逐 token mask、基线、两组训练日志及重载结果。`publish.py` 现在支持明确指定 arm、模型卡和 HF revision，`verify_remote.py` 从对应 manifest 选择数据，而非写死旧3题。

独立训练环境位于 `E:\Evren\Evren_work\.venv-sft`，应用自带 Python 未修改。训练使用 PyTorch + PEFT 的显式循环，不依赖 TRL。顶层版本见 [requirements.txt](requirements.txt)，实际依赖见 [requirements-lock.txt](requirements-lock.txt)。HF 上传使用单独的 `.venv-hf`（huggingface_hub 2.2.0），不把新版 Hub 强行装入固定的训练依赖组合。

CUDA wheel 从官方 cu128 源下载。整包 pip 下载被中止后，[download_torch.py](download_torch.py) 使用分段下载并核验官方索引 SHA-256：`2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30`。该辅助脚本固定 Windows/Python 3.12 的 wheel，不是通用安装器。wheel、基座及 HF 回读缓存均在 Git 仓库外，adapter 位于被忽略的 `artifacts/`。

```powershell
# 从当前目录执行；基座必须已在 train_smoke.py 指定缓存中下载。
# 每次使用新的 run 名，脚本拒绝覆盖已有证据。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 train_smoke.py --run new-run-name
# 先准备和审查文件；只有显式 --execute 才远程上传。
& 'E:\Evren\Evren_work\.venv-hf\Scripts\python.exe' -X utf8 publish.py --run new-run-name
& 'E:\Evren\Evren_work\.venv-hf\Scripts\python.exe' -X utf8 publish.py --run new-run-name --execute
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 verify_remote.py --run new-run-name
```

上传白名单为14个模型相关文件，包含 adapter（4,602,248 bytes）、tokenizer、许可、模型卡、配置及评估结果，不包含训练样例和凭据。`summary.json` 的 `uploaded=false` 是本地训练结束时的状态快照；最终远程状态以 `upload.json` 为准。单元测试还会重算已保存评分、检查证据一致性，这不是重新执行 GPU 或远程访问。

参考：[模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)、[TRL SFT](https://huggingface.co/docs/trl/sft_trainer)、[HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)。
