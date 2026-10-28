# 中文信息抽取微调练习

起点：`Qwen/Qwen3-0.6B` + LoRA。目标是从虚构活动通知提取 `event`、`time`、`location`，缺失值为 JSON `null`。不推断年份、具体日期或地点。

## 当前状态

2026-10-28 完成 [新表达的缺失字段确认](../../daily/2026/10/2026-10-28-sft-missing-field-confirmation.md)：四种表达×四种提供状态16新题，另回放4题，两个模型和两个固定提示共80次GPU生成、0步训练。候选两提示均14/16，追加规则一修复一退化，两提示都把未定说明误作地点；不推广规则。基座8/16降至4/16。[逐题报告](results/2026-10-28-prompt-confirmation-v1/review.json)、[HF回读证据](results/2026-10-28-prompt-confirmation-v1/hf-audit-upload.json)。86项本地测试通过。

四条旧题16份输出逐token复现；新题与回放分开计分。仅追加11个HF诊断JSON，原提示和模型权重保持不变。接下来先整理数据修正、训练/诊断来源隔离方案和人工审查待办，不继续对同一可见题集调提示。

### 10月27日提示诊断与工程验收

2026-10-27 完成 [字段遗漏的提示对照](../../daily/2026/10/2026-10-27-sft-prompt-sensitivity-and-stage-review.md)：20条固定通知（旧12+4组新配对8条）×3提示×2模型，共120次GPU生成、0步训练。候选原提示17/20、调序19/20、追加逐字段规则20/20；调序引入时间遗漏，规则使基座出现格式退化，均不推广为通用修复。[逐题结果](results/2026-10-27-prompt-diagnostic-v1/review.json)、[HF同步证据](results/2026-10-27-prompt-diagnostic-v1/hf-audit-upload.json)。78项本地测试通过。

新增 [工程阶段验收](engineering-acceptance.md)，区分冻结历史入口、受保护入口和独立容器原型；不同模板族和缺失方向确认、独立人工复核仍待完成。HF仅同步13个诊断JSON，权重/main/训练默认提示不变。

### 10月26日单文件提交实验

2026-10-26 再次扩展：新增 [单文件 checkpoint 容器协议](data/atomic-checkpoint-protocol-v1.json) 和 `atomic_experiment.py`。已有 slot 场景下，写入中途/替换前的子进程退出保留旧文件；替换后/正常提交得到完整新文件。5类坏输入均在反序列化前拒绝，容器恢复2步仍与连续参考精确一致。69项本地测试通过。[对照证据](results/2026-10-26-atomic-checkpoint-v1/comparison.json)、[HF文档同步](results/2026-10-26-atomic-checkpoint-v1/hf-audit-upload.json)。

这是单写入者同目录临时写入、刷新后替换的原型，不证明真实断电、目录元数据持久性、多写入者或恶意篡改安全。写入测试old/new仅是提交generation，使用同一step-2 payload。二进制容器/pending/坏文件只存本地，HF仅追加9个工程JSON（commit `ddf7b3f5fda9d0c4dea516e4b8735bc3ba832f4e`），原模型与main不变。

### 当日恢复身份保护

2026-10-26 同日扩展：新增 `guarded_resume.py` 入口和 [恢复身份协议](data/resume-guard-protocol-v1.json)。新前缀2步+受保护恢复2步，8类数据/顺序/版本/tokenizer/模板/配置/字节/缺字段故障全部在 checkpoint 反序列化前拒绝，负例0更新；正常恢复与原连续参考完全一致。共59项本地测试通过。[扩展证据](results/2026-10-26-resume-identity-guard-v1/comparison.json)、[HF同步](results/2026-10-26-resume-identity-guard-v1/hf-audit-upload.json)，仅18个 JSON，模型不变。

身份在前缀训练前采集、结束后核对并绑定本地 checkpoint。故障包含3类临时文件副本变化和5类明确的运行元数据注入，原文件不改；sidecar不提供恶意篡改签名认证或并发/原子写入保证。旧 worker 保留作冻结参考；只有从 guarded 入口进入才启用这些额外检查。

### 当日首轮续训对照

2026-10-26 完成 [精确续训对照](../../daily/2026/10/2026-10-26-sft-exact-training-resume.md)：四个新进程执行连续4步、前缀2步、完整恢复2步、优化器重置2步（合计10次更新）。非零 LoRA dropout=0.1；完整恢复的历史、adapter/optimizer/scheduler/RNG与步数完全一致，adapter最大差0；优化器重置负对照分离。[比较报告](results/2026-10-26-exact-resume-v1/comparison.json)。50项本地测试通过。

仅验证同机同软件单GPU、batch=1、无累积的 optimizer-step 边界恢复，未作质量评估。checkpoint在被忽略的 `artifacts/2026-10-26-exact-resume-v1/`；HF只同步12个工程 JSON（commit `12484003bb33219209a7520022ba85aea5923bf5`），原权重/main不变。[同步证据](results/2026-10-26-exact-resume-v1/hf-audit-upload.json)。下一项是恢复前身份校验与拒绝不匹配状态，地点错误和人工复核保留。

### 前一轮最小对照

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

冻结提示确认入口：`confirm_prompt.py --run 新名称`，协议 [prompt-confirmation-v1.json](data/prompt-confirmation-v1.json)。复用已有生成/评分，分别汇总16新题、4回放及字段错误方向，不改旧证据。`publish_audit.py --kind confirmation --expected-head 已审查分支头` 只同步11个JSON。

提示对照入口：`evaluate_prompt.py --run 新名称`，协议 [prompt-diagnostic-v1.json](data/prompt-diagnostic-v1.json)。`prompt_diagnostic.py` 复用原评分器并重算逐题修复/退化、字段和格式指标，原始键顺序也保留；当前题集不作为独立测试。`publish_audit.py --kind prompt --expected-head 已审查分支头` 仅同步13个JSON诊断报告。

单文件提交实验入口：`atomic_experiment.py --run 新名称`，复用已有可信 step-2 checkpoint。原型限制头部1MiB、payload64MiB，按同一打开句柄校验并交付同一payload字节；为兼容冻结worker仍会导出受控文件，不声称全流程竞态已消除。`publish_audit.py --kind atomic --expected-head 已审查分支头` 只发布 JSON 工程记录。

恢复保护入口为 `guarded_resume.py --run 新名称`，复用原冻结训练 worker；先执行8类拒绝，再验证合法路径。`publish_audit.py --kind resume-guard --expected-head 已审查分支头` 只同步 JSON 工程记录，不上传 checkpoint 或临时副本。

精确续训入口：`resume_experiment.py --run 新名称`，使用 [resume-protocol-v1.json](data/resume-protocol-v1.json)，自动派生四个新子进程。初始配置和协议必须固定；本地受控 checkpoint 使用 `weights_only=True` 加载并核对文件哈希，RNG 在模型/optimizer 重建后恢复。`publish_audit.py --kind resume --expected-head 已审查分支头` 仅同步工程报告，不上传 `.pt` 文件或调试权重。已有候选模型 revision 与本次 fresh-base 工程训练明确区分。

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

以下保留最初的阶段顺序，第1—4、6项已完成对应范围验证。10月24日冻结标注规则，10月26日进一步完成受控精确续训及恢复保护，范围见 [工程阶段验收](engineering-acceptance.md)。第5项中的独立审查和更广泛质量验证仍未完成；10月27日提示实验为开放诊断，不自动增加训练轮数或推广候选。

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
