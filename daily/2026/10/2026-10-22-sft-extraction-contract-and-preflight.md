# 2026年10月22日 中文信息抽取微调：首个 LoRA 训练与 HF 同步

## 今日目标

按昨天确认的方向，开始第一个小模型微调练习：使用 `Qwen/Qwen3-0.6B` 作为候选基座，以 LoRA 学习把中文活动通知转成固定 JSON。先落实输入、标签、评分和环境证据，不重新展开 SFT 算法综述。

今天先完成训练前准备，随后继续跑通真实 GPU 六步 LoRA、本地保存加载、私有 HF 上传与固定 commit 下载加载。模型仓库为 `YueranCao2001/qwen3-0.6b-zh-extraction-lora`。这是流程验收，不是模型质量验收。

## 核心概念

本次把已有 [SFT 笔记](../09/2026-09-07-supervised-fine-tuning-chat-template-lora.md) 和 [小模型实验设计](../09/2026-09-15-small-model-experiment-design.md) 落到一个可检查任务：

- 输出字段固定为 `event`、`time`、`location`；只提取输入原文中的片段，未提供的信息用 JSON `null`，不补全年份或地点。
- JSON 可解析不代表答案正确；分别检查解析、字段及类型、逐字段正确数、整条精确匹配。
- `nvidia-smi` 能看到显卡，不代表当前 PyTorch 能使用 CUDA；驱动显示的 CUDA 支持信息也不是训练包已安装的证据。
- 基座模型、模板、数据、评分规则与生成配置需在训练前固定。一步训练、推理加载、精确续训和模型质量分别验收。

## 示例与练习

### 1. 初次环境检查

执行 GPU 查询、PATH 命令发现及当前可用解释器的包检查。Python 为应用提供的 3.12.14；检查结果保存在 [环境报告](../../../practice/sft-extraction/results/2026-10-22-preflight.json)。

| 检查项 | 实际结果 |
| --- | --- |
| NVIDIA GPU | GeForce RTX 5070，12227 MiB |
| 驱动 | 610.88 |
| 当前解释器训练包 | torch、transformers、peft、trl、datasets、huggingface_hub、accelerate 均未安装 |
| PyTorch CUDA 可用性 | 未检验，缺少 torch；不是已经判定 GPU 不可用 |
| HF 认证与仓库 | 未检查认证、未创建仓库 |

检查只覆盖当前可用解释器，不据此声称整台电脑没有其他环境。PATH 中的 `python.exe` 指向 WindowsApps，因此实际运行使用明确的解释器路径。没有修改应用自带环境或安装 WSL。

这张表是上午准备阶段的快照，不代表下文训练完成后的最终状态；后续建立了独立训练环境并完成认证。

### 2. 六条合成诊断样例

[pilot-v1.json](../../../practice/sft-extraction/data/pilot-v1.json) 是助手编写的虚构活动通知，不包含用户私人资料；固定为3条 train、3条 dev，覆盖完整字段、缺时间、缺地点及两项缺失。不是最终训练数据规模，也没有独立测试集。

例如输入：

```text
摄影分享会的地点是云杉教室，具体时间另行通知。
```

预期标签（人工标注，不是模型生成）：

```json
{"event": "摄影分享会", "time": null, "location": "云杉教室"}
```

[contract.py](../../../practice/sft-extraction/contract.py) 检查 ID、来源组切分、文本重复、标签类型与原文片段。评分时，字段格式正确但时间错误仍记作整条失败；额外字段、非字符串值及代码围栏不能通过当前输出契约。

### 3. 实际运行

从 `practice/sft-extraction/` 执行，以下 `python` 代表本次使用的明确解释器：

```powershell
python -X utf8 preflight.py
python -X utf8 contract.py
python -X utf8 -m unittest -v
```

数据检查输出：

```json
{"version": "pilot-v1", "counts": {"train": 3, "dev": 3}, "model_run": false}
```

最终7项单元测试通过，包括跨切分来源组拒绝、完整匹配、非法 JSON 包装、额外键、错误类型和事实不匹配。它们验证评分程序及合成样例，不是模型准确率。项目入口见 [练习说明](../../../practice/sft-extraction/README.md)。

### 4. 独立 GPU 训练环境与监督边界

训练环境为 `E:\Evren\Evren_work\.venv-sft`，HF 客户端单独位于 `.venv-hf`。没有修改应用自带 Python。训练使用 PyTorch + PEFT 的显式循环，没有安装 TRL 或把 Ollama 权重用于训练。

实测版本：PyTorch 2.7.1+cu128、CUDA runtime 12.8、Transformers 4.57.1、PEFT 0.17.1、Accelerate 1.10.1、huggingface_hub 0.35.3。GPU capability 为12.0，GPU 张量反向传播梯度有限，`pip check` 无依赖冲突。[环境证据](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/environment.json) 与 [依赖快照](../../../practice/sft-extraction/requirements-lock.txt) 已保存。

基座和 tokenizer 固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`，从官方仓库下载。六条真实 tokenizer fixtures 验证了训练与生成前缀一致：

- 使用 `enable_thinking=False`；已有的非思考前缀只作为上下文。
- system/user/assistant 前缀的 labels 全为 `-100`，只监督 JSON 答案和 `<|im_end|>`。
- EOS ID为151645，PAD ID为151643；本次 batch=1，没有 padding，也没有 packing。
- 六条输入都短于512 tokens，不发生截断；逐 token 的 ID、文本和 label 保存在 [token fixtures](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/token-fixtures.json)。

### 5. 固定六步 LoRA 与逐题结果

[训练脚本](../../../practice/sft-extraction/train_smoke.py) 在第一次参数更新前保存了 [基座输出](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/baseline.json)。协议提前固定为6步，不根据 dev 分数追加训练：3条 train 各访问两次，batch=1、AdamW、lr=0.0002、weight_decay=0、梯度裁剪1.0、seed=42、BF16、LoRA rank=8/alpha=16/dropout=0，只训练 q_proj/v_proj。生成固定 greedy、最多128个新 tokens；配置、数据/代码/模板指纹见 [manifest](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/manifest.json)。

| 训练样例 | 第一次访问 loss | 第二次访问 loss |
| --- | --- | --- |
| train-01 | 0.269092 | 0.002264 |
| train-02 | 0.315953 | 0.034488 |
| train-03 | 0.161445 | 0.012591 |

六步 loss 和裁剪前梯度范数均有限。1,146,880个可训练参数对应112个 adapter 张量发生更新；全部冻结基座参数的 SHA-256 前后一致。训练阶段 PyTorch 峰值 allocated 约1.55 GiB，不是整卡占用，也不代表扩大 batch/序列长度仍有相同开销。[训练记录](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/training.json)、[验收摘要](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/summary.json) 均保留。

| dev 样例 | 基座整条匹配 | 六步后整条匹配 | 主要观察 |
| --- | --- | --- | --- |
| dev-01 | 是 | 是 | 正确信息保留，输出变紧凑 |
| dev-02 | 否 | 否 | time/location 修正，但 event 错取为“周六15:00” |
| dev-03 | 是 | 否 | event 从“写作工坊”变成“欢迎参加写作工坊” |

所有训练前后输出都可解析且字段契约通过，但整条正确数从2/3降到1/3。逐字段合计仍为7/9，不能用这个总数掩盖整条样例的退化。[训练后原文](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/after.json) 保留，不临时修改标签或增加步骤。这三个可见诊断样例不能估计泛化准确率；当前证据也不足以唯一确定退化机制。

### 6. 本地重载、HF 上传及远程回读

保存 adapter 后重新载入同一基座，adapter 张量逐项相等，三条生成的 token 序列与保存前完全相同。[本地重载结果](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/reloaded.json) 支持推理加载一致，不支持精确续训已验收。

用户本机完成 HF 登录后，核验账号并创建私有模型仓库。使用 [publish.py](../../../practice/sft-extraction/publish.py) 的显式执行和文件白名单，上传14个文件：adapter（4,602,248 bytes）、必要配置/tokenizer、许可、模型卡和评估记录。不上传基座权重、训练数据、Token 或 optimizer 状态。模型卡明确记录退化，不把产物推广为优于基座的模型。

远程版本：[def6e17e](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f)。按完整 commit 下载后，14个文件 SHA-256 与上传源一致；[verify_remote.py](../../../practice/sft-extraction/verify_remote.py) 加载下载副本，三条 greedy token 序列与本地训练后完全一致。[上传证据](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/upload.json)、[远程重载输出](../../../practice/sft-extraction/results/2026-10-22-lora-smoke-v1/remote-reloaded.json) 已保存，仓库仍为私有。

```powershell
# 从 practice/sft-extraction 执行；本次 run 已存在，不能覆盖。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 train_smoke.py --run 2026-10-22-lora-smoke-v1
& 'E:\Evren\Evren_work\.venv-hf\Scripts\python.exe' -X utf8 publish.py --run 2026-10-22-lora-smoke-v1 --execute
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 verify_remote.py --run 2026-10-22-lora-smoke-v1
```

以上是本次实际入口，重新训练须换新 run 名。最后全部12项本地测试通过：原7项契约测试，加5项已保存证据一致性检查；后者重算评分及检查指纹、mask和重载输出，不会重新执行训练或网络验收。

## 遇到的问题

第一次测试中，跨组负例尝试写系统临时目录，被当前沙箱拒绝，其余6项通过。改为模拟文件读取、直接输入被污染的样例，重新运行后7项通过；这次修正仅消除测试对临时目录权限的依赖，没有绕过组隔离检查。

初次检查发现训练依赖未就绪，后续已通过独立环境解决；Ollama 的量化推理模型不能当作已经可训练的 Transformers 基座。官方 CUDA wheel 的整包 pip 下载进度不稳定，中止后改为分段下载并验证官方 SHA-256，再安装同一官方 wheel。没有切换第三方包源。

Windows 缓存提示不支持 symlink，使用普通文件仍完成了下载；未为消除提示修改系统开发者模式。greedy 生成时提示基座配置中的采样参数被忽略，实际 `do_sample=False` 保持固定。代码使用 `torch_dtype` 的弃用提示未导致运行失败。

六条样例的 dev 标签全部公开，只适合诊断；组 ID 隔离也不保证语义分布独立。训练 loss 显著降低仍伴随活动名提取退化，当前不能把问题简单归结为“再多训几步”。

## 今日总结

完成了从环境、模板与监督 mask，到真实 GPU 六步训练、本地保存重载、私有 HF 上传与远程下载重载的最小闭环。保留基线、逐步 loss、梯度、权重指纹、逐题输出与远程 commit。12项测试通过。

本次真实模型使用合成数据训练，流程通过、诊断质量退化；不宣称能力提升，也不把3条 dev 视为独立测试。当前 adapter 仅作为流程验证产物，基座继续作为质量对照。

GitHub 保存代码与笔记；HF 用于后续模型产物。账号主页不是本地认证成功的证据，Token 不发送到聊天、不写入仓库，也不随产物上传。

## 下一步

- [x] 建立独立训练环境并检查 GPU 反向传播；保存实际依赖版本。
- [x] 固定基座 revision，检查真实 tokenizer 模板、EOS 和监督边界，关闭 packing。
- [x] 保存基线，完成六步 LoRA，验证有限梯度、adapter 更新和基座冻结，再保存及推理重载。
- [x] 本地认证、创建私有 HF 仓库，完成上传、固定 commit 下载哈希与加载验证。
- [ ] 下一次先复盘 dev-02/dev-03 的活动名边界错误，补充“时间在句首”“欢迎参加”等对比样例；先冻结分组与评分，再选择训练预算，不只增加轮数。
- [ ] 扩充按来源组隔离的数据及未用于调参的测试集后再做质量对照；精确续训、基础能力保留评估仍待完成。

## 参考

- [Qwen3-0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)：候选模型与使用接口。
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)：后续 SFT 实现参考，实际版本和监督 mask 仍需本地验证。
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)：后续认证、创建仓库及上传入口。
