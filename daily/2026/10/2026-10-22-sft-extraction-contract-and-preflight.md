# 2026年10月22日 中文信息抽取微调：样例契约与环境检查

## 今日目标

按昨天确认的方向，开始第一个小模型微调练习：使用 `Qwen/Qwen3-0.6B` 作为候选基座，以 LoRA 学习把中文活动通知转成固定 JSON。先落实输入、标签、评分和环境证据，不重新展开 SFT 算法综述。

最终产物计划同步到私有模型仓库 `YueranCao2001/qwen3-0.6b-zh-extraction-lora`。今天先做训练前准备，没有创建 HF 仓库或执行模型上传。

## 核心概念

本次把已有 [SFT 笔记](../09/2026-09-07-supervised-fine-tuning-chat-template-lora.md) 和 [小模型实验设计](../09/2026-09-15-small-model-experiment-design.md) 落到一个可检查任务：

- 输出字段固定为 `event`、`time`、`location`；只提取输入原文中的片段，未提供的信息用 JSON `null`，不补全年份或地点。
- JSON 可解析不代表答案正确；分别检查解析、字段及类型、逐字段正确数、整条精确匹配。
- `nvidia-smi` 能看到显卡，不代表当前 PyTorch 能使用 CUDA；驱动显示的 CUDA 支持信息也不是训练包已安装的证据。
- 基座模型、模板、数据、评分规则与生成配置需在训练前固定。一步训练、推理加载、精确续训和模型质量分别验收。

## 示例与练习

### 1. 实际环境检查

执行 GPU 查询、PATH 命令发现及当前可用解释器的包检查。Python 为应用提供的 3.12.14；检查结果保存在 [环境报告](../../../practice/sft-extraction/results/2026-10-22-preflight.json)。

| 检查项 | 实际结果 |
| --- | --- |
| NVIDIA GPU | GeForce RTX 5070，12227 MiB |
| 驱动 | 610.88 |
| 当前解释器训练包 | torch、transformers、peft、trl、datasets、huggingface_hub、accelerate 均未安装 |
| PyTorch CUDA 可用性 | 未检验，缺少 torch；不是已经判定 GPU 不可用 |
| HF 认证与仓库 | 未检查认证、未创建仓库 |

检查只覆盖当前可用解释器，不据此声称整台电脑没有其他环境。PATH 中的 `python.exe` 指向 WindowsApps，因此实际运行使用明确的解释器路径。没有修改应用自带环境或安装 WSL。

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

## 遇到的问题

第一次测试中，跨组负例尝试写系统临时目录，被当前沙箱拒绝，其余6项通过。改为模拟文件读取、直接输入被污染的样例，重新运行后7项通过；这次修正仅消除测试对临时目录权限的依赖，没有绕过组隔离检查。

当前训练依赖未就绪，不能把 Ollama 的量化推理模型当作已经可训练的 Transformers 基座。六条样例的 dev 标签全部公开，只适合诊断；组 ID 隔离也不保证语义分布独立。

## 今日总结

完成了微调任务选择、输出契约、六条合成诊断样例、七项评分与切分测试，以及当前解释器和 GPU 检查。没有运行真实模型，尚无基线输出、loss、梯度、adapter 或质量提升证据。

GitHub 保存代码与笔记；HF 用于后续模型产物。账号主页不是本地认证成功的证据，Token 不发送到聊天、不写入仓库，也不随产物上传。

## 下一步

- [ ] 建立独立训练环境，选择兼容 RTX 5070 的 PyTorch/CUDA 组合并检查 GPU 张量、反向传播；记录实际依赖版本。
- [ ] 下载候选基座前固定 revision，检查 tokenizer、非思考模式模板、EOS 和 assistant 监督边界；先关闭 packing。
- [ ] 保存训练前的固定 dev 原始输出，执行一步 LoRA，检查有限梯度、adapter 更新和基座冻结，再保存及重新加载。
- [ ] 扩充并按来源组隔离数据后再做质量对照，不用六条诊断样例声称泛化改善。
- [ ] 本地完成最小权限认证，创建约定的私有 HF 仓库；上传审核后的模型文件与说明，核验远程 commit 和下载加载。

## 参考

- [Qwen3-0.6B 官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)：候选模型与使用接口。
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)：后续 SFT 实现参考，实际版本和监督 mask 仍需本地验证。
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)：后续认证、创建仓库及上传入口。
