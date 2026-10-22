# 中文信息抽取微调练习

起点：`Qwen/Qwen3-0.6B` + LoRA。目标是从虚构活动通知提取 `event`、`time`、`location`，缺失值为 JSON `null`。不推断年份、具体日期或地点。

## 当前状态

2026-10-22 完成六条合成样例、GPU 六步 LoRA 训练、本地保存加载，以及 HF 上传与固定 commit 下载加载验证。3 train + 3 dev 仅用于检查训练管线，不足以证明泛化。所有 dev 标签已可见，不是封存测试集。组 ID 检查不等于语义去重完成。

私有模型仓库：[qwen3-0.6b-zh-extraction-lora](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora)，验证 commit 为 `def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f`。不要向聊天发送 Token。

逐题整条匹配从基座的2/3降为 adapter 的1/3；虽然训练 loss 降低、输出格式紧凑了，但事实抽取有退化，不推广替代基座。证据保存在 [本次运行](results/2026-10-22-lora-smoke-v1/summary.json) 和 [上传验证](results/2026-10-22-lora-smoke-v1/upload.json)。

## 离线检查

以下命令从本目录执行，用实际可用 Python 替换 `python`：

```powershell
python -X utf8 preflight.py
python -X utf8 contract.py
python -X utf8 -m unittest -v
```

`preflight.py` 不读取凭据，只输出当前解释器和训练包、GPU 可见性。报告中的 `null` CUDA 表示未检验，而不是 GPU 不可用。`contract.py` 检查样例身份、按组隔离、原文片段标签；`score()` 区分 JSON 解析、字段契约、逐字段和整条精确匹配。不允许代码围栏或额外解释，按解析后的值比较，不比较键顺序。该评分器仅适用于当前简单协议，不能代替生产级 JSON 安全解析或语义标注审查。

## 后续验收顺序

以下为阶段清单：第1—4项的 GPU、模板、六步训练及推理重载已完成，第6项也完成了私有上传及下载加载。精确续训仍未验证；接下来优先分析退化与补充数据，不自动增加训练轮数。

1. 建独立虚拟环境，不修改应用自带 Python；安装兼容 RTX 5070 的训练依赖，执行 GPU 张量及反向传播检查，记录版本。
2. 固定基座/tokenizer revision，审核非思考模式模板、EOS、assistant 监督 mask 和截断；先 `packing=False`。
3. 在训练前保存固定 dev 的原始生成结果；执行一步 LoRA，验证有限 loss/梯度、adapter 更新和基座冻结。
4. 保存 adapter 并重新加载，检查固定输入输出。推理加载通过不等于精确续训通过；续训还需 optimizer、scheduler、随机状态等。
5. 补充按来源组隔离的数据后再训练和比较；不为了展示提升重复调整 dev 标签。
6. 本地认证后创建私有 HF 模型仓库，只上传审核后的 adapter、必要配置/tokenizer、模型卡和逐题评估摘要，排除数据、凭据及中间状态。检查远程 commit 和下载加载后再称同步成功。GitHub 只存代码、诊断数据与笔记。

## 本次真实训练的复现入口

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
