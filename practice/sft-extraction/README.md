# 中文信息抽取微调练习

起点：`Qwen/Qwen3-0.6B` + LoRA。目标是从虚构活动通知提取 `event`、`time`、`location`，缺失值为 JSON `null`。不推断年份、具体日期或地点。

## 当前状态

2026-10-22 完成环境检查、六条人工合成诊断样例和评分程序；没有下载训练模型、执行训练或上传模型。3 train + 3 dev 仅用于检查训练管线，不足以证明泛化。所有 dev 标签已可见，不是封存测试集。组 ID 检查不等于语义去重完成。

仓库目标为 `YueranCao2001/qwen3-0.6b-zh-extraction-lora`，用户已选择私有；尚未创建或核验认证。不要向聊天发送 Token。

## 离线检查

以下命令从本目录执行，用实际可用 Python 替换 `python`：

```powershell
python -X utf8 preflight.py
python -X utf8 contract.py
python -X utf8 -m unittest -v
```

`preflight.py` 不读取凭据，只输出当前解释器和训练包、GPU 可见性。报告中的 `null` CUDA 表示未检验，而不是 GPU 不可用。`contract.py` 检查样例身份、按组隔离、原文片段标签；`score()` 区分 JSON 解析、字段契约、逐字段和整条精确匹配。不允许代码围栏或额外解释，按解析后的值比较，不比较键顺序。该评分器仅适用于当前简单协议，不能代替生产级 JSON 安全解析或语义标注审查。

## 后续验收顺序

1. 建独立虚拟环境，不修改应用自带 Python；安装兼容 RTX 5070 的训练依赖，执行 GPU 张量及反向传播检查，记录版本。
2. 固定基座/tokenizer revision，审核非思考模式模板、EOS、assistant 监督 mask 和截断；先 `packing=False`。
3. 在训练前保存固定 dev 的原始生成结果；执行一步 LoRA，验证有限 loss/梯度、adapter 更新和基座冻结。
4. 保存 adapter 并重新加载，检查固定输入输出。推理加载通过不等于精确续训通过；续训还需 optimizer、scheduler、随机状态等。
5. 补充按来源组隔离的数据后再训练和比较；不为了展示提升重复调整 dev 标签。
6. 本地认证后创建私有 HF 模型仓库，只上传审核后的 adapter、必要配置/tokenizer、模型卡和逐题评估摘要，排除数据、凭据及中间状态。检查远程 commit 和下载加载后再称同步成功。GitHub 只存代码、诊断数据与笔记。

参考：[模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)、[TRL SFT](https://huggingface.co/docs/trl/sft_trainer)、[HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)。
