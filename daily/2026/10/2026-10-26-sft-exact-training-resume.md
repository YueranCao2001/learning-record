# 2026年10月26日 中文信息抽取微调：训练中断后的精确续训对照

## 今日目标

接着 [昨天的工程待办](2026-10-25-sft-location-minimal-pairs.md)，验证同一训练设置下“连续4步”和“2步后保存、退出进程，再恢复2步”能否一致。此前只验证了推理重载，不能据此声称 optimizer/RNG 等训练状态恢复正确。

今天使用真实 GPU LoRA 训练做工程对照，不处理地点错误、不评估质量增益，也不更换已发布候选。

完成首轮后，按用户“今天多做一点”的要求，继续实现恢复前身份保护，并用8类不匹配负例和一次真实受保护恢复验证；扩展内容仍保留在今天这篇笔记中。

## 核心概念

- 推理权重与续训 checkpoint 不同。除 adapter 外，本次还保存优化器、scheduler、完成步数、参数顺序和各类随机状态。[PyTorch checkpoint 教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- 恢复的是中断位置的随机状态，不是重新设置初始 seed。Python 采样、NumPy/CPU 探针与 CUDA dropout 的状态分开管理。
- 先重建模型和优化器等对象，再恢复 RNG，避免初始化消耗随机数改变后续序列。
- 本次只在 optimizer-step 结束、梯度清空后保存，不涉及梯度累积中途恢复。精确结果绑定同一设备和软件组合，不能外推为跨版本、跨平台一致。[PyTorch 2.7 可复现说明](https://docs.pytorch.org/docs/2.7/notes/randomness.html)

## 示例与练习

### 1. 提前固定四条执行路径

[resume-protocol-v1.json](../../../practice/sft-extraction/data/resume-protocol-v1.json) 在训练前固定预算、状态清单及验收条件。四个阶段由 [resume_experiment.py](../../../practice/sft-extraction/resume_experiment.py) 顺序启动为四个新 Python 进程：

| 阶段 | 实际执行 | 目的 |
| --- | --- | --- |
| continuous | 从相同初始状态连续4步 | 参考路径 |
| prefix | 独立重跑前2步，保存后进程退出 | 构造真实跨进程恢复点 |
| resume | 从 checkpoint 完整恢复，再跑第3—4步 | 正对照 |
| optimizer-reset | 恢复相同权重、RNG、步数、scheduler/LR，但故意不恢复 AdamW 历史 | 负对照 |

实际共执行10个 optimizer steps，最终各条完整路径长度都是4步，不是训练10步后的模型质量比较。`prefix` 的前2步与参考前2步相同，完整恢复和负对照都从该 checkpoint 分叉。

### 2. 训练配置与 checkpoint 内容

基座固定 `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`，读取已有本地缓存；数据使用不变的 [pilot-v1](../../../practice/sft-extraction/data/pilot-v1.json) 中3条 train。Python `random.randrange` 按有放回方式选样，本次实际顺序为 train-03、train-01、train-01、train-03；不声称遍历了完整 epoch。

沿用 RTX 5070、Windows、独立 `.venv-sft`、PyTorch 2.7.1+cu128、Transformers 4.57.1、PEFT 0.17.1，`pip check` 无冲突，无新依赖。LoRA rank=8、alpha=16，目标 q_proj/v_proj；本次 **dropout=0.1**，与此前质量候选的0不同，用于使 CUDA RNG 恢复有实际训练意义，而非宣称是质量改进。

其余设置：seed=42、BF16基座、batch=1、AdamW lr=0.0002、weight_decay=0、梯度裁剪1.0、`foreach=False/fused=False`；StepLR 每2步将 LR 乘0.5。固定 deterministic algorithms、math SDPA、关闭 TF32，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。沿用原模板与 assistant 监督，未启用 packing、padding、梯度累积或 GradScaler。

在 step 2 的 optimizer 和 scheduler 更新后、梯度清空时保存：

- adapter 张量、AdamW state_dict（含动量与步数）、scheduler state_dict；
- 有序参数名、已完成步数与历史记录、协议指纹；
- Python、NumPy、Torch CPU 和所有可见 CUDA RNG 状态。

[resume_state.py](../../../practice/sft-extraction/resume_state.py) 将 NumPy 状态转为普通容器，张量克隆到 CPU；只对本次本地受控 checkpoint 使用 `torch.load(weights_only=True)`，加载前核对文件 SHA-256。不加载远程任意 pickle，也不保存整个 Python 模型对象。

本地 `artifacts/2026-10-26-exact-resume-v1/` 中 checkpoint 及三个最终状态文件各约13.9MB，被 Git 忽略，不上传 HF。重建模型和参数顺序后，完整路径恢复优化器与 scheduler，最后恢复 RNG；负对照仅清空优化器历史，保留同一 LR/scheduler 和 RNG。

```powershell
# 从 practice/sft-extraction 执行。该 run 已完成，复现须换新名称。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 resume_experiment.py --run 2026-10-26-exact-resume-v1
```

### 3. 连续训练与完整恢复的实际结果

[对照摘要](../../../practice/sft-extraction/results/2026-10-26-exact-resume-v1/comparison.json) 保存验收结果，[manifest](../../../practice/sft-extraction/results/2026-10-26-exact-resume-v1/manifest.json) 绑定协议、数据与源码指纹。四个子进程 PID 不同，两条从头路径的初始 adapter 指纹相同，每个阶段的冻结基座参数哈希均保持不变。

| 步骤 | 样例 | 更新前 LR | 连续训练 loss | 完整恢复 loss |
| --- | --- | --- | --- | --- |
| 1 | train-03 | 0.0002 | 0.4462213814 | 前缀重跑相同 |
| 2 | train-01 | 0.0002 | 0.1484066844 | 前缀重跑相同 |
| 3 | train-01 | 0.0001 | 0.0270694699 | 完全相同 |
| 4 | train-03 | 0.0001 | 0.0957139283 | 完全相同 |

连续与完整恢复的完整历史记录逐项相同，包括 loss、梯度范数、采样 ID、LR、NumPy/CPU 探针。最终 adapter 张量逐项相等，最大绝对差为 **0**；optimizer、scheduler、RNG、步数与参数名的规范化状态指纹也一致。不是仅设置一个容差后称“接近”。

实际记录：[continuous](../../../practice/sft-extraction/results/2026-10-26-exact-resume-v1/continuous/history.json)、[prefix](../../../practice/sft-extraction/results/2026-10-26-exact-resume-v1/prefix/history.json)、[resume](../../../practice/sft-extraction/results/2026-10-26-exact-resume-v1/resume/history.json)。

### 4. 仅重置优化器的负对照

负对照恢复了相同 adapter、样本位置、RNG、scheduler 和学习率，只让 AdamW 动量与步数重新开始：

- 第3步 forward loss 仍为0.0270694699，梯度范数也相同；因为此时参数、输入与 dropout 序列相同。
- 第3步更新后路径分离，第4步 loss 为0.1014072075，而参考为0.0957139283。
- 最终 adapter 最大绝对差为0.0003402730；采样 ID、LR、探针及最终 RNG 指纹仍与参考一致。

[负对照原始记录](../../../practice/sft-extraction/results/2026-10-26-exact-resume-v1/optimizer-reset/history.json) 支持的结论是：在本次控制设置中，遗漏优化器历史会改变续训轨迹；第一步 loss 相同不足以验收恢复成功。没有据负对照的 loss 高低评判模型质量，也没有另跑“只遗漏 RNG”等额外消融。

### 5. HF 仅同步工程证据

复用 [publish_audit.py](../../../practice/sft-extraction/publish_audit.py) 的 `resume` 类型，只同步12个 JSON 工程报告至私有实验分支的 `engineering/2026-10-26-exact-resume-v1/`。checkpoint、调试 adapter、优化器张量和 RNG 原始状态留在本地，报告只包含指纹与数值记录。

HF 工程文档 commit：[12484003](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/12484003bb33219209a7520022ba85aea5923bf5/engineering/2026-10-26-exact-resume-v1)。12个文件回读哈希一致，原 HF adapter 权重和 `main` 均未改变，仓库保持私有。[同步证据](../../../practice/sft-extraction/results/2026-10-26-exact-resume-v1/hf-audit-upload.json) 将当前工程调试与原候选区分开，不将本次训练称为新质量候选发布。

### 6. 当日扩展：恢复前身份保护与八类拒绝

首轮 checkpoint 能恢复正确，不代表拿错数据、模板或状态文件时也会安全停止。为此新增 [guarded_resume.py](../../../practice/sft-extraction/guarded_resume.py) 入口，复用冻结的训练 worker，不修改第一轮脚本和证据。[预先固定的扩展协议](../../../practice/sft-extraction/data/resume-guard-protocol-v1.json) 限定新前缀2步、受保护恢复2步，以及8类故障注入；不调整训练预算或标签。

新前缀训练前采集运行身份，结束后再次检查一致，再通过 `binding.json` 绑定新 checkpoint 的 SHA-256 和有序参数名指纹。身份包括：

- 基座 ID/revision、实际 config 和权重文件 SHA-256；
- 原始数据文件、训练样本顺序、实际 token IDs/labels 指纹；
- tokenizer 文件、chat template、EOS/PAD；
- 训练协议、执行代码、Python/依赖版本及 GPU/CUDA 信息。

[resume_guard.py](../../../practice/sft-extraction/resume_guard.py) 先核对 checkpoint 字节哈希和身份，再允许 `weights_only=True` 反序列化；加载后继续核对状态结构、步数、协议和参数顺序，最后才派发训练。正常路径的模型/优化器初始化与 RNG 恢复仍复用原逻辑。

| 负例 | 实际操作 | 结果 |
| --- | --- | --- |
| 数据内容变化 | 修改临时数据副本，原文件不变 | 拒绝 data/token 指纹不匹配 |
| 样本顺序变化 | 交换临时数据副本前两条记录 | 拒绝数据、顺序与 token 指纹不匹配 |
| 基座 revision 不符 | 运行身份元数据注入，不下载错误模型 | 拒绝 base_revision |
| tokenizer 文件指纹不符 | 运行身份指纹注入，不改原 tokenizer | 拒绝 tokenizer_files |
| 模板指纹不符 | 运行身份指纹注入，不改原模板 | 拒绝 chat_template_sha256 |
| LoRA 配置变化 | 运行身份 rank 注入，不构建错误 adapter | 拒绝 training_protocol |
| checkpoint 字节变化 | 向临时 checkpoint 副本追加字节 | 拒绝 checkpoint_sha256 |
| 身份字段缺失 | 从运行身份元数据移除字段 | 拒绝 identity_fields |

8类负例均在 checkpoint 反序列化前被拒绝，`checkpoint_deserialized=false`、`training_dispatched=false`、`optimizer_updates=0`。其中3类使用真实文件副本，5类为明确标注的元数据故障注入；不将后者描述为已经加载过错误模型或真的更换了 tokenizer。[逐类拒绝记录](../../../practice/sft-extraction/results/2026-10-26-resume-identity-guard-v1/rejections/data-order.json) 与 [完整协议中的类型说明](../../../practice/sft-extraction/results/2026-10-26-resume-identity-guard-v1/guard-protocol.json) 可核查。

合法身份随后通过保护，实际继续训练第3—4步。与第一轮连续参考比较，adapter、optimizer、scheduler、RNG、完整 history、步数及参数名仍完全一致，adapter 最大绝对差为0；原 checkpoint 哈希保持不变。[扩展对照摘要](../../../practice/sft-extraction/results/2026-10-26-resume-identity-guard-v1/comparison.json)、[合法放行记录](../../../practice/sft-extraction/results/2026-10-26-resume-identity-guard-v1/valid-guard.json) 已保存。扩展新增4次 optimizer 更新，负例无更新；加上首轮10次，当天两轮合计14次实际更新，不是一个模型连续训练14步。

```powershell
# 从 practice/sft-extraction 执行。新 run 名会创建新的前缀和受保护恢复。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 guarded_resume.py --run 2026-10-26-resume-identity-guard-v1
```

扩展仅同步18个 JSON 工程文件，HF commit 为 [b28bcea7](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/b28bcea7e7737451f916f1b9414a2304ae29f0d2/engineering/2026-10-26-resume-identity-guard-v1)。下载哈希通过，原候选权重和 `main` 不变；checkpoint、损坏副本和临时数据副本均留本地。[同步证据](../../../practice/sft-extraction/results/2026-10-26-resume-identity-guard-v1/hf-audit-upload.json) 与首轮分别保存。

## 遇到的问题与边界

此前推理加载一致，只能说明权重可用于同样的生成，缺少真实续训证据。今天用四个独立进程、非零 dropout 和优化器重置负对照，补上了固定设置下的工程验收。

范围仍有限：只验证同机同软件、单GPU、batch=1、普通有放回采样、optimizer-step 边界的保存恢复；未验证多 worker、分布式、AMP scaler、累积梯度中途、强制杀进程导致的半写文件或跨平台恢复。可复现设置也不是吞吐性能测量。

身份保护覆盖的是新的 guarded 入口，历史 worker 保留作冻结参考，直接运行旧入口不会自动获得全部保护。sidecar 是本地可信来源的绑定记录，不是数字签名；未承诺防止攻击者同时篡改 checkpoint 和绑定记录，也没有并发文件锁、检查与使用间竞态或断电原子写入保证。

## 今日总结

完成真实 GPU 精确续训对照：完整恢复逐项一致，优化器负对照在更新后分离；进一步完成身份保护、8类反序列化前拒绝和真实受保护恢复。微调项目本地测试从首轮50项增至59项，覆盖 checkpoint 容器、RNG 回放、身份拒绝、指纹、正负对照及两轮文档同步证据。

今天完成的是受控训练工程能力，不是活动抽取质量改善；地点错误、独立人工复核和更广泛能力评估继续保留。

## 下一步

- 已补齐新的 guarded 入口及8类拒绝。下一步考虑将绑定信息纳入正式 checkpoint 格式，并完善原子写入与中断损坏处理，而不是将此次 sidecar 直接视为生产级安全机制。
- 继续区分工程验收和质量验收；补充训练样例前先审查来源分组、标注边界与独立评估计划，不因续训一致就推广当前候选。
