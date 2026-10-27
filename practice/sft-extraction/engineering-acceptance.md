# 微调工程阶段验收

截至2026年10月27日，单GPU受控训练、推理重载、精确续训、身份拒绝和单文件提交已有实际证据。质量验收仍有明确地点遗漏与独立人工审查待办，工程通过不能据此升级模型。

## 入口与验证范围

以下命令从本目录执行，使用已有 `.venv-sft`。重跑须换新 `--run`，不得覆盖已保存证据；旧源码受 manifest 指纹约束，扩展入口不改写旧实现。

| 入口 | 已验证行为 | 保护范围和限制 |
| --- | --- | --- |
| `train_smoke.py` / `boundary_experiment.py` | 模板监督、真实更新、冻结基座、adapter 保存与推理重载 | 历史质量实验参考；不包含完整续训保护 |
| `resume_experiment.py --run 新名称` | 连续4步与2+2跨进程恢复完全一致；optimizer-reset负对照分离 | 同机同软件、step边界；历史 worker 不自动启用新 guard |
| `guarded_resume.py --run 新名称` | 8类不匹配在反序列化前拒绝；合法恢复精确一致 | 数据/tokenizer/模板/配置/状态身份绑定；sidecar无签名，无文件发布原子性承诺 |
| `atomic_experiment.py --run 新名称` | 4类提交边界、5类坏容器与容器恢复通过 | 同目录单写入者原型；头部1MiB、payload64MiB；复用既有可信step-2状态 |
| `evaluate_prompt.py --run 新名称` | 固定模型和提示的只读诊断 | 不创建 optimizer，不替换任何模型；质量状态以逐题报告为准 |

前述多个入口各有验证范围，当前没有一个统一训练入口自动接通所有保护。容器读取对同一打开句柄的字节做校验再反序列化；兼容旧 worker 时仍导出受控文件，不应宣称全链路竞态已消除。

## 已保存的验收证据

| 证据 | 可支持的判断 |
| --- | --- |
| [连续与恢复对照](results/2026-10-26-exact-resume-v1/comparison.json) | adapter、optimizer、scheduler、RNG、history和步数一致；优化器遗漏确实改变轨迹 |
| [身份保护对照](results/2026-10-26-resume-identity-guard-v1/comparison.json) | 8类已定义故障拒绝，合法路径仍一致；含真实副本变化与元数据注入，二者不混称 |
| [写入与容器对照](results/2026-10-26-atomic-checkpoint-v1/comparison.json) | 指定边界的进程主动退出仍能读取完整正式slot；不支持真实断电结论 |
| [开放审计](results/2026-10-24-open-audit-v1/review.json) | 候选明确地点遗漏仍需处理；独立人工审查pending |

离线运行 `python -X utf8 -m unittest discover -s .` 会重算保存结果和指纹等断言，不会重跑上述 GPU 实验，也不会自动访问 HF。GPU 重跑、离线回放和远程回读应分别记录，不能互相替代。

## 发布状态

私有 HF 仓库的 `main` 仍为最初 smoke test（`def6e17e`）；质量候选权重固定于实验分支的 `e3b24d11`。后续 evaluations/engineering 提交只追加报告，每次核对分支头、私有状态、文档回读哈希、根 adapter 哈希以及 main 未变。

训练 checkpoint、RNG/optimizer 张量、容器、pending和坏文件副本仅保存在被忽略的本地 `artifacts/`。报告同步成功不等于候选质量通过，也不等于本地大文件已备份到远程。

## 待完成

- 独立人工标注复核、来源与模板族隔离的评估，以及更广泛的能力保留检查。现有数据由助手编写且标签可见。
- 梯度累积中途、AMP scaler、多worker/分布式、跨平台与版本变化的恢复；现有精确结果仅覆盖固定组合。
- 多写入者协调、目录与硬件持久性、真实断电、随机时点外部强杀，以及更大payload与全流程竞态。
- 正式统一入口和清理策略。目前容器是独立原型，不替换已经冻结的历史 worker。

优先继续解释和验证已观察到的质量错误；没有新训练/运行需求时，不为凑进度继续叠加存储原型。
