# 2026年10月25日 中文信息抽取微调：地点遗漏的最小对照

## 今日目标

围绕 [昨天开放审计](2026-10-24-sft-annotation-rules-open-audit.md) 的 `audit-03`，检查候选为何在明确提供“潮汐教室”时仍将地点输出为 `null`。今天先比较可控文本变体，不追加训练、不改变模型提示、不更换 checkpoint。

目标是缩小后续排查范围，而不是凭一次表面文本实验确定模型内部原因。所有变体来自同一已知失败通知，不作为独立测试题计算泛化准确率。

## 核心概念

- 最小对照中的每项修改都相对同一个原题，而不是在前一个成功版本上累计修改。活动名和真实地点始终不变。
- 文本只改一个预定因素，仍可能改变 token 数、位置和生成轨迹；观察到输出变化不等于确定了单一内部机制。
- 地点字段正确与整条匹配分开记录。模型可以提取对地点，却仍遗漏时间的一部分。
- 同一来源通知的改写全部使用同一个来源组，不能日后分别混入训练与独立测试。

## 示例与练习

### 1. 运行前固定六种条件

原题保持为：

```text
面向初学者的活动已确定：活动名称为地图绘制体验课。场地设在潮汐教室，开课时间另行公布。
```

[location-minimal-v1.json](../../../practice/sft-extraction/data/location-minimal-v1.json) 声明原题及五项修改，[构造函数](../../../practice/sft-extraction/location_minimal.py) 按字段生成实际文本并拒绝未声明因素。六种条件都归入 `notice-18`，父样例为 `audit-03`。

| ID | 唯一预定修改 | 预期 time |
| --- | --- | --- |
| minimal-00 | 不修改，重现原题 | `null` |
| minimal-01 | “开课时间另行公布”改为“开课时间待定” | `null` |
| minimal-02 | 时间分句改为“开课时间为11月20日14:00” | `11月20日14:00` |
| minimal-03 | 交换地点和时间分句，连接逗号与结尾句号保留 | `null` |
| minimal-04 | “场地设在潮汐教室”改为“地点是潮汐教室” | `null` |
| minimal-05 | 开头缩为“活动名称为地图绘制体验课。” | `null` |

全部预期 `event` 为“地图绘制体验课”、`location` 为“潮汐教室”。只有明确补入时间的条件改变 time 标签。没有增加新活动、新地点或基于输出改写目标。

[预先固定的协议](../../../practice/sft-extraction/data/location-minimal-protocol-v1.json) 限定基座先、候选后，各生成六次，共12次；不追加变体、不调提示、不训练。

### 2. 实际执行与来源绑定

沿用 RTX 5070、独立 `.venv-sft`、PyTorch 2.7.1+cu128、Transformers 4.57.1、PEFT 0.17.1，`pip check` 无冲突，没有安装新依赖。基座 revision 仍为 `c1899de289a04d12100db370d81485cdf75e47ca`，候选 adapter 仍为 HF `e3b24d1115c21cf69feaa5e6e9cf85a98f502c84`。

实际使用昨天已校验的模型下载副本，运行前重新核验14个模型文件哈希。保持原 instruction、非思考模板、BF16、greedy、最多128个新 tokens；标注规则没有加入 prompt。

[evaluate_location.py](../../../practice/sft-extraction/evaluate_location.py) 复用已有加载、生成与参数哈希函数，实际执行12次 GPU 生成、0步训练。两模型各自运行前后的参数 SHA-256 均相同。[manifest](../../../practice/sft-extraction/results/2026-10-25-location-minimal-v1/manifest.json) 绑定协议、输入、源码、原规则及 adapter 指纹，[cases.json](../../../practice/sft-extraction/results/2026-10-25-location-minimal-v1/cases.json) 保存实际展开文本。

```powershell
# 从 practice/sft-extraction 执行。本次 run 已存在，复现须换新名称。
& 'E:\Evren\Evren_work\.venv-sft\Scripts\python.exe' -X utf8 evaluate_location.py --run 2026-10-25-location-minimal-v1
```

### 3. 实际逐条件结果

| 条件 | 基座地点 | 候选地点 | 基座整条匹配 | 候选整条匹配 |
| --- | --- | --- | --- | --- |
| 原题 | 潮汐教室 | `null` | 是 | 否 |
| 时间缺失措辞 | 潮汐教室 | `null` | 是 | 否 |
| 明确提供时间 | 潮汐教室 | 潮汐教室 | 否 | 是 |
| 交换分句顺序 | 潮汐教室 | `null` | 是 | 否 |
| 地点引导词 | 潮汐教室 | 潮汐教室 | 是 | 是 |
| 缩短开头 | 潮汐教室 | `null` | 是 | 否 |

原题在基座与候选上的生成 token 都与昨天完全一致，候选错误得到重现。[基座原文](../../../practice/sft-extraction/results/2026-10-25-location-minimal-v1/base-extraction.json)、[候选原文](../../../practice/sft-extraction/results/2026-10-25-location-minimal-v1/candidate-extraction.json) 和 [逐条件审查](../../../practice/sft-extraction/results/2026-10-25-location-minimal-v1/review.json) 均保留。

明确提供时间时，基座的地点正确，但 time 仅输出 `11月20日`，漏掉 `14:00`，所以整条不匹配；候选完整提取时间和地点。不能把基座这一行的整条失败误记为地点失败。

目前支持的局部观察：

- 对这个通知，仅把“另行公布”换成“待定”、交换分句或缩短开头，没有修复候选的地点遗漏。
- 保持时间缺失，只将地点引导词换成“地点是”，候选也能提取地点。因此不能断言“缺时间必然导致地点被清空”。
- 补入明确时间也使地点恢复，但同时改变了时间内容和 token 序列；不能据此唯一确定错误机制。

这是一个来源组、一个 seed、六个预定输入上的条件比较，不报告为六条独立样本的准确率，也不把成功写法当作通用修复。

### 4. HF 仅同步诊断文档

复用 [publish_audit.py](../../../practice/sft-extraction/publish_audit.py)，增加 `minimal` 类型和明确的 `--expected-head`，在现有私有实验分支追加7个诊断文件。若分支头与预期不同则停止写入，不自动覆盖他人修改。

```powershell
& 'E:\Evren\Evren_work\.venv-hf\Scripts\python.exe' -X utf8 publish_audit.py --run 2026-10-25-location-minimal-v1 --kind minimal --expected-head 4890a0afc0d0eb3f9561d34f7220e5b0ffeaf6ad --execute
```

HF 文档 commit：[5f582e5a](https://huggingface.co/YueranCao2001/qwen3-0.6b-zh-extraction-lora/tree/5f582e5a3da1d7a86c4116bb0512b9613f0d2ead/evaluations/2026-10-25-location-minimal-v1)。下载的7个文档哈希一致，adapter 权重仍与被测 `e3b24d11` 相同，`main` 仍为 `def6e17e1d38ae6d618c72dc55ddaaf961a4fb8f`，仓库保持私有。[同步证据](../../../practice/sft-extraction/results/2026-10-25-location-minimal-v1/hf-audit-upload.json) 已保存。没有上传新权重、训练数据或凭据。

## 遇到的问题

时间缺失写法和分句顺序都不是这次测试中的单步修复；地点引导词变化却使输出翻转。结果提示后续应同时关注地点表述与缺失字段组合，但还没有解释训练中哪一条样例或哪个参数造成变化。

这些改写参考了已知错误，人工独立复核仍为 `pending`。未扩大到其他活动、其他地点或多个 seed，也未修改生产提示或补训练。

## 今日总结

完成一个来源组上的六种固定对照、12次真实 GPU 生成和逐条件结果核验。原题重现；两种修改下候选地点恢复，其余三种修改仍遗漏。参数和默认模型未改变，仅同步 HF 诊断记录。

42项微调项目本地测试通过，检查变体因素限制、来源组、标签、源码指纹、评分重算、参数不变与文档同步证据。测试通过不代表地点问题已通用修复。

## 下一步

- 将本轮条件差异保留为待审数据需求，独立审查后再决定是否补充覆盖“地点表述×缺失字段组合”的训练样例；不把同源改写拆入训练和独立测试。
- 地点问题暂存为已定位、未通用修复。工程侧下一项优先补齐精确续训对照：固定样本顺序，在 optimizer-step 边界保存并恢复 optimizer/RNG 等状态，对照连续训练结果。
- 人工复核、更广泛质量评估及基础能力保留仍未完成；已有成功条件不作为推广候选的依据。
