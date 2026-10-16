# 离线 RAG 检索基线

将三个文本检索配置整理为共享代码，并接入输入证据、原文引用与 Ollama 本地单题生成。当前没有压缩或自动语义支持评价。

## 固定输入

[data/baseline-v1.json](data/baseline-v1.json) 保存四个合成片段、读者权限和五道问题。四题有必需证据，一题无答案。原文不是实际商家政策。词典与分析器保存在 [retrieval.py](retrieval.py)，gold 标注仅由评估层读取。

## 重跑命令

从仓库根目录执行，Python 3.10 或更高版本，只使用标准库：

```powershell
python -m unittest discover -s practice/rag-baseline -p "test_*.py" -v
python practice/rag-baseline/run.py
```

第二条命令向标准输出打印 UTF-8 JSON 内容，包含每题片段 ID、分数、证据召回和完整覆盖。启动位置不影响数据定位。若终端无法显示中文，可启用 Python UTF-8 模式：`python -X utf8 practice/rag-baseline/run.py`。

## 保存结果

[results/2026-10-11-baseline.json](results/2026-10-11-baseline.json) 是本次运行的快照，包含数据和代码的 SHA-256 及 Python 版本。仅换行转换也会改变原始字节指纹；跨平台比较时先区分字节差异与检索结果差异。

改变代码或数据后重新运行并保存新快照，不把旧报告覆盖成新配置的结果。报告不输出 private 片段原文；本练习数据全部为合成内容，真实数据还需独立设计报告访问权限。

## 当前限制

v1 的三种配置完整证据覆盖率相同，都是 3/4。近义表达仍漏召回，这不是重构后应隐藏的失败。无答案题的召回与完整覆盖为 null，另外记录是否返回片段。

v2 的机制诊断与原文引用检查见下节。同义词扩展与真实生成评估仍未完成。

## 机制诊断数据

2026 年 10 月 12 日新增 [baseline-v2.json](data/baseline-v2.json)。保留 v1 的原片段与问题，新增词频、长度和词边界三组诊断。每组用独立读者隔离三个可见片段，避免新增片段改变原问题的 BM25 统计量。检索代码与词典不变。

新增问题中的 required 是预先指定的诊断目标，用于观察目标位置，不是独立标注的唯一语义证据。尤其长度案例中的长短片段都包含同一事实，不能将长片段排名靠前解释为回答必然错误。数据中的 design 记录运行前的预期方向；这些案例不是独立质量测试集。

从仓库根目录执行：

```powershell
python -X utf8 practice/rag-baseline/run.py --dataset practice/rag-baseline/data/baseline-v2.json --k 1
python -X utf8 practice/rag-baseline/run.py --dataset practice/rag-baseline/data/baseline-v2.json --k 2
```

保存的实际报告：[Top-1](results/2026-10-12-v2-top1.json)、[Top-2](results/2026-10-12-v2-top2.json)。每份包含三种配置、八道问题，共 24 条记录。Top-1 排名有差异；Top-2 均覆盖三个新增目标，近义漏召回仍存在。

不直接比较 v1 与 v2 的总分：问题和诊断目标已经变化。v1 的历史报告保持不变；运行器新增命令行选项会改变代码指纹，但默认 v1 的检索结果保持一致。

[当天笔记](../../daily/2026/10/2026-10-11-reusable-retrieval-baseline.md)

## 原文引用检查

[citations.py](citations.py) 从检索结果组装绑定来源版本的证据包。引用必须来自实际输入，满足当前允许集合、版本、范围与原文匹配。固定来源在 [source-snapshots-v1.json](data/source-snapshots-v1.json)，来自同一合成 v1 语料，不改动原检索数据。

```powershell
python -m unittest discover -s practice/rag-baseline -p "test_*.py" -v
python -X utf8 practice/rag-baseline/citation_demo.py
```

[实际报告](results/2026-10-13-citations.json) 保存十个案例：八个异常引用被拒绝，正常引用和真实引文搭配错误结论均通过原文检查。共 21 项测试通过。这不是生成质量或语义支持评价。

偏移为 Python 字符串的左闭右开区间。允许集合、来源快照和输入证据由可信应用层提供，调用者不得将模型自报的权限当作允许集合。证据摘要不是来源认证签名；本地撤权案例不证明真实并发权限同步已经实现。

原文检查的边界见 [10 月 13 日笔记](../../daily/2026/10/2026-10-13-source-citation-checks.md)。

## 生成协议与审查草案

[generation_contract.py](generation_contract.py) 已复用引用检查，校验请求绑定、响应状态与逐主张引用。使用手工响应夹具的七项新增测试后，整个项目共 28 项测试通过。

[生成协议](generation-protocol.md) 说明具体字段、状态与运行清单；[五题草案](data/generation-review-draft.json) 尚待独立审查，版本冲突题缺少来源，不能直接运行或当作未见确认集。

协议准备阶段见 [10 月 14 日笔记](../../daily/2026/10/2026-10-14-generation-contract-review-draft.md)，本地模型接入状态见下节。

## 生成请求准备

[试跑候选](data/generation-pilot-candidate-v1.json) 与 [试跑来源](data/generation-pilot-sources-v1.json) 补齐五题的控制上下文、评分依据和日期版本来源；状态为自查完成、待独立审查。

```powershell
python -m unittest discover -s practice/rag-baseline -p "test_*.py" -v
python -X utf8 practice/rag-baseline/prepare_generation.py
```

共 33 项测试通过。[本次实际准备结果](results/2026-10-15-prepared-requests.json) 含五份请求、model_calls=0。评分依据不会进入请求；上下文人为指定，不代表实际检索或模型回答结果。

准备阶段见 [10 月 15 日笔记](../../daily/2026/10/2026-10-15-pilot-sources-and-scoring-rubric.md)。五题候选仍待独立审查。

## Ollama 本地单题入口

[ollama_smoke.py](ollama_smoke.py) 固定访问 127.0.0.1:11434，使用本机 qwen3:8b。每次显式执行最多发起一次生成请求，不下载模型、不访问云端、不自动重试。模型先通过 Ollama 的安装流程下载到本机，不提交到 Git。

```powershell
ollama pull qwen3:8b
python -X utf8 practice/rag-baseline/ollama_smoke.py
```

入口选择第一道普通退货题，使用人工固定上下文。模型返回 answer、status、citations、reason；引用仅接受原输入 ID 和唯一匹配的原文，应用层确定性计算字符位置并复用原文检查。不会猜测缺失引用或修正错误原文；当前整段短答案作为一个主张，并不自动拆解多主张。

输出保留原始 Ollama 响应、实际提示与输入、参数、模型摘要、用量、耗时、规范化结果和失败原因。temperature=0、seed=42、num_ctx=4096、num_predict=512、think=false；后端正常停止才进入规范化，截断结果不算完成。成本未知为 null，本地运行仍消耗计算资源。

接口依据 [Ollama Chat API](https://docs.ollama.com/api/chat) 与 [结构化输出说明](https://docs.ollama.com/capabilities/structured-outputs)。后续修改提示或重跑必须另存记录，不能覆盖首次输出。

2026 年 10 月 16 日首题真实调用正常停止，回答为普通设备可以在30天内退货，引用检查通过。[原始报告](results/2026-10-16-ollama-smoke.json) 与 [助手单题审查](results/2026-10-16-ollama-smoke-review.json) 分别保存，不冒称独立评估。37 项本地测试通过，其他四题和真实检索链路待运行。详见 [当天笔记](../../daily/2026/10/2026-10-16-ollama-first-local-generation.md)。
