# 离线 RAG 检索基线

将 2026 年 10 月 10 日的三个文本检索配置整理为共享代码。当前只有检索与证据覆盖评估，没有生成、压缩或语义引用核验。

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

三种配置的完整证据覆盖率相同，都是 3/4。近义表达仍漏召回，这不是重构后应隐藏的失败。无答案题的召回与完整覆盖为 null，另外记录是否返回片段。

下一步补充有区分力的案例并保存逐题排名，不先增加同义词扩展。原文引用检查的共享模块也仍待接入。

[当天笔记](../../daily/2026/10/2026-10-11-reusable-retrieval-baseline.md)
