# 本地大模型推理服务与压测

## 复现首条 HTTP 回复

从本目录打开 PowerShell。需要本机 Ollama 和已缓存的 `qwen3:8b`。实测版本为 `0.40.17`。用 `ollama list` 检查模型，实测时的模型 ID 为 `500a1f067a9f`。若电脑尚无此模型，可以运行 `ollama pull qwen3:8b` 下载。[Ollama 模型页](https://ollama.com/library/qwen3:8b)

在第一个 PowerShell 窗口运行服务。实验使用回环地址和独立端口，通常可以与默认的 `11434` 端口并行使用。

```powershell
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_NUM_PARALLEL = '1'
ollama serve
```

在第二个 PowerShell 窗口运行验证。脚本先读取 `/api/version`，再调用 `/api/chat`，检查 HTTP 200、模型名称、回复正文 `4`、`done: true` 和 `done_reason: stop`。

```powershell
.\smoke.ps1
```

用 `.\smoke.ps1 -OutputPath .\smoke-result.json` 将一次请求和原始响应另存为 UTF-8 JSON。返回成功后可在服务窗口按 `Ctrl+C` 退出。

## 复现流式对照

启动同一服务，从本目录使用可运行的 Python 3 执行以下命令。脚本仅依赖标准库。它先做一次预热，随后串行运行 5 对流式和普通请求，交替先后顺序，并验证每条完整回复。

```powershell
python .\stream_probe.py --pairs 5 --output .\stream-result.json
```

本机可用 `..\..\..\.venv-sft\Scripts\python.exe` 替换 `python`。输出包含逐次首段时间、完整时间、服务端测试计数和中位数。客户端首段时间取第一条含非空白正文的 NDJSON 事件，普通响应只记录完整时间。模型和固定提示均用于预热，回答耗时受本次前缀缓存影响。[实测结果](results/2026-11-23-stream.json) · [日期笔记](../../daily/2026/11/2026-11-23-local-inference-http-smoke.md) · [Ollama Chat API](https://docs.ollama.com/api/chat)

## 复现单并发基线

用相同环境启动一个新服务实例，保持 `OLLAMA_NUM_PARALLEL=1`。脚本先发一条独立的模型预热请求，再执行短提示与长提示各 6 对。每对先发送带新开头的提示，随后立即原样重发。短、长提示的组间顺序轮换，模型在整轮中常驻。

```powershell
python .\baseline_probe.py --output .\baseline-result.json
```

脚本按服务端计数验收缓存条件。首次请求的缓存比例须不超过 25%，重复请求须至少达到 90%。每条回复还要通过 HTTP、正文和结束标记校验。生成吞吐按服务端 `eval_count / eval_duration` 换算为 token/s。逐次文件保留原始计数、提示模板和四组中位数。[基线结果](results/2026-11-23-baseline.json)

## 复现并发 1 与 2 对照

用相同的回环地址启动新服务，将 `$env:OLLAMA_NUM_PARALLEL` 改为 `'2'`。两个客户端条件共用这一个服务实例。脚本先预热，再交替运行 4 组配对试验。每个条件每轮发送 2 个短提示和 2 个长提示，各用不同的三位数字开头。并发上限指客户端同时在途的请求数。

```powershell
python .\concurrency_probe.py --output .\concurrency-result.json
```

逐次文件记录每轮峰值在途数、失败数、请求墙钟时间和服务端缓存计数。请求与生成 token 的整轮吞吐以墙钟时间为分母，预热单独保存。[并发结果](results/2026-11-23-concurrency.json)

### 从新服务实例复跑

关闭手动启动的 `11435` 服务，运行下面的 PowerShell 命令。脚本每次独立启动并关闭一个 Ollama 实例，采集模型摘要、槽配置和模型显存，再重复同一套并发对照。`manifest.json` 与逐次 JSON 可用于复核，服务日志留在指定的本地输出目录。

```powershell
.\reproduce_concurrency.ps1 -Python python -OutputDirectory .\concurrency-recheck -Trials 3
```

若 `python` 指向 Windows 商店别名，把 `-Python` 值换成实际解释器路径。[阶段报告](stage-report.md)

## 复现短输出与长输出对照

从新实例启动 Ollama，保持回环地址和两个槽。测试提示同时包含两个待复制的文本，末尾选择 `short` 或 `long`。输入仍使用短、长两种背景长度，每个条件每轮各发送两条，按轮次轮换四个条件的执行次序。

```powershell
python .\output_length_probe.py --output .\output-length-result.json
```

脚本验证每条回复与目标文本逐字一致且正常结束。JSON 用 SHA-256 引用去重后的实际正文，并保留 prompt token、生成 token、缓存计数、单条延迟和整轮吞吐。[逐次结果](results/2026-11-23-output-length.json)

## 复现并发超过服务槽数

保持 `OLLAMA_NUM_PARALLEL=2`，使用上一节的长输出任务。客户端在途上限分别设为 2 和 4，交替运行 6 组配对，每轮 4 条短输入和 4 条长输入。

```powershell
python .\queue_probe.py --output .\queue-result.json
```

P90 取各条件有效请求排好序后第 `ceil(0.90 × N)` 个值。首段正文和完整回复都从客户端发出请求开始计时，因此包含客户端观察到的等待。[逐次结果](results/2026-11-23-queue.json)

### 对照客户端放行上限

添加 `--batch-arrival` 后，每批以同一时钟点提交 4 条请求，比较客户端最多放行 2 条与全部放行。两种方式始终使用同一个有两个槽的服务实例。

```powershell
python .\queue_probe.py --batch-arrival --output .\admission-result.json
```

`start_ms` 是从整批提交到开始 HTTP 调用的时间，`first_text_ms` 是 HTTP 调用至首段正文的时间，`end_ms` 是从整批提交到该条回复完成的时间。从同一批次起点计算首段正文 P90，可以把发送前的等待也算进去。[逐次结果](results/2026-11-23-admission.json)
