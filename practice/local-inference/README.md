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
