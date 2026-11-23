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

当天检查的是一次非流式请求的功能链路。流式消费、首个可见文本片段的时延和重复压测在后续步骤推进。[日期笔记](../../daily/2026/11/2026-11-23-local-inference-http-smoke.md) · [Ollama Chat API](https://docs.ollama.com/api/chat)
