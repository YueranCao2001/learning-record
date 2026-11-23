# 2026-11-23：本地推理实作：跑通第一条 HTTP 回复

## 今日目标

启动新的本地大模型推理服务与压测项目。今天只验证一个问题：本机服务能否加载模型，并通过 HTTP 返回一条完整回复？完成条件是服务只监听回环地址，固定请求收到 HTTP 200、回复正文 `4` 和正常停止标记，并留下可复查的响应文件。[练习入口](../../../practice/local-inference/README.md)

## 核心概念

- `/api/version` 成功说明 HTTP 服务已启动。第一次 `/api/chat` 还要加载模型并完成生成，才能验证整条推理链路。
- `stream: false` 使客户端一次性收到完整 JSON，适合验证正文和停止原因。下一次启用流式响应后，客户端才能记录首个可见文本片段的到达时间。[Ollama Chat API](https://docs.ollama.com/api/chat)

## 示例与练习

在 Windows 11 和 RTX 5070 上启动独立进程，监听 `127.0.0.1:11435`。使用本机缓存的 `qwen3:8b`，设置单路并行、上下文 2048、温度为 0、最多生成 32 token。关闭思考输出，请求 `What is 2 + 2? Reply with just the number.`。[验证脚本](../../../practice/local-inference/smoke.ps1)

| 检查项 | 实测结果 |
| --- | --- |
| 服务版本 | Ollama `0.40.17` |
| 模型与设备 | `qwen3:8b`，ID `500a1f067a9f`，`ollama ps` 显示 `100% GPU`，runner 为 `llamacpp` |
| 非流式响应 | HTTP 200，`assistant` 回复 `4`，`done: true`，`done_reason: stop` |
| 服务端计数 | 30 个提示 token、2 个生成 token |
| 该次首请求服务端耗时 | 总计 44.66 秒，其中模型加载 24.08 秒 |

[请求和响应 JSON](../../../practice/local-inference/results/2026-11-23-smoke.json)

## 遇到的问题

原计划使用独立 llama.cpp Windows 包和 Qwen3-0.6B GGUF。官方运行包未能及时下载，Hugging Face GGUF 下载连接超时。检查时发现 Ollama 已安装且有 `qwen3:8b` 缓存，因此首日用这套组合完成同一个本机 HTTP 验证。后续性能对照需要固定服务实现、模型和运行配置。

## 今日总结

完成从本机 HTTP 请求到 GPU 推理和完整回复的功能验收。检查只有一次冷启动请求，耗时包含模型加载。稳定延迟和吞吐要在固定环境、完成预热后通过重复请求统计。

## 下一步

沿用相同本地服务与模型，添加一个最小流式客户端。用单请求验证能读取完整响应，并分别记录发送请求到首个非空正文片段、到结束事件的耗时。
