# 本地大模型推理服务与压测

## 一键回归验收

从本目录打开 PowerShell。需要能运行的 Python 3、本机 Ollama 和已缓存的 `qwen3:8b`。默认回环端口 `11435` 需要空闲，也可用 `--port` 指定一个空闲端口。脚本自行启动、预热并关闭一个单槽服务。

```powershell
python .\acceptance_probe.py --output .\acceptance-result.json
```

本机可用 `..\..\..\.venv-sft\Scripts\python.exe` 替换 `python`。五项通过时退出码为 `0`。出现失败时写入已完成项与首个失败项，并以 `1` 退出。结果 JSON 可用于核对流式正文、超时阶段、任务号和槽释放。[阶段收尾报告](stage-report.md#第一阶段回归验收)

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

### 复现首段正文超时

维持两个服务槽和客户端最多两条在途。每轮按 40、600、4000 ms 顺序提交三个批次，每批 4 条请求。截止时间从整批提交起算，覆盖等待放行、HTTP 连接和读取首段正文。到期而未放行的请求直接返回，已开始 HTTP 的请求关闭连接。通过首段验收后，客户端继续读取并验证完整正文。

```powershell
python .\deadline_probe.py --output .\deadline-result.json
```

输出记录每条请求是否开始 HTTP、是否收到状态码、退出时间和结果类别。首段正文到达后的流式读操作使用独立的 30 秒空闲超时。[逐次结果](results/2026-11-23-deadline.json)

### 核对断开后的服务槽释放

关闭该端口上的手动服务，然后运行脚本。它会启动独立的单槽 Ollama 实例，让目标请求占用槽，在收到首段正文时提交后续请求，再对照主动断开目标连接和读完目标回复两种情况。单槽让后续请求只能复用同一个槽，便于核对 runner 的任务号和槽号。

```powershell
python .\cancellation_probe.py --output .\cancellation-result.json
```

输出保留筛选后的 runner 事件，并用同一单调时钟记录客户端事件和 stderr 日志的接收时间。日志接收有传输和调度开销。[逐次结果](results/2026-11-23-cancellation.json)

加 `--dual-slot` 可对照两个槽同时忙碌的情况。脚本等待两条流都出现首段正文，再提交后续请求。取消时只关闭目标连接，另一条流继续读完；同时按任务号核对后续请求是否复用目标槽。

```powershell
python .\cancellation_probe.py --dual-slot --output .\dual-cancellation-result.json
```

[双槽时间线](results/2026-11-23-dual-cancellation.json)

### 核对首段正文之前取消

新脚本会启动单槽实例，轮换测试三种情况：持有槽的请求仍在生成时取消排队请求；观察到候选请求进入槽后立即断开；保留排队请求直到它正常完成。取消前检查候选连接是否有可读响应字节，按 runner 任务日志区分是否进入过槽。

```powershell
python .\pre_first_cancel_probe.py --output .\pre-first-cancel-result.json
```

每次都再发送一条后续请求验收槽仍可用。逐次 JSON 保存客户端时点和对应的任务及槽事件。[首段前取消时间线](results/2026-11-23-pre-first-cancel.json)

加 `--length-sweep` 可以固定在 runner 记录候选入槽后断开，只改变提示背景的长度。四档提示每档 8 次，轮换顺序。逐次记录第一次报告的缓存位置，后续分段处理记录仍保留在事件列表中。

```powershell
python .\pre_first_cancel_probe.py --length-sweep --output .\cancel-length-result.json
```

[四档提示逐次结果](results/2026-11-23-cancel-length.json)

用 `--cache-pair` 可固定 1570-token 提示，配对比较低缓存的新开头与预热后原样重发。脚本每对先测新开头，再读完一次同一提示以填充缓存，随即原样重发。两次都在看到入槽日志后、候选响应仍不可读时断开。每对用提示摘要和 runner 首次缓存位置核对两种条件。

```powershell
python .\pre_first_cancel_probe.py --cache-pair --output .\cancel-cache-result.json
```

[缓存对照逐次结果](results/2026-11-23-cancel-cache.json)

用 `--cache-sweep` 可以比较五档前缀命中。每次都使用新的 1570-token 候选提示。部分缓存条件先读完一条只在第 8、16 或 24 行换一个词的提示，再发送候选。无预热和完整预热作为两端对照。脚本记录实际缓存 token 数、未复用 token 数和槽释放时间。

```powershell
python .\pre_first_cancel_probe.py --cache-sweep --output .\cancel-partial-cache-result.json
```

[部分缓存逐次结果](results/2026-11-23-cancel-partial-cache.json)

用 `--matched-suffix` 可以让 882-token 和 1570-token 提示在两种条件下分别留下 347 或 691 个未复用 token。两条提示在相同的段内位置与各自预热提示分叉，分叉后的候选文本相同。程序核对 runner 实际记录的 token 数，每种条件测 8 次，并按轮次配对计算长提示减短提示的释放时间差。

```powershell
python .\pre_first_cancel_probe.py --matched-suffix --output .\cancel-matched-suffix-result.json
```

[等量未缓存逐次结果](results/2026-11-23-cancel-matched-suffix.json)

跨实例复验可连续调用同一命令并换两个输出路径。脚本每次都启动并关闭独立的 Ollama 实例，顺序运行可以避免两个实例争用同一张 GPU。

```powershell
foreach ($trial in 1, 2) {
    python .\pre_first_cancel_probe.py --matched-suffix --output ".\matched-suffix-recheck-$trial.json"
    if ($LASTEXITCODE) { break }
}
```

[复验 1](results/matched-suffix-recheck/trial-01.json) · [复验 2](results/matched-suffix-recheck/trial-02.json)

## 固定到达速率测试

第二阶段使用独立的两槽回环服务，客户端最多两条请求在途。每轮按固定时间表提交 10 条请求，交替使用短、长输入，输出验收保持不变。分别测试每秒 1 条和 2 条，交替先后顺序，各跑两轮。下一条请求的提交时刻由时间表确定。

```powershell
python .\arrival_probe.py --output .\arrival-result.json
```

输出区分计划提交、实际提交、开始 HTTP 和回复完成四个时点。客户端排队从实际提交算到开始 HTTP。停止提交后继续等待剩余回复，记录队列排空耗时。[逐次结果](results/2026-11-23-arrival.json)

加 `--capacity` 后，每秒 1.25 和 1.5 条各跑两轮，每轮 32 条。结果另存每轮前后半程的排队中位数、P90 和最大已有排队条数。

```powershell
python .\arrival_probe.py --capacity --output .\capacity-result.json
```

[容量实测逐次结果](results/2026-11-24-capacity.json)

补测中间速率时，用 `--capacity-midpoint` 以每秒 1.375 条各提交两轮，每轮 48 条。模型、槽数、输入和输出验收与容量对照相同。

```powershell
python .\arrival_probe.py --capacity-midpoint --output .\capacity-midpoint-result.json
```

[中间速率逐次结果](results/2026-11-24-capacity-midpoint.json)

用 `--capacity-equal-window` 对照 1.25 和 1.375 条/s。两档都在同一个 32 秒计划窗口内提交，分别提交 40 和 44 条，短、长输入各占一半。两档各跑两轮，第二轮反转测试顺序。

```powershell
python .\arrival_probe.py --capacity-equal-window --output .\capacity-equal-window-result.json
```

[等时长对照逐次结果](results/2026-11-24-capacity-equal-window.json)

用 `--arrival-shape` 对照均匀提交和成批到达。两组都计划在 32 秒内提交 40 条。均匀组每 0.8 秒发一条，批量组每 3.2 秒发四条。输入交替、两个服务槽和两条在途上限保持相同，第二轮反转测试顺序。

```powershell
python .\arrival_probe.py --arrival-shape --output .\arrival-shape-result.json
```

比较平均到达速率和完成量时使用固定 32 秒窗口。`actual_arrivals_per_second` 仍按第一条和最后一条实际提交的间隔计算，适用于核对提交间隔。成批到达时另外使用 `arrivals_per_second_in_window` 做两组对照。[到达形态逐次结果](results/2026-11-24-arrival-shape.json)

### 验证突发到达时的有限等待队列

从本目录运行，脚本启用独立双槽回环实例，按每 3.2 秒四条的轨迹比较基线与最多一条等待、1000 ms 放行预算。两种策略各跑两轮并交换顺序。

```powershell
python .\admission_probe.py --output .\bounded-admission-result.json
```

报告保留逐条提交时间、放行截止、HTTP 开始、终态和实际回复。成功延迟分位数以成功回复为样本，完成率以全部提交为分母。[11 月 25 日逐次结果](results/2026-11-25-bounded-admission.json)

加 `--policy-breakdown` 后，比较基线、仅限制等待队列、仅设置放行预算以及组合策略。每轮五批共 20 条，四种策略各跑两轮，第二轮反转顺序。报告保存每种策略的实际配置和每条请求的结果。

```powershell
python .\admission_probe.py --policy-breakdown --output .\admission-breakdown-result.json
```

[队列与预算逐次结果](results/2026-11-25-admission-breakdown.json)

用 `--budget-sweep` 固定一条等待名额，仅改变放行预算。1.0、1.4、1.6 秒各跑两轮，每轮五批共 20 条，保留逐条截止时间和实际放行时刻。

```powershell
python .\admission_probe.py --budget-sweep --output .\admission-budget-sweep-result.json
```

[预算扫描逐次结果](results/2026-11-25-admission-budget-sweep.json)

用 `--order-check` 固定一条等待、1.6 秒放行预算，比较每批短输入先提交、长输入先提交，以及批次间交替。每组跑两轮，每轮五批，各长度的提交数在三组都相同。报告按输入长度给出完成分子和提交分母。

```powershell
python .\admission_probe.py --order-check --output .\admission-input-order-result.json
```

[输入顺序逐次结果](results/2026-11-25-admission-input-order.json)

### 复核等待预算

在本目录按原始记录的 `client_wait_ms` 统计达到各档阈值的请求。该表用于选择下一次过载测试的参数。

```powershell
$report = Get-Content -Raw -Encoding UTF8 .\results\2026-11-24-arrival-shape.json | ConvertFrom-Json
foreach ($shape in 'paced', 'burst') {
    $requests = ($report.rounds | Where-Object arrival_shape -eq $shape).requests
    foreach ($budget in 250, 500, 1000, 1500) {
        [PSCustomObject]@{
            Shape = $shape
            BudgetMs = $budget
            Reached = @($requests | Where-Object client_wait_ms -ge $budget).Count
            Total = $requests.Count
        }
    }
}
```
