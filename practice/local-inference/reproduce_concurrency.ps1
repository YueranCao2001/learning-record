param(
    [Parameter(Mandatory)]
    [string]$Python,
    [Parameter(Mandatory)]
    [string]$OutputDirectory,
    [ValidateRange(1, 10)]
    [int]$Trials = 3,
    [ValidateRange(1024, 65535)]
    [int]$Port = 11435
)

$ErrorActionPreference = 'Stop'
$ollama = (Get-Command ollama -ErrorAction Stop).Source
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
$output = [System.IO.Directory]::CreateDirectory($outputPath).FullName
$baseUrl = "http://127.0.0.1:$Port"
$previousHost = $env:OLLAMA_HOST
$previousParallel = $env:OLLAMA_NUM_PARALLEL
$runs = @()

try {
    $env:OLLAMA_HOST = "127.0.0.1:$Port"
    $env:OLLAMA_NUM_PARALLEL = '2'
    for ($trial = 1; $trial -le $Trials; $trial++) {
        $portCheck = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        try { $portCheck.Start() } finally { $portCheck.Stop() }

        $stem = 'trial-{0:D2}' -f $trial
        $resultFile = Join-Path $output "$stem.json"
        $stderr = Join-Path $output "$stem.stderr.log"
        $process = Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $output "$stem.stdout.log") -RedirectStandardError $stderr
        try {
            $version = $null
            for ($attempt = 0; $attempt -lt 30; $attempt++) {
                if ($process.HasExited) { throw "Ollama exited during startup for $stem. See $stderr" }
                try {
                    $version = (Invoke-RestMethod "$baseUrl/api/version" -TimeoutSec 2).version
                    if ($version) { break }
                } catch { Start-Sleep -Milliseconds 250 }
            }
            if (-not $version) { throw "Ollama did not become ready for $stem" }
            $model = (Invoke-RestMethod "$baseUrl/api/tags" -TimeoutSec 10).models |
                Where-Object name -eq 'qwen3:8b' | Select-Object -First 1
            if (-not $model.digest) { throw 'The cached qwen3:8b model is unavailable' }

            & $Python -B (Join-Path $PSScriptRoot 'concurrency_probe.py') --port $Port --output $resultFile
            if ($LASTEXITCODE) { throw "Concurrency probe failed for $stem" }

            $slots = Select-String -Path $stderr -Pattern 'n_slots\s*=\s*2,\s*n_ctx_slot\s*=\s*2048' |
                Select-Object -First 1 -ExpandProperty Line
            if (-not $slots) { throw "Expected two 2048-token slots for $stem. See $stderr" }
            $running = (Invoke-RestMethod "$baseUrl/api/ps" -TimeoutSec 10).models |
                Where-Object name -eq 'qwen3:8b' | Select-Object -First 1
            if ($running.digest -ne $model.digest) { throw "Running model digest mismatch for $stem" }

            $runs += [ordered]@{
                result = "$stem.json"
                ollama_version = $version
                model = $model.name
                model_digest = $model.digest
                quantization = $model.details.quantization_level
                slots = 2
                context_per_slot = 2048
                model_size_bytes = $running.size
                model_vram_bytes = $running.size_vram
            }
        } finally {
            if (-not $process.HasExited) {
                Stop-Process -Id $process.Id
                if (-not $process.WaitForExit(10000)) { throw "Ollama did not exit after $stem" }
            }
            $process.Dispose()
        }
    }
} finally {
    $env:OLLAMA_HOST = $previousHost
    $env:OLLAMA_NUM_PARALLEL = $previousParallel
}

$manifest = [ordered]@{ parallel_slots = 2; warmup_requests_per_trial = 4; trials = $runs }
$json = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText((Join-Path $output 'manifest.json'), $json + "`n", [System.Text.UTF8Encoding]::new($false))
