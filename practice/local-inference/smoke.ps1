param(
    [ValidateRange(1, 65535)]
    [int]$Port = 11435,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$baseUrl = "http://127.0.0.1:$Port"
$version = Invoke-RestMethod -Uri "$baseUrl/api/version" -TimeoutSec 10
if (-not $version.version) {
    throw 'The server did not report an Ollama version.'
}

$request = @{
    model = 'qwen3:8b'
    messages = @(@{
        role = 'user'
        content = 'What is 2 + 2? Reply with just the number.'
    })
    stream = $false
    think = $false
    options = @{ temperature = 0; num_predict = 32; num_ctx = 2048 }
}

$response = Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/api/chat" `
    -Method Post -ContentType 'application/json; charset=utf-8' `
    -Body ($request | ConvertTo-Json -Depth 6 -Compress) -TimeoutSec 60
$completion = $response.Content | ConvertFrom-Json
$content = [string]$completion.message.content
if ($response.StatusCode -ne 200 -or $completion.model -ne $request.model -or
    $completion.message.role -ne 'assistant' -or $content.Trim() -ne '4' -or
    $completion.done -ne $true -or $completion.done_reason -ne 'stop') {
    throw "Unexpected completion: $($response.Content)"
}

$result = [ordered]@{
    base_url = $baseUrl
    ollama_version = $version.version
    http_status = [int]$response.StatusCode
    request = $request
    response = $completion
}
$json = $result | ConvertTo-Json -Depth 8
if ($OutputPath) {
    $fullPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputPath)
    [System.IO.File]::WriteAllText($fullPath, $json + "`n", [System.Text.UTF8Encoding]::new($false))
}
$json
