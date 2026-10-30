param([int]$Port = 4187)
$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config.json') -Raw | ConvertFrom-Json
$previewPython = Join-Path $config.pilot_task '.venv\Scripts\python.exe'
$previewRoot = Join-Path $PSScriptRoot 'visualization\dist'
if (-not (Test-Path -LiteralPath $previewRoot -PathType Container)) { throw "Missing visualization build: $previewRoot" }
if (-not (Test-Path -LiteralPath $previewPython -PathType Leaf)) { throw "Missing Python: $previewPython" }
if ($Port -lt 1024 -or $Port -gt 65000) { throw 'Choose a port between 1024 and 65000.' }
while (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    $Port++
    if ($Port -gt 65535) { throw 'No available preview port.' }
}
$previewUrl = "http://127.0.0.1:$Port/?view=1&tab=calibration"
Write-Host "Local preview: $previewUrl"
Write-Host 'Ctrl+C stops this server. Existing servers are left unchanged.'
Start-Process $previewUrl
& $previewPython -m http.server $Port --bind 127.0.0.1 --directory $previewRoot
