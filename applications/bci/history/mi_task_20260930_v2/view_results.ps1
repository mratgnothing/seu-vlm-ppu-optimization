param([int]$Port=4187)
$ErrorActionPreference='Stop'
$datasetRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath=Join-Path $datasetRoot 'training\mi_pilot_20260930_v1\.venv\Scripts\python.exe'
$distPath=Join-Path $PSScriptRoot 'visualization\dist'
if (!(Test-Path -LiteralPath (Join-Path $distPath 'index.html'))) {throw 'Dashboard build missing'}
while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {$Port++}
$url="http://127.0.0.1:$Port/"
Write-Host "结果页：$url"
Write-Host '此窗口维持本地服务；结束时按 Ctrl+C。仅监听本机，不公开数据。'
Start-Process $url
& $pythonPath -m http.server $Port --bind 127.0.0.1 --directory $distPath
