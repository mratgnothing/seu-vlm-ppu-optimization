param([int]$Port = 8877)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
python (Join-Path $ProjectRoot "test_app\server.py") --port $Port
