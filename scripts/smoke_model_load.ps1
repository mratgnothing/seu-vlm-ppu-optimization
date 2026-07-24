[CmdletBinding()]
param(
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "configs\local.psd1"
}
$config = Import-PowerShellDataFile -LiteralPath $ConfigPath
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project environment not found: $python"
}
if (-not (Test-Path -LiteralPath $config.ModelPath -PathType Container)) {
    throw "Model directory not found: $($config.ModelPath)"
}

$output = Join-Path $repoRoot "artifacts\model-load-smoke.json"
& $python (Join-Path $PSScriptRoot "smoke_model_load.py") `
    --model-path $config.ModelPath `
    --output $output
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

