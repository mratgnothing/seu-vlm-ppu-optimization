[CmdletBinding()]
param(
    [ValidateSet("cn", "en")]
    [string]$Language = "cn",
    [int]$Position = 0,
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "configs\local.psd1"
}
$config = Import-PowerShellDataFile -LiteralPath $ConfigPath
$datasetPath = if ($Language -eq "cn") {
    $config.DatasetCnPath
}
else {
    $config.DatasetEnPath
}
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$output = Join-Path $repoRoot "artifacts\debug-$Language-position-$Position.json"

& $python (Join-Path $PSScriptRoot "debug_single_sample.py") `
    --dataset-path $datasetPath `
    --model-path $config.ModelPath `
    --position $Position `
    --output $output
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

