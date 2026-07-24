[CmdletBinding()]
param(
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "configs\local.psd1"
}
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Local config not found: $ConfigPath"
}

$config = Import-PowerShellDataFile -LiteralPath $ConfigPath
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
}
elseif (Test-Path -LiteralPath "D:\Anaconda3\python.exe") {
    "D:\Anaconda3\python.exe"
}
else {
    "python"
}

& (Join-Path $PSScriptRoot "check_organizer_files.ps1")
if (-not $?) {
    exit 1
}

$output = Join-Path $repoRoot "artifacts\dataset-audit.json"
& $python (Join-Path $PSScriptRoot "audit_datasets.py") `
    $config.DatasetCnPath `
    $config.DatasetEnPath `
    --decode-limit 100 `
    --output $output
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Dataset audit written to $output"
