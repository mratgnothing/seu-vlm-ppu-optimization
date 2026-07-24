[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

& $python (Join-Path $PSScriptRoot "summarize_results.py") `
    --input-dir (Join-Path $repoRoot "results\raw") `
    --output (Join-Path $repoRoot "results\summary.csv")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

