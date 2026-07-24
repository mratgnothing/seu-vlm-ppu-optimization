[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Baseline,

    [Parameter(Mandatory = $true)]
    [string]$Optimized,

    [string]$Output
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
$scriptPath = Join-Path $PSScriptRoot "compare_results.py"

$arguments = @(
    $scriptPath,
    "--baseline", $Baseline,
    "--optimized", $Optimized
)
if ($Output) {
    $arguments += @("--output", $Output)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Result comparison failed with exit code $LASTEXITCODE"
}
