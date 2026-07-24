[CmdletBinding()]
param(
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
}
else {
    "python"
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $repoRoot "artifacts\submission-source.zip"
}

& $python (Join-Path $PSScriptRoot "package_submission.py") `
    --root $repoRoot `
    --output $OutputPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
