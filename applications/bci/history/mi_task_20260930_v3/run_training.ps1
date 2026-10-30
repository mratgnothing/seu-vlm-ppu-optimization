$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config.json') -Raw | ConvertFrom-Json
$trainingPython = Join-Path $config.pilot_task '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $trainingPython -PathType Leaf)) { throw "Missing training environment: $trainingPython" }
Push-Location $PSScriptRoot
try {
    & $trainingPython -u (Join-Path $PSScriptRoot 'train_v3.py')
    if ($LASTEXITCODE -ne 0) { throw "Training failed; retained logs in runs. Exit code: $LASTEXITCODE" }
} finally { Pop-Location }
