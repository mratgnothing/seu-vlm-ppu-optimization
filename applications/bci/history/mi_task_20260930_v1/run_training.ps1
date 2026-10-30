param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run setup_environment.ps1 first' }
$trainingArgs = @('-u', (Join-Path $PSScriptRoot 'train.py'), '--config', (Join-Path $PSScriptRoot 'config.json'))
if ($Cpu) { $trainingArgs += '--cpu' }
& $pythonPath @trainingArgs
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect the run log and status.json' }
