$ErrorActionPreference='Stop'
$datasetRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath=Join-Path $datasetRoot 'training\mi_pilot_20260930_v1\.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonPath)) { throw 'Missing pilot environment' }
& $pythonPath -u (Join-Path $PSScriptRoot 'train_expanded.py')
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect latest_run.json' }
