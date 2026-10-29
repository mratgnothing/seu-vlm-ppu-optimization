$ErrorActionPreference='Stop'
$config=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config.json') -Raw | ConvertFrom-Json
$trainingPython=Join-Path $config.pilot_task '.venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $trainingPython -PathType Leaf)){throw "Missing Python: $trainingPython"}
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
& $trainingPython -u (Join-Path $PSScriptRoot 'train_v4.py')
if($LASTEXITCODE -ne 0){throw "Training failed, logs retained. Exit code $LASTEXITCODE"}
