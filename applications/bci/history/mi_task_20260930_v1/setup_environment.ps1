$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskRoot = $PSScriptRoot
$venvPath = Join-Path $taskRoot '.venv'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'
$cachePath = Join-Path $taskRoot '.uv-cache'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & uv --cache-dir $cachePath venv --python 'D:\Anaconda3\python.exe' $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& uv --cache-dir $cachePath pip install --python $pythonPath -r (Join-Path $taskRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'EEG dependency installation failed' }
Copy-Item -LiteralPath (Join-Path $taskRoot 'borrowed_torch.pth') -Destination (Join-Path $venvPath 'Lib\site-packages\borrowed_torch.pth') -Force
& $pythonPath -c "import torch,mne,sklearn,numpy,matplotlib; print({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'mne':mne.__version__,'sklearn':sklearn.__version__,'numpy':numpy.__version__}); assert torch.cuda.is_available()"
if ($LASTEXITCODE -ne 0) { throw 'CUDA/EEG environment smoke failed' }
