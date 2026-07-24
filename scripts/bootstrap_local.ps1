[CmdletBinding()]
param(
    [string]$Python = "3.12",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required but was not found on PATH."
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    uv venv $venvPath --python $Python
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment."
    }
}

uv pip install --python $pythonPath torch torchvision --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the CUDA PyTorch stack."
}

uv pip install --python $pythonPath -r (Join-Path $repoRoot "requirements-local.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install local project requirements."
}

& $pythonPath (Join-Path $PSScriptRoot "check_environment.py")
if ($LASTEXITCODE -ne 0) {
    throw "Environment validation failed."
}

