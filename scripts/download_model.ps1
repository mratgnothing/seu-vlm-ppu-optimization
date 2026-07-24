[CmdletBinding()]
param(
    [string]$Destination
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $repoRoot "configs\model-lock.json"
$lock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json

if (-not $Destination) {
    $Destination = Join-Path $repoRoot "models\Qwen3.5-2B"
}

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "The Hugging Face hf CLI is required but was not found on PATH."
}

hf download $lock.repo_id `
    --revision $lock.revision `
    --local-dir $Destination
if ($LASTEXITCODE -ne 0) {
    throw "Model download failed."
}

hf cache verify $lock.repo_id `
    --revision $lock.revision `
    --local-dir $Destination `
    --fail-on-missing-files `
    --fail-on-extra-files
if ($LASTEXITCODE -ne 0) {
    throw "Model checksum verification failed."
}

Write-Host "Verified model: $Destination"

