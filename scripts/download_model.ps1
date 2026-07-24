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
    --fail-on-missing-files
if ($LASTEXITCODE -ne 0) {
    throw "Model checksum verification failed."
}

$verifiedFiles = @()
foreach ($property in $lock.files.PSObject.Properties) {
    $path = Join-Path $Destination $property.Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Locked model file is missing: $path"
    }
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    $actualSize = (Get-Item -LiteralPath $path).Length
    if ($actualHash -ne $property.Value.sha256 -or $actualSize -ne $property.Value.size_bytes) {
        throw "Locked model file failed verification: $($property.Name)"
    }
    $verifiedFiles += @{
        file = $property.Name
        sha256 = $actualHash
        size_bytes = $actualSize
    }
}

$integrityReport = @{
    repo_id = $lock.repo_id
    revision = $lock.revision
    verified_files = $verifiedFiles
    verified_at = (Get-Date).ToString("o")
}
$reportPath = Join-Path $repoRoot "artifacts\model-integrity.json"
$integrityReport | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Host "Verified model: $Destination"
