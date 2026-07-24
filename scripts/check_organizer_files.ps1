[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$lock = Get-Content -LiteralPath (Join-Path $repoRoot "configs\organizer-lock.json") `
    -Raw -Encoding UTF8 | ConvertFrom-Json
$failures = @()

foreach ($property in $lock.files.PSObject.Properties) {
    $path = Join-Path $repoRoot $property.Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $failures += "missing: $($property.Name)"
        continue
    }

    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $property.Value) {
        $failures += "hash mismatch: $($property.Name) expected=$($property.Value) actual=$actual"
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Host "Organizer files match $($lock.package)."

