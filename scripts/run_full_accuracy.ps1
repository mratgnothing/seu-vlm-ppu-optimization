[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cn", "en")]
    [string]$Language,

    [ValidateRange(1, 1000)]
    [int]$ChunkSize = 200,

    [ValidateRange(0, 1000)]
    [int]$WarmupSamples = 2,

    [ValidateSet("o0_no_grad", "o1_inference_mode")]
    [string]$OptimizationProfile = "o1_inference_mode",

    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "configs\local.psd1"
}
$config = Import-PowerShellDataFile -LiteralPath $ConfigPath
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
}
else {
    "python"
}
$dataset = if ($Language -eq "cn") {
    $config.DatasetCnPath
}
else {
    $config.DatasetEnPath
}
$model = $config.ModelPath
$chunkDirectory = Join-Path $repoRoot "data\derived\full-$Language-chunks-$ChunkSize"
$manifestPath = Join-Path $repoRoot "artifacts\full-$Language-chunks-$ChunkSize.json"
$resultPrefix = "full_$OptimizationProfile`_$Language"
$resultDirectory = Join-Path $repoRoot "results\raw"
$summaryPath = Join-Path $repoRoot "artifacts\$resultPrefix.json"

$reuseManifest = $false
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    try {
        $candidate = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $sourceHash = (
            Get-FileHash -LiteralPath $dataset -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        $sourceMatches = (
            [System.IO.Path]::GetFullPath($candidate.source_path) -eq
            [System.IO.Path]::GetFullPath($dataset)
        )
        $allChunksValid = $true
        foreach ($chunk in $candidate.chunks) {
            $path = Join-Path $chunkDirectory $chunk.file
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                $allChunksValid = $false
                break
            }
            $chunkHash = (
                Get-FileHash -LiteralPath $path -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            if ($chunkHash -ne $chunk.sha256) {
                $allChunksValid = $false
                break
            }
        }
        if (
            $sourceMatches -and
            $sourceHash -eq $candidate.source_sha256 -and
            [int]$candidate.chunk_size -eq $ChunkSize -and
            $allChunksValid
        ) {
            $reuseManifest = $true
        }
    }
    catch {
        Write-Warning "Existing chunk manifest is invalid and will be rebuilt."
    }
}
if ($reuseManifest) {
    Write-Host "Reusing verified dataset chunks."
}
else {
    & $python (Join-Path $PSScriptRoot "prepare_dataset_chunks.py") `
        --dataset $dataset `
        --output-dir $chunkDirectory `
        --manifest $manifestPath `
        --chunk-size $ChunkSize
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$runScript = Join-Path $PSScriptRoot "run_benchmark.ps1"
New-Item -ItemType Directory -Path $resultDirectory -Force | Out-Null

foreach ($chunk in $manifest.chunks) {
    $chunkIndex = [int]$chunk.index
    $chunkTag = "{0:D4}" -f $chunkIndex
    $chunkPath = Join-Path $chunkDirectory $chunk.file
    $outputPath = Join-Path $resultDirectory "$resultPrefix`_chunk-$chunkTag.json"
    $skip = $false
    if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
        try {
            $existing = Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json
            $profiles = @(
                $existing.answers |
                    ForEach-Object { $_.meta.optimization_profile } |
                    Sort-Object -Unique
            )
            if (
                [int]$existing.sample_count -eq [int]$chunk.sample_count -and
                $existing.backend -eq "transformers" -and
                $profiles.Count -eq 1 -and
                $profiles[0] -eq $OptimizationProfile
            ) {
                $skip = $true
            }
        }
        catch {
            Write-Warning "Existing chunk is invalid and will be rerun: $outputPath"
        }
    }
    if ($skip) {
        Write-Host "Skipping completed chunk $chunkIndex/$($manifest.chunk_count)"
        continue
    }

    Write-Host "Running chunk $chunkIndex/$($manifest.chunk_count)"
    & $runScript `
        -DatasetPath $chunkPath `
        -ModelPath $model `
        -Backend transformers `
        -NumSamples ([int]$chunk.sample_count) `
        -WarmupSamples $WarmupSamples `
        -OptimizationProfile $OptimizationProfile `
        -OutputPath $outputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Full accuracy chunk failed: $chunkIndex"
    }
}

$resultGlob = Join-Path $resultDirectory "$resultPrefix`_chunk-*.json"
& $python (Join-Path $PSScriptRoot "merge_benchmark_chunks.py") `
    --manifest $manifestPath `
    --results-glob $resultGlob `
    --output $summaryPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Full accuracy summary written to $summaryPath"
