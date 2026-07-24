[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetPath,

    [string]$ModelPath = ".\models\Qwen3.5-2B",

    [ValidateSet("dummy", "transformers", "auto")]
    [string]$Backend = "transformers",

    [int]$NumSamples = 20,

    [int]$WarmupSamples = 2,

    [string]$OutputPath = ".\results\raw\result_public.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$benchmarkPath = Join-Path $repoRoot "benchmark_public.py"

if (-not (Test-Path -LiteralPath $DatasetPath)) {
    throw "Dataset not found: $DatasetPath"
}

if ($Backend -eq "transformers" -and -not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model directory not found: $ModelPath"
}

$resolvedOutput = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
$outputDirectory = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$arguments = @(
    $benchmarkPath,
    "--dataset-path", $DatasetPath,
    "--model-path", $ModelPath,
    "--backend", $Backend,
    "--num-samples", $NumSamples,
    "--warmup-samples", $WarmupSamples,
    "--output", $resolvedOutput
)

python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark failed with exit code $LASTEXITCODE"
}

Write-Host "Result written to $resolvedOutput"

