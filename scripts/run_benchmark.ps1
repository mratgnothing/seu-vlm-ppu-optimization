[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetPath,

    [string]$ModelPath = ".\models\Qwen3.5-2B",

    [ValidateSet("dummy", "transformers", "auto")]
    [string]$Backend = "transformers",

    [int]$NumSamples = 20,

    [int]$WarmupSamples = 2,

    [ValidateSet("o0_no_grad", "o1_inference_mode")]
    [string]$OptimizationProfile = "o1_inference_mode",

    [string]$OutputPath = ".\results\raw\result_public.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$benchmarkPath = Join-Path $repoRoot "benchmark_public.py"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

if (-not (Test-Path -LiteralPath $DatasetPath)) {
    throw "Dataset not found: $DatasetPath"
}

if ($Backend -eq "transformers" -and -not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model directory not found: $ModelPath"
}

$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    [System.IO.Path]::GetFullPath($OutputPath)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
}
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

$previousProfile = $env:VLM_OPT_PROFILE
try {
    $env:VLM_OPT_PROFILE = $OptimizationProfile
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $previousProfile) {
        Remove-Item Env:VLM_OPT_PROFILE -ErrorAction SilentlyContinue
    }
    else {
        $env:VLM_OPT_PROFILE = $previousProfile
    }
}

Write-Host "Result written to $resolvedOutput"
