[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetPath,

    [string]$ModelPath = ".\models\Qwen3.5-2B",

    [int]$SampleOffset = 0,

    [int]$WarmupSamples = 1,

    [string]$OutputPath = ".\artifacts\profile-single.json",

    [string]$TraceOutputPath = ".\artifacts\profile-single.trace.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }
$scriptPath = Join-Path $PSScriptRoot "profile_single.py"

& $python $scriptPath `
    --dataset-path $DatasetPath `
    --model-path $ModelPath `
    --sample-offset $SampleOffset `
    --warmup-samples $WarmupSamples `
    --output $OutputPath `
    --trace-output $TraceOutputPath

if ($LASTEXITCODE -ne 0) {
    throw "Profiler failed with exit code $LASTEXITCODE"
}
