[CmdletBinding()]
param(
    [ValidateSet("o0_no_grad", "o1_inference_mode")]
    [string[]]$Profiles = @("o0_no_grad", "o1_inference_mode"),

    [ValidateSet("cn", "en")]
    [string[]]$Languages = @("cn", "en"),

    [ValidateRange(1, 100000)]
    [int]$NumSamples = 20,

    [ValidateRange(0, 1000)]
    [int]$WarmupSamples = 2,

    [ValidateRange(1, 100)]
    [int]$Repeats = 1,

    [string]$RunLabel = "m1",

    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "configs\local.psd1"
}
$config = Import-PowerShellDataFile -LiteralPath $ConfigPath
$runScript = Join-Path $PSScriptRoot "run_benchmark.ps1"

$datasets = @{
    cn = $config.DatasetCnPath
    en = $config.DatasetEnPath
}

foreach ($profile in $Profiles) {
    foreach ($language in $Languages) {
        foreach ($repeat in 1..$Repeats) {
            $name = "$RunLabel`_$profile`_$language`_n$NumSamples`_r$repeat"
            $output = Join-Path $repoRoot "results\raw\$name.json"
            Write-Host "Running $name"
            & $runScript `
                -DatasetPath $datasets[$language] `
                -ModelPath $config.ModelPath `
                -Backend transformers `
                -NumSamples $NumSamples `
                -WarmupSamples $WarmupSamples `
                -OptimizationProfile $profile `
                -OutputPath $output
            if ($LASTEXITCODE -ne 0) {
                throw "Matrix run failed: $name"
            }
        }
    }
}
