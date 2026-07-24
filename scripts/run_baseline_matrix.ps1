[CmdletBinding()]
param(
    [ValidateSet("dummy", "transformers", "auto")]
    [string]$Backend = "transformers",
    [int]$NumSamples = 20,
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "configs\local.psd1"
}
$config = Import-PowerShellDataFile -LiteralPath $ConfigPath
$runScript = Join-Path $PSScriptRoot "run_benchmark.ps1"

$runs = @(
    @{ Name = "cn"; Dataset = $config.DatasetCnPath },
    @{ Name = "en"; Dataset = $config.DatasetEnPath }
)

foreach ($run in $runs) {
    $output = ".\results\raw\baseline_$($Backend)_$($run.Name)_n$NumSamples.json"
    & $runScript `
        -DatasetPath $run.Dataset `
        -ModelPath $config.ModelPath `
        -Backend $Backend `
        -NumSamples $NumSamples `
        -OutputPath $output
    if (-not $?) {
        exit 1
    }
}
