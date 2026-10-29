param(
    [string]$Serial = "",
    [string]$RemoteRoot = "/sdcard/Download/ZhimanModels"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $ProjectRoot "tools\platform-tools\adb.exe"
$ModelDir = Join-Path $ProjectRoot "models\Qwen3.5-2B-MNN"
$Required = @(
    "config.json",
    "llm_config.json",
    "llm.mnn",
    "llm.mnn.weight",
    "tokenizer.txt",
    "visual.mnn",
    "visual.mnn.weight"
)

if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }
foreach ($Name in $Required) {
    $Candidate = Join-Path $ModelDir $Name
    if (-not (Test-Path -LiteralPath $Candidate)) {
        throw "Model download is incomplete; missing $Name"
    }
}

$AdbPrefix = @()
if ($Serial) { $AdbPrefix += @("-s", $Serial) }
$RemoteDir = "$RemoteRoot/Qwen3.5-2B-MNN"
& $Adb @AdbPrefix shell mkdir -p $RemoteDir
if ($LASTEXITCODE -ne 0) { throw "Could not create $RemoteDir" }
& $Adb @AdbPrefix push $ModelDir $RemoteRoot
if ($LASTEXITCODE -ne 0) { throw "Model push failed." }
Write-Host "Model copied to $RemoteDir"
