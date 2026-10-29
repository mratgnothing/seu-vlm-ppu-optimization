param([string]$Serial = "")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $ProjectRoot "tools\platform-tools\adb.exe"
$Apk = Join-Path $ProjectRoot "artifacts\mnn_chat_0_8_3.apk"

if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }
if (-not (Test-Path -LiteralPath $Apk)) { throw "APK not found: $Apk" }

$AdbArgs = @()
if ($Serial) { $AdbArgs += @("-s", $Serial) }
$AdbArgs += @("install", "-r", $Apk)

& $Adb @AdbArgs
if ($LASTEXITCODE -ne 0) { throw "MNN Chat installation failed." }
