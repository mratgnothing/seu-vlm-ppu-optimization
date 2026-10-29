param(
    [string]$Serial = "",
    [int]$Port = 8877
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $ProjectRoot "tools\platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }

$Prefix = @()
if ($Serial) { $Prefix += @("-s", $Serial) }
& $Adb @Prefix reverse "tcp:$Port" "tcp:$Port"
if ($LASTEXITCODE -ne 0) { throw "ADB reverse failed." }
& $Adb @Prefix shell am start -a android.intent.action.VIEW -d "http://127.0.0.1:$Port"
if ($LASTEXITCODE -ne 0) { throw "Could not open the test app URL on the phone." }
