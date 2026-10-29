param(
    [string]$Serial = "",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $ProjectRoot "tools\platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }

$Prefix = @()
if ($Serial) { $Prefix += @("-s", $Serial) }
if (-not $OutputPath) {
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputPath = Join-Path $ProjectRoot "evidence\runtime-$Stamp.json"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $ProjectRoot $OutputPath
}

function A([string[]]$Args) {
    $value = & $Adb @Prefix @Args
    return (($value -join "`n").Trim())
}

$Snapshot = [ordered]@{
    captured_at = (Get-Date).ToString("o")
    foreground_activity = A @("shell", "dumpsys", "activity", "activities")
    mnn_process = A @("shell", "dumpsys", "meminfo", "com.alibaba.mnnllm.android")
    battery = A @("shell", "dumpsys", "battery")
    thermal = A @("shell", "dumpsys", "thermalservice")
}

$Json = $Snapshot | ConvertTo-Json -Depth 3
$Parent = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $Parent -Force | Out-Null
Set-Content -LiteralPath $OutputPath -Value $Json -Encoding utf8
$Json
