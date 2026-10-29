param(
    [string]$Serial = "",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $ProjectRoot "tools\platform-tools\adb.exe"

if (-not (Test-Path -LiteralPath $Adb)) {
    throw "ADB not found: $Adb"
}

$DeviceLines = @(& $Adb devices -l) | Where-Object { $_ -match "\sdevice(\s|$)" }
if ($Serial) {
    $DeviceLines = @($DeviceLines | Where-Object { $_ -like "$Serial *" })
}

if ($DeviceLines.Count -eq 0) {
    throw "No authorized Android device found. Enable USB debugging, reconnect the cable, and approve the computer on the phone."
}
if ($DeviceLines.Count -gt 1 -and -not $Serial) {
    throw "Multiple devices found. Re-run with -Serial <device-serial>."
}

if (-not $Serial) {
    $Serial = ($DeviceLines[0] -split "\s+")[0]
}

function Invoke-AdbText {
    param([string[]]$Arguments)
    $value = & $Adb -s $Serial @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "ADB command failed: $($Arguments -join ' ')"
    }
    return (($value -join "`n").Trim())
}

function Get-AndroidProp {
    param([string]$Name)
    return Invoke-AdbText -Arguments @("shell", "getprop", $Name)
}

$PackageResult = & $Adb -s $Serial shell pm list packages com.alibaba.mnnllm.android
$MnnInstalled = (($PackageResult -join "`n") -match "com\.alibaba\.mnnllm\.android")

$Report = [ordered]@{
    captured_at = (Get-Date).ToString("o")
    adb_version = ((& $Adb version) -join " ").Trim()
    serial = $Serial
    manufacturer = Get-AndroidProp "ro.product.manufacturer"
    model = Get-AndroidProp "ro.product.model"
    device = Get-AndroidProp "ro.product.device"
    soc_manufacturer = Get-AndroidProp "ro.soc.manufacturer"
    soc_model = Get-AndroidProp "ro.soc.model"
    hardware = Get-AndroidProp "ro.hardware"
    android_version = Get-AndroidProp "ro.build.version.release"
    api_level = Get-AndroidProp "ro.build.version.sdk"
    abi = Get-AndroidProp "ro.product.cpu.abi"
    abi_list = Get-AndroidProp "ro.product.cpu.abilist"
    total_memory = Invoke-AdbText -Arguments @("shell", "grep", "MemTotal", "/proc/meminfo")
    data_storage = Invoke-AdbText -Arguments @("shell", "df", "-k", "/data")
    battery = Invoke-AdbText -Arguments @("shell", "dumpsys", "battery")
    mnn_chat_installed = $MnnInstalled
}

$Json = $Report | ConvertTo-Json -Depth 4
$Json

if ($OutputPath) {
    $ResolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
        $OutputPath
    } else {
        Join-Path $ProjectRoot $OutputPath
    }
    $Parent = Split-Path -Parent $ResolvedOutput
    if ($Parent) {
        New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    }
    Set-Content -LiteralPath $ResolvedOutput -Value $Json -Encoding utf8
}
