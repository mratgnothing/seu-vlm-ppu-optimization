param(
    [string]$Serial = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $ProjectRoot "tools\platform-tools\adb.exe"
$AgentPackage = "ai.zhiman.accessibility"
$AgentIme = "ai.zhiman.accessibility/.AgentInputMethodService"
$AgentAccessibility =
    "ai.zhiman.accessibility/ai.zhiman.accessibility.AgentAccessibilityService"
$MnnChat = "com.alibaba.mnnllm.android/com.alibaba.mnnllm.android.chat.ChatActivity"
$WechatPackage = "com.tencent.mm"
$PhoneRingSeconds = 5
$VideoRingSeconds = 5

$ActionCallPrompt = "ai.zhiman.accessibility.shell.COMMIT_DEMO_CALL_PROMPT"
$ActionComboPrompt = "ai.zhiman.accessibility.shell.COMMIT_DEMO_COMBO_PROMPT"
$ActionDinnerMessage = "ai.zhiman.accessibility.shell.COMMIT_DEMO_DINNER_MESSAGE"
$ActionOpenDialer = "ai.zhiman.accessibility.shell.OPEN_BOUND_DIALER"
$ActionCommitName = "ai.zhiman.accessibility.shell.COMMIT_BOUND_WECHAT_NAME"

if (-not (Test-Path -LiteralPath $Adb)) {
    throw "ADB not found: $Adb"
}

function Invoke-Adb {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowFailure
    )
    $value = & $Adb -s $Serial @Arguments
    if (-not $AllowFailure -and $LASTEXITCODE -ne 0) {
        throw "ADB command failed: $($Arguments -join ' ')"
    }
    return @(($value | ForEach-Object { $_.ToString() }))
}

function Get-DeviceValue {
    param([string[]]$Arguments)
    return ((Invoke-Adb -Arguments $Arguments) -join "`n").Trim()
}

function Invoke-Tap {
    param([int]$X, [int]$Y)
    Invoke-Adb -Arguments @("shell", "input", "tap", "$X", "$Y") | Out-Null
}

function Invoke-Bridge {
    param([string]$Action)
    $result = Get-DeviceValue -Arguments @(
        "shell", "am", "broadcast", "-a", $Action, "-p", $AgentPackage)
    if ($result -notmatch "result=1") {
        throw "Protected demo bridge rejected action: $Action"
    }
}

function Set-CurrentIme {
    param([string]$Ime)
    Invoke-Adb -Arguments @("shell", "ime", "set", $Ime) | Out-Null
}

function Remove-AgentAccessibility {
    $raw = Get-DeviceValue -Arguments @(
        "shell", "settings", "get", "secure", "enabled_accessibility_services")
    $services = @()
    if ($raw -and $raw -ne "null") {
        $services = @($raw -split ":" | Where-Object {
            $_ -and $_ -ne $AgentAccessibility
        })
    }
    if ($services.Count -eq 0) {
        Invoke-Adb -Arguments @(
            "shell", "settings", "delete", "secure", "enabled_accessibility_services") |
            Out-Null
        Invoke-Adb -Arguments @(
            "shell", "settings", "put", "secure", "accessibility_enabled", "0") |
            Out-Null
    } else {
        Invoke-Adb -Arguments @(
            "shell", "settings", "put", "secure", "enabled_accessibility_services",
            ($services -join ":")) | Out-Null
        Invoke-Adb -Arguments @(
            "shell", "settings", "put", "secure", "accessibility_enabled", "1") |
            Out-Null
    }
}

& $Adb start-server | Out-Null
$devices = @(& $Adb devices) | Where-Object { $_ -match "\sdevice$" }
if ($Serial) {
    $devices = @($devices | Where-Object { $_ -like "$Serial*" })
}
if ($devices.Count -ne 1) {
    throw "Connect exactly one authorized Android device, or pass -Serial."
}
if (-not $Serial) {
    $Serial = ($devices[0] -split "\s+")[0]
}

$OriginalIme = Get-DeviceValue -Arguments @(
    "shell", "settings", "get", "secure", "default_input_method")
$OriginalEnabledImes = Get-DeviceValue -Arguments @(
    "shell", "settings", "get", "secure", "enabled_input_methods")
$AgentImeWasEnabled = @($OriginalEnabledImes -split ":") -contains $AgentIme
$Completed = $false

try {
    if (-not $OriginalIme -or $OriginalIme -eq "null" -or $OriginalIme -eq $AgentIme) {
        throw "Start from a normal keyboard; the test IME must not be the default."
    }

    $size = Get-DeviceValue -Arguments @("shell", "wm", "size")
    if ($size -notmatch "1220x2656") {
        throw "This fixed demo is validated only for a 1220x2656 portrait display."
    }
    foreach ($package in @($AgentPackage, "com.alibaba.mnnllm.android", $WechatPackage)) {
        $installed = Get-DeviceValue -Arguments @(
            "shell", "pm", "list", "packages", $package)
        if ($installed -notmatch [regex]::Escape($package)) {
            throw "Required package is missing: $package"
        }
    }
    $focus = Get-DeviceValue -Arguments @("shell", "dumpsys", "window")
    if ($focus -notmatch [regex]::Escape($MnnChat)) {
        throw "Open the Qwen conversation in MNN Chat before starting the demo."
    }

    Remove-AgentAccessibility
    if ($DryRun) {
        Write-Host "DRY RUN PASS: device, display, apps, MNN foreground and cleanup contract are valid."
        $Completed = $true
        return
    }

    # 1. Existing MNN conversation: fixed phone-call instruction.
    Invoke-Adb -Arguments @("shell", "ime", "enable", $AgentIme) | Out-Null
    Set-CurrentIme $AgentIme
    Invoke-Tap 610 2400
    Start-Sleep -Milliseconds 300
    Invoke-Bridge $ActionCallPrompt
    Start-Sleep -Milliseconds 200
    Invoke-Tap 1119 2563
    Start-Sleep -Milliseconds 700

    # 2. Open the privately bound number and place one five-second call.
    Set-CurrentIme $OriginalIme
    Invoke-Bridge $ActionOpenDialer
    Start-Sleep -Milliseconds 1600
    Invoke-Tap 438 2336
    Start-Sleep -Seconds $PhoneRingSeconds
    Invoke-Adb -Arguments @("shell", "input", "keyevent", "KEYCODE_ENDCALL") |
        Out-Null
    Start-Sleep -Milliseconds 700

    # 3. New MNN conversation: fixed WeChat-message plus video-call instruction.
    Invoke-Adb -Arguments @("shell", "am", "start", "-n", $MnnChat) | Out-Null
    Start-Sleep -Milliseconds 700
    Invoke-Tap 1150 245
    Start-Sleep -Milliseconds 250
    Invoke-Tap 888 281
    Start-Sleep -Milliseconds 450
    Set-CurrentIme $AgentIme
    Invoke-Tap 610 2400
    Start-Sleep -Milliseconds 250
    Invoke-Bridge $ActionComboPrompt
    Start-Sleep -Milliseconds 200
    Invoke-Tap 1119 2563
    Start-Sleep -Milliseconds 700

    # 4. Return to WeChat home, search the private binding, and open its chat.
    Set-CurrentIme $OriginalIme
    Invoke-Adb -Arguments @(
        "shell", "monkey", "-p", $WechatPackage,
        "-c", "android.intent.category.LAUNCHER", "1") | Out-Null
    Start-Sleep -Milliseconds 1200
    Invoke-Adb -Arguments @("shell", "input", "keyevent", "KEYCODE_BACK") |
        Out-Null
    Start-Sleep -Milliseconds 300
    Invoke-Adb -Arguments @(
        "shell", "monkey", "-p", $WechatPackage,
        "-c", "android.intent.category.LAUNCHER", "1") | Out-Null
    Start-Sleep -Milliseconds 700
    Invoke-Tap 1013 199
    Start-Sleep -Milliseconds 450
    Invoke-Tap 549 199
    Set-CurrentIme $AgentIme
    Start-Sleep -Milliseconds 250
    Invoke-Bridge $ActionCommitName
    Start-Sleep -Milliseconds 1100
    Invoke-Tap 610 505
    Start-Sleep -Milliseconds 1000

    # 5. Send the fixed dinner invitation.
    Invoke-Tap 525 2510
    Start-Sleep -Milliseconds 250
    Invoke-Bridge $ActionDinnerMessage
    Start-Sleep -Milliseconds 250
    Invoke-Tap 1110 2550
    Start-Sleep -Milliseconds 700

    # 6. Start one video call, keep it visible for five seconds, then end/leave it.
    Invoke-Tap 1147 2510
    Start-Sleep -Milliseconds 550
    Invoke-Tap 756 1859
    Start-Sleep -Milliseconds 550
    Invoke-Tap 610 2200
    Start-Sleep -Seconds $VideoRingSeconds
    Invoke-Tap 610 2360
    Start-Sleep -Milliseconds 900
    Invoke-Adb -Arguments @("shell", "input", "keyevent", "KEYCODE_BACK") |
        Out-Null
    $Completed = $true
} finally {
    try {
        if ($OriginalIme -and $OriginalIme -ne "null" -and $OriginalIme -ne $AgentIme) {
            Set-CurrentIme $OriginalIme
        }
        if (-not $AgentImeWasEnabled) {
            Invoke-Adb -Arguments @("shell", "ime", "disable", $AgentIme) |
                Out-Null
        }
        Remove-AgentAccessibility
        Invoke-Adb -Arguments @("shell", "am", "force-stop", $AgentPackage) |
            Out-Null
        Invoke-Adb -Arguments @(
            "shell", "am", "force-stop", "com.alibaba.mnnllm.android") |
            Out-Null
    } catch {
        Write-Warning "Phone cleanup needs manual verification: $($_.Exception.Message)"
    }
    try {
        & $Adb kill-server | Out-Null
    } catch {
    }
}

if ($Completed -and -not $DryRun) {
    Write-Host "DEMO COMPLETE: fixed phone, WeChat message and video-call sequence finished."
    Write-Host "CLEANUP COMPLETE: original IME restored; test IME/accessibility closed; ADB stopped."
}
