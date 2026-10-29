param(
    [string]$Serial = "",
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Adb = Join-Path $ProjectRoot "tools\platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $Adb)) { throw "ADB not found: $Adb" }

if (-not $OutputPath) {
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputPath = Join-Path $ProjectRoot "evidence\screen-$Stamp.png"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $ProjectRoot $OutputPath
}

$Parent = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Path $Parent -Force | Out-Null
$Arguments = [System.Collections.Generic.List[string]]::new()
if ($Serial) {
    $Arguments.Add("-s")
    $Arguments.Add($Serial)
}
$Arguments.Add("exec-out")
$Arguments.Add("screencap")
$Arguments.Add("-p")

# Native stdout redirection through Windows PowerShell can transcode binary
# PNG bytes as UTF-16. Copy the process byte stream directly instead.
$StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $Adb
$StartInfo.UseShellExecute = $false
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true
$StartInfo.Arguments = ($Arguments -join " ")

$Process = [System.Diagnostics.Process]::new()
$Process.StartInfo = $StartInfo
$null = $Process.Start()
$Stream = [System.IO.File]::Create($OutputPath)
try {
    $Process.StandardOutput.BaseStream.CopyTo($Stream)
}
finally {
    $Stream.Dispose()
}
$Process.WaitForExit()
if ($Process.ExitCode -ne 0) {
    throw "Screenshot capture failed: $($Process.StandardError.ReadToEnd())"
}
Write-Output $OutputPath
