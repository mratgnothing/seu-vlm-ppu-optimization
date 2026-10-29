param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ToolsRoot = Join-Path $ProjectRoot 'tools\android-build'
$SdkRoot = Join-Path $ToolsRoot 'sdk'
$GradleRoot = Join-Path $ToolsRoot 'gradle-9.6.0'
$Downloads = Join-Path $ToolsRoot 'downloads'
$JavaHome = Join-Path $ToolsRoot 'temurin-17'

$CmdlineZip = Join-Path $Downloads 'commandlinetools-win-15859902_latest.zip'
$CmdlineUrl = 'https://dl.google.com/android/repository/commandlinetools-win-15859902_latest.zip'
$CmdlineSha256 = '90ae805d20434428bffcb699c290860f19bb5f66a67e6b330067e3de801fb04a'

$GradleZip = Join-Path $Downloads 'gradle-9.6.0-bin.zip'
$GradleUrl = 'https://services.gradle.org/distributions/gradle-9.6.0-bin.zip'
$GradleSha256 = 'bbaeb2fef8710818cf0e261201dab964c572f92b942812df0c3620d62a529a01'

function Assert-Sha256([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. expected=$Expected actual=$actual"
    }
}

function Download-Verified([string]$Url, [string]$Path, [string]$Sha256) {
    if (Test-Path -LiteralPath $Path) {
        try {
            Assert-Sha256 $Path $Sha256
            Write-Host "Verified cached download: $Path"
            return
        } catch {
            Remove-Item -LiteralPath $Path -Force
        }
    }
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Path
    Assert-Sha256 $Path $Sha256
}

function Install-Temurin17([string]$Destination) {
    $javaExe = Join-Path $Destination 'bin\java.exe'
    $jlinkExe = Join-Path $Destination 'bin\jlink.exe'
    if ((Test-Path -LiteralPath $javaExe) -and (Test-Path -LiteralPath $jlinkExe)) {
        Write-Host "Using cached full JDK: $Destination"
        return
    }

    $api = 'https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&heap_size=normal&image_type=jdk&jvm_impl=hotspot&os=windows&page_size=1&project=jdk&vendor=eclipse'
    Write-Host 'Resolving latest Eclipse Temurin 17 JDK from Adoptium API...'
    $assets = @(Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'zhi-ban-mobile-vla-bootstrap/1.0' })
    if ($assets.Count -lt 1) {
        throw 'Adoptium API returned no Windows x64 JDK 17 asset.'
    }

    $package = $assets[0].binary.package
    if (-not $package.link -or -not $package.checksum -or -not $package.name) {
        throw 'Adoptium API response is missing package link/checksum/name.'
    }

    $jdkZip = Join-Path $Downloads $package.name
    Download-Verified $package.link $jdkZip $package.checksum

    $unpack = Join-Path $ToolsRoot 'temurin-unpack'
    if (Test-Path -LiteralPath $unpack) {
        Remove-Item -LiteralPath $unpack -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $unpack | Out-Null
    Expand-Archive -LiteralPath $jdkZip -DestinationPath $unpack -Force

    $jdkRoot = Get-ChildItem -LiteralPath $unpack -Directory |
        Where-Object {
            (Test-Path -LiteralPath (Join-Path $_.FullName 'bin\java.exe')) -and
            (Test-Path -LiteralPath (Join-Path $_.FullName 'bin\jlink.exe'))
        } |
        Select-Object -First 1

    if (-not $jdkRoot) {
        throw 'Downloaded Temurin archive does not contain a full JDK with jlink.exe.'
    }

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Move-Item -LiteralPath $jdkRoot.FullName -Destination $Destination
    Remove-Item -LiteralPath $unpack -Recurse -Force

    if (-not (Test-Path -LiteralPath $jlinkExe)) {
        throw "Full JDK installation failed: jlink.exe missing at $jlinkExe"
    }
    Write-Host "Installed full JDK: $Destination"
}

New-Item -ItemType Directory -Force -Path $ToolsRoot,$SdkRoot,$Downloads | Out-Null
Install-Temurin17 $JavaHome

Download-Verified $CmdlineUrl $CmdlineZip $CmdlineSha256
Download-Verified $GradleUrl $GradleZip $GradleSha256

$SdkManager = Join-Path $SdkRoot 'cmdline-tools\latest\bin\sdkmanager.bat'
if (-not (Test-Path -LiteralPath $SdkManager)) {
    $unpack = Join-Path $ToolsRoot 'cmdline-unpack'
    if (Test-Path -LiteralPath $unpack) {
        Remove-Item -LiteralPath $unpack -Recurse -Force
    }
    Expand-Archive -LiteralPath $CmdlineZip -DestinationPath $unpack -Force
    $latest = Join-Path $SdkRoot 'cmdline-tools\latest'
    New-Item -ItemType Directory -Force -Path (Split-Path $latest -Parent) | Out-Null
    if (Test-Path -LiteralPath $latest) {
        Remove-Item -LiteralPath $latest -Recurse -Force
    }
    Move-Item -LiteralPath (Join-Path $unpack 'cmdline-tools') -Destination $latest
    Remove-Item -LiteralPath $unpack -Recurse -Force
}

if (-not (Test-Path -LiteralPath (Join-Path $GradleRoot 'bin\gradle.bat'))) {
    Expand-Archive -LiteralPath $GradleZip -DestinationPath $ToolsRoot -Force
}

$env:JAVA_HOME = $JavaHome
$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
$env:Path = "$JavaHome\bin;$SdkRoot\platform-tools;$SdkRoot\build-tools\36.0.0;" + $env:Path

Write-Host 'Accepting Android SDK licenses...'
1..30 | ForEach-Object { 'y' } | & $SdkManager --sdk_root=$SdkRoot --licenses | Out-Host

Write-Host 'Installing Android SDK packages...'
& $SdkManager --sdk_root=$SdkRoot 'platform-tools' 'platforms;android-36' 'build-tools;36.0.0'
if ($LASTEXITCODE -ne 0) {
    throw "sdkmanager failed with exit code $LASTEXITCODE"
}

$Gradle = Join-Path $GradleRoot 'bin\gradle.bat'
$AndroidProject = Join-Path $ProjectRoot 'android\accessibility-agent'

Write-Host 'Building accessibility-agent debug APK...'
& $Gradle -p $AndroidProject :app:assembleDebug --stacktrace
if ($LASTEXITCODE -ne 0) {
    throw "Gradle build failed with exit code $LASTEXITCODE"
}

$Apk = Join-Path $AndroidProject 'app\build\outputs\apk\debug\app-debug.apk'
if (-not (Test-Path -LiteralPath $Apk)) {
    throw "Build completed but APK was not found at $Apk"
}

Write-Host "APK=$Apk"
Get-FileHash -Algorithm SHA256 -LiteralPath $Apk
