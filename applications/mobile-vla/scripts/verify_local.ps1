$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ManifestPath = Join-Path $ProjectRoot "artifacts\manifest.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$Checked = @()

foreach ($Entry in $Manifest.files) {
    $Candidate = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $ManifestPath) $Entry.path))
    if (-not (Test-Path -LiteralPath $Candidate)) {
        throw "Missing required artifact: $Candidate"
    }
    $File = Get-Item -LiteralPath $Candidate
    if ($File.Length -ne [int64]$Entry.bytes) {
        throw "Size mismatch: $Candidate"
    }
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Candidate).Hash
    if ($Hash -ne $Entry.sha256) {
        throw "SHA-256 mismatch: $Candidate"
    }
    $Checked += [ordered]@{ path = $Candidate; bytes = $File.Length; sha256 = $Hash }
}

$ModelDir = Join-Path $ProjectRoot "models\Qwen3.5-2B-MNN"
$RuntimeConfig = Get-Content -LiteralPath (Join-Path $ModelDir "config.json") -Raw | ConvertFrom-Json
$LlmConfig = Get-Content -LiteralPath (Join-Path $ModelDir "llm_config.json") -Raw | ConvertFrom-Json
$ExportArgs = Get-Content -LiteralPath (Join-Path $ModelDir "export_args.json") -Raw | ConvertFrom-Json

if ($RuntimeConfig.backend_type -ne "cpu") { throw "Unexpected default backend" }
if ($LlmConfig.model_type -ne "qwen3_5" -or -not $LlmConfig.is_visual) {
    throw "Downloaded model is not the expected visual Qwen3.5 package"
}
if ($ExportArgs.quant_bit -ne 4 -or $ExportArgs.quant_block -ne 64) {
    throw "Unexpected model quantization settings"
}

$BenchmarkPath = Join-Path $ProjectRoot "benchmarks\senior_tasks.jsonl"
$Tasks = @(Get-Content -LiteralPath $BenchmarkPath | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
if ($Tasks.Count -ne 20) { throw "Expected 20 benchmark tasks, found $($Tasks.Count)" }
if (@($Tasks.task_id | Sort-Object -Unique).Count -ne 20) { throw "Task IDs are not unique" }

$TestsPath = Join-Path $ProjectRoot "tests"
python -m unittest discover -s $TestsPath -v
if ($LASTEXITCODE -ne 0) { throw "Python tests failed" }
$PythonTestCount = [int](python -c "import sys, unittest; print(unittest.TestLoader().discover(sys.argv[1]).countTestCases())" $TestsPath)
if ($LASTEXITCODE -ne 0 -or $PythonTestCount -le 0) {
    throw "Could not determine Python test count"
}

[ordered]@{
    status = "pass"
    checked_artifacts = $Checked.Count
    model_type = $LlmConfig.model_type
    is_visual = [bool]$LlmConfig.is_visual
    quant_bit = [int]$ExportArgs.quant_bit
    quant_block = [int]$ExportArgs.quant_block
    benchmark_tasks = $Tasks.Count
    python_tests = $PythonTestCount
} | ConvertTo-Json -Depth 4
