# ============================================
# make_tex_data.ps1
# Split combined_frontier.csv into per-task .csv files
# used by LaTeX PGFPlots in efficient_adaptive_transformer.tex
# ============================================

$ErrorActionPreference = "Stop"

# --- Normalize working directory
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$TablesDir = Join-Path $ProjectRoot "results\tables"
$Combined  = Join-Path $TablesDir "combined_frontier.csv"

if (-not (Test-Path $Combined)) {
    Write-Host "❌ Combined CSV not found: $Combined" -ForegroundColor Red
    exit 1
}

# --- Load combined CSV
$df = Import-Csv -Path $Combined
if ($df.Count -eq 0) {
    Write-Host "❌ No data found in $Combined" -ForegroundColor Red
    exit 1
}

# --- Group by task and export individual CSVs
$tasks = $df | Select-Object -ExpandProperty task -Unique
foreach ($task in $tasks) {
    $taskName = $task.Trim()
    if (-not $taskName) { continue }

    $out = Join-Path $TablesDir ("frontier_{0}.csv" -f $taskName)
    $df | Where-Object { $_.task -eq $taskName } |
        Export-Csv -Path $out -NoTypeInformation -Encoding UTF8

    Write-Host "✅ Wrote $out" -ForegroundColor Green
}

Write-Host "All per-task CSVs written to $TablesDir" -ForegroundColor Cyan
