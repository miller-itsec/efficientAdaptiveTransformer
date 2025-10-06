# ============================================
# Collect and merge all timing logs (JSONL → CSV)
# Robust: uses FullName, adds debug output
# ============================================

$ErrorActionPreference = "Stop"

# --- Normalize working directory to project root
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

# --- Input / output
$LogDir    = Join-Path $ProjectRoot "results\logs"
$TablesDir = Join-Path $ProjectRoot "results\tables"
$OutCsv    = Join-Path $TablesDir "combined_frontier.csv"

Write-Host "[collect_all] ProjectRoot = $ProjectRoot"
Write-Host "[collect_all] LogDir      = $LogDir"
Write-Host "[collect_all] TablesDir   = $TablesDir"

# --- Ensure dirs
if (-not (Test-Path -LiteralPath $LogDir)) {
    Write-Host "[collect_all] No log directory found at $LogDir" -ForegroundColor Yellow
    exit 0
}
New-Item -ItemType Directory -Force -Path $TablesDir | Out-Null

# --- Gather all JSONL files (use FullName explicitly)
$JsonFiles = Get-ChildItem -LiteralPath $LogDir -Filter "*.jsonl" -File -Recurse
if (-not $JsonFiles -or $JsonFiles.Count -eq 0) {
    Write-Host "[collect_all] No JSONL logs found in $LogDir" -ForegroundColor Yellow
    exit 0
}

Write-Host "[collect_all] Found $($JsonFiles.Count) JSONL file(s):" -ForegroundColor Cyan
$JsonFiles | ForEach-Object { Write-Host "  - $($_.FullName)" }

# --- Accumulate rows as PSCustomObjects
$rows = New-Object System.Collections.Generic.List[object]
$parsedCount = 0
$skipCount = 0

foreach ($file in $JsonFiles) {
    $full = $file.FullName
    Write-Host "[collect_all] Processing: $full"

    try {
        # Read line-by-line (JSONL)
        Get-Content -LiteralPath $full | ForEach-Object {
            $line = $_.Trim()
            if ($line.Length -gt 0) {
                try {
                    $entry = $line | ConvertFrom-Json

                    $rows.Add([pscustomobject]@{
                        task          = $entry.task
                        model_type    = $entry.model_type
                        tau           = $(if ($null -ne $entry.tau) { $entry.tau } else { "" })
                        latency_ms    = $entry.latency_ms
                        accuracy      = $entry.accuracy
                        avg_depth     = $(if ($null -ne $entry.avg_depth) { $entry.avg_depth } else { "" })
                        avg_retention = $(if ($null -ne $entry.avg_retention) { $entry.avg_retention } else { "" })
                        bs            = $entry.bs
                        device        = $entry.device
                    })
                    $parsedCount++
                } catch {
                    $skipCount++
                    Write-Host "[collect_all] Skipping invalid JSON line in $full" -ForegroundColor Yellow
                }
            }
        }
    } catch {
        Write-Host "[collect_all] ERROR reading $full : $($_.Exception.Message)" -ForegroundColor Red
    }
}

if ($rows.Count -eq 0) {
    Write-Host "[collect_all] No valid entries parsed from logs in $LogDir" -ForegroundColor Yellow
    exit 0
}

Write-Host "[collect_all] Parsed rows: $parsedCount; Skipped lines: $skipCount"

# --- De-duplicate and sort
$rowsUnique = $rows |
    Sort-Object task, model_type, tau, latency_ms, accuracy -Unique |
    Sort-Object task, model_type, tau, latency_ms

# --- Write combined CSV (stable header)
$rowsUnique |
  Select-Object task, model_type, tau, latency_ms, accuracy, avg_depth, avg_retention, bs, device |
  Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8

# --- Also write per-task CSVs for plotting convenience
$tasks = $rowsUnique | Select-Object -ExpandProperty task -Unique
foreach ($t in $tasks) {
    $outTask = Join-Path $TablesDir ("frontier_{0}.csv" -f $t)
    $rowsUnique | Where-Object { $_.task -eq $t } |
      Select-Object task, model_type, tau, latency_ms, accuracy, avg_depth, avg_retention, bs, device |
      Export-Csv -Path $outTask -NoTypeInformation -Encoding UTF8
}

Write-Host "✅ Combined CSV: $OutCsv" -ForegroundColor Green
Write-Host "✅ Per-task CSVs: $TablesDir\frontier_{sst2,qqp,mnli}.csv" -ForegroundColor Green
