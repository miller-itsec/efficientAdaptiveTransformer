# In scripts/run_everything.ps1

$ErrorActionPreference = "Stop"

# --- Add this block for robustness ---
# Normalize working directory to project root
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot
# ------------------------------------

Write-Host "Starting the full EAT experiment pipeline..." -ForegroundColor Green

# --- Step 1: Run all per-task experiments ---
Write-Host "Running SST-2 experiments..."
.\scripts\run_all_sst2.ps1

Write-Host "Running QQP experiments..."
.\scripts\run_all_qqp.ps1

Write-Host "Running MNLI experiments..."
.\scripts\run_all_mnli.ps1

# --- Step 2: Build LaTeX CSVs ---
Write-Host "Building CSVs..."
.\scripts\collect_all.ps1
.\scripts\make_tex_data.ps1

# --- Step 3: Generate final summaries and plots ---
Write-Host "Creating summary tables and plots..."
python src/summarize_results.py
python src/plot_ablation.py
python src/plot_frontiers.py

Write-Host "Full pipeline complete. Results are in the /results folder." -ForegroundColor Green
