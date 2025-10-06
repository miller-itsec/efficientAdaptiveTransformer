# ============================================
# SST-2 end-to-end pipeline: train baselines + EAT, timing, FLOPs
# Works from any current directory. ASCII-only (no smart quotes).
# ============================================

$ErrorActionPreference = "Stop"

# --- Normalize working directory to project root (parent of this script)
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

# --- Console encoding safety
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# --- Helpers
function ToUnix([string]$p) { return ($p -replace '\\','/') }
function ModelReady([string]$dir) {
    $hasModel = (Test-Path (Join-Path $dir "pytorch_model.bin")) -or (Test-Path (Join-Path $dir "model.safetensors"))
    $hasTok   = (Test-Path (Join-Path $dir "tokenizer.json")) -and (Test-Path (Join-Path $dir "vocab.txt"))
    return ($hasModel -and $hasTok)
}

# --- Activate venv (create if missing)
$Activate1 = Join-Path $ProjectRoot "env\Scripts\activate"
$Activate2 = Join-Path $ProjectRoot "env\Scripts\Activate.ps1"
if (Test-Path $Activate1) {
    & $Activate1
} elseif (Test-Path $Activate2) {
    & $Activate2
} else {
    Write-Host "Virtual environment not found - creating..." -ForegroundColor Yellow
    python -m venv (Join-Path $ProjectRoot "env")
    & $Activate1
}

# --- Paths
$TrainPy = Join-Path $ProjectRoot "src\train.py"
$TimePy  = Join-Path $ProjectRoot "src\time_infer.py"
$FlopsPy = Join-Path $ProjectRoot "src\flops.py"

# --- Ensure output dirs
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "models")         | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "results")        | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "results\logs")   | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "results\tables") | Out-Null

# --- Show CUDA availability (diagnostic)
Write-Host ("CUDA available: " + (python -c "import torch; print(torch.cuda.is_available())"))

# -------- Train (fast, AMP, skip only if model+tokenizer exist) --------
# Tip: reduce --max_len to 128 for SST-2 for extra speed.

# BERT-base
$BertDir = Join-Path $ProjectRoot "models\sst2_bert_seed42"
if (ModelReady $BertDir) {
    Write-Host "Skipping BERT SST-2 (model+tokenizer found): $BertDir" -ForegroundColor Yellow
} else {
    python $TrainPy --task sst2 --model_type bert `
        --epochs 2 --batch_size 48 --lr 2e-5 --seed 42 `
        --fp16 --max_len 128 --save_each_epoch
}

# DistilBERT
$DistilDir = Join-Path $ProjectRoot "models\sst2_distilbert_seed42"
if (ModelReady $DistilDir) {
    Write-Host "Skipping DistilBERT SST-2 (model+tokenizer found): $DistilDir" -ForegroundColor Yellow
} else {
    python $TrainPy --task sst2 --model_type distilbert `
        --epochs 2 --batch_size 64 --lr 2e-5 --seed 42 `
        --fp16 --max_len 128 --save_each_epoch
}

# EAT
$EatDir = Join-Path $ProjectRoot "models\sst2_eat_seed42"
if (ModelReady $EatDir) {
    Write-Host "Skipping EAT SST-2 (model+tokenizer found): $EatDir" -ForegroundColor Yellow
} else {
    python $TrainPy --task sst2 --model_type eat `
        --epochs 2 --batch_size 48 --lr 2e-5 --seed 42 `
        --fp16 --max_len 128 `
        --anneal_prune --target_prune 0.30 --save_each_epoch
}

# -------- Latency (bs=1) --------
$BertPathUnix   = ToUnix (Join-Path $ProjectRoot "models\sst2_bert_seed42")
$DistilPathUnix = ToUnix (Join-Path $ProjectRoot "models\sst2_distilbert_seed42")
$EatPathUnix    = ToUnix (Join-Path $ProjectRoot "models\sst2_eat_seed42")

python $TimePy --task sst2 --model_type bert       --model_dir $BertPathUnix   --bs 1 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\timing_sst2.jsonl")
python $TimePy --task sst2 --model_type distilbert --model_dir $DistilPathUnix --bs 1 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\timing_sst2.jsonl")
foreach ($tau in 0.80, 0.85, 0.90, 0.95) {
    python $TimePy --task sst2 --model_type eat --model_dir $EatPathUnix --bs 1 --n_eval 1000 --warmup 50 --tau $tau `
        --detailed_log (Join-Path $ProjectRoot "results\tables\detailed_log.csv") `
        --log (Join-Path $ProjectRoot "results\logs\timing_sst2.jsonl")
}

# -------- Throughput (bs=32) --------
python $TimePy --task sst2 --model_type bert       --model_dir $BertPathUnix   --bs 32 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\throughput_sst2.jsonl")
python $TimePy --task sst2 --model_type distilbert --model_dir $DistilPathUnix --bs 32 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\throughput_sst2.jsonl")
python $TimePy --task sst2 --model_type eat        --model_dir $EatPathUnix    --bs 32 --n_eval 1000 --warmup 50 --tau 0.90 --log (Join-Path $ProjectRoot "results\logs\throughput_sst2.jsonl")

# -------- FLOPs (SST-2 uses max seq length 128) --------
python $FlopsPy --task sst2 --seq_len 128 --log (Join-Path $ProjectRoot "results\tables\flops_sst2.csv")

Write-Host "SST-2 pipeline complete." -ForegroundColor Green
