# ============================================
# MNLI (matched) end-to-end pipeline: train baselines + EAT, timing, FLOPs
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

# --- MNLI (3-way classification) ---

# BERT-base
$BertDir = Join-Path $ProjectRoot "models\mnli_bert_seed42"
if (ModelReady $BertDir) {
    Write-Host "Skipping BERT MNLI (model+tokenizer found): $BertDir" -ForegroundColor Yellow
} else {
    python $TrainPy --task mnli --model_type bert `
        --epochs 2 --batch_size 48 --lr 3e-5 --seed 42 `
        --fp16 --max_len 320 --save_each_epoch
}

# DistilBERT
$DistilDir = Join-Path $ProjectRoot "models\mnli_distilbert_seed42"
if (ModelReady $DistilDir) {
    Write-Host "Skipping DistilBERT MNLI (model+tokenizer found): $DistilDir" -ForegroundColor Yellow
} else {
    python $TrainPy --task mnli --model_type distilbert `
        --epochs 2 --batch_size 64 --lr 3e-5 --seed 42 `
        --fp16 --max_len 320 --save_each_epoch
}

# EAT
$EatDir = Join-Path $ProjectRoot "models\mnli_eat_seed42"
if (ModelReady $EatDir) {
    Write-Host "Skipping EAT MNLI (model+tokenizer found): $EatDir" -ForegroundColor Yellow
} else {
    python $TrainPy --task mnli --model_type eat `
        --epochs 2 --batch_size 48 --lr 3e-5 --seed 42 `
        --fp16 --max_len 320 `
        --anneal_prune --target_prune 0.30 --save_each_epoch
}

# -------- Latency (bs=1) --------
$BertPathUnix   = ToUnix (Join-Path $ProjectRoot "models\mnli_bert_seed42")
$DistilPathUnix = ToUnix (Join-Path $ProjectRoot "models\mnli_distilbert_seed42")
$EatPathUnix    = ToUnix (Join-Path $ProjectRoot "models\mnli_eat_seed42")

python $TimePy --task mnli --model_type bert       --model_dir $BertPathUnix   --bs 1 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\timing_mnli.jsonl")
python $TimePy --task mnli --model_type distilbert --model_dir $DistilPathUnix --bs 1 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\timing_mnli.jsonl")
foreach ($tau in 0.80, 0.85, 0.90, 0.95) {
    python $TimePy --task mnli --model_type eat --model_dir $EatPathUnix --bs 1 --n_eval 1000 --warmup 50 --tau $tau `
        --detailed_log (Join-Path $ProjectRoot "results\tables\detailed_log.csv") `
        --log (Join-Path $ProjectRoot "results\logs\timing_mnli.jsonl")
}

# -------- Throughput (bs=32) --------
python $TimePy --task mnli --model_type bert       --model_dir $BertPathUnix   --bs 32 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\throughput_mnli.jsonl")
python $TimePy --task mnli --model_type distilbert --model_dir $DistilPathUnix --bs 32 --n_eval 1000 --warmup 50 --log (Join-Path $ProjectRoot "results\logs\throughput_mnli.jsonl")
python $TimePy --task mnli --model_type eat        --model_dir $EatPathUnix    --bs 32 --n_eval 1000 --warmup 50 --tau 0.90 --log (Join-Path $ProjectRoot "results\logs\throughput_mnli.jsonl")

# -------- FLOPs (MNLI uses max seq length 320) --------
python $FlopsPy --task mnli --seq_len 320 --log (Join-Path $ProjectRoot "results\tables\flops_mnli.csv")

Write-Host "MNLI pipeline complete." -ForegroundColor Green
