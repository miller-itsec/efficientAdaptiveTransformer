# Efficient Adaptive Transformer (EAT)

This repository contains training, timing, and analysis scripts for the **Efficient Adaptive Transformer (EAT)** benchmark suite — comparing **BERT**, **DistilBERT**, and **EAT** across GLUE tasks (**SST-2**, **QQP**, **MNLI**). It also includes a short **cybersecurity application guide** (anti-phishing and file-type detection), a **reproducibility checklist**, and a LaTeX-friendly workflow for paper figures.

---

## 🔧 1) Environment Setup (Windows-friendly)

```powershell
python -m venv env
env\Scripts\Activate.ps1
pip install -r src\requirements.txt
```

**CUDA (optional, recommended):**
- Verify GPU: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`  
- Ensure your CUDA enabled torch/torchaudio/torchvision builds match the driver.

**Hugging Face cache (optional):**
- To avoid re-downloading: set env var `HF_HOME` to a persistent location.

---

## 🏃 2) End-to-End Training Pipelines

Each GLUE task has a PowerShell pipeline under `scripts/` that **trains** models (skips if a matching checkpoint exists), **times** inference, and records logs.

```powershell
pwsh scripts/run_all_sst2.ps1
pwsh scripts/run_all_qqp.ps1
pwsh scripts/run_all_mnli.ps1
```

Checkpoints are saved under `models/` (e.g., `models/sst2_eat_seed42/`).  
All timing logs go to `results/logs/` (JSONL). FLOPs tables go to `results/tables/`.

> **Tip:** The scripts accept `--max_len` (e.g., 128 for SST-2) and `--fp16` to accelerate training. They also use “skip-if-exists” logic so you **don’t retrain twice** if you rerun pipelines.

---

## ⏱️ 3) Timing and FLOPs (included in the task scripts)

- Per-task latency (batch=1) and throughput (batch=32) are measured by `src/time_infer.py`.
- FLOPs estimates written by `src/flops.py`.
- EAT timing can sweep exit thresholds `--tau` (e.g., 0.80 … 0.95).

---

## 📊 4) Postprocessing & Plots (CSV-driven)

After running all tasks, **merge** logs, **summarize** metrics, and **plot** the frontiers.

### 4.1 Merge logs → one CSV
```powershell
pwsh scripts/collect_all.ps1
```
**Output:** `results/tables/combined_frontier.csv`

### 4.2 Summaries from the combined CSV
```bash
python scripts/summarize_results.py
```
**Output:** `results/tables/summary_metrics.csv`

### 4.3 Plot the ablation study plot
```bash
python scripts/plot_ablation.py
```
**Output:**  
- `results/plots/ablation_sst2.pdf`

### 4.4 Plot accuracy–latency frontiers
```bash
python scripts/plot_frontiers.py
```
**Output:**  
- `results/plots/frontier_sst2.pdf`  
- `results/plots/frontier_qqp.pdf`  
- `results/plots/frontier_mnli.pdf`

> These PDFs can be included directly in LaTeX (see Section 8).  
> The **old `make_tex_data.ps1` is retired** — plots are now generated from `combined_frontier.csv`.

---

## 🔁 5) Using Trained Models (Inference)

Quick checks on a saved checkpoint (example: SST-2 EAT, τ=0.90):

```bash
python src/time_infer.py --task sst2 --model_type eat   --model_dir "./models/sst2_eat_seed42"   --bs 1 --n_eval 1000 --warmup 50 --tau 0.90   --log "./results/logs/timing_sst2.jsonl"
```

**Common flags:**
- `--tau` : EAT early-exit threshold (higher = slower, often more accurate). Omit for BERT/DistilBERT.
- `--ignore_classifier_mismatch` : allows timing a checkpoint trained for a different label count (accuracy is **not** meaningful; use for profiling only).
- For Windows path issues, prefer quotes: `--model_dir "C:\path\to\model"`.

---

## 🛡️ 6) Security Applications (Recipes)

EAT’s adaptive depth + pruning make it attractive for **latency-sensitive** cyber pipelines.

### 6.1 Anti-phishing triage (email/URL/metadata)
- **Goal:** Flag likely-benign messages quickly; spend more compute only on “hard” cases.
- **Recipe:** Fine-tune EAT on a labeled phishing dataset (subject + body, tokenized as a pair if helpful).  
  Use `--tau` ~ **0.85–0.92** so easy examples exit early (avg depth ~4–5), while ambiguous samples go to the final layer.
- **Integration:** Run EAT as the first gate. If `confidence<τ` or **suspicious class**, hand off to slower analyzers (link unshortening, sandboxing, vision OCR on attachments).
- **Benefit:** Cut p50 latency significantly while keeping high recall via downstream escalation.

### 6.2 File-type or document family detection (metadata-first)
- **Goal:** Quickly classify likely file families before deeper content inspection.
- **Recipe:** Feed **filename + MIME + short header bytes** converted to a compact textual representation.  
  Pruning sheds uninformative tokens; sparse attention keeps local patterns.
- **Integration:** Use EAT outputs to route samples: benign → lightweight scanning; unknown/suspicious → deep static/dynamic analysis.
- **Benefit:** Boosts overall pipeline throughput with measured accuracy latency tradeoffs.

> **Deployment tip:** For production, freeze a **task-specific checkpoint** (e.g., `phishing_eat_seedN/`) and set a fixed `τ` after validation. Log **avg executed layers** & **final retention** from `time_infer.py` to monitor drift.

---

## 🧪 7) Reproducibility Checklist

- **Seeds**: Default seed = 42. The scripts set Python/NumPy/Torch seeds (`utils.set_seed`).  
- **Determinism** (optional): set `CUBLAS_WORKSPACE_CONFIG=:4096:8` & `torch.set_deterministic_debug_mode("warn")` if truly needed.
- **Software versions**: Pin `torch`, `transformers`, `datasets`, `safetensors`, `tqdm`. Store `pip freeze` in `results/ENVIRONMENT.txt`.
- **Hardware**: Note GPU (e.g., RTX 2080 Ti), driver, and whether FP16 was used.
- **Dataset provenance**: Loaded via `datasets.load_dataset("glue", subset)` and cached under `./data/`.  
- **Logging**: All timing logs are JSONL (one row per run) and merged into `combined_frontier.csv` for plots & tables.
- **Skip-if-exists**: Pipelines won’t retrain if `models/<task>_*_seed42/` already exists (saves time across reruns).

---

## 📝 8) LaTeX Paper Integration

In your `.tex`, include the generated figures:

```latex
\includegraphics[width=0.32\textwidth]{results/plots/frontier_sst2.pdf}
\includegraphics[width=0.32\textwidth]{results/plots/frontier_qqp.pdf}
\includegraphics[width=0.32\textwidth]{results/plots/frontier_mnli.pdf}
```

Tables can be built from `results/tables/summary_metrics.csv` (PGFPlots/pgfplotstable or pasted values).  
Security appendix can cite the application recipes above.

---

## 📦 9) Model Export (Optional)

**ONNX (CPU/GPU inference via ONNX Runtime):**
```bash
pip install optimum onnxruntime-gpu
python - <<'PY'
from optimum.exporters.onnx import export
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, pathlib
mdir = "models/sst2_eat_seed42"
tok  = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
mdl  = AutoModelForSequenceClassification.from_pretrained(mdir, local_files_only=True).eval()
out  = pathlib.Path("models/sst2_eat_seed42_onnx")
out.mkdir(parents=True, exist_ok=True)
export(task="text-classification", model=mdl, output=out, tokenizer=tok)
print("Exported to", out)
PY
```

**TorchScript (quick CPU/GPU deploy):**
```python
import torch
from transformers import AutoModelForSequenceClassification
m = AutoModelForSequenceClassification.from_pretrained("models/sst2_eat_seed42", local_files_only=True).eval()
example = { "input_ids": torch.ones(1,128,dtype=torch.long),
            "attention_mask": torch.ones(1,128,dtype=torch.long) }
ts = torch.jit.trace(m, (example["input_ids"], example["attention_mask"]))
ts.save("models/sst2_eat_seed42.ts")
```

> EAT’s early-exit threshold is a runtime flag in our scripts; if you need the same in ONNX/TorchScript, wrap the forward() in a small driver function that applies early-exit logic before export.

---

## 🩺 10) Troubleshooting (Windows)

- **Tokenizer path errors** (`Can't load tokenizer for 'C:\...models\qqp_bert_seed42'`)  
  → Ensure the folder has `tokenizer.json`, `vocab.txt`, `tokenizer_config.json` and use quoted paths.
- **Label mismatch** (e.g., “size mismatch for bias … 3 vs 2”)  
  → You tried to load an **MNLI (3-label)** checkpoint for a **2-label** task. Use the matching `models/<task>_*` folder or add `--ignore_classifier_mismatch` for timing-only profiling.
- **CUDA available but slow** (WDDM contention)  
  → Close heavy GPU apps; keep batch small; ensure FP16 is enabled (`--fp16`) and `pin_memory` is False on CPU-only.
- **PowerShell JSON quirks**  
  → Our scripts avoid `ConvertFrom-Json -AsArray` for compatibility; paths are normalized internally.
- **`multiprocessing` lambda pickling error**  
  → Avoid lambdas in DataLoader workers on Windows; our code uses safe patterns already.

---

## 📂 11) Expected Output Layout

```
├── models/
│   ├── sst2_eat_seed42/           # etc.
│   ├── qqp_eat_seed42/
│   └── mnli_eat_seed42/
├── results/
│   ├── logs/
│   │   └── timing_mnli.jsonl
│   │   ├── timing_sst2.jsonl
│   │   ├── timing_qqp.jsonl
│   ├── tables/
│   │   ├── combined_frontier.csv
│   │   ├── summary_ablation_sst2.csv
│   │   └── flops_mnli.csv
│   │   ├── flops_qqp.csv
│   │   ├── flops_sst2.csv
│   └── plots/
│       ├── frontier_sst2.pdf
│       └── frontier_mnli.pdf
│       ├── frontier_qqp.pdf
│       └── ablation_sst2.pdf
```

---

## 📄 12) Licensing & Data

- GLUE datasets are loaded via the `datasets` library (“glue”, subsets `sst2`, `qqp`, `mnli`). Respect the GLUE license.
- Include proper citations in your LaTeX (`references.bib` already contains BERT/DistilBERT, pruning, sparse attention, early exit works).

---

**Maintainer:** Jan Miller (OPSWAT) • `jan.miller@opswat.com`  
**Last updated:** October 2025
