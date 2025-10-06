"""
Timing / accuracy harness for BERT, DistilBERT, and EAT.

- Measures latency on bs=1 and throughput on bs>1 over N dev examples with warmup.
- Supports EAT early-exit threshold sweep via --tau.
- Requests runtime stats from EAT (avg executed depth, final retention).
- Writes JSONL logs and appends compact CSV rows.
"""

# ---- Make transformers import fast/safe on Windows (no torchvision), and avoid hub telemetry
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import csv
import pathlib
import argparse
import time
import json
import glob
import safetensors.torch as st
from functools import partial
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig, BertTokenizerFast, DistilBertTokenizerFast,
    BertForSequenceClassification, DistilBertForSequenceClassification
)
from eat_model import EATForSequenceClassification

# (Keep all helper functions from the original script: normalize_path, has_tokenizer_files, etc.)
# ... (all functions down to `load_model` remain the same) ...
# NOTE: For brevity, only the modified `main` function is shown below. You should replace
# the `main` function in your `time_infer.py` with this new one and keep all the helper
# functions above it as they were.

# ------------------------- Path utilities -------------------------

def normalize_path(p):
    """Normalize to an absolute pathlib.Path and avoid hub validation on Windows."""
    if p is None:
        return None
    p = str(p).strip().strip('"').strip("'")
    return pathlib.Path(p).expanduser().resolve()

def has_tokenizer_files(path: pathlib.Path) -> bool:
    """Check minimal tokenizer files to ensure local load succeeds."""
    needed = ["tokenizer.json", "vocab.txt", "tokenizer_config.json"]
    return path.is_dir() and all((path / n).exists() for n in needed)


# ------------------------- Device utils -------------------------

def device_name() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    return "CPU"


# ------------------------- Loaders -------------------------

def build_tokenizer(model_dir, is_distil=False):
    path = normalize_path(model_dir)
    # Prefer local folder when present (bypass hub validation)
    if path and path.exists() and has_tokenizer_files(path):
        if is_distil:
            return DistilBertTokenizerFast.from_pretrained(path, use_fast=True, local_files_only=True)
        return BertTokenizerFast.from_pretrained(path, use_fast=True, local_files_only=True)

    # Safe fallback to base tokenizer (keeps script running if folder is incomplete)
    base = "distilbert-base-uncased" if is_distil else "bert-base-uncased"
    if is_distil:
        return DistilBertTokenizerFast.from_pretrained(base, use_fast=True)
    return BertTokenizerFast.from_pretrained(base, use_fast=True)


def infer_ckpt_num_labels(model_dir: str):
    """Infer num_labels from config.json (num_labels or id2label/label2id),
    else from the classifier tensor shape inside the checkpoint weights."""
    p = normalize_path(model_dir)
    if not (p and p.exists() and p.is_dir()):
        return None  # remote or non-local -> skip

    # 1) Try config.json
    cfg_path = p / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "num_labels" in cfg and cfg["num_labels"] is not None:
                return int(cfg["num_labels"])
            # fallbacks via label maps
            if "label2id" in cfg and isinstance(cfg["label2id"], dict) and len(cfg["label2id"]) > 0:
                return int(len(cfg["label2id"]))
            if "id2label" in cfg and isinstance(cfg["id2label"], dict) and len(cfg["id2label"]) > 0:
                return int(len(cfg["id2label"]))
        except Exception:
            pass

    # 2) Inspect weights for classifier out_features
    # Support .bin (PyTorch) and .safetensors
    cand_bias_keys = [
        "classifier.bias",                  # HF BERT/DistilBERT style
        "classifier.out_proj.bias",         # alt patterns
        "score.bias",                       # some models use "score"
        "classifier_4.bias",                # early-exit head names (if any)
        "cls.bias",                         # generic
    ]
    cand_weight_keys = [
        "classifier.weight",
        "classifier.out_proj.weight",
        "score.weight",
        "classifier_4.weight",
        "cls.weight",
    ]

    # Try safetensors first (fast, safe)
    try:
        import safetensors.torch as st
        st_paths = list(glob.glob(str(p / "*.safetensors")))
        if st_paths:
            state = st.load_file(st_paths[0])
            for k in cand_bias_keys:
                if k in state:
                    return int(state[k].shape[-1])
            for k in cand_weight_keys:
                if k in state:
                    # weight: [out_features, in_features]
                    return int(state[k].shape[0])
    except Exception:
        pass

    # Fallback: pytorch .bin
    try:
        import torch as _torch
        bin_path = p / "pytorch_model.bin"
        if bin_path.exists():
            sd = _torch.load(str(bin_path), map_location="cpu")
            for k in cand_bias_keys:
                if k in sd:
                    return int(sd[k].shape[-1])
            for k in cand_weight_keys:
                if k in sd:
                    return int(sd[k].shape[0])
    except Exception:
        pass

    return None


def _ensure_cls_out_features(model, num_labels: int):
    # Works for typical HF classifiers (Linear head called "classifier")
    head = getattr(model, "classifier", None)
    if head is None:
        return
    if getattr(head, "out_features", None) != num_labels:
        in_f = head.in_features
        device = next(model.parameters()).device
        new_head = torch.nn.Linear(in_f, num_labels).to(device)
        model.classifier = new_head  # replace in-place


def _load_state_dict_from_dir(path: pathlib.Path):
    """Return a state_dict from a .safetensors or .bin in the folder."""
    st_paths = list(glob.glob(str(path / "*.safetensors")))
    if st_paths:
        return st.load_file(st_paths[0])
    bin_path = path / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    return None


def load_model(model_type, model_dir, num_labels, allow_mismatch=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = normalize_path(model_dir)
    local = bool(path and path.exists() and path.is_dir())
    local_kwargs = {"local_files_only": True} if local else {}

    if model_type in ("bert", "distilbert"):
        source = path if local else ("distilbert-base-uncased" if model_type == "distilbert" else "bert-base-uncased")
        cls = DistilBertForSequenceClassification if model_type == "distilbert" else BertForSequenceClassification
        m = cls.from_pretrained(source, num_labels=num_labels, **local_kwargs)
        return m.to(device).eval()

    # ---- EAT branch (the tricky one)
    source = path if local else "bert-base-uncased"
    cfg = AutoConfig.from_pretrained(source, local_files_only=local)
    cfg.num_labels = int(num_labels)  # hard-set

    # 1) Try standard from_pretrained WITH num_labels passed through to __init__
    try:
        m = EATForSequenceClassification.from_pretrained(
            source,
            config=cfg,
            num_labels=num_labels,                 # <-- force init head size
            ignore_mismatched_sizes=allow_mismatch,
            **local_kwargs
        )
        # sanity: make sure the instantiated head really has the right size
        _ensure_cls_out_features(m, num_labels)
        return m.to(device).eval()
    except RuntimeError as e:
        # 2) Fallback: manual construct + manual state_dict load
        #    This avoids HF trying to load before we can fix the head.
        m = EATForSequenceClassification(cfg, num_labels=num_labels)
        _ensure_cls_out_features(m, num_labels)
        if local:
            sd = _load_state_dict_from_dir(path)
            if sd is not None:
                # Load matching keys; skip any leftover mismatches
                m.load_state_dict(sd, strict=False)
        return m.to(device).eval()

# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sst2", "qqp", "mnli"], required=True)
    parser.add_argument("--model_type", choices=["bert", "distilbert", "eat"], required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--bs", type=int, default=1, help="Batch size. Use 1 for latency; >1 for throughput.")
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--tau", type=float, default=None, help="EAT early-exit threshold (inference)")
    parser.add_argument("--log", default="./results/logs/timing.jsonl")
    parser.add_argument("--detailed_log", default=None, help="Path to new CSV for detailed per-example stats.")
    parser.add_argument("--ignore_classifier_mismatch", action="store_true",
                        help="Proceed even if checkpoint label count differs from task; "
                             "classifier will be re-initialized (accuracy invalid).")
    args = parser.parse_args()

    meta = {
        "sst2": ("glue", "sst2", "sentence", 2, 256, "validation"),
        "qqp":  ("glue", "qqp", ("question1", "question2"), 2, 256, "validation"),
        "mnli": ("glue", "mnli", ("premise", "hypothesis"), 3, 320, "validation_matched"),
    }
    dname, subset, fields, num_labels, maxlen, split = meta[args.task]

    ckpt_labels = infer_ckpt_num_labels(args.model_dir)
    allow_mismatch = False
    if ckpt_labels is not None and ckpt_labels != num_labels:
        if args.ignore_classifier_mismatch:
            allow_mismatch = True
        else:
            raise SystemExit(f"\n[ERROR] Label mismatch: task='{args.task}' expects {num_labels}, but ckpt has {ckpt_labels}.\n")

    ds = load_dataset(dname, subset, split=split, cache_dir="./data").shuffle(seed=1234)
    total_needed = min(args.n_eval + args.warmup, len(ds))
    ds = ds.select(range(total_needed))

    is_distil = args.model_type == "distilbert"
    tok = build_tokenizer(args.model_dir, is_distil)
    model = load_model(args.model_type, args.model_dir, num_labels, allow_mismatch=allow_mismatch)
    if args.model_type == "eat":
        print("Compiling the EAT model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
	device = next(model.parameters()).device

    detailed_log_file = None
    csv_writer = None
    if args.detailed_log:
        detailed_log_file = open(args.detailed_log, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(detailed_log_file)
        # Write header for the new CSV
        csv_writer.writerow(["task", "tau", "input_len", "final_retention", "executed_depth"])

    def run_model_on_batch(batch_encoding):
        with torch.no_grad():
            batch_encoding = {k: v.to(device, non_blocking=True) for k, v in batch_encoding.items()}
            if args.model_type == "eat":
                out = model(**batch_encoding, exit_threshold=args.tau, return_stats=True)
            else:
                out = model(**batch_encoding)
            
            if isinstance(out, dict):
                return out["logits"], out.get("stats")
            return out[0], out[2] if len(out) >= 3 else None

    times, preds, labels, depths, retentions = [], [], [], [], []

    if args.bs == 1: # LATENCY PATH
        warmup_ds = ds.select(range(min(args.warmup, len(ds))))
        eval_ds = ds.select(range(min(args.warmup, len(ds)), total_needed))
        
        for ex in warmup_ds:
            if isinstance(fields, str):
                # Handle single-sentence tasks
                enc = tok(ex[fields], truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
            else:
                # Handle sentence-pair tasks
                enc = tok(ex[fields[0]], ex[fields[1]], truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
            _ = run_model_on_batch(enc)

        for ex in eval_ds:
            example_text = ex[fields] if isinstance(fields, str) else ex[fields[0]] + " " + ex[fields[1]]
            input_len = len(tok.encode(example_text))

            if isinstance(fields, str):
                # Handle single-sentence tasks
                enc = tok(ex[fields], truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
            else:
                # Handle sentence-pair tasks
                enc = tok(ex[fields[0]], ex[fields[1]], truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, stats = run_model_on_batch(enc)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

            preds.append(torch.argmax(logits, dim=-1).item())
            labels.append(ex["label"])
            
            if stats:
                d = stats.get("executed_depth", np.nan)
                r = stats.get("final_retention", np.nan)
                depths.append(d)
                retentions.append(r)
                if csv_writer:
                    csv_writer.writerow([args.task, args.tau, input_len, r, d])

    else: # THROUGHPUT PATH
        def collate_fn(batch):
            if isinstance(fields, str):
                # Handle single-sentence tasks (e.g., SST-2)
                text_inputs = [ex[fields] for ex in batch]
                enc = tok(text_inputs, truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
            else:
                # Handle sentence-pair tasks (e.g., QQP, MNLI)
                text_inputs_1 = [ex[fields[0]] for ex in batch]
                text_inputs_2 = [ex[fields[1]] for ex in batch]
                enc = tok(text_inputs_1, text_inputs_2, truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
            
            return enc, torch.tensor([ex["label"] for ex in batch])

        loader = DataLoader(ds, batch_size=args.bs, collate_fn=collate_fn)
        
        warmup_batches = args.warmup // args.bs
        for i, (batch_enc, _) in enumerate(loader):
            if i >= warmup_batches: break
            _ = run_model_on_batch(batch_enc)

        eval_batches = args.n_eval // args.bs
        batch_count = 0
        for batch_enc, batch_labels in loader:
            if batch_count >= eval_batches: break
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, stats = run_model_on_batch(batch_enc)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            # Latency per example in the batch
            times.extend([((t1 - t0) * 1000.0) / args.bs] * args.bs)

            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch_labels.tolist())
            if stats: # For EAT, stats are per-batch averages
                depths.append(stats.get("executed_depth", np.nan))
                retentions.append(stats.get("final_retention", np.nan))
            batch_count += 1

    lat_ms = float(np.mean(times)) if times else float("nan")
    acc = float(np.mean(np.array(preds) == np.array(labels))) if preds else float("nan")
    avg_depth = float(np.nanmean(depths)) if depths else None
    avg_retention = float(np.nanmean(retentions)) if retentions else None

    result = {
        "task": args.task, "model_type": args.model_type, "model_dir": str(normalize_path(args.model_dir)),
        "bs": args.bs, "n": len(preds), "latency_ms": lat_ms, "accuracy": acc, "tau": args.tau,
        "device": device_name(), "avg_depth": avg_depth, "avg_retention": avg_retention,
    }

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    
    csv_dir = "./results/tables"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"frontier_{args.task}.csv")
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as w:
        if header_needed:
            w.write("model_type,tau,latency_ms,accuracy,avg_depth,avg_retention,bs,device\n")
        tau_str = "" if args.tau is None else f"{args.tau:.2f}"
        w.write(
            f"{args.model_type},{tau_str},{lat_ms:.4f},{acc:.4f},"
            f"{'' if avg_depth is None else f'{avg_depth:.3f}'},"
            f"{'' if avg_retention is None else f'{avg_retention:.3f}'},"
            f"{args.bs},{result['device']}\n"
        )
    if detailed_log_file:
        detailed_log_file.close()
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()