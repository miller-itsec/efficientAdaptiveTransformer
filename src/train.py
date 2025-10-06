import os, argparse, time, json
# --- IMPORTANT: disable torchvision import paths in transformers (set BEFORE importing transformers)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from functools import partial

from transformers import (
    BertForSequenceClassification, BertTokenizerFast,
    DistilBertForSequenceClassification, DistilBertTokenizerFast,
    BertConfig, get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from torch.optim import AdamW  # use PyTorch AdamW
from tqdm import tqdm

from utils import set_seed, ensure_dir, log_jsonl, metric_sst2, metric_qqp, metric_mnli, device_name
from eat_model import EATForSequenceClassification

# ----------------------- Task registry -----------------------

TASK2INFO = {
    "sst2": {"name": "glue", "subset": "sst2", "text": ("sentence",), "metric": metric_sst2, "num_labels": 2, "maxlen": 256},
    "qqp":  {"name": "glue", "subset": "qqp",  "text": ("question1", "question2"), "metric": metric_qqp, "num_labels": 2, "maxlen": 256},
    "mnli": {"name": "glue", "subset": "mnli", "text": ("premise", "hypothesis"),   "metric": metric_mnli, "num_labels": 3, "maxlen": 320, "split":"validation_matched"}
}

# ----------------------- Builders -----------------------

def build_tokenizer(model_name):
    if "distilbert" in model_name:
        return DistilBertTokenizerFast.from_pretrained(model_name, use_fast=True)
    return BertTokenizerFast.from_pretrained(model_name, use_fast=True)

def build_model(model_type, model_name, num_labels, args):
    if model_type == "bert":
        return BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif model_type == "distilbert":
        return DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif model_type == "eat":
        # 1. Load a standard BERT config and add our custom EAT parameters
        cfg = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
        
        # --- EAT Architecture Config ---
        cfg.window_size = 32
        cfg.prune_layers = (2, 4)
        cfg.exit_layer = 4
        
        # --- Training Hyperparameters ---
        cfg.prune_ratio = args.target_prune  # annealed if --anneal_prune
        cfg.exit_loss_weight = 0.3
        
        # 2. Instantiate the EAT model with the modified config
        model = EATForSequenceClassification(cfg)
        
        # 3. Initialize EAT's BERT backbone with pretrained bert-base-uncased weights
        # This ensures we start from a good checkpoint (transfer learning)
        base_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        model.bert.load_state_dict(base_bert.bert.state_dict(), strict=False)
        
        # Optional: also initialize the final classifier head for faster convergence
        model.classifier.load_state_dict(base_bert.classifier.state_dict(), strict=False)
        
        return model
    else:
        raise ValueError("Unknown model_type")

# --- TOP-LEVEL collate (picklable on Windows) ---
def collate_batch(batch, tokenizer, task, maxlen):
    texts = TASK2INFO[task]["text"]
    if len(texts)==1:
        enc = tokenizer([x[texts[0]] for x in batch], truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
    else:
        enc = tokenizer([x[texts[0]] for x in batch], [x[texts[1]] for x in batch],
                        truncation=True, padding=True, max_length=maxlen, return_tensors="pt")
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return enc, labels

# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sst2","qqp","mnli"], required=True)
    parser.add_argument("--model_type", choices=["bert","distilbert","eat"], required=True)
    parser.add_argument("--model_name", default=None)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--logdir", default="./results/logs")
    parser.add_argument("--savedir", default="./models")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to existing checkpoint dir to resume")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip training if save_dir already exists")

    # Efficiency / speed
    parser.add_argument("--fp16", action="store_true", help="Enable AMP float16")
    parser.add_argument("--bf16", action="store_true", help="Enable AMP bfloat16 (if supported)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (Windows-safe default=0)")
    parser.add_argument("--max_len", type=int, default=None, help="Override task default max length")

    # EAT options
    parser.add_argument("--anneal_prune", action="store_true", help="EAT: anneal pruning during training")
    parser.add_argument("--target_prune", type=float, default=0.30, help="EAT: final prune ratio at each prune layer (default 0.30)")

    # Saving cadence
    parser.add_argument("--save_each_epoch", action="store_true", help="Save checkpoint each epoch")
    args = parser.parse_args()

    set_seed(args.seed)
    info = TASK2INFO[args.task]
    split_valid = info.get("split","validation")
    task_maxlen = info["maxlen"] if args.max_len is None else args.max_len

    # Dataset (cached under ./data)
    dataset = load_dataset(info["name"], info["subset"], cache_dir="./data")
    train_ds = dataset["train"]
    valid_ds = dataset[split_valid]

    # Model/Tokenizer selection
    if args.model_type == "distilbert":
        model_name = args.model_name or "distilbert-base-uncased"
    else:
        model_name = args.model_name or "bert-base-uncased"

    tokenizer = build_tokenizer(model_name)

    # Canonical save dir used by timing scripts
    run_id = f"{args.task}_{args.model_type}_seed{args.seed}"
    os.makedirs(args.savedir, exist_ok=True)
    save_dir = os.path.join(args.savedir, run_id)

    # Skip if exists (prevents retraining by multiple scripts)
    if args.skip_if_exists and os.path.isdir(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[train.py] Skip requested: existing model at {save_dir}.")
        return

    # Build / resume model
    if args.resume_from:
        print(f"[train.py] Resuming from {args.resume_from}")
        if args.model_type == "bert":
            model = BertForSequenceClassification.from_pretrained(args.resume_from, num_labels=info["num_labels"])
        elif args.model_type == "distilbert":
            model = DistilBertForSequenceClassification.from_pretrained(args.resume_from, num_labels=info["num_labels"])
        else:
            model = EATForSequenceClassification.from_pretrained(args.resume_from, num_labels=info["num_labels"])
    else:
        model = build_model(args.model_type, model_name, info["num_labels"], args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optional torch.compile
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", backend="inductor")
            print("[train.py] torch.compile enabled.")
        except Exception as e:
            print(f"[train.py] torch.compile not available/failed: {e}")

    # AMP setup
    use_fp16 = bool(args.fp16 and torch.cuda.is_available())
    use_bf16 = bool(args.bf16 and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)  # bf16 typically doesn't need scaling
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # --- DataLoaders (Windows-safe; pin only if CUDA; workers optional) ---
    use_cuda = torch.cuda.is_available()
    pin_mem = bool(use_cuda)
    persist = (args.num_workers > 0) and use_cuda  # only worthwhile with CUDA + workers

    from functools import partial
    collate = partial(collate_batch, tokenizer=tokenizer, task=args.task, maxlen=task_maxlen)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,          # default 0 on Windows in our script
        pin_memory=pin_mem,                    # avoid warning when no accelerator
        persistent_workers=persist
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist
    )

    # Optimizer / scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    t_total = max(1, args.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*t_total), t_total)

    # Logging
    log_path = os.path.join(args.logdir, f"{run_id}.jsonl")
    ensure_dir(os.path.dirname(log_path))

    # -------- Training loop --------
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for (enc, labels) in pbar:
            enc = {k:v.to(device, non_blocking=True) for k,v in enc.items()}
            labels = labels.to(device, non_blocking=True)

            # EAT pruning anneal: epoch 1 -> 0.0; progressively reach target at later epochs
            if args.model_type == "eat" and args.anneal_prune:
                if epoch == 1:
                    model.keep_ratio = 1.0
                else:
                    steps = max(1, args.epochs - 1)
                    progress = min(1.0, (epoch - 1) / steps)
                    target_keep = 1.0 - float(args.target_prune)
                    model.keep_ratio = float(1.0 - progress * (1.0 - target_keep))

            optimizer.zero_grad(set_to_none=True)
            if use_fp16 or use_bf16:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    out = model(**enc, labels=labels)
                    loss = out["loss"] if isinstance(out, dict) else out[1]
                if use_fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                out = model(**enc, labels=labels)
                loss = out["loss"] if isinstance(out, dict) else out[1]
                loss.backward()
                optimizer.step()

            scheduler.step()
            losses.append(float(loss.item()))
            pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

        # -------- Eval --------
        model.eval()
        preds, golds = [], []
        with torch.no_grad():
            for (enc, labels) in tqdm(valid_loader, desc="Eval"):
                enc = {k:v.to(device, non_blocking=True) for k,v in enc.items()}
                logits = model(**enc)["logits"]
                pred = torch.argmax(logits, dim=-1)
                preds.extend(pred.cpu().tolist())
                golds.extend(labels.tolist())
        metric = info["metric"](np.array(preds), np.array(golds))
        log_jsonl(log_path, {"epoch": epoch, "val": metric, "device": device_name()})
        print(f"[{run_id}] epoch {epoch} val: {metric}")

        # Optional per-epoch save (useful for resume/early stop)
        if args.save_each_epoch:
            ensure_dir(save_dir)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[{run_id}] saved checkpoint to {save_dir} after epoch {epoch}")

    # -------- Final save --------
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved final model to {save_dir}")


if __name__ == "__main__":
    main()
