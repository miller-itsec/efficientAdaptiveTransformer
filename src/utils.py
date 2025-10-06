import os, json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def log_jsonl(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def metric_sst2(preds, labels):
    return {"accuracy": float(accuracy_score(labels, preds))}

def metric_mnli(preds, labels):
    return {"accuracy": float(accuracy_score(labels, preds))}

def metric_qqp(preds, labels):
    # QQP official often logs accuracy + F1
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds))
    }

def device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def now_ms():
    return int(time.time() * 1000)
