"""
Summarize results from the combined CSV into per-task summaries.
Input:  results/tables/combined_frontier.csv
Output: results/tables/summary_<task>.csv (+ optional summary_ablation_sst2.csv)
"""

import os
import pandas as pd
import math

CSV_PATH = "results/tables/combined_frontier.csv"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_COLS = ["tau", "latency_ms", "accuracy", "avg_depth", "avg_retention"]

def _clean_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("", "nan", "none", "null") else s

def _to_num_series(s):
    # Accept both "1.23" and "1,23" (EU decimal) and strip spaces
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )

def _fmt(v, ndigits):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return f"{v:.{ndigits}f}"

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Combined CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Required columns
    required = {"task", "model_type", "latency_ms", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Clean string columns
    df["task"] = df["task"].map(_clean_str)
    df["model_type"] = df["model_type"].map(_clean_str)

    # Normalize numerics from EU/US decimal formats
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = _to_num_series(df[c])

    # Drop rows that can’t be summarized
    df = df.dropna(subset=["latency_ms", "accuracy"])
    df = df[(df["task"] != "") & (df["model_type"] != "")]

    tasks = sorted([t for t in df["task"].unique() if isinstance(t, str) and t])
    if not tasks:
        raise ValueError(
            "No valid tasks after cleaning. Inspect your CSV for decimal commas and empty 'task'/'model_type'."
        )

    # ---- Write one summary_<task>.csv per task (BERT baseline + DistilBERT + EAT sweep) ----
    for task in tasks:
        sub = df[df["task"] == task].copy()
        if sub.empty:
            continue

        # Reference = best BERT accuracy for this task
        ref = sub[sub["model_type"] == "bert"].sort_values("accuracy", ascending=False).head(1)
        if ref.empty:
            print(f"[skip] no BERT baseline for {task}")
            continue
        ref_row = ref.iloc[0]
        ref_lat = float(ref_row["latency_ms"])
        ref_acc = float(ref_row["accuracy"])

        outp = os.path.join(OUT_DIR, f"summary_{task}.csv")
        with open(outp, "w", encoding="utf-8") as w:
            w.write("model,tau,latency_ms,accuracy,speedup_vs_bert,delta_acc_pp,avg_depth,avg_retention\n")
            # BERT baseline row
            w.write(f"bert,,{_fmt(ref_lat,2)},{_fmt(ref_acc,4)},1.00x,+0.00,,\n")

            # DistilBERT + EAT rows
            for model in ["distilbert", "eat"]:
                models = sub[sub["model_type"] == model].copy()
                if models.empty:
                    continue
                if model == "eat":
                    models = models.sort_values("tau", na_position="last")
                for _, r in models.iterrows():
                    lat = float(r["latency_ms"])
                    acc = float(r["accuracy"])
                    tau = float(r["tau"]) if pd.notna(r.get("tau", float("nan"))) else 0.0
                    sp = (ref_lat / lat) if lat > 0 else float("nan")
                    dacc = (acc - ref_acc) * 100.0
                    tau_str = f"{tau:.2f}" if tau > 0 else ""
                    avg_depth = _fmt(float(r["avg_depth"]), 2) if pd.notna(r.get("avg_depth")) else ""
                    avg_ret = _fmt(float(r["avg_retention"]), 3) if pd.notna(r.get("avg_retention")) else ""
                    w.write(
                        f"{model},{tau_str},{_fmt(lat,2)},{_fmt(acc,4)},"
                        f"{_fmt(sp,2)}x,{dacc:+.2f},{avg_depth},{avg_ret}\n"
                    )
        print(f"[write] {outp}")

    # ---- Optional: ablation CSV (SST-2). Uses same formatting & derived metrics. ----
    # If you want other tasks, replicate the block and change 'sst2' to the task name.
    task_ablation = "sst2"
    sub = df[df["task"] == task_ablation].copy()
    if not sub.empty:
        # BERT refs for this task
        ref = sub[sub["model_type"] == "bert"].sort_values("accuracy", ascending=False).head(1)
        if not ref.empty:
            ref_row = ref.iloc[0]
            ref_lat = float(ref_row["latency_ms"])
            ref_acc = float(ref_row["accuracy"])

            outp_ab = os.path.join(OUT_DIR, f"summary_ablation_{task_ablation}.csv")
            with open(outp_ab, "w", encoding="utf-8") as w:
                w.write("model,tau,latency_ms,accuracy,speedup_vs_bert,delta_acc_pp,avg_depth,avg_retention\n")
                # You can choose which variants to include; here we include everything logged for the task
                # so the ablation table stays fully data-driven.
                # Order: BERT baseline, DistilBERT, then EAT τ sweep.
                # BERT baseline row
                w.write(f"bert,,{_fmt(ref_lat,2)},{_fmt(ref_acc,4)},1.00x,+0.00,,\n")

                # DistilBERT
                dist = sub[sub["model_type"] == "distilbert"].copy()
                for _, r in dist.iterrows():
                    lat = float(r["latency_ms"]); acc = float(r["accuracy"])
                    sp = (ref_lat / lat) if lat > 0 else float("nan")
                    dacc = (acc - ref_acc) * 100.0
                    avg_depth = _fmt(float(r["avg_depth"]), 2) if pd.notna(r.get("avg_depth")) else ""
                    avg_ret = _fmt(float(r["avg_retention"]), 3) if pd.notna(r.get("avg_retention")) else ""
                    w.write(f"distilbert,,{_fmt(lat,2)},{_fmt(acc,4)},{_fmt(sp,2)}x,{dacc:+.2f},{avg_depth},{avg_ret}\n")

                # EAT τ sweep
                eat = sub[sub["model_type"] == "eat"].copy().sort_values("tau", na_position="last")
                for _, r in eat.iterrows():
                    lat = float(r["latency_ms"]); acc = float(r["accuracy"])
                    tau = float(r["tau"]) if pd.notna(r.get("tau", float("nan"))) else 0.0
                    sp = (ref_lat / lat) if lat > 0 else float("nan")
                    dacc = (acc - ref_acc) * 100.0
                    tau_str = f"{tau:.2f}" if tau > 0 else ""
                    avg_depth = _fmt(float(r["avg_depth"]), 2) if pd.notna(r.get("avg_depth")) else ""
                    avg_ret = _fmt(float(r["avg_retention"]), 3) if pd.notna(r.get("avg_retention")) else ""
                    w.write(f"eat,{tau_str},{_fmt(lat,2)},{_fmt(acc,4)},{_fmt(sp,2)}x,{dacc:+.2f},{avg_depth},{avg_ret}\n")

            print(f"[write] {outp_ab}")
        else:
            print("[skip] no BERT baseline found for ablation; not writing summary_ablation_sst2.csv")
    else:
        print("[skip] no SST-2 rows; not writing summary_ablation_sst2.csv")

if __name__ == "__main__":
    main()
