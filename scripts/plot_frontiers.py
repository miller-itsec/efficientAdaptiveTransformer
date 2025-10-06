"""
Plot Accuracy–Latency frontiers from the combined CSV (results/tables/combined_frontier.csv).
- Robust to EU-style decimals (e.g., "9,75") and mixed 'task' types.
- Produces PNG + PDF per task with clearer τ labels.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/tables/combined_frontier.csv"
OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight"
})

NUM_COLS = ["latency_ms", "accuracy", "tau"]

def _clean_str(x):
    """Normalize values to safe strings; treat NaN/None/empty as missing."""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("", "nan", "none", "null") else s

def _to_num_series(s):
    """Accept both '1.23' and '1,23' (EU decimal)."""
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )

def load_combined():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Combined CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # String hygiene
    if "task" not in df.columns or "model_type" not in df.columns:
        raise ValueError("CSV must contain 'task' and 'model_type' columns.")
    df["task"] = df["task"].map(_clean_str)
    df["model_type"] = df["model_type"].map(_clean_str)

    # Numeric hygiene (EU/US decimals)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = _to_num_series(df[c])

    # Keep only usable rows
    df = df[(df["task"] != "") & (df["model_type"] != "")]
    df = df.dropna(subset=["latency_ms", "accuracy"])
    if "tau" not in df.columns:
        df["tau"] = 0.0
    df["tau"] = df["tau"].fillna(0.0)
    df["tau_label"] = df["tau"].apply(lambda x: f"τ={x:.2f}" if x > 0 else "")
    return df

def plot_task(df, task):
    sub = df[df["task"] == task]
    if sub.empty:
        print(f"[skip] No entries for task {task}")
        return

    plt.figure(figsize=(6.5, 4.3))

    # Plot BERT and DistilBERT as points
    for model, color, marker, label in [
        ("bert", "black", "o", "BERT"),
        ("distilbert", "tab:blue", "s", "DISTILBERT"),
    ]:
        d = sub[sub["model_type"] == model].copy()
        if d.empty:
            continue
        plt.scatter(d["latency_ms"], d["accuracy"], color=color, marker=marker, s=64, label=label, zorder=3)

    # Plot EAT as a line with labeled points in τ order
    eat = sub[sub["model_type"] == "eat"].copy()
    if not eat.empty:
        eat = eat.sort_values("tau", na_position="last")
        plt.plot(eat["latency_ms"], eat["accuracy"], color="tab:orange", marker="D",
                 linewidth=2, label="EAT (τ sweep)", zorder=2)
        # Improved label placement: stagger labels to reduce overlap
        for i, (_, r) in enumerate(eat.iterrows()):
            if r["tau"] <= 0:
                continue
            dx = 0.10  # ms
            dy = 0.004 if (i % 2 == 0) else -0.004
            plt.text(float(r["latency_ms"]) + dx,
                     float(r["accuracy"]) + dy,
                     r["tau_label"],
                     fontsize=9, color="gray",
                     ha="left", va="center", zorder=4,
                     clip_on=False)

    # Axes & cosmetics
    plt.xlabel("Latency (ms, batch=1)")
    plt.ylabel("Accuracy")
    plt.title(task.upper())
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    plt.legend(frameon=False, loc="lower left")

    # Tight y-limits with a little padding
    ymin = max(0.0, sub["accuracy"].min() - 0.01)
    ymax = min(1.0, sub["accuracy"].max() + 0.01)
    if ymin < ymax:
        plt.ylim(ymin, ymax)

    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"frontier_{task}.png")
    out_pdf = os.path.join(OUT_DIR, f"frontier_{task}.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[write] {out_png} / {out_pdf}")

def main():
    df = load_combined()
    tasks = sorted([t for t in df["task"].unique() if isinstance(t, str) and t])
    if not tasks:
        raise ValueError("No valid tasks found in combined CSV after cleaning.")
    for task in tasks:
        plot_task(df, task)

if __name__ == "__main__":
    main()
