"""
Plot Ablation Study from a summary CSV.
- Reads a CSV with ablation variants (e.g., EAT-prune-only).
- Calculates accuracy delta and speedup relative to a 'bert' baseline.
- Generates a bar chart showing the trade-offs.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CSV_PATH = "results/tables/summary_ablation_sst2.csv"
OUT_DIR = "results/plots"
BASELINE_MODEL = "bert"

# --- Plotting Style ---
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.bbox": "tight"
})

def load_ablation_data():
    """Loads and cleans the ablation data from the CSV."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Ablation CSV not found: {CSV_PATH}. "
            "Please generate it from your experiment logs."
        )
    df = pd.read_csv(CSV_PATH)

    # Basic data validation
    if BASELINE_MODEL not in df["model"].values:
        raise ValueError(f"Baseline model '{BASELINE_MODEL}' not found in the CSV.")

    # Ensure numeric types
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df = df.dropna(subset=["latency_ms", "accuracy"])
    return df

def process_ablation_metrics(df):
    """Calculates accuracy delta and speedup relative to the baseline."""
    baseline = df[df["model"] == BASELINE_MODEL].iloc[0]
    baseline_latency = baseline["latency_ms"]
    baseline_accuracy = baseline["accuracy"]

    df["speedup"] = baseline_latency / df["latency_ms"]
    df["acc_delta_pp"] = (df["accuracy"] - baseline_accuracy) * 100
    
    # Filter out the baseline itself for plotting
    variants = df[df["model"] != BASELINE_MODEL].copy()
    variants = variants.sort_values("speedup", ascending=False)
    return variants

def plot_ablation_chart(variants):
    """Generates and saves the ablation bar chart."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    models = variants["model"].str.replace("_", " ").str.title()
    x = np.arange(len(models))
    width = 0.35

    # Bar for Speedup (left y-axis)
    color_speedup = "tab:blue"
    bars1 = ax1.bar(x - width/2, variants["speedup"], width, label="Speedup vs BERT", color=color_speedup)
    ax1.set_ylabel("Speedup (Higher is Better)", color=color_speedup)
    ax1.tick_params(axis='y', labelcolor=color_speedup)
    ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, zorder=0) # Baseline speedup

    # Line for Accuracy Delta (right y-axis)
    ax2 = ax1.twinx()
    color_acc = "tab:red"
    ax2.plot(x, variants["acc_delta_pp"], color=color_acc, marker='o', linestyle='--', label="Accuracy Delta (pp)")
    ax2.set_ylabel("Accuracy Delta (pp vs BERT)", color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=0) # Baseline accuracy

    # Add labels and title
    ax1.set_xlabel("Model Variant")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    plt.title("SST-2 Ablation: Speedup and Accuracy vs BERT Baseline")
    
    # Add data labels
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f'{yval:.2f}x', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    
    # Save files
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "ablation_sst2.png")
    out_pdf = os.path.join(OUT_DIR, "ablation_sst2.pdf")
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[write] Generated ablation plots: {out_png} / {out_pdf}")


def main():
    """Main execution function."""
    try:
        df = load_ablation_data()
        variants = process_ablation_metrics(df)
        plot_ablation_chart(variants)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Could not generate ablation plot.")

if __name__ == "__main__":
    main()
