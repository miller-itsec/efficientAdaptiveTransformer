# In src/plot_retention.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot token retention vs. input length.")
    parser.add_argument("--input_csv", required=True, help="Path to the detailed log CSV from time_infer.py")
    parser.add_argument("--output_dir", default="./results/plots", help="Directory to save the plot.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df.dropna(subset=['final_retention', 'input_len'], inplace=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='input_len', y='final_retention', alpha=0.3)
    
    plt.title('Token Retention vs. Input Length')
    plt.xlabel('Input Sequence Length (Tokens)')
    plt.ylabel('Final Token Retention (%)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "retention_vs_length.pdf")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
	
    output_path_png = os.path.join(args.output_dir, "retention_vs_length.png")
    plt.savefig(output_path_png, dpi=300) # dpi=300 ensures high quality
    print(f"Plot saved to {output_path_png}")

if __name__ == "__main__":
    main()