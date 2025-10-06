# In src/plot_exit_distribution.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot the distribution of early-exit layers.")
    parser.add_argument("--input_csv", required=True, help="Path to the detailed log CSV from time_infer.py")
    parser.add_argument("--output_dir", default="./results/plots", help="Directory to save the plot.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df.dropna(subset=['executed_depth'], inplace=True)
    
    # We only care about EAT runs which have a tau value
    df = df[df['tau'].notna()]

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='executed_depth', hue='task', stat='percent')

    plt.title('Distribution of Model Exit Layers by Task (tau=0.90)')
    plt.xlabel('Exit Layer (Executed Depth)')
    plt.ylabel('Percentage of Examples (%)')
    plt.legend(title='Task')
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "exit_distribution.pdf")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
	
    output_path_png = os.path.join(args.output_dir, "exit_distribution.png")
    plt.savefig(output_path_png, dpi=300) # dpi=300 ensures high quality
    print(f"Plot saved to {output_path_png}")

if __name__ == "__main__":
    main()