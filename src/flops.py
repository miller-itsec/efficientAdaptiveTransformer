# In src/flops.py

import argparse
import torch
import os
from fvcore.nn import FlopCountAnalysis, parameter_count
from transformers import BertConfig

# --- Important: Import your custom model ---
from eat_model import EATForSequenceClassification

def main():
    parser = argparse.ArgumentParser(description="Calculate FLOPs and parameters for EAT model.")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for dummy input.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., sst2) for logging.")
    parser.add_argument("--log", type=str, required=True, help="Path to output CSV log file.")
    args = parser.parse_args()

    print(f"--- Analyzing EAT Model for task: {args.task} ---")

    # 1. Create the model configuration
    cfg = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
    model = EATForSequenceClassification(cfg, prune_eval=False)
    model.eval()

    # 2. Create a dummy input tensor
    dummy_input_ids = torch.randint(0, cfg.vocab_size, (1, args.seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, args.seq_len, dtype=torch.long)
    
    # --- FIX: Pass inputs as a TUPLE, not a dictionary ---
    inputs = (dummy_input_ids, dummy_attention_mask)

    # 3. Use fvcore to analyze the model
    flop_analyzer = FlopCountAnalysis(model, inputs)
    total_flops = flop_analyzer.total()
    total_params = parameter_count(model)[""]

    gflops = total_flops / 1e9
    m_params = total_params / 1e6

    print(f"\nAnalysis for sequence length: {args.seq_len}")
    print(f"Total Parameters: {m_params:.2f} M")
    print(f"Total GFLOPs (unpruned): {gflops:.3f}")

    # 4. Log results to the CSV file, appending if it exists
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    header_needed = not os.path.exists(args.log)
    with open(args.log, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("task,seq_len,gflops,m_params\n")
        f.write(f"{args.task},{args.seq_len},{gflops:.4f},{m_params:.4f}\n")
    
    print(f"FLOPs results appended to {args.log}")


if __name__ == "__main__":
    main()