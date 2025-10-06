"""
EAT (Efficient Adaptive Transformer) for sequence classification.

Key features:
- Longformer-style sparse attention mask: fixed local window (k=32) + global [CLS]
- Progressive token pruning after layers 2 and 4 using L2-norm importance; [CLS] never pruned
- Early-exit head at layer 4 with confidence threshold (tau) at inference
- Pruning active during training AND (optionally) at inference (prune_eval=True)
- Returns optional runtime stats: executed depth and final token retention

Dependencies: transformers==4.43+ (BertModel/BertConfig APIs)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel


# ----------------------------- Sparse attention utilities -----------------------------

def build_sparse_mask(attn_mask: torch.Tensor, window: int = 32, cls_global: bool = True) -> torch.Tensor:
    """
    Create a boolean attention adjacency mask with a symmetric local window and optional global [CLS].

    Args:
        attn_mask: (bsz, L)  1 for valid tokens, 0 for padding (and pruned positions).
        window:    total window size (e.g., 32 => ±16 around each token).
        cls_global: if True, token 0 ([CLS]) can attend to all tokens and all tokens can attend to [CLS].

    Returns:
        allowed: (bsz, L, L) boolean mask (True = allowed to attend).
    """
    bsz, L = attn_mask.shape
    device = attn_mask.device
    allowed = torch.zeros((bsz, L, L), dtype=torch.bool, device=device)

    half = max(1, window // 2)
    idx = torch.arange(L, device=device)

    # Local windows
    for i in range(L):
        left = max(0, i - half)
        right = min(L, i + half + 1)
        allowed[:, i, left:right] = True

    valid = attn_mask.bool()
    # Only allow attention among valid tokens
    allowed &= valid.unsqueeze(1).expand(-1, L, -1)
    allowed &= valid.unsqueeze(2).expand(-1, -1, L)

    # Global CLS connectivity
    if cls_global and L > 0:
        # CLS attends to all valid tokens
        allowed[:, 0, :] = valid
        # Everyone can attend to CLS (even if they are padding they won't query due to valid gating above)
        allowed[:, :, 0] = True

    return allowed  # (bsz, L, L)


# ----------------------------- Token pruning utilities -----------------------------

def token_importance(hidden: torch.Tensor) -> torch.Tensor:
    """
    L2-norm importance per token.

    Args:
        hidden: (bsz, L, d)

    Returns:
        scores: (bsz, L)
    """
    return torch.norm(hidden, dim=-1)


def prune_tokens_vectorized(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_ratio: float,
    protect_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized pruning of tokens by zero-masking low-importance ones.
    """
    bsz, L, _ = hidden.shape
    device = hidden.device

    # 1. Calculate importance scores and mask out invalid tokens
    scores = torch.norm(hidden, dim=-1)
    scores.masked_fill_(attention_mask == 0, -1e9)  # Give padding tokens a very low score

    # 2. Always protect the [CLS] token
    if protect_idx is not None:
        scores[:, protect_idx] = 1e9 # Give protected token a very high score

    # 3. Determine how many tokens to keep for each item in the batch
    n_valid = attention_mask.sum(dim=1).int()
    n_keep = (n_valid * keep_ratio).ceil().int().clamp(min=1)

    # 4. Find the top-k tokens to keep for the whole batch
    # topk will return the values and indices of the most important tokens
    _, keep_indices = torch.topk(scores, k=n_keep.max(), dim=1)

    # 5. Create the new attention mask
    new_mask = torch.zeros_like(attention_mask, device=device)
    # Use scatter to efficiently mark the kept tokens with '1'
    # For each row, we gather the indices from `keep_indices` up to `n_keep` for that row
    for i in range(bsz):
        new_mask[i].scatter_(0, keep_indices[i, :n_keep[i]], 1)

    new_hidden = hidden * new_mask.unsqueeze(-1)
    return new_hidden, new_mask


# ----------------------------- EAT Model -----------------------------

class EATForSequenceClassification(BertPreTrainedModel):
    """
    EAT on top of BERT:
      - Sparse attention mask (window + global CLS)
      - Token pruning after {2,4} with keep_ratio = 1 - prune_ratio
      - Early exit at layer=4 with threshold tau at inference
      - Optional runtime stats for avg executed depth and final retention
    """

    def __init__(self, config: BertConfig, **kwargs):
        """
        Initializes the EAT model. All EAT-specific hyperparameters are
        expected to be attributes of the `config` object.
        """
        super().__init__(config)
        self.config = config
        self.num_labels = getattr(config, "num_labels", 2)

        # --- EAT Hyperparameters (read from config with defaults) ---
        self.window_size = getattr(config, "window_size", 32)
        self.prune_layers = set(getattr(config, "prune_layers", (2, 4)))
        self.keep_ratio = 1.0 - getattr(config, "prune_ratio", 0.30)
        self.exit_layer = getattr(config, "exit_layer", 4)
        self.exit_loss_weight = getattr(config, "exit_loss_weight", 0.3)
        self.prune_eval = getattr(config, "prune_eval", True)

        # --- Model Layers ---
        self.bert = BertModel(config, add_pooling_layer=False)
        hidden_size = config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.exit_classifier = nn.Linear(hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        exit_threshold: Optional[float] = None,
        return_dict: bool = True,
        return_stats: bool = False,
        **kwargs,
    ) -> Dict[str, Any] | tuple:
        """
        If exit_threshold is provided (e.g., 0.9), do early-exit at layer=exit_layer when max prob >= threshold.
        Training: provide labels; loss = final_ce + exit_loss_weight * exit_ce.
        Optionally return runtime stats: executed_depth and final_retention.
        """
        assert input_ids is not None and attention_mask is not None, "input_ids and attention_mask required"
        bsz, L = input_ids.shape

        device = input_ids.device
        dtype = self.bert.dtype if hasattr(self.bert, "dtype") else torch.float32

        # Initial sparse mask (boolean allowed connections)
        allowed = build_sparse_mask(attention_mask, window=self.window_size, cls_global=True)  # (bsz, L, L)
        # Convert to additive mask for HF attention: 0 for allowed, -inf for disallowed
        attn_additive = torch.where(
            allowed, torch.zeros_like(allowed, dtype=dtype), torch.full_like(allowed, fill_value=-1e4, dtype=dtype)
        ).unsqueeze(1)  # (bsz, 1, L, L) – broadcastable to (bsz, heads, L, L)

        # Track stats
        executed_depth = 0
        init_valid = attention_mask.sum(dim=1).clamp(min=1)  # (bsz,) avoid div-by-zero

        # Embeddings
        hidden_states = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        exit_logits = None
        ce = nn.CrossEntropyLoss()
        total_loss = None

        # Iterate encoder layers manually to inject pruning & custom masks
        for idx, layer_module in enumerate(self.bert.encoder.layer, start=1):
            executed_depth = idx

            # Run layer with custom 4D attention mask (broadcasted across heads)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attn_additive,
                head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]  # (bsz, L, hidden)

            # Progressive pruning after specific layers (train + optional eval)
            if idx in self.prune_layers and (self.training or self.prune_eval):
                hidden_states, attention_mask = prune_tokens_vectorized(
                    hidden_states, attention_mask, keep_ratio=self.keep_ratio, protect_idx=0
                )
                # Rebuild sparse mask with updated attention_mask
                allowed = build_sparse_mask(attention_mask, window=self.window_size, cls_global=True)
                attn_additive = torch.where(
                    allowed, torch.zeros_like(allowed, dtype=dtype), torch.full_like(allowed, fill_value=-1e4, dtype=dtype)
                ).unsqueeze(1)

            # Early-exit head at configured layer
            if idx == self.exit_layer:
                cls_vec = hidden_states[:, 0, :]  # [CLS]
                exit_logits = self.exit_classifier(cls_vec)

                # Inference-time early exit
                if exit_threshold is not None and not self.training:
                    probs = F.softmax(exit_logits, dim=-1)
                    maxp, _ = probs.max(dim=-1)
                    all_confident = torch.all(maxp >= exit_threshold).item()

                    if all_confident:
                        # Return early
                        out_stats = None
                        if return_stats:
                            final_ret = (attention_mask.sum(dim=1) / init_valid).mean().item()
                            out_stats = {
                                "executed_depth": float(executed_depth),
                                "final_retention": float(final_ret),
                            }
                        if labels is None:
                            if return_dict:
                                return {"logits": exit_logits, "stats": out_stats}
                            return (exit_logits, out_stats)

                        exit_loss = ce(exit_logits, labels)
                        if return_dict:
                            return {"logits": exit_logits, "loss": exit_loss, "stats": out_stats}
                        return (exit_logits, exit_loss, out_stats)

        # Final head (no early exit triggered or training)
        cls_vec = hidden_states[:, 0, :]
        logits = self.classifier(cls_vec)

        if labels is None:
            if return_stats:
                final_ret = (attention_mask.sum(dim=1) / init_valid).mean().item()
                stats = {"executed_depth": float(executed_depth), "final_retention": float(final_ret)}
                if return_dict:
                    return {"logits": logits, "stats": stats}
                return (logits, stats)
            if return_dict:
                return {"logits": logits}
            return (logits,)

        # Compute loss (final + weighted exit head if available)
        final_loss = ce(logits, labels)
        if exit_logits is not None:
            exit_loss = ce(exit_logits, labels)
            final_loss = final_loss + self.exit_loss_weight * exit_loss

        if return_stats:
            final_ret = (attention_mask.sum(dim=1) / init_valid).mean().item()
            stats = {"executed_depth": float(executed_depth), "final_retention": float(final_ret)}
            if return_dict:
                return {"logits": logits, "loss": final_loss, "stats": stats}
            return (logits, final_loss, stats)

        if return_dict:
            return {"logits": logits, "loss": final_loss}
        return (logits, final_loss)
