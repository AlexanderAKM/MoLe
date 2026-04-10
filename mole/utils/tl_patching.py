"""Activation patching utilities for ChemBERTa + TransformerLens workflows.

This module implements a lightweight, pairwise activation patching setup:
copy activations from a source molecule into a target molecule at one layer,
then measure how the target prediction changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformer_lens as tl
from transformers import RobertaTokenizerFast

from .tl_conversion import FaithfulTLRegressor, get_final_encoder_output


def _predict_single(
    regressor: FaithfulTLRegressor,
    tokenizer: RobertaTokenizerFast,
    smiles: str,
    device: str,
    denormalize: bool = True,
) -> float:
    """Predict a scalar property for one SMILES string."""
    inputs = tokenizer(smiles, return_tensors="pt").to(device)
    with torch.no_grad():
        prediction = regressor(
            inputs["input_ids"],
            inputs["attention_mask"],
            denormalize=denormalize,
        )
    return float(prediction.squeeze().item())


def select_same_length_pair(
    data: pd.DataFrame,
    tokenizer: RobertaTokenizerFast,
    smiles_column: str = "smiles",
    target_column: str = "solubility",
    min_target_gap: float = 0.5,
) -> Tuple[str, str]:
    """Pick two molecules with equal tokenized length and sufficiently different targets.

    Returns:
        (source_smiles, target_smiles)
    """
    if smiles_column not in data.columns or target_column not in data.columns:
        raise ValueError(f"Expected columns '{smiles_column}' and '{target_column}' in data.")

    working = data[[smiles_column, target_column]].dropna().copy()
    working["token_len"] = working[smiles_column].apply(
        lambda smi: int(tokenizer(smi, return_tensors="pt")["input_ids"].shape[-1])
    )
    working = working.sort_values(target_column).reset_index(drop=True)

    # For each token length, try min/max target pair.
    for token_len, group in working.groupby("token_len"):
        if len(group) < 2:
            continue
        low = group.iloc[0]
        high = group.iloc[-1]
        if abs(float(high[target_column]) - float(low[target_column])) >= min_target_gap:
            # source = higher target, target = lower target (arbitrary but consistent)
            return str(high[smiles_column]), str(low[smiles_column])

    raise ValueError(
        "Could not find a same-length pair meeting min_target_gap. "
        "Try lowering min_target_gap or pass a manual pair."
    )


def run_pair_activation_patching(
    tl_model: tl.HookedEncoder,
    regressor: FaithfulTLRegressor,
    tokenizer: RobertaTokenizerFast,
    source_smiles: str,
    target_smiles: str,
    device: Optional[str] = None,
    token_position: int = 0,
    patch_all_positions: bool = False,
    denormalize: bool = True,
) -> Dict:
    """Patch source activations into target one layer at a time.

    Strategy:
    - Cache source activations at `blocks.{layer}.hook_normalized_resid_post`.
    - For each layer, run the target with a forward hook replacing either:
      - CLS token only (default), or
      - all token positions.
    - Read out predictions from the shared regression head.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    source_inputs = tokenizer(source_smiles, return_tensors="pt").to(device)
    target_inputs = tokenizer(target_smiles, return_tensors="pt").to(device)

    source_len = int(source_inputs["input_ids"].shape[-1])
    target_len = int(target_inputs["input_ids"].shape[-1])
    if source_len != target_len:
        raise ValueError(
            f"Token length mismatch (source={source_len}, target={target_len}). "
            "Use equal-length pairs or enable padding to equal size before patching."
        )

    source_pred = _predict_single(regressor, tokenizer, source_smiles, device, denormalize=denormalize)
    target_pred = _predict_single(regressor, tokenizer, target_smiles, device, denormalize=denormalize)

    with torch.no_grad():
        _, source_cache = tl_model.run_with_cache(
            source_inputs["input_ids"],
            one_zero_attention_mask=source_inputs["attention_mask"],
        )

    n_layers = tl_model.cfg.n_layers
    layer_results: List[Dict] = []

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_normalized_resid_post"
        source_activation = source_cache[hook_name]

        def patch_hook(activation: torch.Tensor, hook) -> torch.Tensor:
            patched = activation.clone()
            if patch_all_positions:
                patched[:, :, :] = source_activation[:, :, :]
            else:
                patched[:, token_position, :] = source_activation[:, token_position, :]
            return patched

        with torch.no_grad():
            with tl_model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                hidden = get_final_encoder_output(
                    tl_model,
                    target_inputs["input_ids"],
                    one_zero_attention_mask=target_inputs["attention_mask"],
                )
                cls_token = hidden[:, 0, :]
                norm_pred = regressor.mlp_head(regressor.dropout(cls_token)).squeeze(-1)
                if denormalize and getattr(regressor, "scaler", None) is not None:
                    pred = regressor.denormalize_predictions(norm_pred)
                else:
                    pred = norm_pred
                patched_pred = float(pred.squeeze().item())

        layer_results.append(
            {
                "layer": layer + 1,
                "patched_prediction": patched_pred,
                "delta_vs_target": patched_pred - target_pred,
                "delta_vs_source": patched_pred - source_pred,
            }
        )

    return {
        "source_smiles": source_smiles,
        "target_smiles": target_smiles,
        "source_prediction": source_pred,
        "target_prediction": target_pred,
        "token_position": token_position,
        "patch_all_positions": patch_all_positions,
        "layer_results": layer_results,
    }


def run_head_activation_patching(
    tl_model: tl.HookedEncoder,
    regressor: FaithfulTLRegressor,
    tokenizer: RobertaTokenizerFast,
    source_smiles: str,
    target_smiles: str,
    device: Optional[str] = None,
    token_position: int = 0,
    patch_all_positions: bool = False,
    denormalize: bool = True,
) -> Dict:
    """Patch source attention-head outputs into target, one (layer, head) at a time.

    Uses hook point `blocks.{layer}.attn.hook_z`, whose shape is:
    [batch, pos, head_index, d_head].
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    source_inputs = tokenizer(source_smiles, return_tensors="pt").to(device)
    target_inputs = tokenizer(target_smiles, return_tensors="pt").to(device)

    source_len = int(source_inputs["input_ids"].shape[-1])
    target_len = int(target_inputs["input_ids"].shape[-1])
    if source_len != target_len:
        raise ValueError(
            f"Token length mismatch (source={source_len}, target={target_len}). "
            "Use equal-length pairs for clean per-head patching."
        )

    source_pred = _predict_single(regressor, tokenizer, source_smiles, device, denormalize=denormalize)
    target_pred = _predict_single(regressor, tokenizer, target_smiles, device, denormalize=denormalize)

    with torch.no_grad():
        _, source_cache = tl_model.run_with_cache(
            source_inputs["input_ids"],
            one_zero_attention_mask=source_inputs["attention_mask"],
        )

    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads
    head_results: List[Dict] = []

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        source_activation = source_cache[hook_name]

        for head in range(n_heads):

            def patch_hook(activation: torch.Tensor, hook, head_idx: int = head) -> torch.Tensor:
                patched = activation.clone()
                if patch_all_positions:
                    patched[:, :, head_idx, :] = source_activation[:, :, head_idx, :]
                else:
                    patched[:, token_position, head_idx, :] = source_activation[:, token_position, head_idx, :]
                return patched

            with torch.no_grad():
                with tl_model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
                    hidden = get_final_encoder_output(
                        tl_model,
                        target_inputs["input_ids"],
                        one_zero_attention_mask=target_inputs["attention_mask"],
                    )
                    cls_token = hidden[:, 0, :]
                    norm_pred = regressor.mlp_head(regressor.dropout(cls_token)).squeeze(-1)
                    if denormalize and getattr(regressor, "scaler", None) is not None:
                        pred = regressor.denormalize_predictions(norm_pred)
                    else:
                        pred = norm_pred
                    patched_pred = float(pred.squeeze().item())

            head_results.append(
                {
                    "layer": layer + 1,
                    "head": head,
                    "patched_prediction": patched_pred,
                    "delta_vs_target": patched_pred - target_pred,
                    "delta_vs_source": patched_pred - source_pred,
                }
            )

    return {
        "source_smiles": source_smiles,
        "target_smiles": target_smiles,
        "source_prediction": source_pred,
        "target_prediction": target_pred,
        "token_position": token_position,
        "patch_all_positions": patch_all_positions,
        "head_results": head_results,
    }


def plot_head_patching_heatmap(
    head_results_df: pd.DataFrame,
    output_path: Path | str,
    value_column: str = "delta_vs_target",
    title: str = "Attention Head Activation Patching (delta vs target)",
) -> None:
    """Plot and save a layer x head heatmap from head patching results."""
    required_cols = {"layer", "head", value_column}
    missing = required_cols.difference(set(head_results_df.columns))
    if missing:
        raise ValueError(f"Missing required columns for heatmap: {sorted(missing)}")

    matrix_df = head_results_df.pivot(index="layer", columns="head", values=value_column).sort_index()
    matrix = matrix_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1e-8)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns.tolist())
    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index.tolist())

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_column)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
