"""Extract per-layer R² and variance ratio from regression lens for all datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parents[1]

from mole.utils.tl_conversion import load_chemberta_models
from mole.utils.tl_regression import run_regression_lens

DEVICE = "cpu"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"

DATASETS = {
    "ESOL": {
        "model": _REPO / "trained_models" / "train_esol" / "chemberta" / "chemberta_model_final.bin",
        "scaler": _REPO / "trained_models" / "train_esol" / "chemberta" / "normalization_scaler.pkl",
        "train": _REPO / "clustered_data" / "esol" / "train_esol.csv",
        "test": _REPO / "clustered_data" / "esol" / "test_esol.csv",
        "target": "solubility",
    },
    "QM9": {
        "model": _REPO / "trained_models" / "train_qm9_1" / "chemberta" / "chemberta_model_final.bin",
        "scaler": _REPO / "trained_models" / "train_qm9_1" / "chemberta" / "normalization_scaler.pkl",
        "train": _REPO / "clustered_data" / "qm9" / "train_qm9.csv",
        "test": _REPO / "clustered_data" / "qm9" / "test_qm9.csv",
        "target": "dga",
    },
    "HCE": {
        "model": _REPO / "trained_models" / "train_hce" / "chemberta" / "chemberta_model_final.bin",
        "scaler": _REPO / "trained_models" / "train_hce" / "chemberta" / "normalization_scaler.pkl",
        "train": _REPO / "clustered_data" / "hce" / "train_hce.csv",
        "test": _REPO / "clustered_data" / "hce" / "test_hce.csv",
        "target": "pce_1",
    },
}


def get_or_rebuild_scaler(info: dict) -> StandardScaler:
    """Load saved scaler or reconstruct from training data."""
    import pickle
    scaler_path = info["scaler"]
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"  Loaded scaler from {scaler_path}")
        return scaler
    print(f"  Scaler not found at {scaler_path}, rebuilding from training data...")
    train_df = pd.read_csv(info["train"])
    scaler = StandardScaler()
    scaler.fit(train_df[info["target"]].values.reshape(-1, 1))
    print(f"  Rebuilt scaler: mean={scaler.mean_[0]:.6f}, scale={scaler.scale_[0]:.6f}")
    return scaler


def run_dataset(name: str, info: dict):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    scaler = get_or_rebuild_scaler(info)

    test_df = pd.read_csv(info["test"])
    smiles = test_df["smiles"].tolist()
    targets = test_df[info["target"]].values.astype(float)

    _, _, tokenizer, _, tl_regressor, _ = load_chemberta_models(
        str(info["model"]), TOKENIZER_NAME, DEVICE, str(info["scaler"]),
    )
    if tl_regressor.scaler is None:
        tl_regressor.scaler = scaler

    tl_encoder = tl_regressor.tl_model

    n_molecules = min(len(smiles), 500)
    sample_smiles = smiles[:n_molecules]
    sample_targets = targets[:n_molecules]

    results = run_regression_lens(
        tl_encoder, tl_regressor, scaler, sample_smiles, tokenizer,
        device=DEVICE, batch_size=64,
    )

    layer_names = list(next(iter(results.values())).keys())
    preds_by_layer = {}
    for layer in layer_names:
        preds_by_layer[layer] = np.array([results[s][layer] for s in sample_smiles])

    target_var = np.var(sample_targets)

    print(f"\nUsing {n_molecules} molecules from test set")
    print(f"Target variance: {target_var:.4f}")
    print(f"\n{'Layer':<15} {'R²':>10} {'Var Ratio':>12} {'Pred Var':>12}")
    print("-" * 52)

    summary = {}
    for layer in layer_names:
        preds = preds_by_layer[layer]
        r2 = r2_score(sample_targets, preds)
        pred_var = np.var(preds)
        vr = pred_var / target_var if target_var > 0 else 0.0
        print(f"{layer:<15} {r2:>10.4f} {vr:>12.4f} {pred_var:>12.4f}")
        summary[layer] = {"r2": float(r2), "variance_ratio": float(vr), "pred_var": float(pred_var)}

    summary["_meta"] = {
        "dataset": name,
        "n_molecules": n_molecules,
        "target_variance": float(target_var),
    }
    return summary


if __name__ == "__main__":
    all_results = {}
    for name, info in DATASETS.items():
        all_results[name] = run_dataset(name, info)

    out_path = _REPO / "results" / "regression_lens_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
