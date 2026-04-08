# %%
from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from mole.utils.chemberta_workflows import ChembertaRegressor, evaluate_chemberta_model
from mole.utils.normalizing import normalize_csv

# %%
REPO = Path(__file__).resolve().parents[2]

# %%
# Change this if you want a different QM9 model folder.
QM9_MODEL_DIR = REPO / "trained_models" / "train_qm9_05" / "chemberta"
USE_LATEST_HCE_CHECKPOINT = False

CONFIGS = {
    "esol": {
        "test_csv": REPO / "clustered_data" / "esol" / "test_esol.csv",
        "target_col": "solubility",
        "model_dir": REPO / "trained_models" / "train_esol" / "chemberta",
    },
    "hce": {
        "test_csv": REPO / "clustered_data" / "hce" / "test_hce.csv",
        "target_col": "pce_1",
        "model_dir": REPO / "trained_models" / "train_hce" / "chemberta",
    },
    "qm9": {
        "test_csv": REPO / "clustered_data" / "qm9" / "test_qm9.csv",
        "target_col": "dga",
        "model_dir": QM9_MODEL_DIR,
    },
}

# %%
def evaluate_dataset(name: str, batch_size: int = 32):
    cfg = CONFIGS[name]
    model_dir = cfg["model_dir"]

    with open(model_dir / "normalization_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "hyperparameters.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)

    df_test = pd.read_csv(cfg["test_csv"])
    norm_test, _ = normalize_csv(df_test, target_col=cfg["target_col"], scaler=scaler, fit_scaler=False)

    model = ChembertaRegressor(
        pretrained="DeepChem/ChemBERTa-77M-MLM",
        dropout=hparams.get("dropout", 0.3),
        hidden_channels=hparams.get("hidden_channels", 128),
        num_mlp_layers=hparams.get("num_mlp_layers", 1),
        freeze_encoder=False,
    )
    state_path = model_dir / "chemberta_model_final.bin"
    if name == "hce" and USE_LATEST_HCE_CHECKPOINT:
        checkpoints = sorted(model_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            ckpt_state = checkpoints[0] / "model.safetensors"
            if ckpt_state.exists():
                state_path = ckpt_state

    if state_path.suffix == ".safetensors":
        model.load_state_dict(load_file(str(state_path), device="cpu"))
    else:
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    print(f"\n=== Evaluating {name.upper()} ===")
    print(f"Using weights: {state_path}")
    scaler_path = model_dir / "normalization_scaler.pkl"
    print(f"Scaler path: {scaler_path}")
    print(f"Weights mtime: {state_path.stat().st_mtime:.0f} | Scaler mtime: {scaler_path.stat().st_mtime:.0f}")
    print(f"Scaler mean={float(scaler.mean_[0]):.4f}, scale={float(scaler.scale_[0]):.4f}")

    if name == "hce":
        train_csv = REPO / "clustered_data" / "hce" / "train_hce.csv"
        if train_csv.exists():
            y = pd.to_numeric(pd.read_csv(train_csv)["pce_1"], errors="coerce").dropna().values
            print(
                "Current train_hce stats:",
                f"n={len(y)} mean={float(np.mean(y)):.4f} std={float(np.std(y)):.4f} min={float(np.min(y)):.4f} max={float(np.max(y)):.4f}",
            )

    results = evaluate_chemberta_model(
        model=model,
        dataset=norm_test,
        scaler=scaler,
        target_column=cfg["target_col"],
        output_dir=str(model_dir),
        batch_size=batch_size,
        plot_filename="preds_vs_targets_recheck_2.pdf",
    )

    if name == "hce":
        preds = np.asarray(results["predictions"], dtype=float)
        targets_orig = np.asarray(results["targets"], dtype=float)
        preds_norm = (preds - float(scaler.mean_[0])) / float(scaler.scale_[0])
        targets_norm = (targets_orig - float(scaler.mean_[0])) / float(scaler.scale_[0])
        print(
            "HCE prediction sanity:",
            f"min={preds.min():.4f} p1={np.quantile(preds, 0.01):.4f} p99={np.quantile(preds, 0.99):.4f} max={preds.max():.4f}",
        )
        print(
            "HCE normalized prediction sanity:",
            f"min={preds_norm.min():.4f} p1={np.quantile(preds_norm, 0.01):.4f} p99={np.quantile(preds_norm, 0.99):.4f} max={preds_norm.max():.4f}",
        )
        print(
            "HCE normalized target range:",
            f"min={targets_norm.min():.4f} p1={np.quantile(targets_norm, 0.01):.4f} p99={np.quantile(targets_norm, 0.99):.4f} max={targets_norm.max():.4f}",
        )
        print(
            "HCE cutoffs:",
            f"pct<=1.0={100.0 * np.mean(preds <= 1.0):.2f}% pct>=9.0={100.0 * np.mean(preds >= 9.0):.2f}%",
        )
    return results

# %%
# Run all three
esol_eval = evaluate_dataset("esol")

# %%
hce_eval = evaluate_dataset("hce")

# %%
qm9_eval = evaluate_dataset("qm9")

# %%
