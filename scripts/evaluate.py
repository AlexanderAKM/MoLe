# %%
from __future__ import annotations

from pathlib import Path
import json
import pickle
import sys

import pandas as pd
import torch

# %%
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.chemberta_workflows import ChembertaRegressor, evaluate_chemberta_model
from utils.normalizing import normalize_csv

# %%
# Change this if you want a different QM9 model folder.
QM9_MODEL_DIR = REPO / "trained_models" / "train_qm9_05" / "chemberta"

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
    model.load_state_dict(torch.load(model_dir / "chemberta_model_final.bin", map_location="cpu"))
    model.eval()

    print(f"\n=== Evaluating {name.upper()} ===")
    return evaluate_chemberta_model(
        model=model,
        dataset=norm_test,
        scaler=scaler,
        target_column=cfg["target_col"],
        output_dir=str(model_dir),
        batch_size=batch_size,
        plot_filename="preds_vs_targets_recheck.pdf",
    )

# %%
# Run all three
esol_eval = evaluate_dataset("esol")

# %%
hce_eval = evaluate_dataset("hce")

# %%
qm9_eval = evaluate_dataset("qm9")
