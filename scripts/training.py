# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_CLUSTER = _REPO / "clustered_data"
_TRAINED = _REPO / "trained_models"

from utils.normalizing import normalize_csv
from utils.chemberta_workflows import train_chemberta_model
# %%
RANDOM_SEED = 19237

# %%
# =============================================================================
# Shared hyperparameters (identical across all datasets)
# =============================================================================
_SHARED_HPARAMS = {
    'smiles_column': 'smiles',
    'output_dir': str(_TRAINED),
    'epochs': 100,
    'batch_size': 32,
    'lr': 5e-5,
    'l1_lambda': 0.0,
    'l2_lambda': 0.01,
    'dropout': 0.3,
    'hidden_channels': 128,
    'num_mlp_layers': 1,
    'early_stopping_patience': 10,
    'random_seed': RANDOM_SEED,
}


def _load_and_normalize(dataset_dir, target_col):
    """Load train/val/test CSVs and normalize using the training scaler."""
    train = pd.read_csv(_CLUSTER / dataset_dir / f"train_{dataset_dir}.csv")
    val = pd.read_csv(_CLUSTER / dataset_dir / f"validation_{dataset_dir}.csv")
    test = pd.read_csv(_CLUSTER / dataset_dir / f"test_{dataset_dir}.csv")
    norm_train, scaler = normalize_csv(train, target_col=target_col)
    norm_val, _ = normalize_csv(val, target_col=target_col, scaler=scaler)
    norm_test, _ = normalize_csv(test, target_col=target_col, scaler=scaler)
    return norm_train, norm_val, norm_test, scaler


# %%
# =============================================================================
# ESOL
# =============================================================================
esol_parser = argparse.Namespace(**{**_SHARED_HPARAMS, 'target_column': 'solubility'})
norm_train_esol, norm_val_esol, norm_test_esol, esol_scaler = _load_and_normalize("esol", esol_parser.target_column)

# %%
esol_results = train_chemberta_model(
    esol_parser, norm_train_esol, norm_test_esol, esol_scaler,
    df_val=norm_val_esol, dataset_name="train_esol", model_name="chemberta",
)
esol_results

# %%
# =============================================================================
# HCE
# =============================================================================
hce_parser = argparse.Namespace(**{**_SHARED_HPARAMS, 'target_column': 'pce_1'})
norm_train_hce, norm_val_hce, norm_test_hce, hce_scaler = _load_and_normalize("hce", hce_parser.target_column)

# %%
hce_results = train_chemberta_model(
    hce_parser, norm_train_hce, norm_test_hce, hce_scaler,
    df_val=norm_val_hce, dataset_name="train_hce", model_name="chemberta",
)
hce_results

# %%
# =============================================================================
# QM9
# =============================================================================
qm9_parser = argparse.Namespace(**{**_SHARED_HPARAMS, 'target_column': 'dga'})
norm_train_qm9, norm_val_qm9, norm_test_qm9, qm9_scaler = _load_and_normalize("qm9", qm9_parser.target_column)

# %%
qm9_results = train_chemberta_model(
    qm9_parser, norm_train_qm9, norm_test_qm9, qm9_scaler,
    df_val=norm_val_qm9, dataset_name="train_qm9", model_name="chemberta",
)
qm9_results
