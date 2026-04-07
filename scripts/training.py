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
# ESOL Training Configuration
# =============================================================================

esol_defaults = {
    'train_csv': str(_CLUSTER / 'esol' / 'train_esol.csv'),
    'test_csv': str(_CLUSTER / 'esol' / 'test_esol.csv'),
    'target_column': 'solubility',
    'smiles_column': 'smiles',
    'output_dir': str(_TRAINED),
    'epochs': 20,
    'batch_size': 16,
    'lr': 0.001,
    'l1_lambda': 0.0,
    'l2_lambda': 0.01,
    'dropout': 0.3,  # dropout for single linear layer
    'hidden_channels': 256,  # Not used when num_mlp_layers=1
    'num_mlp_layers': 1,  # Single linear layer: 384 -> 1
    'random_seed': RANDOM_SEED,
}

esol_parser = argparse.Namespace(**esol_defaults)

# %%
train_esol = pd.read_csv(esol_parser.train_csv)
test_esol = pd.read_csv(esol_parser.test_csv)
norm_train_esol, esol_scaler = normalize_csv(train_esol, target_col=esol_parser.target_column)
norm_test_esol, _ = normalize_csv(test_esol, target_col=esol_parser.target_column, scaler=esol_scaler)

# %%
# ESOL with FROZEN encoder (only regression head trained)
# Architecture: Frozen ChemBERTa encoder -> Linear(384, 1)
esol_frozen_results = train_chemberta_model(
    esol_parser, 
    norm_train_esol, 
    norm_test_esol, 
    esol_scaler,
    dataset_name="train_esol",
    freeze_encoder=True,
    model_name="chemberta_frozen"
)
print("\n" + "="*60)
print("ESOL FROZEN ENCODER TRAINING COMPLETE")
print("Model saved to: trained_models/train_esol/chemberta_frozen/")
print("="*60)
esol_frozen_results

# %%
# ESOL with FULL finetuning (encoder + regression head trained)
# Architecture: Trainable ChemBERTa encoder -> Linear(384, 1)
esol_finetuned_results = train_chemberta_model(
    esol_parser, 
    norm_train_esol, 
    norm_test_esol, 
    esol_scaler,
    dataset_name="train_esol",
    freeze_encoder=False,
    model_name="chemberta"
)
print("\n" + "="*60)
print("ESOL FULL FINETUNING TRAINING COMPLETE")
print("Model saved to: trained_models/train_esol/chemberta/")
print("="*60)
esol_finetuned_results

# %%
# =============================================================================
# HCE and QM9 training 
# =============================================================================

# %%
# For hce
hce_defaults = {
    'train_csv': str(_CLUSTER / 'hce' / 'train_hce.csv'),
    'test_csv': str(_CLUSTER / 'hce' / 'test_hce.csv'),
    'target_column': 'pce_1',
    'smiles_column': 'smiles',
    'output_dir': str(_TRAINED),
    'epochs': 40,
    'batch_size': 16,
    'lr': 0.001,
    'l1_lambda': 0.0,
    'l2_lambda': 0.01,
    'dropout': 0.3,
    'hidden_channels': 128,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

hce_parser = argparse.Namespace(**hce_defaults)


train_hce = pd.read_csv(hce_parser.train_csv)
test_hce = pd.read_csv(hce_parser.test_csv)
norm_train_hce, hce_scaler = normalize_csv(train_hce, hce_parser.target_column)
norm_test_hce, _ = normalize_csv(test_hce, hce_parser.target_column, scaler=hce_scaler)

# %%
hce_results = train_chemberta_model(hce_parser, norm_train_hce, norm_test_hce, hce_scaler)
hce_results

# %% 
# For qm9
qm9_defaults = {
    'train_csv': str(_CLUSTER / 'qm9' / 'train_qm9.csv'),
    'test_csv': str(_CLUSTER / 'qm9' / 'test_qm9.csv'),
    'target_column': 'dga',
    'smiles_column': 'smiles',
    'output_dir': str(_TRAINED),
    'epochs': 20,
    'batch_size': 16,
    'lr': 0.001,
    'l1_lambda': 0.0,
    'l2_lambda': 0.01,
    'dropout': 0.3,
    'hidden_channels': 128,
    'num_mlp_layers': 1,
    'random_seed': RANDOM_SEED,
}

qm9_parser = argparse.Namespace(**qm9_defaults)

train_qm9 = pd.read_csv(_CLUSTER / "qm9" / "train_qm9.csv")
val_qm9 = pd.read_csv(_CLUSTER / "qm9" / "validation_qm9.csv")
test_qm9 = pd.read_csv(_CLUSTER / "qm9" / "test_qm9.csv")
norm_train_qm9, qm9_scaler = normalize_csv(train_qm9, target_col=qm9_parser.target_column)
norm_val_qm9, _ = normalize_csv(val_qm9, target_col=qm9_parser.target_column, scaler=qm9_scaler)
norm_test_qm9, _ = normalize_csv(test_qm9, target_col=qm9_parser.target_column, scaler=qm9_scaler)

norm_train_qm9_05 = norm_train_qm9.sample(n=int(0.005 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 0.5%
norm_train_qm9_1 = norm_train_qm9.sample(n=int(0.01 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 1%
norm_train_qm9_2 = norm_train_qm9.sample(n=int(0.02 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 2%
norm_train_qm9_5 = norm_train_qm9.sample(n=int(0.05 * len(norm_train_qm9)), random_state=RANDOM_SEED) # 5%

# %%
qm9_results_05 = train_chemberta_model(qm9_parser, norm_train_qm9_05, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_05")
qm9_results_1 = train_chemberta_model(qm9_parser, norm_train_qm9_1, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_1")
qm9_results_2 = train_chemberta_model(qm9_parser, norm_train_qm9_2, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_2")
qm9_results_5 = train_chemberta_model(qm9_parser, norm_train_qm9_5, norm_val_qm9, qm9_scaler, dataset_name="train_qm9_5")

# %%
# The point where improvement kind of stalls is X, so we're retraining on that percentage and testing on test set and keeping that as 
# the final model
qm9_results = train_chemberta_model(qm9_parser, norm_train_qm9_05, norm_test_qm9, qm9_scaler)