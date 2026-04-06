# %%
from __future__ import annotations

from pathlib import Path

import importlib.util
import random
import sys
import urllib.request

import numpy as np
import pandas as pd
import rdkit.Chem as rdc
import torch_geometric.datasets as tcgd

# Repo root (so `python scripts/load_data.py` works from any cwd)
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Structure-based clustering from scripts/preprocessing.py (random reference FP subsample; not label-quantile refs)
_pp_spec = importlib.util.spec_from_file_location(
    "dataset_preprocessing", _REPO / "scripts" / "preprocessing.py"
)
assert _pp_spec and _pp_spec.loader
_preprocessing = importlib.util.module_from_spec(_pp_spec)
_pp_spec.loader.exec_module(_preprocessing)
clustering = _preprocessing.clustering
clustering_hce = _preprocessing.clustering_hce

random.seed(0)
np.random.seed(0)

# URLs for datasets
ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
HCE_URL = "https://raw.githubusercontent.com/aspuru-guzik-group/Tartarus/refs/heads/main/datasets/hce.csv"

def download_dataset(url, output_path: Path | str):
    """Download a dataset from URL."""
    output_path = Path(output_path)
    print(f"Downloading {output_path.name} from {url}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(output_path))
    print(f"Downloaded to {output_path}")
# %%
# Load QM9
# database = tcgd.QM9(root=str(_REPO / "qm9_data"))

# # Extract SMILES and Gibbs free energy of atomization from QM9
# smiles_list = [di.smiles for di in database]
# data_qm9 = pd.DataFrame(smiles_list, columns=['smiles'])
# data_qm9['dga'] = [di.y[:,15].item() for di in database] # Place 15 is Gibbs free energy
# print(f"Dataset size: {len(data_qm9.index)}")

# # Filter QM9 with roundtrip
# data_qm9['mol'] = [rdc.MolFromSmiles(si) for si in list(data_qm9['smiles'].values)]
# data_qm9 = data_qm9.dropna(ignore_index=True)
# data_qm9 = data_qm9.drop(columns=['mol'])
# print(f"Dataset size after filtering: {len(data_qm9.index)}")

# # Apply clustering to QM9
# data_qm9 = clustering(
#     data_qm9,
#     target_column="dga",
#     output_dir=str(_REPO / "clustered_data" / "qm9"),
#     dataset_name="qm9",
# )

# %%
# ESOL
download_dataset(ESOL_URL, _REPO / "data" / "esol.csv")

database = pd.read_csv(_REPO / "data" / "esol.csv")
data_esol = pd.DataFrame()
data_esol['smiles'] = database['smiles'].values
data_esol['solubility'] = database['measured log solubility in mols per litre'].values
print(f"Dataset size: {len(data_esol.index)}")

# Apply clustering to ESOL
data_esol = clustering(
    data_esol,
    target_column="solubility",
    output_dir=str(_REPO / "clustered_data" / "esol"),
    dataset_name="esol",
)
# %%
# HCE — AtomPair fingerprint clustering (see scripts/preprocessing.clustering_hce)
download_dataset(HCE_URL, _REPO / "data" / "hce.csv")

database = pd.read_csv(_REPO / "data" / "hce.csv")
data_hce = pd.DataFrame()
data_hce['smiles'] = database['smiles'].values
data_hce['pce_1'] = database['pce_1'].values
print(f"Dataset size (raw): {len(data_hce.index)}")

# Exact-zero PCE: non-physical / invalid-device placeholders in the source table; drop for regression.
pce_num = pd.to_numeric(data_hce['pce_1'], errors='coerce')
n_zero = int((pce_num.notna() & (pce_num == 0.0)).sum())
if n_zero:
    data_hce = data_hce.loc[~(pce_num.notna() & (pce_num == 0.0))].reset_index(drop=True)
    print(f"Dropped {n_zero} rows with pce_1 == 0 (HCE). New size: {len(data_hce.index)}")
else:
    print(f"Dataset size: {len(data_hce.index)}")

data_hce = clustering_hce(
    data_hce,
    target_column="pce_1",
    output_dir=str(_REPO / "clustered_data" / "hce"),
    dataset_name="hce",
)
print("\nAll datasets processed successfully!")
# %%
