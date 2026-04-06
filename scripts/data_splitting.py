# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_REPO = Path(__file__).resolve().parents[1]
_CLUSTER = _REPO / "clustered_data"

# %%
RANDOM_SEED = 19237
# %%
# FOR ESOL
esol = pd.read_csv(_CLUSTER / "esol" / "esol.csv")
_esol_drop = [c for c in ("quantile",) if c in esol.columns]
if _esol_drop:
    esol = esol.drop(columns=_esol_drop)
train_esol, test_esol = train_test_split(esol, test_size=0.2, random_state=RANDOM_SEED, stratify=esol['cluster'])
train_esol.to_csv(_CLUSTER / "esol" / "train_esol.csv", index=False)
test_esol.to_csv(_CLUSTER / "esol" / "test_esol.csv", index=False)
# %%
# FOR QM9
qm9 = pd.read_csv(_CLUSTER / "qm9" / "qm9.csv")
drop_cols = [c for c in ("mol", "quantile") if c in qm9.columns]
if drop_cols:
    qm9 = qm9.drop(columns=drop_cols)
train_qm9, test_qm9 = train_test_split(qm9, test_size=0.2, random_state=RANDOM_SEED, stratify=qm9['cluster'])
train_qm9, validation_qm9 = train_test_split(train_qm9, test_size=0.25, random_state=RANDOM_SEED, stratify=train_qm9['cluster']) # 60/20/20 split
train_qm9.to_csv(_CLUSTER / "qm9" / "train_qm9.csv", index=False)
validation_qm9.to_csv(_CLUSTER / "qm9" / "validation_qm9.csv", index=False)
test_qm9.to_csv(_CLUSTER / "qm9" / "test_qm9.csv", index=False)
# %%
hce = pd.read_csv(_CLUSTER / "hce" / "hce.csv")
hce
train_hce, test_hce = train_test_split(hce, test_size=0.2, random_state=RANDOM_SEED, stratify=hce['cluster'])
train_hce.to_csv(_CLUSTER / "hce" / "train_hce.csv", index=False)
test_hce.to_csv(_CLUSTER / "hce" / "test_hce.csv", index=False)
# %%
