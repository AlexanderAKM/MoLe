"""Profile token lengths for all three datasets to quantify truncation impact."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast

_REPO = Path(__file__).resolve().parents[1]

TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
MAX_LENGTH = 512

DATASETS = {
    "ESOL": {
        "full": _REPO / "clustered_data" / "esol" / "esol.csv",
        "train": _REPO / "clustered_data" / "esol" / "train_esol.csv",
        "test": _REPO / "clustered_data" / "esol" / "test_esol.csv",
        "smiles_col": "smiles",
    },
    "QM9": {
        "full": _REPO / "clustered_data" / "qm9" / "qm9.csv",
        "train": _REPO / "clustered_data" / "qm9" / "train_qm9.csv",
        "test": _REPO / "clustered_data" / "qm9" / "test_qm9.csv",
        "smiles_col": "smiles",
    },
    "HCE": {
        "full": _REPO / "clustered_data" / "hce" / "hce.csv",
        "train": _REPO / "clustered_data" / "hce" / "train_hce.csv",
        "test": _REPO / "clustered_data" / "hce" / "test_hce.csv",
        "smiles_col": "smiles",
    },
}


def profile_dataset(name: str, info: dict, tokenizer: RobertaTokenizerFast):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    for split_name in ("full", "train", "test"):
        path = info[split_name]
        if not path.exists():
            print(f"  [{split_name}] file not found: {path}")
            continue
        df = pd.read_csv(path)
        smiles_list = df[info["smiles_col"]].tolist()
        encodings = tokenizer(smiles_list, add_special_tokens=True)
        lengths = [len(ids) for ids in encodings["input_ids"]]
        lengths = np.array(lengths)

        n_total = len(lengths)
        n_truncated = int((lengths > MAX_LENGTH).sum())
        pct_truncated = 100.0 * n_truncated / n_total if n_total else 0.0

        print(f"\n  Split: {split_name}  (n={n_total})")
        print(f"    Min tokens:     {lengths.min()}")
        print(f"    Mean tokens:    {lengths.mean():.1f}")
        print(f"    Median tokens:  {int(np.median(lengths))}")
        print(f"    Max tokens:     {lengths.max()}")
        print(f"    Std tokens:     {lengths.std():.1f}")
        print(f"    >512 (truncated): {n_truncated}  ({pct_truncated:.2f}%)")

        if n_truncated > 0:
            trunc_lengths = lengths[lengths > MAX_LENGTH]
            print(f"    Truncated max:  {trunc_lengths.max()}")
            print(f"    Truncated mean: {trunc_lengths.mean():.1f}")


if __name__ == "__main__":
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_NAME)
    for name, info in DATASETS.items():
        profile_dataset(name, info, tokenizer)
