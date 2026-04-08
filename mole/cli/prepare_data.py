"""CLI entry point: mole-prepare-data

Download, cluster, and split datasets for training.

Examples
--------
# Prepare ESOL dataset
mole-prepare-data --dataset esol

# Prepare all datasets
mole-prepare-data --dataset all

# Custom output directory
mole-prepare-data --dataset esol --output-dir ./my_data
"""
from __future__ import annotations

import argparse
import random
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit.Chem as rdc
from sklearn.model_selection import train_test_split

ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
HCE_URL  = "https://raw.githubusercontent.com/aspuru-guzik-group/Tartarus/refs/heads/main/datasets/hce.csv"

RANDOM_SEED = 19237


def _resolve_repo_root() -> Path:
    for anchor in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (anchor / "clustered_data").is_dir() or (anchor / "mole").is_dir():
            return anchor
    return Path.cwd()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} from {url} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  -> {dest}")


def prepare_esol(output_dir: Path, raw_dir: Path) -> None:
    from mole.utils.clustering import clustering

    csv_path = raw_dir / "esol.csv"
    download(ESOL_URL, csv_path)

    db = pd.read_csv(csv_path)
    data = pd.DataFrame({
        "smiles": db["smiles"].values,
        "solubility": db["measured log solubility in mols per litre"].values,
    })
    print(f"ESOL size: {len(data)}")

    cluster_dir = output_dir / "esol"
    data = clustering(data, target_column="solubility",
                      output_dir=str(cluster_dir), dataset_name="esol")

    _split_and_save(data, cluster_dir, "esol")


def prepare_hce(output_dir: Path, raw_dir: Path) -> None:
    from mole.utils.clustering import clustering_hce

    csv_path = raw_dir / "hce.csv"
    download(HCE_URL, csv_path)

    db = pd.read_csv(csv_path)
    data = pd.DataFrame({
        "smiles": db["smiles"].values,
        "pce_1": db["pce_1"].values,
    })
    print(f"HCE raw size: {len(data)}")

    pce_num = pd.to_numeric(data["pce_1"], errors="coerce")
    n_zero = int((pce_num.notna() & (pce_num == 0.0)).sum())
    if n_zero:
        data = data.loc[~(pce_num.notna() & (pce_num == 0.0))].reset_index(drop=True)
        print(f"Dropped {n_zero} rows with pce_1 == 0. New size: {len(data)}")

    cluster_dir = output_dir / "hce"
    data = clustering_hce(data, target_column="pce_1",
                          output_dir=str(cluster_dir), dataset_name="hce")

    _split_and_save(data, cluster_dir, "hce")


def _split_and_save(data: pd.DataFrame, cluster_dir: Path, name: str) -> None:
    drop_cols = [c for c in ("quantile",) if c in data.columns]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    train, test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED, stratify=data["cluster"])
    train, val = train_test_split(
        train, test_size=0.25, random_state=RANDOM_SEED, stratify=train["cluster"])

    train.to_csv(cluster_dir / f"train_{name}.csv", index=False)
    val.to_csv(cluster_dir / f"validation_{name}.csv", index=False)
    test.to_csv(cluster_dir / f"test_{name}.csv", index=False)
    print(f"  Saved train({len(train)})/val({len(val)})/test({len(test)}) to {cluster_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mole-prepare-data",
        description="Download, cluster, and split datasets for MoLe training.",
    )
    p.add_argument(
        "--dataset", choices=["esol", "hce", "all"], default="all",
        help="Which dataset to prepare (default: all).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Root output directory for clustered data (default: clustered_data/)",
    )
    p.add_argument(
        "--raw-dir", type=Path, default=None,
        help="Directory for raw downloaded files (default: data/)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    random.seed(0)
    np.random.seed(0)

    repo = _resolve_repo_root()
    output_dir = args.output_dir or repo / "clustered_data"
    raw_dir    = args.raw_dir    or repo / "data"

    datasets = ["esol", "hce"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        print(f"\n{'=' * 40}\nPreparing {ds.upper()}\n{'=' * 40}")
        if ds == "esol":
            prepare_esol(output_dir, raw_dir)
        elif ds == "hce":
            prepare_hce(output_dir, raw_dir)

    print("\nAll datasets prepared successfully!")


if __name__ == "__main__":
    main()
