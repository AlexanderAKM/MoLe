"""CLI entry point: mole-train

Train a ChemBERTa regressor on one of the bundled datasets or a custom CSV.

Examples
--------
# Train on ESOL (from repo with pre-split data)
mole-train --dataset esol

# Train on a custom dataset
mole-train --train-csv data/train.csv --test-csv data/test.csv \
           --val-csv data/val.csv --target-column solubility \
           --output-dir trained_models/custom
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATASET_CONFIGS = {
    "esol": {"target_column": "solubility", "dir": "esol"},
    "hce":  {"target_column": "pce_1",      "dir": "hce"},
    "qm9":  {"target_column": "dga",        "dir": "qm9"},
}


def _resolve_repo_root() -> Path:
    """Best-effort: walk up from CWD or this file looking for clustered_data/."""
    for anchor in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (anchor / "clustered_data").is_dir():
            return anchor
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mole-train",
        description="Train a ChemBERTa regression model.",
    )

    g = p.add_argument_group("dataset")
    g.add_argument(
        "--dataset", choices=list(DATASET_CONFIGS), default=None,
        help="Bundled dataset name (looks for pre-split CSVs under clustered_data/). "
             "Mutually exclusive with --train-csv / --test-csv.",
    )
    g.add_argument("--train-csv", type=Path, default=None)
    g.add_argument("--test-csv",  type=Path, default=None)
    g.add_argument("--val-csv",   type=Path, default=None)
    g.add_argument("--target-column", type=str, default=None)
    g.add_argument("--smiles-column", type=str, default="smiles")

    h = p.add_argument_group("hyperparameters")
    h.add_argument("--epochs",           type=int,   default=100)
    h.add_argument("--batch-size",       type=int,   default=32)
    h.add_argument("--lr",               type=float, default=5e-5)
    h.add_argument("--l1-lambda",        type=float, default=0.0)
    h.add_argument("--l2-lambda",        type=float, default=0.01)
    h.add_argument("--dropout",          type=float, default=0.3)
    h.add_argument("--hidden-channels",  type=int,   default=128)
    h.add_argument("--num-mlp-layers",   type=int,   default=1)
    h.add_argument("--early-stopping-patience", type=int, default=10)
    h.add_argument("--freeze-encoder",   action="store_true")
    h.add_argument("--random-seed",      type=int,   default=19237)

    o = p.add_argument_group("output")
    o.add_argument("--output-dir", type=Path, default=None,
                   help="Root output directory (default: trained_models/)")
    o.add_argument("--model-name", type=str, default="chemberta",
                   help="Subdirectory name inside the dataset output folder")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    import pandas as pd
    from mole.utils.normalizing import normalize_csv
    from mole.utils.chemberta_workflows import train_chemberta_model

    repo = _resolve_repo_root()

    if args.dataset:
        cfg = DATASET_CONFIGS[args.dataset]
        target_col = args.target_column or cfg["target_column"]
        d = cfg["dir"]
        cluster_dir = repo / "clustered_data" / d
        train_csv = args.train_csv or cluster_dir / f"train_{d}.csv"
        test_csv  = args.test_csv  or cluster_dir / f"test_{d}.csv"
        val_csv   = args.val_csv   or cluster_dir / f"validation_{d}.csv"
        dataset_name = f"train_{d}"
    else:
        if not args.train_csv or not args.test_csv or not args.target_column:
            parser.error("Provide --dataset OR (--train-csv, --test-csv, --target-column)")
        train_csv  = args.train_csv
        test_csv   = args.test_csv
        val_csv    = args.val_csv
        target_col = args.target_column
        dataset_name = train_csv.stem

    output_dir = args.output_dir or repo / "trained_models"

    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    norm_train, scaler = normalize_csv(df_train, target_col=target_col)
    norm_test, _       = normalize_csv(df_test,  target_col=target_col, scaler=scaler)

    df_val = None
    if val_csv and Path(val_csv).exists():
        df_val_raw = pd.read_csv(val_csv)
        df_val, _ = normalize_csv(df_val_raw, target_col=target_col, scaler=scaler)

    ns = argparse.Namespace(
        smiles_column=args.smiles_column,
        target_column=target_col,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_mlp_layers=args.num_mlp_layers,
        early_stopping_patience=args.early_stopping_patience,
        random_seed=args.random_seed,
        train_csv=str(train_csv),
    )

    results = train_chemberta_model(
        ns, norm_train, norm_test, scaler,
        df_val=df_val,
        dataset_name=dataset_name,
        model_name=args.model_name,
        freeze_encoder=args.freeze_encoder,
    )
    print(f"\nTraining complete. Outputs in: {results['output_dir']}")


if __name__ == "__main__":
    main()
