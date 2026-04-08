"""CLI entry point: mole-evaluate

Evaluate a trained ChemBERTa model on a test set.

Examples
--------
# Evaluate the bundled ESOL model
mole-evaluate --dataset esol

# Evaluate a custom model
mole-evaluate --model-dir trained_models/custom/chemberta \
              --test-csv data/test.csv --target-column solubility
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

DATASET_CONFIGS = {
    "esol": {"target_column": "solubility", "dir": "esol"},
    "hce":  {"target_column": "pce_1",      "dir": "hce"},
    "qm9":  {"target_column": "dga",        "dir": "qm9"},
}


def _resolve_repo_root() -> Path:
    for anchor in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (anchor / "clustered_data").is_dir():
            return anchor
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mole-evaluate",
        description="Evaluate a trained ChemBERTa regression model.",
    )
    p.add_argument(
        "--dataset", choices=list(DATASET_CONFIGS), default=None,
        help="Bundled dataset (auto-finds model, scaler, and test CSV).",
    )
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Directory containing chemberta_model_final.bin, "
                        "normalization_scaler.pkl, and hyperparameters.json")
    p.add_argument("--test-csv",  type=Path, default=None)
    p.add_argument("--target-column", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--plot-filename", type=str, default="preds_vs_targets.pdf")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    import torch
    from safetensors.torch import load_file
    from mole.utils.chemberta_workflows import ChembertaRegressor, evaluate_chemberta_model
    from mole.utils.normalizing import normalize_csv

    repo = _resolve_repo_root()

    if args.dataset:
        cfg = DATASET_CONFIGS[args.dataset]
        target_col = args.target_column or cfg["target_column"]
        d = cfg["dir"]
        model_dir = args.model_dir or repo / "trained_models" / f"train_{d}" / "chemberta"
        test_csv  = args.test_csv  or repo / "clustered_data" / d / f"test_{d}.csv"
    else:
        if not args.model_dir or not args.test_csv or not args.target_column:
            parser.error("Provide --dataset OR (--model-dir, --test-csv, --target-column)")
        model_dir  = args.model_dir
        test_csv   = args.test_csv
        target_col = args.target_column

    import pandas as pd

    with open(model_dir / "normalization_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "hyperparameters.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)

    df_test = pd.read_csv(test_csv)
    norm_test, _ = normalize_csv(df_test, target_col=target_col, scaler=scaler, fit_scaler=False)

    model = ChembertaRegressor(
        pretrained="DeepChem/ChemBERTa-77M-MLM",
        dropout=hparams.get("dropout", 0.3),
        hidden_channels=hparams.get("hidden_channels", 128),
        num_mlp_layers=hparams.get("num_mlp_layers", 1),
    )

    state_path = model_dir / "chemberta_model_final.bin"
    if state_path.suffix == ".safetensors":
        model.load_state_dict(load_file(str(state_path), device="cpu"))
    else:
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    results = evaluate_chemberta_model(
        model=model,
        dataset=norm_test,
        scaler=scaler,
        target_column=target_col,
        output_dir=str(model_dir),
        batch_size=args.batch_size,
        plot_filename=args.plot_filename,
    )
    print(f"\nEvaluation complete. Results saved to {model_dir}")


if __name__ == "__main__":
    main()
