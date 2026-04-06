"""
Dataset quality control for QM9 (ChemBERTa) and HCE.

QM9: flags molecules that often cause embedding / training artifacts:
  - RDKit sanitization failures
  - non-canonical SMILES vs RDKit canonical SMILES
  - token length vs ChemBERTa (truncation at 512 in training)
  - optional: [CLS] hidden-state norm outliers (base ChemBERTa, no task head)

HCE: identify exact-zero PCE labels (invalid / non-emissive devices in source data).

Run from repo root, e.g.:
  python scripts/dataset_qc.py qm9 --csv clustered_data/qm9/qm9.csv
  python scripts/dataset_qc.py qm9 --csv clustered_data/qm9/qm9.csv --embedding-outliers --device cuda
  python scripts/dataset_qc.py hce --csv data/hce.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import rdkit.Chem as rdc
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def _repo_root() -> Path:
    return _REPO


def _canonical_smiles(smiles: str) -> str | None:
    mol = rdc.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rdc.SanitizeMol(mol)
    except Exception:
        return None
    return rdc.MolToSmiles(mol, canonical=True)


def qc_qm9_row(raw_smiles: str, tokenizer) -> dict:
    out = {
        "sanitize_ok": True,
        "canonical": None,
        "not_canonical_input": False,
        "token_len": None,
        "tokens_exceed_512": False,
        "tokens_near_limit": False,
    }
    mol = rdc.MolFromSmiles(raw_smiles)
    if mol is None:
        out["sanitize_ok"] = False
        return out
    try:
        rdc.SanitizeMol(mol)
    except Exception:
        out["sanitize_ok"] = False
        return out

    can = rdc.MolToSmiles(mol, canonical=True)
    out["canonical"] = can
    out["not_canonical_input"] = raw_smiles.strip() != can.strip()

    enc = tokenizer.encode(raw_smiles, add_special_tokens=True)
    out["token_len"] = len(enc)
    out["tokens_exceed_512"] = len(enc) > 512
    out["tokens_near_limit"] = len(enc) > 480

    return out


def run_qm9(
    csv_path: Path,
    sample_print: int,
    embedding_outliers: bool,
    device: str,
    batch_size: int,
    write_filtered: Path | None,
    exclude_flags: list[str],
) -> None:
    from transformers import RobertaModel, RobertaTokenizerFast

    from utils.chemberta_workflows import DEFAULT_PRETRAINED_NAME

    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "dga" not in df.columns:
        raise SystemExit("QM9 CSV must contain columns: smiles, dga")

    tokenizer = RobertaTokenizerFast.from_pretrained(DEFAULT_PRETRAINED_NAME)
    records = []
    for idx, row in df.iterrows():
        s = str(row["smiles"])
        q = qc_qm9_row(s, tokenizer)
        flags = []
        if not q["sanitize_ok"]:
            flags.append("sanitize_fail")
        if q["not_canonical_input"]:
            flags.append("not_canonical_input")
        if q["tokens_exceed_512"]:
            flags.append("tokens_gt_512")
        if q["tokens_near_limit"] and not q["tokens_exceed_512"]:
            flags.append("tokens_gt_480")
        records.append(
            {
                "index": idx,
                "smiles": s,
                "dga": row["dga"],
                "canonical_smiles": q["canonical"],
                "token_len": q["token_len"],
                "flags": ";".join(flags) if flags else "",
            }
        )

    out = pd.DataFrame.from_records(records)
    n = len(out)
    print(f"QM9 rows: {n}")
    fc: dict[str, int] = {}
    for cell in out["flags"]:
        for p in (cell.split(";") if cell else []):
            if p:
                fc[p] = fc.get(p, 0) + 1
    print("Flag counts (before embedding):", fc)

    if embedding_outliers:
        import torch

        model = RobertaModel.from_pretrained(DEFAULT_PRETRAINED_NAME)
        model.eval()
        model.to(device)
        norms = np.zeros(n, dtype=np.float32)
        texts = out["smiles"].tolist()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = texts[start : start + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                hidden = model(**enc).last_hidden_state
                # [CLS] at index 0
                cls_vec = hidden[:, 0, :]
                norms[start : start + len(batch)] = (
                    cls_vec.norm(dim=-1).float().cpu().numpy()
                )
        out["cls_norm"] = norms
        lo, hi = np.percentile(norms, [1.0, 99.0])
        emb_flags = []
        for v in norms:
            f = []
            if v < lo:
                f.append("cls_norm_low")
            if v > hi:
                f.append("cls_norm_high")
            emb_flags.append(";".join(f) if f else "")
        out["embedding_flags"] = emb_flags
        efc: dict[str, int] = {}
        for cell in out["embedding_flags"]:
            for p in (cell.split(";") if cell else []):
                if p:
                    efc[p] = efc.get(p, 0) + 1
        print("Embedding flag counts:", efc)
        print(f"[CLS] L2 norm: min={norms.min():.4f} max={norms.max():.4f} p1={lo:.4f} p99={hi:.4f}")

    # --- Print examples for manual review (Robert) ---
    def show_subset(mask: pd.Series, title: str) -> None:
        sub = out.loc[mask].head(sample_print)
        if sub.empty:
            print(f"\n--- {title}: (none) ---\n")
            return
        print(f"\n--- {title} (up to {sample_print}) ---\n")
        cols = ["smiles", "dga", "token_len", "flags"]
        if "cls_norm" in sub.columns:
            cols.append("cls_norm")
        print(sub[cols].to_string(index=False))

    show_subset(out["flags"].str.contains("sanitize_fail", na=False), "sanitize_fail")
    show_subset(out["flags"].str.contains("not_canonical_input", na=False), "not_canonical_input")
    show_subset(out["flags"].str.contains("tokens_gt_512", na=False), "tokens_gt_512")
    show_subset(out["flags"].str.contains("tokens_gt_480", na=False), "tokens_gt_480 (<=512)")
    if embedding_outliers:
        show_subset(
            out["embedding_flags"].str.contains("cls_norm_low|cls_norm_high", na=False),
            "embedding norm outliers (p1–p99 band)",
        )

    if write_filtered:
        def row_drop(row) -> bool:
            parts = []
            if row["flags"]:
                parts.extend(row["flags"].split(";"))
            if embedding_outliers and row.get("embedding_flags"):
                parts.extend(str(row["embedding_flags"]).split(";"))
            parts = [p for p in parts if p]
            return any(p in exclude_flags for p in parts)

        mask_drop = out.apply(row_drop, axis=1)
        kept = df.iloc[~mask_drop.values].reset_index(drop=True)
        write_filtered.parent.mkdir(parents=True, exist_ok=True)
        kept.to_csv(write_filtered, index=False)
        print(f"\nWrote filtered CSV: {write_filtered} (kept {len(kept)} / {len(df)}, dropped {int(mask_drop.sum())})")


def run_hce(csv_path: Path, sample_print: int, write_filtered: Path | None) -> None:
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns or "pce_1" not in df.columns:
        raise SystemExit("HCE CSV must contain columns: smiles, pce_1")

    pce = pd.to_numeric(df["pce_1"], errors="coerce")
    is_zero = pce.notna() & (pce == 0.0)
    n_zero = int(is_zero.sum())
    print(f"HCE rows: {len(df)} | exact zero pce_1: {n_zero}")

    zeros = df.loc[is_zero]
    if n_zero and sample_print:
        print(f"\n--- Sample exact-zero PCE rows (up to {sample_print}) ---\n")
        print(zeros.head(sample_print)[["smiles", "pce_1"]].to_string(index=False))

    if write_filtered:
        kept = df.loc[~is_zero].reset_index(drop=True)
        write_filtered.parent.mkdir(parents=True, exist_ok=True)
        kept.to_csv(write_filtered, index=False)
        print(f"\nWrote filtered CSV: {write_filtered} (kept {len(kept)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="QM9 / HCE dataset QC")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_q = sub.add_parser("qm9", help="QM9 SMILES / token / optional embedding QC")
    p_q.add_argument("--csv", type=Path, default=_repo_root() / "clustered_data/qm9/qm9.csv")
    p_q.add_argument("--sample-print", type=int, default=25, help="Rows per flag category to print")
    p_q.add_argument("--embedding-outliers", action="store_true", help="Flag [CLS] norm outliers (slower)")
    p_q.add_argument("--device", type=str, default="cpu")
    p_q.add_argument("--batch-size", type=int, default=32)
    p_q.add_argument(
        "--write-filtered",
        type=Path,
        default=None,
        help="Write CSV with rows removed when any exclude-flag matches",
    )
    p_q.add_argument(
        "--exclude-flags",
        type=str,
        default="sanitize_fail,tokens_gt_512,cls_norm_low,cls_norm_high",
        help="Comma-separated flags used when --write-filtered is set",
    )

    p_h = sub.add_parser("hce", help="HCE exact-zero PCE report / filter")
    p_h.add_argument("--csv", type=Path, default=_repo_root() / "data/hce.csv")
    p_h.add_argument("--sample-print", type=int, default=25)
    p_h.add_argument("--write-filtered", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "qm9":
        excl = [x.strip() for x in args.exclude_flags.split(",") if x.strip()]
        run_qm9(
            args.csv.resolve(),
            args.sample_print,
            args.embedding_outliers,
            args.device,
            args.batch_size,
            args.write_filtered.resolve() if args.write_filtered else None,
            excl,
        )
    else:
        run_hce(
            args.csv.resolve(),
            args.sample_print,
            args.write_filtered.resolve() if args.write_filtered else None,
        )


if __name__ == "__main__":
    main()
