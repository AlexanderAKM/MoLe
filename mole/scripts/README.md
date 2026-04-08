## Pipeline scripts

These scripts reproduce the paper results end-to-end. Run them from the
repo root after `pip install -e .`, or use the CLI commands instead.

### Data preparation

1. **`load_data.py`** — downloads ESOL and HCE from online sources and clusters
   them using `mole.utils.clustering` (`clustering` for ESOL,
   `clustering_hce` for HCE with AtomPair fingerprints). The QM9 block is
   commented out (requires `torch-geometric`). Rows with HCE `pce_1 == 0`
   are dropped before clustering.

2. **`data_splitting.py`** — splits each clustered dataset into
   train / validation / test (60/20/20, stratified by cluster).

### Training & evaluation

3. **`training.py`** — trains ChemBERTa regressors on ESOL, HCE, and QM9
   with shared hyperparameters.

4. **`evaluate.py`** — loads trained checkpoints and evaluates on test sets.

### Interpretability

5. **`TL_chem.py`** (in `mole/`) — runs the full interpretability pipeline
   (ablation studies + regression lens) for all three datasets.

### CLI alternatives

Instead of running scripts directly, you can use:

```bash
mole-prepare-data --dataset all    # replaces load_data.py + data_splitting.py
mole-train --dataset esol           # replaces training.py (one dataset at a time)
mole-evaluate --dataset esol        # replaces evaluate.py (one dataset at a time)
```

### Utility scripts (in repo-root `scripts/`)

- `dataset_qc.py` — QM9/HCE quality checks (sanitization, token length, embedding outliers)
- `token_length_profile.py` — tokenization length statistics
- `extract_regression_lens.py` — extract per-layer R² from regression lens
