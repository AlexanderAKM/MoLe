# MoLe: Molecular Lens

**Mechanistic Interpretability for Chemistry using TransformerLens and ChemBERTa**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

MoLe (Molecular Lens) is a framework for applying mechanistic interpretability techniques to molecular property prediction models. This repository provides tools to understand how transformer-based chemistry models (specifically ChemBERTa) process molecular representations and make predictions.

### Key Features

- **Regression Lens Analysis**: Track how predictions evolve through transformer layers
- **Component Ablation**: Systematically ablate MLP neurons and attention heads to measure their importance
- **Model Conversion**: Convert HuggingFace RoBERTa models to TransformerLens format for interpretability
- **Multiple Datasets**: Pre-configured experiments on ESOL, QM9, and HCE datasets
- **Visualization Tools**: Publication-quality plots for interpretability results

### What is Regression Lens?

Regression lens extends the concept of "logit lens" from language models to regression tasks. By applying the prediction head after each transformer layer, we can observe how the model's prediction evolves as information flows through the network. This reveals:
- Which layers contribute most to the final prediction
- How molecular representations are refined through the network
- Whether certain properties are captured early or late in processing

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training and analysis), although trained models are provided

### Quick Install (pip)

```bash
# Clone and install in one go
git clone https://github.com/yourusername/Mechanistic-interpretability-for-chemistry.git
cd Mechanistic-interpretability-for-chemistry
pip install -e .
```

### HPC / Cluster Install

On HPC systems where PyTorch/CUDA are provided as modules, load them first
then install the rest:

```bash
module load python/3.10 cuda/12.1 pytorch/2.2   # adjust to your site
git clone https://github.com/yourusername/Mechanistic-interpretability-for-chemistry.git
cd Mechanistic-interpretability-for-chemistry
pip install --no-deps -e .          # skip deps already satisfied by modules
pip install -r requirements.txt     # or cherry-pick missing packages
```

> **Tip:** If `torch-geometric` installation fails on HPC, install the core
> package first (`pip install -e .`) and then install PyG following
> [their platform guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Setup with Virtual Environment

```bash
# Using conda
conda create -n mole python=3.10
conda activate mole

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install
pip install -e .
```

### Verify Installation

```bash
python -c "from mole.utils.tl_conversion import load_chemberta_models; print('OK')"
```

---

## Quick Start

### 1. Load a Pre-trained Model

```python
from mole.utils.tl_conversion import load_chemberta_models

MODEL_PATH = "trained_models/train_esol/chemberta/chemberta_model_final.bin"
SCALER_PATH = "trained_models/train_esol/chemberta/normalization_scaler.pkl"
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"

hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, device="cuda", scaler_path=SCALER_PATH
)
```

### 2. Run Regression Lens Analysis

```python
from mole.utils.tl_regression import run_regression_lens, plot_individual_molecules_regression_lens
from pathlib import Path

molecules = ["CCO", "c1ccccc1", "CC(=O)O"]  # Ethanol, Benzene, Acetic acid
results = run_regression_lens(tl_encoder, tl_regressor, scaler, molecules, tokenizer)

plot_individual_molecules_regression_lens(
    results, 
    results_dir=Path("results/example"),
    molecule_labels=["Ethanol", "Benzene", "Acetic Acid"]
)
```

### 3. Run Ablation Studies

```python
from mole.utils.tl_ablation import run_ablation_analysis_with_metrics
import pandas as pd

test_data = pd.read_csv("clustered_data/esol/test_esol.csv")

results = run_ablation_analysis_with_metrics(
    tl_encoder, 
    tl_regressor, 
    tokenizer, 
    test_data,
    target_column="solubility",
    output_dir=Path("results/esol"),
    n_seeds=10,
    scaler=scaler
)
```

---

## CLI Commands

After installation, three commands are available:

### `mole-train` — Train a model

```bash
# Train on a bundled dataset
mole-train --dataset esol
mole-train --dataset hce --epochs 50 --lr 1e-4

# Train on custom CSVs
mole-train --train-csv my_train.csv --test-csv my_test.csv \
           --val-csv my_val.csv --target-column property
```

### `mole-evaluate` — Evaluate a trained model

```bash
# Evaluate a bundled model
mole-evaluate --dataset esol

# Evaluate a custom model
mole-evaluate --model-dir trained_models/custom/chemberta \
              --test-csv data/test.csv --target-column solubility
```

### `mole-prepare-data` — Download and preprocess datasets

```bash
# Prepare all bundled datasets (ESOL + HCE)
mole-prepare-data

# Prepare a specific dataset
mole-prepare-data --dataset esol
```

---

## Repository Structure

```
.
├── mole/                            # Installable Python package
│   ├── models/                      # Model architectures
│   │   ├── chemberta_regressor.py
│   │   ├── encoder_mlp.py
│   │   └── simple_mlp.py
│   ├── utils/                       # Core utilities & interpretability
│   │   ├── tl_conversion.py         # HF → TransformerLens conversion
│   │   ├── tl_regression.py         # Regression lens analysis
│   │   ├── tl_ablation.py           # Ablation studies
│   │   ├── tl_validation.py         # Model validation
│   │   ├── chemberta_workflows.py   # Training & evaluation orchestration
│   │   ├── clustering.py            # Molecular clustering
│   │   └── plotting.py              # Visualization
│   ├── scripts/                     # Core pipeline scripts
│   │   ├── load_data.py             # Data downloading and clustering
│   │   ├── data_splitting.py        # Train/test splits
│   │   ├── training.py              # Full training pipeline
│   │   └── evaluate.py              # Model evaluation
│   ├── cli/                         # Command-line entry points
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── prepare_data.py
│   └── TL_chem.py                   # Main interpretability analysis script
├── scripts/                         # Non-essential utility scripts
│   ├── token_length_profile.py      # Tokenization stats
│   ├── dataset_qc.py               # Data quality checks
│   └── extract_regression_lens.py   # Regression lens extraction
├── clustered_data/                  # Processed datasets (ESOL, QM9, HCE)
├── trained_models/                  # Pre-trained model checkpoints
├── results/                         # Analysis results and plots
├── pyproject.toml                   # Package configuration
└── requirements.txt                 # Legacy dependency list
```

---

## Usage Examples

### Comparing Molecule Groups

```python
from mole.utils.tl_regression import compare_molecule_groups_regression_lens

molecule_groups = {
    "Alcohols": ["CCO", "CC(C)O", "CCCO"],
    "Aromatics": ["c1ccccc1", "c1ccc(C)cc1"],
    "Acids": ["CC(=O)O", "CCC(=O)O"]
}

group_results = compare_molecule_groups_regression_lens(
    tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer
)
```

### Validation of Model Conversion

```python
from mole.utils.tl_validation import validate_conversion, test_prediction_equivalence

test_smiles = "CCO"
inputs = tokenizer(test_smiles, return_tensors="pt").to(device)

conversion_results = validate_conversion(
    hf_encoder, tl_encoder, 
    inputs["input_ids"], 
    inputs["attention_mask"]
)
print(f"Max difference: {conversion_results['final_output']:.6f}")
# Should be < 1e-5 for faithful conversion
```

## Reproducibility

To reproduce all results from the paper:

1. **Data Preparation**
```bash
mole-prepare-data --dataset all
```

2. **Model Training**
```bash
mole-train --dataset esol
mole-train --dataset hce
mole-train --dataset qm9
```

3. **Run Analysis**
```bash
python mole/TL_chem.py
```

---

## Key Components

### TransformerLens Integration

This project uses [TransformerLens](https://github.com/neelnanda-io/TransformerLens) to access intermediate activations from ChemBERTa models. The conversion process:

1. Loads a fine-tuned HuggingFace RoBERTa model
2. Converts weights to TransformerLens format
3. Validates numerical equivalence (< 1e-5 difference)
4. Provides hook points for interpretability

**Note**: We primarily use TransformerLens for its clean API to access layer-wise activations via `run_with_cache()`. Advanced features like activation patching are planned for future work.

### Regression Lens

The regression lens technique applies the prediction head after each transformer layer:

```
Embedding → [Apply Head] → Prediction₀
    ↓
Block 1 → [Apply Head] → Prediction₁
    ↓
Block 2 → [Apply Head] → Prediction₂
    ↓
Block 3 → [Apply Head] → Prediction₃ (final)
```

This reveals how predictions evolve and which layers are most important for property prediction.

### Ablation Studies

Systematic removal of model components:
- **MLP Ablation**: Zero out random percentages of MLP neurons
- **Attention Ablation**: Zero out random percentages of attention heads
- **Combined Ablation**: Ablate both components simultaneously

Metrics tracked: MAE, RMSE, R² degradation across ablation percentages.

---

## Citation

If you use this code in your research, please cite:

```
[to be filled in when paper is on ArXiv]
```

Paper citation (when published):
```
[to be filled in when paper is on ArXiv]

```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **ChemBERTa**: Pre-trained model from [DeepChem](https://github.com/deepchem/deepchem)
- **TransformerLens**: Interpretability framework by [Neel Nanda](https://github.com/neelnanda-io/TransformerLens)
- **MoleculeNet**: Dataset source for ESOL and QM9
- **Tartarus**: Dataset source for HCE
