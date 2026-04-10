# %% [markdown]
# # ChemBERTa × TransformerLens: mechanistic interpretability notebook
#
# **Goal** – Load a fine‑tuned ChemBERTa checkpoint, port its encoder into
# [TransformerLens](https://github.com/neelnanda‑io/TransformerLens) (TL),
# validate it is functionally the same as the original model, and run a
# round of mechanistic interpretability analyses and visualizations
# (neuron ablations and regression-lens probes).  
# This notebook is used for development. After a technique works, it is moved to
# an independent Python file in utils/ and imported to ensure modularity.
#
# Essentially, this is just a file which runs the techniques shown in the paper
# For **all three datasets**. There is lots of repetitive code, but 
# this way it's easy to go through the methods and develop them.
# %%
from pathlib import Path
import torch
import pandas as pd
import os

_REPO = Path(__file__).resolve().parents[1]

from mole.utils.tl_conversion import load_chemberta_models
from mole.utils.tl_validation import validate_conversion, test_prediction_equivalence
from mole.utils.tl_ablation import run_ablation_analysis_with_metrics, plot_ablation_metrics
from mole.utils.tl_regression import run_regression_lens, plot_individual_molecules_regression_lens
from mole.utils.tl_regression import compare_molecule_groups_regression_lens, plot_group_molecules_regression_lens
from mole.utils.tl_patching import (
    run_pair_activation_patching,
    run_head_activation_patching,
    plot_head_patching_heatmap,
    select_same_length_pair,
)

# %%
# For ESOL
MODEL_PATH = str(_REPO / "trained_models/train_esol/chemberta/chemberta_model_final.bin")
FULL_PATH = str(_REPO / "clustered_data/esol/esol.csv")
TEST_PATH = str(_REPO / "clustered_data/esol/test_esol.csv")
TRAIN_PATH = str(_REPO / "clustered_data/esol/train_esol.csv")
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = str(_REPO / "trained_models/train_esol/chemberta/normalization_scaler.pkl")
TARGET_COLUMN = "solubility"
print(DEVICE)

# %%
full_data = pd.read_csv(FULL_PATH)
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)
# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO" # arbitrary
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results['final_output'] < 0.001}")

prediction_results = test_prediction_equivalence(hf_regressor, tl_regressor, [test_smiles], tokenizer, DEVICE)
print(f"The predictions are equivalent: {prediction_results['is_equivalent']}")
# %% [markdown]
# Let's run ablation studies to see the effect of misssing components
test_data = pd.read_csv(TEST_PATH)
test_molecules = test_data['smiles'].to_list()
targets = test_data[TARGET_COLUMN].to_list()

print(f"Testing ablation on {len(test_molecules)} molecules")
print(f"Target range: {min(targets):.3f} to {max(targets):.3f}")

esol_results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, target_column=TARGET_COLUMN, output_dir=_REPO / "results/esol", n_seeds=10, scaler=scaler)
plot_ablation_metrics(esol_results, _REPO / "results/esol")

# %%
import pickle

# Load saved ablation results
with open(_REPO / "results/esol/ablation/all_results.pkl", "rb") as f:
    esol_results = pickle.load(f)
    
plot_ablation_metrics(esol_results, _REPO / "results/esol")

# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest, smallest, and median target value to showcase the technique
# on the training data
median_idx = len(full_data) // 2
median_molecule = full_data.sort_values(TARGET_COLUMN).iloc[median_idx]["smiles"]

# Flatten the list to get a simple list of SMILES strings
max_smiles = full_data.nlargest(1128//2, TARGET_COLUMN)["smiles"].to_list()
min_smiles = full_data.nsmallest(1128//2, TARGET_COLUMN)["smiles"].to_list()
min_max_median_molecules = max_smiles + [median_molecule] + min_smiles

# Get actual target values for these molecules (also flattened)
max_targets = full_data.nlargest(1128//2, TARGET_COLUMN)[TARGET_COLUMN].to_list()
median_target = full_data.sort_values(TARGET_COLUMN).iloc[median_idx][TARGET_COLUMN]
min_targets = full_data.nsmallest(1128//2, TARGET_COLUMN)[TARGET_COLUMN].to_list()
actual_targets = max_targets + [median_target] + min_targets

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_median_molecules, tokenizer)
plot_individual_molecules_regression_lens(
    results, 
    results_dir=_REPO / "results/esol/example_regression_lens", 
    molecule_labels=[f"Molecule {i+1}" for i in range(1129)], 
    actual_targets=actual_targets, 
    target_labels=[f"rank {i+1}" for i in range(1129)]
)

results
# %% [markdown]
# Now we do regression lens on groups of molecules
# First example group
# example_molecule_groups = {
#     "Simple Alcohols": ["CCO", "CC(C)O", "CCCO"],
#     "Aromatic": ["c1ccccc1", "c1ccc(C)cc1", "c1ccc(O)cc1"],  
#     "Carboxylic Acids": ["CC(=O)O", "CCC(=O)O", "c1ccc(C(=O)O)cc1"],
#     "Alkanes": ["CC", "CCC", "CCCCCCCCCC"]
# }
# example_group_results = compare_molecule_groups_regression_lens(tl_encoder, tl_regressor, scaler, example_molecule_groups, tokenizer, DEVICE)
# plot_group_molecules_regression_lens(example_group_results, results_dir=_REPO / "results/ESOL/example_regression_lens")

# With clustering - build both groups and targets in one pass to ensure alignment
molecule_groups = {}
ordered_targets = []
for cluster, group in full_data.groupby('cluster'):
    cluster_name = f"Cluster {cluster + 1}"
    molecule_groups[cluster_name] = group['smiles'].tolist()
    ordered_targets.extend(group[TARGET_COLUMN].tolist())

group_results = compare_molecule_groups_regression_lens(
    tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer, 
    targets=ordered_targets, results_dir=str(_REPO / "results/esol/regression_lens"), device=DEVICE
)
plot_group_molecules_regression_lens(group_results, results_dir=_REPO / "results/esol/regression_lens")

# %%
# Activation patching on ESOL:
# pick a same-token-length pair and patch layer-by-layer.
# Default patches CLS token stream only (token_position=0), which is what the head reads.
source_smiles, target_smiles = select_same_length_pair(
    full_data,
    tokenizer,
    smiles_column="smiles",
    target_column=TARGET_COLUMN,
    min_target_gap=0.5,
)
print("Activation patch pair:")
print(f"  source: {source_smiles}")
print(f"  target: {target_smiles}")

patching_results = run_pair_activation_patching(
    tl_encoder,
    tl_regressor,
    tokenizer,
    source_smiles=source_smiles,
    target_smiles=target_smiles,
    device=DEVICE,
    token_position=0,
    patch_all_positions=False,
    denormalize=True,
)

patching_df = pd.DataFrame(patching_results["layer_results"])
print(
    f"source pred={patching_results['source_prediction']:.4f}, "
    f"target pred={patching_results['target_prediction']:.4f}"
)
print(patching_df)

esol_patch_dir = _REPO / "results/esol/activation_patching"
os.makedirs(esol_patch_dir, exist_ok=True)
patching_df.to_csv(esol_patch_dir / "layerwise_patching_results.csv", index=False)

# %%
# Attention-head activation patching on ESOL:
# patches one head at a time at attn.hook_z and saves heatmap-ready outputs.
head_patching_results = run_head_activation_patching(
    tl_encoder,
    tl_regressor,
    tokenizer,
    source_smiles=source_smiles,
    target_smiles=target_smiles,
    device=DEVICE,
    token_position=0,
    patch_all_positions=False,
    denormalize=True,
)

head_df = pd.DataFrame(head_patching_results["head_results"])
print(
    f"[Head patching] source pred={head_patching_results['source_prediction']:.4f}, "
    f"target pred={head_patching_results['target_prediction']:.4f}"
)
print(head_df.head())

head_df.to_csv(esol_patch_dir / "headwise_patching_results.csv", index=False)

# Heatmap-ready matrix: rows=layer, cols=head, values=delta_vs_target
head_heatmap_df = head_df.pivot(index="layer", columns="head", values="delta_vs_target")
head_heatmap_df.to_csv(esol_patch_dir / "headwise_delta_vs_target_matrix.csv")
plot_head_patching_heatmap(
    head_df,
    esol_patch_dir / "headwise_delta_vs_target_heatmap.pdf",
    value_column="delta_vs_target",
    title="ESOL Head-wise Activation Patching (delta vs target)",
)



# %%


# %% 
# Now for **qm9 dataset**
MODEL_PATH = str(_REPO / "trained_models/train_qm9_1/chemberta/chemberta_model_final.bin")
FULL_PATH = str(_REPO / "clustered_data/qm9/qm9.csv")
TEST_PATH = str(_REPO / "clustered_data/qm9/test_qm9.csv")
TRAIN_PATH = str(_REPO / "clustered_data/qm9/train_qm9.csv")
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = str(_REPO / "trained_models/train_qm9_1/chemberta/normalization_scaler.pkl")
TARGET_COLUMN = "dga"
print(DEVICE)
# %%
full_data = pd.read_csv(FULL_PATH)
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)

# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO"
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results['final_output'] < 0.001}")

prediction_results = test_prediction_equivalence(hf_regressor, tl_regressor, [test_smiles], tokenizer, DEVICE)
print(f"The predictions are equivalent: {prediction_results['is_equivalent']}")
# %% [markdown]
# Let's run ablation studies to see the effect of misssing components
test_data = pd.read_csv(TEST_PATH)
test_molecules = test_data['smiles'].to_list()
targets = test_data[TARGET_COLUMN].to_list()

print(f"Testing ablation on {len(test_molecules)} molecules")
print(f"Target range: {min(targets):.3f} to {max(targets):.3f}")

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, target_column=TARGET_COLUMN, output_dir=_REPO / "results/qm9", n_seeds=10, scaler=scaler)
# %% [markdown]
# We move on to regression lens
# We pick the molecules with the largest, smallest, and median target value to showcase the technique
# on the training data
import pickle
with open(_REPO / "results/qm9_1/ablation/all_results.pkl", "rb") as f:
    qm9_results = pickle.load(f)
    
plot_ablation_metrics(qm9_results, _REPO / "results/qm9_1", title = "QM9")

# %%
median_idx = len(full_data) // 2
median_molecule = full_data.sort_values(TARGET_COLUMN).iloc[median_idx]["smiles"]
min_max_median_molecules = [
    full_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # max
    median_molecule, # median
    full_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # min
]
# Get actual target values for these molecules
actual_targets = [
    full_data.nlargest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # max value
    full_data.sort_values(TARGET_COLUMN).iloc[median_idx][TARGET_COLUMN],  # median value
    full_data.nsmallest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # min value
]
min_max_median_molecules

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_median_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=_REPO / "results/qm9_1/example_regression_lens", molecule_labels = ["Molecule 4", "Molecule 5", "Molecule 6"], y_label = "Gibbs Free Energies of Atomization At 298K", title = "QM9", actual_targets=actual_targets, target_labels=["maximum", "median", "minimum"])

# %% 
# With clustering - build both groups and targets in one pass to ensure alignment
molecule_groups = {}
ordered_targets = []
for cluster, group in full_data.groupby('cluster'):
    cluster_name = f"Cluster {cluster + 1}"
    molecule_groups[cluster_name] = group['smiles'].tolist()
    ordered_targets.extend(group[TARGET_COLUMN].tolist())

group_results = compare_molecule_groups_regression_lens(
    tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer,
    targets=ordered_targets, results_dir=str(_REPO / "results/qm9_1/regression_lens"), device=DEVICE
)
plot_group_molecules_regression_lens(group_results, results_dir=_REPO / "results/qm9_1/regression_lens", mean_y_label = "Mean Gibbs Free Energies of Atomization At 298K", var_y_label = "Variance Gibbs Free Energies of Atomization At 298K", title = "QM9")

# %%
# Now for **hce dataset**
MODEL_PATH = str(_REPO / "trained_models/train_hce/chemberta/chemberta_model_final.bin")
FULL_PATH = str(_REPO / "clustered_data/hce/hce.csv")
TEST_PATH = str(_REPO / "clustered_data/hce/test_hce.csv")
TRAIN_PATH = str(_REPO / "clustered_data/hce/train_hce.csv")
TOKENIZER_NAME = "DeepChem/ChemBERTa-77M-MLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALER_PATH = str(_REPO / "trained_models/train_hce/chemberta/normalization_scaler.pkl")
TARGET_COLUMN = "pce_1"
print(DEVICE)
# %%
full_data = pd.read_csv(FULL_PATH)
train_data = pd.read_csv(TRAIN_PATH)
hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler = load_chemberta_models(
    MODEL_PATH, TOKENIZER_NAME, DEVICE, SCALER_PATH, train_data=train_data
)
print(hf_encoder, tl_encoder, tokenizer, hf_regressor, tl_regressor, scaler)

# %% [markdown]
# Validating conversation (check whether the two models have the same internals and output, extremely important!)
# First check internal and then output
# %%
test_smiles = "CCO"
inputs = tokenizer(test_smiles, return_tensors="pt").to(DEVICE)

conversion_results = validate_conversion(hf_encoder, tl_encoder, inputs["input_ids"], inputs["attention_mask"])
print(f"The difference between the final embeddings are less than 0.001: {conversion_results['final_output'] < 0.001}")

prediction_results = test_prediction_equivalence(hf_regressor, tl_regressor, [test_smiles], tokenizer, DEVICE)
print(f"The predictions are equivalent: {prediction_results['is_equivalent']}")
# %% [markdown]
# Let's run ablation studies to see the effect of misssing components
test_data = pd.read_csv(TEST_PATH)
test_molecules = test_data['smiles'].to_list()
targets = test_data[TARGET_COLUMN].to_list()

print(f"Testing ablation on {len(test_molecules)} molecules")
print(f"Target range: {min(targets):.3f} to {max(targets):.3f}")

results = run_ablation_analysis_with_metrics(tl_encoder, tl_regressor, tokenizer, test_data, target_column=TARGET_COLUMN, output_dir=_REPO / "results/hce", n_seeds=10, scaler=scaler)

plot_ablation_metrics(results, _REPO / "results/hce")
# %% 
# We move on to regression lens
# We pick the molecules with the largest, smallest, and median target value to showcase the technique
# on the training data
import pickle
with open(_REPO / "results/hce/ablation/all_results.pkl", "rb") as f:
    hce_results = pickle.load(f)
    
plot_ablation_metrics(hce_results, _REPO / "results/hce", title = "HCE")

# %%
median_idx = len(full_data) // 2
median_molecule = full_data.sort_values(TARGET_COLUMN).iloc[median_idx]["smiles"]
min_max_median_molecules = [
    full_data.nlargest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # max
    median_molecule, # median
    full_data.nsmallest(1, TARGET_COLUMN)["smiles"].to_list()[0],  # min
]
# Get actual target values for these molecules
actual_targets = [
    full_data.nlargest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # max value
    full_data.sort_values(TARGET_COLUMN).iloc[median_idx][TARGET_COLUMN],  # median value
    full_data.nsmallest(1, TARGET_COLUMN)[TARGET_COLUMN].to_list()[0],  # min value
]
min_max_median_molecules

results = run_regression_lens(tl_encoder, tl_regressor, scaler, min_max_median_molecules, tokenizer)
plot_individual_molecules_regression_lens(results, results_dir=_REPO / "results/hce/example_regression_lens", molecule_labels = ["Molecule 7", "Molecule 8", "Molecule 9"], y_label = "Power Conversion Efficiency", title = "HCE", actual_targets=actual_targets, target_labels=["maximum", "median", "minimum"])

# %% 
# With clustering - build both groups and targets in one pass to ensure alignment
molecule_groups = {}
ordered_targets = []
for cluster, group in full_data.groupby('cluster'):
    cluster_name = f"Cluster {cluster + 1}"
    molecule_groups[cluster_name] = group['smiles'].tolist()
    ordered_targets.extend(group[TARGET_COLUMN].tolist())

group_results = compare_molecule_groups_regression_lens(
    tl_encoder, tl_regressor, scaler, molecule_groups, tokenizer,
    targets=ordered_targets, results_dir=str(_REPO / "results/hce/regression_lens"), device=DEVICE
)
plot_group_molecules_regression_lens(group_results, results_dir=_REPO / "results/hce/regression_lens", mean_y_label = "Mean Power Conversion Efficiency", var_y_label = "Variance Power Conversion Efficiency", title = "HCE")


# %%

# TODO: activation patching, see thesis repo
