#!/bin/bash
#SBATCH --job-name=training_job
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "$HOME/MoLe"
source .venv/bin/activate

# Force venv/user-site precedence; block cluster site-package leakage
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

# Optional but useful
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import sys, typing_extensions; print(sys.executable); print(typing_extensions.__file__)"
python mole/TL_chem.py