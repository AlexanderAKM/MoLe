#!/bin/bash
#SBATCH --job-name=training_job
#SBATCH --time=12:00:00
#SBATCH --mem=12GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

cd $HOME/MoLe
mole-train --dataset esol
mole-train --dataset hce
mole-train --dataset qm9
