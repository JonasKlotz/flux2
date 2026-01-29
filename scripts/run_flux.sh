#!/bin/bash
#SBATCH --job-name=flux2_syn_images
#SBATCH --output=logs/main-%j.out
#SBATCH --error=logs/main-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=80
#SBATCH --mem=128G
#SBATCH --time=150:00:00


export PROJECT_ROOT="/home/jonas/PycharmProjects/flux2"
source "$PROJECT_ROOT/.venv/bin/activate"

# run create synthetic_images.py with arguments
python "src/create_syn_images.py"