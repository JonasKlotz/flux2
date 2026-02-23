#!/bin/bash
#SBATCH --job-name=flux2_syn_images
#SBATCH --output=logs/cub/main-%j.out
#SBATCH --error=logs/cub/main-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=128G
#SBATCH --time=150:00:00


export PROJECT_ROOT="/home/jonas/PycharmProjects/flux2"
source "$PROJECT_ROOT/.venv/bin/activate"
cd $PROJECT_ROOT
# run create synthetic_images.py with arguments
python "src/create_syn_images.py"