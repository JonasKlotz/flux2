#!/bin/bash
#SBATCH --job-name=coco
#SBATCH --output=logs/coco/main-%A_%a.out
#SBATCH --error=logs/coco/main-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=128G
#SBATCH --time=150:00:00
#SBATCH --array=0-0        # adjust depending on how many chunks you want

export PROJECT_ROOT="/home/jonas/PycharmProjects/flux2"
source "$PROJECT_ROOT/.venv/bin/activate"
cd $PROJECT_ROOT

CHUNK_SIZE=1000
START_POINT=4000

START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + START_POINT))
END_IDX=$((START_IDX + CHUNK_SIZE))

python src/create_syn_images_coco.py --start_idx ${START_IDX} --end_idx ${END_IDX}
