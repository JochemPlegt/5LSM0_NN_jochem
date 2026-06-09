#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

srun apptainer exec --nv \
    --env-file .env \
    --env EXPERIMENT="${EXPERIMENT:-unet-no-augment}" \
    --env MODEL="${MODEL:-unet}" \
    --env AUGMENT_FLAGS="${AUGMENT_FLAGS:-}" \
    --env FINETUNE_FLAGS="${FINETUNE_FLAGS:-}" \
    --env SEED="${SEED:-42}" \
    container.sif /bin/bash main.sh
