#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

BASE_ARGS="--data-dir ./data/cityscapes --batch-size 64 --epochs 100 --lr 0.001 --num-workers 10 --seed 42"

EXPERIMENT=${EXPERIMENT:-"unet-no-augment"}

srun apptainer exec --nv --env-file .env container.sif python3 train.py \
    $BASE_ARGS \
    --experiment-id "$EXPERIMENT" \
    $AUGMENT_FLAGS
