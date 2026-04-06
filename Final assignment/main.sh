#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00

wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "deeplabv3-no-augment"