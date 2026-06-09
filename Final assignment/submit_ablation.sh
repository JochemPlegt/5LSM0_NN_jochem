#!/bin/bash
# Submit all ablation runs to Snellius.
# Usage: bash submit_ablation.sh
#
# Experiments:
#   1. Original ablation (seed 42) — U-Net variants + DeepLabV3 frozen
#   2. Frozen vs. fine-tuned backbone comparison
#   3. Multi-seed runs (seeds 0 and 1) for key configurations

# ---------------------------------------------------------------
# 1. Original ablation (seed 42)
# ---------------------------------------------------------------
sbatch --export=ALL,EXPERIMENT="unet-no-augment",MODEL="unet",AUGMENT_FLAGS="",SEED=42                                          jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="unet-flip-only",MODEL="unet",AUGMENT_FLAGS="--augment-flip",SEED=42                             jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="unet-jitter-only",MODEL="unet",AUGMENT_FLAGS="--augment-jitter",SEED=42                         jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="unet-flip-and-jitter",MODEL="unet",AUGMENT_FLAGS="--augment-flip --augment-jitter",SEED=42      jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="deeplab-frozen-no-augment",MODEL="deeplab",AUGMENT_FLAGS="",SEED=42                             jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="deeplab-frozen-flip-and-jitter",MODEL="deeplab",AUGMENT_FLAGS="--augment-flip --augment-jitter",SEED=42 jobscript_slurm.sh

# ---------------------------------------------------------------
# 2. Frozen vs. fine-tuned backbone (seed 42)
#    Hypothesis H1: does pretraining (not just architecture) drive robustness?
#    Fine-tuned = same DeepLabV3 architecture, same augmentation, backbone NOT frozen.
# ---------------------------------------------------------------
sbatch --export=ALL,EXPERIMENT="deeplab-finetuned-no-augment",MODEL="deeplab",AUGMENT_FLAGS="",FINETUNE_FLAGS="--finetune-backbone",SEED=42      jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="deeplab-finetuned-flip-and-jitter",MODEL="deeplab",AUGMENT_FLAGS="--augment-flip --augment-jitter",FINETUNE_FLAGS="--finetune-backbone",SEED=42 jobscript_slurm.sh

# ---------------------------------------------------------------
# 3. Multi-seed runs — key configurations only (seeds 0 and 1)
#    Validates that the small clean-data gain and large robustness
#    gain are stable across seeds, not artefacts of seed 42.
# ---------------------------------------------------------------
for SEED in 0 1; do
    sbatch --export=ALL,EXPERIMENT="unet-no-augment-seed${SEED}",MODEL="unet",AUGMENT_FLAGS="",SEED=${SEED}                                          jobscript_slurm.sh
    sbatch --export=ALL,EXPERIMENT="deeplab-frozen-flip-and-jitter-seed${SEED}",MODEL="deeplab",AUGMENT_FLAGS="--augment-flip --augment-jitter",SEED=${SEED} jobscript_slurm.sh
done
