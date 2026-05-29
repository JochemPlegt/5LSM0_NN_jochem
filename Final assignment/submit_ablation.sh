#!/bin/bash
# Submit all augmentation ablation runs to Snellius.
# Usage: bash submit_ablation.sh
# After completion, submit each checkpoint to the robustness challenge server.

sbatch --export=ALL,EXPERIMENT="unet-no-augment",AUGMENT_FLAGS=""                         jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="unet-flip-only",AUGMENT_FLAGS="--augment-flip"            jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="unet-jitter-only",AUGMENT_FLAGS="--augment-jitter"        jobscript_slurm.sh
sbatch --export=ALL,EXPERIMENT="unet-flip-and-jitter",AUGMENT_FLAGS="--augment-flip --augment-jitter" jobscript_slurm.sh
