#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00

# Load .env variables into container via APPTAINERENV_ prefix (avoids --env-file parsing issues)
while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    export "APPTAINERENV_${key}=${val}"
done < .env

srun apptainer exec --nv container.sif python3 generate_multiseed.py
