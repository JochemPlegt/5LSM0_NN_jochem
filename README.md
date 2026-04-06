# 5LSM0 Neural Networks for Computer Vision – Final Assignment

**Student:** Jochem Plegt
**CodaLab username:** JochemPlegt
**TU/e email:** j.j.g.plegt@student.tue.nl

## Project Overview

This repository contains the code for the **Cityscapes Challenge** final assignment of course 5LSM0 at Eindhoven University of Technology. The goal is semantic segmentation on the Cityscapes dataset (19 classes).

Two benchmarks are addressed:
- **Peak Performance** (server 5001): segmentation quality on clean test images
- **Robustness** (server 5002): segmentation quality under common image corruptions

The proposed approach replaces the baseline U-Net with a **DeepLabV3-ResNet50** model using a pretrained backbone, data augmentation, and cosine annealing scheduling.

## Results

| Model | Benchmark | Mean Dice | Mean IoU |
|-------|-----------|-----------|---------|
| Baseline U-Net | Peak Performance | 0.4598 | 0.3758 |
| DeepLabV3-ResNet50 | Peak Performance | 0.4681 | 0.3887 |
| Baseline U-Net | Robustness | 0.2079 | 0.1539 |
| DeepLabV3-ResNet50 | Robustness | 0.4108 | 0.3358 |

## Repository Structure

```
Final assignment/
├── model.py          # Model definition (DeepLabV3-ResNet50 and baseline U-Net)
├── train.py          # Training script with WandB logging
├── predict.py        # Inference script for Docker submission
├── Dockerfile        # Docker container for challenge submission
├── main.sh           # SLURM job script for Snellius HPC cluster
report/
├── main.tex          # LaTeX source of the research paper
├── references.bib    # Bibliography
└── figures/          # Figures used in the paper
```

## Requirements

```bash
pip install torch torchvision wandb
```

## Training

### On Snellius HPC cluster
```bash
sbatch "Final assignment/main.sh"
```

### Local training
```bash
python "Final assignment/train.py" \
  --experiment-id "deeplabv3-final" \
  --lr 0.001 \
  --epochs 100 \
  --augment
```

Training metrics are logged to [Weights & Biases](https://wandb.ai).

## Docker Submission

1. Copy the best checkpoint to `model.pt`
2. Build the Docker image:
```bash
cd "Final assignment"
docker build -t nncv-submission:latest .
```
3. Export:
```bash
docker save -o nncv_submission.tar nncv-submission:latest
```
4. Upload to the challenge server at `http://131.155.126.249:5001` or `http://131.155.126.249:5002`

## Model Details

The proposed model is **DeepLabV3** with a **ResNet-50 backbone** pretrained on ImageNet:

- Backbone is **frozen** during training to preserve pretrained BatchNorm statistics
- Classifier head trained from scratch (19 output classes)
- Data augmentation: random horizontal flip + color jitter
- Optimizer: AdamW, lr=0.001, cosine annealing scheduling
- Loss: Cross-entropy with ignore index 255
- Input: 256×256, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
