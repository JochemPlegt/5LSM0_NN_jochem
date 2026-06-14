"""Generate paper figures: qualitative predictions and per-category Dice barplot."""

import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype

from model import Model as UNetClass, DeepLabModel

# Reuse the same mappings as train.py
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

CITYSCAPES_CLASSES = [
    cls.name for cls in sorted(
        [c for c in Cityscapes.classes if c.train_id not in (255, -1)],
        key=lambda c: c.train_id,
    )
]

DATA_DIR = os.environ.get("DATA_DIR", "./data/cityscapes")
CHECKPOINT_DIR = "./checkpoints"
OUT_DIR = "./figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 19


def convert_to_train_id(label: torch.Tensor) -> torch.Tensor:
    return label.apply_(lambda x: id_to_trainid[x])


def find_best_checkpoint(experiment_id):
    pattern = f"{CHECKPOINT_DIR}/{experiment_id}/best_model-*.pt"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint found: {pattern}")
    return sorted(files)[-1]


def load_model(cls, ckpt_path, **kwargs):
    m = cls(n_classes=N_CLASSES, **kwargs)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(state, strict=True)
    return m.eval().to(DEVICE)


def get_val_dataset():
    transform = Compose([
        ToImage(), Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    target_transform = Compose([ToImage(), Resize((256, 256))])
    return Cityscapes(
        DATA_DIR, split="val", mode="fine", target_type="semantic",
        transform=transform, target_transform=target_transform,
    )


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy() * std + mean
    return (img * 255).clip(0, 255).astype(np.uint8)


def pred_to_color(pred_np):
    rgb = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    for tid, color in train_id_to_color.items():
        rgb[pred_np == tid] = color
    return rgb


def compute_per_class_dice(model, dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    intersect = np.zeros(N_CLASSES)
    sum_pred = np.zeros(N_CLASSES)
    sum_gt = np.zeros(N_CLASSES)

    with torch.no_grad():
        for imgs, labels in loader:
            labels = convert_to_train_id(labels)
            gt = labels.long().squeeze(1).numpy()
            preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu().numpy()
            for c in range(N_CLASSES):
                p, g = preds == c, gt == c
                intersect[c] += (p & g).sum()
                sum_pred[c] += p.sum()
                sum_gt[c] += g.sum()

    return 2 * intersect / (sum_pred + sum_gt + 1e-8)


def find_good_and_bad_indices(model, dataset, n_check=80):
    """Find a high-Dice example and a failure case with many missed persons."""
    indices = np.random.choice(len(dataset), min(n_check, len(dataset)), replace=False)
    best_idx, best_score = None, -1
    fail_idx, fail_score = None, 2.0

    with torch.no_grad():
        for idx in indices:
            img, label = dataset[int(idx)]
            label = convert_to_train_id(label.clone())
            gt = label.long().squeeze(0).numpy()
            pred = model(img.unsqueeze(0).to(DEVICE)).argmax(dim=1)[0].cpu().numpy()

            valid = gt != 255
            if valid.sum() == 0:
                continue

            overall = (pred[valid] == gt[valid]).mean()
            if overall > best_score:
                best_score, best_idx = float(overall), int(idx)

            # Look for cases where person class (11) is present but missed
            gt_p, pred_p = gt == 11, pred == 11
            if gt_p.sum() > 300:
                p_dice = 2 * (gt_p & pred_p).sum() / (gt_p.sum() + pred_p.sum() + 1e-6)
                if p_dice < fail_score:
                    fail_score, fail_idx = float(p_dice), int(idx)

    print(f"Good example: idx={best_idx}, overall_acc={best_score:.3f}")
    print(f"Fail example: idx={fail_idx}, person_dice={fail_score:.3f}")
    return best_idx, fail_idx


def save_prediction_figure(model, dataset, idx, out_path):
    img_t, label = dataset[int(idx)]
    gt = convert_to_train_id(label.clone()).long().squeeze(0).numpy()

    with torch.no_grad():
        pred = model(img_t.unsqueeze(0).to(DEVICE)).argmax(dim=1)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), dpi=120)
    axes[0].imshow(denormalize(img_t))
    axes[0].set_title('Input', fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(pred_to_color(pred))
    axes[1].set_title('Prediction', fontsize=10)
    axes[1].axis('off')
    plt.tight_layout(pad=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def save_per_category_figure(dice_unet, dice_deeplab, out_path):
    order = np.argsort(dice_deeplab)[::-1]
    x = np.arange(N_CLASSES)
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=120)
    ax.bar(x - w/2, dice_unet[order], w, label='U-Net', color='#4878D0', alpha=0.85)
    ax.bar(x + w/2, dice_deeplab[order], w, label='DeepLabV3 (frozen)', color='#EE854A', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([CITYSCAPES_CLASSES[i] for i in order], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Dice Score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Device: {DEVICE}")
    print("Loading dataset...")
    dataset = get_val_dataset()

    print("Loading U-Net checkpoint (unet-no-augment)...")
    unet_ckpt = find_best_checkpoint("unet-no-augment")
    unet = load_model(UNetClass, unet_ckpt)
    print(f"  -> {unet_ckpt}")

    print("Loading DeepLabV3 checkpoint (deeplab-frozen-flip-and-jitter)...")
    dl_ckpt = find_best_checkpoint("deeplab-frozen-flip-and-jitter")
    deeplab = load_model(DeepLabModel, dl_ckpt, freeze_backbone=False, pretrained_backbone=False)
    print(f"  -> {dl_ckpt}")

    print("\nFinding prediction examples...")
    good_idx, fail_idx = find_good_and_bad_indices(deeplab, dataset)

    print("\nSaving prediction figures...")
    save_prediction_figure(deeplab, dataset, good_idx, f"{OUT_DIR}/deeplabv3_pred1.png")
    save_prediction_figure(deeplab, dataset, fail_idx, f"{OUT_DIR}/deeplabv3_pred2.png")

    print("\nComputing per-class Dice (U-Net)...")
    dice_unet = compute_per_class_dice(unet, dataset)

    print("\nComputing per-class Dice (DeepLabV3)...")
    dice_deeplab = compute_per_class_dice(deeplab, dataset)

    print("\nPer-class results:")
    for i, cls in enumerate(CITYSCAPES_CLASSES):
        print(f"  {cls:20s}: U-Net={dice_unet[i]:.3f}  DeepLabV3={dice_deeplab[i]:.3f}")

    print("\nSaving per-category figure...")
    save_per_category_figure(dice_unet, dice_deeplab, f"{OUT_DIR}/robustness_per_category.png")

    print("\nDone! Figures in ./figures/")
