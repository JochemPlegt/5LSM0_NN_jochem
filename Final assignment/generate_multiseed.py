"""Compute validation Mean Dice for multiple seeds to report mean ± std."""

import glob
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype

from model import Model as UNetClass, DeepLabModel

id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

DATA_DIR = os.environ.get("DATA_DIR", "./data/cityscapes")
CHECKPOINT_DIR = "./checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 19


def convert_to_train_id(label):
    return label.apply_(lambda x: id_to_trainid[x])


def get_val_loader():
    transform = Compose([
        ToImage(), Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    target_transform = Compose([ToImage(), Resize((256, 256))])
    dataset = Cityscapes(
        DATA_DIR, split="val", mode="fine", target_type="semantic",
        transform=transform, target_transform=target_transform,
    )
    return DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)


def find_best_checkpoint(experiment_id):
    pattern = f"{CHECKPOINT_DIR}/{experiment_id}/best_model-*.pt"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint: {pattern}")
    return sorted(files)[-1]


def load_model(cls, ckpt, **kwargs):
    m = cls(n_classes=N_CLASSES, **kwargs)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    m.load_state_dict(state, strict=True)
    return m.eval().to(DEVICE)


def compute_mean_dice(model, loader):
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

    return (2 * intersect / (sum_pred + sum_gt + 1e-8)).mean()


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    loader = get_val_loader()

    configs = [
        ("U-Net (seed 42)",     UNetClass,    "unet-no-augment",                      {}),
        ("U-Net (seed 0)",      UNetClass,    "unet-no-augment-seed0",                {}),
        ("U-Net (seed 1)",      UNetClass,    "unet-no-augment-seed1",                {}),
        ("DeepLabV3 (seed 0)",  DeepLabModel, "deeplab-frozen-flip-and-jitter-seed0", {"freeze_backbone": False, "pretrained_backbone": False}),
        ("DeepLabV3 (seed 1)",  DeepLabModel, "deeplab-frozen-flip-and-jitter-seed1", {"freeze_backbone": False, "pretrained_backbone": False}),
    ]

    results = {}
    for name, cls, exp_id, kwargs in configs:
        try:
            ckpt = find_best_checkpoint(exp_id)
            print(f"\nLoading {name}: {ckpt}")
            model = load_model(cls, ckpt, **kwargs)
            dice = compute_mean_dice(model, loader)
            results[name] = dice
            print(f"  Val Mean Dice: {dice:.4f}")
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")

    print("\n=== RESULTS FOR PAPER ===")
    unet_scores = [v for k, v in results.items() if "U-Net" in k]
    dl_scores   = [v for k, v in results.items() if "DeepLabV3" in k]

    if unet_scores:
        print(f"U-Net no-augment  (n={len(unet_scores)}): {np.mean(unet_scores):.4f} +/- {np.std(unet_scores):.4f}")
    if dl_scores:
        print(f"DeepLabV3 frozen  (n={len(dl_scores)}):  {np.mean(dl_scores):.4f} +/- {np.std(dl_scores):.4f}")
