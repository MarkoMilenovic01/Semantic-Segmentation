"""
train.py — Phase 4 (NOS – Semantic Segmentation)

Purpose:
    Train the U-Net model on the synthetic text-segmentation dataset
    using BCEWithLogitsLoss and the Adam optimizer.

Assignment requirements implemented:
    • Optimizer: Adam (lr=1e-4)
    • Loss: BCEWithLogitsLoss (logits → sigmoid inside loss)
    • Scheduler: ReduceLROnPlateau
    • Batch size: 32
    • Image resolution: 256×256
    • Mixed precision (AMP) for faster training
    • 150 epochs
Outputs:
    models/best.pt       — best validation model
    models/last.pt       — last epoch
    models/config.json   — training configuration
    models/history.json  — per-epoch metrics
    models/loss_curves.png
    models/metrics_curves.png
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import UNet


# =============================================================================
# Paths & Hyperparameters
# =============================================================================

DATA_ROOT = Path(r"C:\Users\marko\Desktop\Master\NOS\Semantic Segmentation\data\generated")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR   = DATA_ROOT / "val"

OUT_DIR   = Path(r"C:\Users\marko\Desktop\Master\NOS\Semantic Segmentation\models")

BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
EPOCHS         = 150
IMG_SIZE       = 256
SEED          = 42

USE_AMP = True  # Mixed precision on GPU

# Learning-rate scheduler settings
SCHED_FACTOR   = 0.1
SCHED_PATIENCE = 20
SCHED_MIN_LR   = 1e-6
SCHED_MODE     = "min"


# =============================================================================
# Dataset
# =============================================================================

def list_image_mask_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Match images with masks by filename."""
    image_dir = root / "images"
    mask_dir  = root / "masks"

    pairs = []
    for img_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


class TextSegmentationDataset(Dataset):
    """Loads a dataset of (image, mask) pairs normalized to [0,1]."""

    def __init__(self, root: Path):
        self.pairs = list_image_mask_pairs(root)
        if not self.pairs:
            raise RuntimeError(f"No samples found in: {root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # --- Load image ---
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1))  # (3, H, W)

        # --- Load mask (1 channel) ---
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask[None, ...])  # (1, H, W)

        return img_tensor, mask_tensor


def make_dataloader(root: Path, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader with efficient settings."""
    dataset = TextSegmentationDataset(root)
    print(f"[INFO] Loaded {len(dataset)} samples from {root.name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def sigmoid_binarize(logits: torch.Tensor, threshold: float = 0.5):
    """Apply sigmoid + threshold → binary mask."""
    return (torch.sigmoid(logits) > threshold).float()


@torch.no_grad()
def mean_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """Compute batch-averaged Intersection-over-Union."""
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


@torch.no_grad()
def mean_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """Compute batch-averaged Dice score."""
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (denom + eps)).mean().item()


# =============================================================================
# Training Utilities
# =============================================================================

def set_global_seed(seed: int):
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler):
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0
    total = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss   = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total


@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device):
    """Validate for one epoch. Returns loss, IoU, Dice."""
    model.eval()
    running_loss = 0
    total = 0
    ious, dices = [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        preds = sigmoid_binarize(logits)
        ious.append(mean_iou(preds, masks))
        dices.append(mean_dice(preds, masks))

    val_loss = running_loss / total
    return val_loss, float(np.mean(ious)), float(np.mean(dices))


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    # --- Init ---
    set_global_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Data ---
    train_loader = make_dataloader(TRAIN_DIR, BATCH_SIZE, shuffle=True)
    val_loader   = make_dataloader(VAL_DIR,   BATCH_SIZE, shuffle=False)

    # --- Model setup ---
    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHED_MODE,
        factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE,
        min_lr=SCHED_MIN_LR
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Save config
    config = dict(
        optimizer="Adam",
        lr=LEARNING_RATE,
        loss="BCEWithLogitsLoss",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
    )
    json.dump(config, open(OUT_DIR/"config.json", "w"), indent=2)

    # --- Training ---
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler)
        val_loss, val_iou, val_dice = validate_one_epoch(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        scheduler.step(val_loss)

        print(f"[{epoch:3d}/{EPOCHS}] "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"IoU {val_iou:.3f} | Dice {val_dice:.3f}")

        # Save last.pt
        torch.save(model.state_dict(), OUT_DIR/"last.pt")

        # Save best.pt
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUT_DIR/"best.pt")

    # Save history
    json.dump(history, open(OUT_DIR/"history.json", "w"), indent=2)

    # --- Plots ---
    plt.figure()
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"],   label="Val")
    plt.title("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR/"loss_curves.png")
    plt.close()

    plt.figure()
    plt.plot(history["val_iou"],  label="IoU")
    plt.plot(history["val_dice"], label="Dice")
    plt.title("Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR/"metrics_curves.png")
    plt.close()

    print("\n[DONE] Best val loss:", best_val)


# Entry point
if __name__ == "__main__":
    main()
