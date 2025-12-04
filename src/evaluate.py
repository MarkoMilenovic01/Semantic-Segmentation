"""
evaluate.py — Phase 5 (NOS – Semantic Segmentation)

Goal:
    Evaluate the trained U-Net (best.pt) on the FULL test dataset.

Uses helper functions from train.py:
    - make_dataloader
    - sigmoid_binarize
    - mean_iou
    - mean_dice
    - TextSegmentationDataset

Outputs:
    models/test_results.json    → average + std of IoU and Dice
    models/test_examples.png    → visual comparison (input / mask / prediction)
"""

from __future__ import annotations
from pathlib import Path
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

# Local imports
from model import UNet
from train import (
    make_dataloader,
    sigmoid_binarize,
    mean_iou,
    mean_dice,
    TextSegmentationDataset,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = Path(r"C:\Users\marko\Desktop\Master\NOS\Semantic Segmentation\data\generated")
TEST_DIR  = DATA_ROOT / "test"
MODEL_DIR = Path(r"C:\Users\marko\Desktop\Master\NOS\Semantic Segmentation\models")
MODEL_PATH = MODEL_DIR / "best.pt"

BATCH_SIZE = 32
THRESHOLD  = 0.5  # sigmoid threshold for binarization

# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Compute mean and std for IoU and Dice on the test set."""
    model.eval()
    ious, dices = [], []

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = sigmoid_binarize(model(imgs), THRESHOLD)
        ious.append(mean_iou(preds, masks))
        dices.append(mean_dice(preds, masks))

    return (
        float(np.mean(ious)), float(np.std(ious)),
        float(np.mean(dices)), float(np.std(dices))
    )

@torch.no_grad()
def visualize_examples(model, dataset, device, save_path, n=10):
    """
    Visualize a few test predictions (input / GT / prediction).
    For each example, also compute and display per-image IoU and Dice.
    """
    model.eval()
    n = min(n, len(dataset))
    indices = random.sample(range(len(dataset)), n)

    plt.figure(figsize=(9, 3 * n))
    for i, idx in enumerate(indices):
        img, mask = dataset[idx]               # img: (3,H,W), mask: (1,H,W), both in [0,1]
        img_b = img.unsqueeze(0).to(device)    # (1,3,H,W)
        msk_b = mask.unsqueeze(0).to(device)   # (1,1,H,W)

        # Predict and binarize
        logits = model(img_b)
        pred_b = sigmoid_binarize(logits, THRESHOLD)

        # Per-image metrics (batch size = 1)
        iou_val  = mean_iou(pred_b, msk_b)
        dice_val = mean_dice(pred_b, msk_b)

        # To numpy for display
        img_np  = img.permute(1, 2, 0).cpu().numpy()
        msk_np  = mask.squeeze(0).cpu().numpy()
        pred_np = pred_b.squeeze(0).squeeze(0).cpu().numpy()

        # Plot triplet
        row = i * 3
        plt.subplot(n, 3, row + 1)
        plt.imshow(img_np)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(n, 3, row + 2)
        plt.imshow(msk_np, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(n, 3, row + 3)
        plt.imshow(pred_np, cmap="gray")
        plt.title(f"Prediction\nIoU={iou_val:.3f}  Dice={dice_val:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load FULL test data and model
    test_loader  = make_dataloader(TEST_DIR, BATCH_SIZE, shuffle=False)
    test_dataset = TextSegmentationDataset(TEST_DIR)

    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[INFO] Loaded model: {MODEL_PATH.name}")

    # --- Evaluate ---
    iou_mean, iou_std, dice_mean, dice_std = evaluate_model(model, test_loader, device)

    print("\n================ Evaluation Results ================")
    print(f"IoU   : {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"Dice  : {dice_mean:.4f} ± {dice_std:.4f}")
    print("====================================================\n")

    # --- Save metrics ---
    results = dict(
        iou_mean=iou_mean, iou_std=iou_std,
        dice_mean=dice_mean, dice_std=dice_std,
        n_samples=len(test_dataset),
        threshold=THRESHOLD,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- Visualize predictions with per-image metrics ---
    visualize_examples(model, test_dataset, device, MODEL_DIR / "test_examples.png", n=40)

    print(f"[DONE] Results → {MODEL_DIR / 'test_results.json'}")
    print(f"[DONE] Visualization → {MODEL_DIR / 'test_examples.png'}")

if __name__ == "__main__":
    main()