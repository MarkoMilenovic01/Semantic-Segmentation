from __future__ import annotations
from pathlib import Path
import random
import string

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Path to DIV2K dataset (used for backgrounds and textures)
DIV2K_ROOT = Path(r"C:\Users\marko\Desktop\Master\NOS\Semantic Segmentation\data\archive")

# Output directory for generated dataset
OUT_ROOT   = Path(r"C:\Users\marko\Desktop\Master\NOS\Semantic Segmentation\data\generated")

# We generate at 512×512 and later crop to 256×256
TEMP_SIZE  = 512
FINAL_SIZE = 256

# Number of samples per dataset split
N_TRAIN = 4000
N_VAL   = 500
N_TEST  = 1000

# Minimum percentage of text pixels required in final 256×256 crop
MIN_CROP_COVERAGE = 0.15
MAX_RETRIES        = 8       # attempts to achieve enough text coverage

# OpenCV built-in fonts used for text generation
FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
]


# ============================================================
# UTILITIES
# ============================================================

def load_div2k_images(root: Path) -> list[Path]:
    """
    Load all .png images from DIV2K train and validation folders.

    The DIV2K archive structure is:
        archive/DIV2K_train_HR/DIV2K_train_HR/*.png
        archive/DIV2K_valid_HR/DIV2K_valid_HR/*.png
    """
    paths = []
    for sub in ("DIV2K_train_HR", "DIV2K_valid_HR"):
        nested = root / sub / sub
        paths.extend(nested.glob("*.png"))

    if not paths:
        raise RuntimeError(f"No DIV2K images found at: {root}")

    return paths


def random_patch(img: np.ndarray, size: int = TEMP_SIZE) -> np.ndarray:
    """
    Extract a random square patch from an image.
    If the DIV2K image is too small, upscale it first.
    """
    h, w = img.shape[:2]

    # Upscale small images so we can crop a 512×512 region
    if h < size or w < size:
        img = cv2.resize(img, (max(size, w), max(size, h)), cv2.INTER_LINEAR)
        h, w = img.shape[:2]

    # Select a random top-left coordinate
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)

    return img[y:y + size, x:x + size]


def center_crop(img: np.ndarray, size: int = FINAL_SIZE) -> np.ndarray:
    """
    Crop the center of an image to the given size (256×256).
    Used after augmentation on the 512×512 canvas.
    """
    h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def augment(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply random flips and small rotation.
    Reflect padding is used to avoid black borders.
    """
    # 50% chance of horizontal flip
    if random.random() < 0.5:
        img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)

    # 50% chance of vertical flip
    if random.random() < 0.5:
        img, mask = cv2.flip(img, 0), cv2.flip(mask, 0)

    # 60% chance of rotation in range [-15°, +15°]
    if random.random() < 0.6:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((TEMP_SIZE / 2, TEMP_SIZE / 2), angle, 1.0)

        img  = cv2.warpAffine(img,  M, (TEMP_SIZE, TEMP_SIZE), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (TEMP_SIZE, TEMP_SIZE), borderMode=cv2.BORDER_REFLECT)

    return img, mask


def random_text(min_len=3, max_len=8) -> str:
    """
    Generate a random alphanumeric string of random length.
    Used to create diverse synthetic text.
    """
    chars = string.ascii_letters + string.digits
    L = random.randint(min_len, max_len)
    return ''.join(random.choice(chars) for _ in range(L))


def draw_dense_text_mask(size: int = TEMP_SIZE) -> np.ndarray:
    """
    Create a dense 512×512 mask by drawing many rows of random text.

    The entire canvas is filled with:
        • random fonts
        • random font scales
        • random thickness
        • random spacing between characters and rows
    """
    mask = np.zeros((size, size, 1), np.uint8)

    # Starting row position
    y = random.randint(30, 60)

    while y < size - 20:
        font  = random.choice(FONTS)
        scale = random.uniform(1.2, 2.4)
        thick = random.randint(2, 6)

        # Slight jitter in horizontal position
        x = random.randint(-40, 20)

        # Fill row with repeated random string segments
        while x < size - 10:
            text = random_text()
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

            cv2.putText(mask, text, (x, y), font, scale, (255,), thick, cv2.LINE_AA)
            x += tw + random.randint(12, 40)

        # Move downwards to the next row
        y += max(16, int(th * random.uniform(1.2, 1.8)))

    return mask.astype(np.float32) / 255.0


def blend(bg: np.ndarray, texture: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend text texture into background using the soft mask.

        I = (1 - M) * B + M * F

    This preserves anti-aliasing and creates realistic text boundaries.
    """
    return ((1 - mask) * bg.astype(np.float32) + mask * texture.astype(np.float32)) / 255.0


def save_pair(img: np.ndarray, mask: np.ndarray, out_dir: Path, idx: int) -> None:
    """
    Save an image–mask pair to disk.

    Paths:
        images/00001.png
        masks/00001.png
    """
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"images/{idx:05d}.png"), (img * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / f"masks/{idx:05d}.png"),  (mask * 255).astype(np.uint8))


def crop_coverage(mask: np.ndarray, size: int = FINAL_SIZE) -> float:
    """
    Compute percentage of text pixels (mask >= 0.5)
    in the final 256×256 center crop.
    """
    crop = center_crop(mask, size)

    # If mask has shape (H,W,1) → reduce to (H,W)
    if crop.ndim == 3:
        crop = crop[..., 0]

    return float((crop >= 0.5).mean())


# ============================================================
# SAMPLE GENERATION
# ============================================================

def make_sample(bg_img: np.ndarray, tx_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a single (image, mask) pair.

    Steps:
      1. Extract random 512×512 patches from background + texture images.
      2. Generate dense text mask.
      3. Blend text into background.
      4. Apply augmentations.
      5. Ensure final 256×256 crop contains enough text.
    """
    for _ in range(MAX_RETRIES):
        bg = random_patch(bg_img)
        tx = random_patch(tx_img)

        mask = draw_dense_text_mask()
        img  = blend(bg, tx, mask)

        img, mask = augment(img, mask)

        if crop_coverage(mask) >= MIN_CROP_COVERAGE:
            return center_crop(img), center_crop(mask)

    # If all retries fail, return the last crop anyway
    return center_crop(img), center_crop(mask)


def generate_split(name: str, n_samples: int, pool: list[Path]) -> None:
    """
    Generate a dataset split (train / val / test).

    For each sample:
        • pick two DIV2K images (background + texture)
        • generate a sample
        • save to disk
    """
    print(f"[INFO] Generating {name} ({n_samples} samples)…")
    out_dir = OUT_ROOT / name

    for i in tqdm(range(1, n_samples + 1)):
        bg_path, tx_path = random.sample(pool, 2)

        bg_img = cv2.imread(str(bg_path))
        tx_img = cv2.imread(str(tx_path))

        # Skip unreadable files (very rare)
        if bg_img is None or tx_img is None:
            continue

        img, mask = make_sample(bg_img, tx_img)
        save_pair(img, mask, out_dir, i)

    print(f"[DONE] {name} saved → {out_dir}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Main function: load DIV2K and generate all dataset splits."""
    print("[INFO] Scanning DIV2K…")
    all_images = load_div2k_images(DIV2K_ROOT)
    print(f"[INFO] Found {len(all_images)} DIV2K images")

    generate_split("train", N_TRAIN, all_images)
    generate_split("val",   N_VAL,   all_images)
    generate_split("test",  N_TEST,  all_images)

    print("\nAll splits generated successfully.")
    print(f"Dataset saved to: {OUT_ROOT}")


if __name__ == "__main__":
    random.seed(42)
    np.random.default_rng(42)
    main()
