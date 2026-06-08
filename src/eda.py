"""
Exploratory Data Analysis for the GCP dataset.

Run from the drone/ root:
    python src/eda.py --data_dir data/train_dataset --annotations data/train_dataset/curated_gcp_marks.json

Outputs:
  - Console stats (class counts, coordinate stats, missing files)
  - eda_output/class_distribution.png
  - eda_output/keypoint_scatter.png
  - eda_output/sample_grid.png  (first 9 annotated images)
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CLASSES


def draw_keypoint(img, x, y, radius=20, color=(0, 255, 0)):
    img = img.copy()
    cv2.circle(img, (int(x), int(y)), radius, color, 4)
    cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_CROSS, 40, 4)
    return img


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.annotations) as f:
        ann = json.load(f)

    print(f"Total annotations: {len(ann)}")

    # Filter out entries without verified_shape (4 in production dataset)
    ann = {k: v for k, v in ann.items() if 'verified_shape' in v}
    print(f"Entries with verified_shape: {len(ann)}")

    shapes = [v['verified_shape'] for v in ann.values()]
    xs = [v['mark']['x'] for v in ann.values()]
    ys = [v['mark']['y'] for v in ann.values()]

    # ── Class distribution ─────────────────────────────────────────────────────
    counts = Counter(shapes)
    print("\nClass distribution:")
    for cls in CLASSES:
        print(f"  {cls:12s}: {counts.get(cls, 0):5d} ({counts.get(cls, 0)/len(shapes)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([c for c in CLASSES], [counts.get(c, 0) for c in CLASSES], color=['#4C72B0', '#DD8452', '#55A868'])
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    for i, c in enumerate(CLASSES):
        ax.text(i, counts.get(c, 0) + 0.5, str(counts.get(c, 0)), ha='center')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'class_distribution.png'), dpi=150)
    plt.close(fig)

    # ── Coordinate distribution ────────────────────────────────────────────────
    print(f"\nKeypoint X: min={min(xs):.1f}  max={max(xs):.1f}  mean={np.mean(xs):.1f}  std={np.std(xs):.1f}")
    print(f"Keypoint Y: min={min(ys):.1f}  max={max(ys):.1f}  mean={np.mean(ys):.1f}  std={np.std(ys):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(xs, ys, alpha=0.4, s=10, c='steelblue')
    axes[0].set_title('Keypoint Scatter (pixel coords)')
    axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
    axes[0].invert_yaxis()

    axes[1].hist2d(xs, ys, bins=30, cmap='Blues')
    axes[1].set_title('Keypoint Density Heatmap')
    axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
    axes[1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'keypoint_scatter.png'), dpi=150)
    plt.close(fig)

    # ── Missing file check ─────────────────────────────────────────────────────
    missing = []
    for rel_path in ann.keys():
        full = os.path.join(args.data_dir, rel_path.replace('/', os.sep))
        if not os.path.exists(full):
            missing.append(rel_path)

    print(f"\nMissing image files: {len(missing)}")
    if missing:
        for m in missing[:10]:
            print(f"  {m}")

    # ── Image dimension check ──────────────────────────────────────────────────
    print("\nChecking image dimensions (sampling up to 20 images)...")
    items = list(ann.items())
    np.random.seed(0)
    np.random.shuffle(items)
    dims = set()
    for rel_path, _ in items[:20]:
        full = os.path.join(args.data_dir, rel_path.replace('/', os.sep))
        if os.path.exists(full):
            img = cv2.imread(full)
            if img is not None:
                dims.add((img.shape[1], img.shape[0]))  # (W, H)

    print(f"  Unique (W, H) values: {dims}")

    # ── Sample grid ────────────────────────────────────────────────────────────
    n_samples = min(9, len(items))
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    loaded = 0
    for rel_path, meta in items:
        if loaded >= n_samples:
            break
        full = os.path.join(args.data_dir, rel_path.replace('/', os.sep))
        if not os.path.exists(full):
            continue
        img = cv2.imread(full)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y = meta['mark']['x'], meta['mark']['y']
        img = draw_keypoint(img, x, y)

        # Crop around keypoint for better visibility
        h, w = img.shape[:2]
        half = 200
        x1, y1 = max(0, int(x) - half), max(0, int(y) - half)
        x2, y2 = min(w, int(x) + half), min(h, int(y) + half)
        crop = img[y1:y2, x1:x2]

        ax = axes[loaded]
        ax.imshow(crop)
        ax.set_title(f"{meta['verified_shape']}\n{os.path.basename(rel_path)}", fontsize=8)
        ax.axis('off')
        loaded += 1

    for i in range(loaded, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Sample Annotated Images (cropped around GCP)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'sample_grid.png'), dpi=150)
    plt.close(fig)

    print(f"\nEDA outputs saved to: {args.out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/train_dataset')
    parser.add_argument('--annotations', default='data/train_dataset/curated_gcp_marks.json')
    parser.add_argument('--out_dir', default='eda_output')
    args = parser.parse_args()
    main(args)
