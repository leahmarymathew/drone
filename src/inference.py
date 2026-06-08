"""
Run the trained GCPNet on the test dataset and write predictions.json.

Usage:
    python src/inference.py --test_dir data/test_dataset --checkpoint checkpoints/best_model.pth

With TTA (test-time augmentation, recommended):
    python src/inference.py --test_dir data/test_dataset --checkpoint checkpoints/best_model.pth --tta
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GCPTestDataset, IDX_TO_CLASS
from model import GCPNet


@torch.no_grad()
def predict_batch(model, imgs, device, orig_ws, orig_hs, img_size, use_tta=False):
    """
    Returns lists of (x_orig, y_orig, shape_str) for each image in the batch.

    The model predicts normalised coords in [0,1] relative to the resized
    (img_size x img_size) space. Because we used A.Resize (stretching), the
    inverse is simply: x_orig = x_norm * orig_w, y_orig = y_norm * orig_h.
    """
    imgs = imgs.to(device)

    if use_tta:
        # Original + horizontal flip + vertical flip — average keypoints, sum logits
        pred_kp, pred_cls = model(imgs)

        imgs_hflip = torch.flip(imgs, dims=[3])
        kp_h, cls_h = model(imgs_hflip)
        kp_h[:, 0] = 1.0 - kp_h[:, 0]  # unflip x

        imgs_vflip = torch.flip(imgs, dims=[2])
        kp_v, cls_v = model(imgs_vflip)
        kp_v[:, 1] = 1.0 - kp_v[:, 1]  # unflip y

        pred_kp = (pred_kp + kp_h + kp_v) / 3.0
        pred_cls = pred_cls + cls_h + cls_v
    else:
        pred_kp, pred_cls = model(imgs)

    cls_idx = pred_cls.argmax(dim=1).cpu().numpy()
    kp = pred_kp.cpu().numpy()  # (B, 2)

    results = []
    for i in range(len(imgs)):
        x_norm, y_norm = float(kp[i, 0]), float(kp[i, 1])
        orig_w = int(orig_ws[i])
        orig_h = int(orig_hs[i])

        x_orig = round(x_norm * orig_w, 2)
        y_orig = round(y_norm * orig_h, 2)

        # Clamp to valid pixel range
        x_orig = max(0.0, min(x_orig, orig_w - 1))
        y_orig = max(0.0, min(y_orig, orig_h - 1))

        results.append((x_orig, y_orig, IDX_TO_CLASS[int(cls_idx[i])]))

    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get('args', {})
    backbone = saved_args.get('backbone', args.backbone)
    img_size = saved_args.get('img_size', args.img_size)

    model = GCPNet(backbone=backbone, num_classes=3, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}) "
          f"— PCK@25={checkpoint.get('pck', {}).get('pck@25', '?'):.3f}, "
          f"F1={checkpoint.get('f1', '?'):.3f}")

    test_ds = GCPTestDataset(args.test_dir, img_size=img_size)
    print(f"Found {len(test_ds)} test images")

    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    predictions = {}
    for imgs, rel_paths, orig_ws, orig_hs in tqdm(loader, desc='Inference'):
        results = predict_batch(model, imgs, device, orig_ws, orig_hs, img_size, use_tta=args.tta)
        for rel_path, (x, y, shape) in zip(rel_paths, results):
            predictions[rel_path] = {
                'mark': {'x': x, 'y': y},
                'verified_shape': shape,
            }

    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nWrote {len(predictions)} predictions to: {args.output}")

    shape_counts = {}
    for v in predictions.values():
        shape_counts[v['verified_shape']] = shape_counts.get(v['verified_shape'], 0) + 1
    print("Shape distribution:", shape_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference and produce predictions.json')
    parser.add_argument('--test_dir', default='data/test_dataset')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--output', default='predictions.json')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Fallback image size if not stored in checkpoint')
    parser.add_argument('--backbone', default='efficientnet_b4',
                        help='Fallback backbone if not stored in checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (flips; improves accuracy ~1-2%%)')
    args = parser.parse_args()
    main(args)
