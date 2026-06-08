import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import GCPDataset, IDX_TO_CLASS
from model import GCPNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Reference image width for PCK — images are 4096px wide.
# Normalised keypoints (x_norm = x_orig / orig_w) scale correctly to this space.
ORIG_REF_W = 4096


def compute_pck(pred_kp: torch.Tensor, gt_kp: torch.Tensor, thresholds=(10, 25, 50)):
    """PCK in original image pixel space. pred/gt are normalised [0,1] where 1 = orig image width/height."""
    pred_px = pred_kp * ORIG_REF_W
    gt_px = gt_kp * ORIG_REF_W
    dist = torch.norm(pred_px - gt_px, dim=-1)
    return {f'pck@{t}': (dist <= t).float().mean().item() for t in thresholds}


def train_one_epoch(model, loader, optimizer, kp_crit, cls_crit, device, kp_w, cls_w):
    model.train()
    total = 0.0
    for imgs, kps, labels, _ in tqdm(loader, desc='Train', leave=False):
        imgs, kps, labels = imgs.to(device), kps.to(device), labels.to(device)

        pred_kp, pred_cls = model(imgs)
        loss = kp_w * kp_crit(pred_kp, kps) + cls_w * cls_crit(pred_cls, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, kp_crit, cls_crit, device, kp_w, cls_w):
    model.eval()
    total = 0.0
    all_pred_kp, all_gt_kp = [], []
    all_pred_cls, all_gt_cls = [], []

    for imgs, kps, labels, _ in tqdm(loader, desc='Val  ', leave=False):
        imgs, kps, labels = imgs.to(device), kps.to(device), labels.to(device)

        pred_kp, pred_cls = model(imgs)
        loss = kp_w * kp_crit(pred_kp, kps) + cls_w * cls_crit(pred_cls, labels)
        total += loss.item()

        all_pred_kp.append(pred_kp.cpu())
        all_gt_kp.append(kps.cpu())
        all_pred_cls.append(pred_cls.argmax(dim=1).cpu())
        all_gt_cls.append(labels.cpu())

    pred_kp_t = torch.cat(all_pred_kp)
    gt_kp_t = torch.cat(all_gt_kp)
    pred_cls_np = torch.cat(all_pred_cls).numpy()
    gt_cls_np = torch.cat(all_gt_cls).numpy()

    pck = compute_pck(pred_kp_t, gt_kp_t)
    f1 = f1_score(gt_cls_np, pred_cls_np, average='macro', zero_division=0)

    return total / len(loader), pck, float(f1)


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.annotations) as f:
        annotations = json.load(f)

    # Drop the 4 entries that have no verified_shape label
    annotations = {k: v for k, v in annotations.items() if 'verified_shape' in v}

    paths = list(annotations.keys())
    shapes = [annotations[p]['verified_shape'] for p in paths]

    train_paths, val_paths = train_test_split(
        paths, test_size=args.val_split, random_state=args.seed, stratify=shapes
    )

    train_ann = {p: annotations[p] for p in train_paths}
    val_ann = {p: annotations[p] for p in val_paths}

    print(f"Train: {len(train_ann)}  Val: {len(val_ann)}")

    train_ds = GCPDataset(args.data_dir, train_ann, args.img_size, is_train=True)
    val_ds = GCPDataset(args.data_dir, val_ann, args.img_size, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = GCPNet(backbone=args.backbone, num_classes=3, pretrained=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    kp_crit = nn.SmoothL1Loss(beta=0.01)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_score = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, kp_crit, cls_crit,
            device, args.kp_weight, args.cls_weight,
        )
        val_loss, pck, f1 = evaluate(
            model, val_loader, kp_crit, cls_crit,
            device, args.kp_weight, args.cls_weight,
        )
        scheduler.step()

        # Combined score: PCK@25 + macro-F1
        score = pck['pck@25'] + f1

        print(
            f"Epoch {epoch:03d} | Train {tr_loss:.4f} | Val {val_loss:.4f} | "
            f"PCK@10={pck['pck@10']:.3f} PCK@25={pck['pck@25']:.3f} "
            f"PCK@50={pck['pck@50']:.3f} | F1={f1:.3f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            ckpt = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'pck': pck,
                'f1': f1,
                'args': vars(args),
            }, ckpt)
            print(f"  -> Saved checkpoint (score={best_score:.4f})")

    print(f"\nBest checkpoint at epoch {best_epoch} (score={best_score:.4f})")
    print(f"Saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GCP keypoint + classification model')
    parser.add_argument('--data_dir', default='data/train_dataset',
                        help='Root directory of the training dataset')
    parser.add_argument('--annotations', default='data/train_dataset/gcp_marks.json')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--backbone', default='efficientnet_b4',
                        help='timm backbone name (e.g. efficientnet_b0, efficientnet_b4)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--kp_weight', type=float, default=1.0,
                        help='Weight for keypoint loss component')
    parser.add_argument('--cls_weight', type=float, default=0.5,
                        help='Weight for classification loss component')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
