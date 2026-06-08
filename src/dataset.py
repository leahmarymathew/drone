import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ['Cross', 'L-Shape', 'Square']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, is_train: bool) -> A.Compose:
    kp_params = A.KeypointParams(format='xy', remove_invisible=False)

    aug = [A.Resize(img_size, img_size)]

    if is_train:
        aug += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03, scale_limit=0.1, rotate_limit=15,
                border_mode=0, value=0, p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.25),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
        ]

    aug += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(aug, keypoint_params=kp_params)


class GCPDataset(Dataset):
    def __init__(self, root_dir: str, annotations: dict, img_size: int = 640, is_train: bool = True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transforms = build_transforms(img_size, is_train)

        self.samples = []
        missing = 0
        skipped = 0
        for rel_path, ann in annotations.items():
            if 'verified_shape' not in ann:
                skipped += 1
                continue
            full_path = os.path.join(root_dir, rel_path.replace('/', os.sep))
            if not os.path.exists(full_path):
                missing += 1
                continue
            self.samples.append({
                'path': full_path,
                'rel_path': rel_path,
                'x': float(ann['mark']['x']),
                'y': float(ann['mark']['y']),
                'shape': ann['verified_shape'],
            })

        if missing:
            print(f"Warning: {missing} annotation paths not found on disk (partial download).")
        if skipped:
            print(f"Warning: {skipped} entries skipped — no verified_shape label.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = cv2.imread(s['path'])
        if img is None:
            raise RuntimeError(f"Failed to load image: {s['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=img, keypoints=[(s['x'], s['y'])])
        img_tensor = transformed['image']

        kps = transformed['keypoints']
        if kps:
            kx, ky = kps[0]
        else:
            # Keypoint shifted out of frame by augmentation — fall back to center
            kx, ky = self.img_size / 2, self.img_size / 2

        kx_norm = float(np.clip(kx / self.img_size, 0.0, 1.0))
        ky_norm = float(np.clip(ky / self.img_size, 0.0, 1.0))

        keypoint = torch.tensor([kx_norm, ky_norm], dtype=torch.float32)
        label = torch.tensor(CLASS_TO_IDX[s['shape']], dtype=torch.long)

        return img_tensor, keypoint, label, s['rel_path']


class GCPTestDataset(Dataset):
    """Dataset for unlabelled test images."""

    def __init__(self, root_dir: str, img_size: int = 640):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        self.samples = []
        for root, _, files in os.walk(root_dir):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, root_dir).replace('\\', '/')
                    self.samples.append((full_path, rel_path))

        self.samples.sort(key=lambda x: x[1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_path, rel_path = self.samples[idx]

        img = cv2.imread(full_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {full_path}")
        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=img)
        img_tensor = transformed['image']

        return img_tensor, rel_path, orig_w, orig_h
