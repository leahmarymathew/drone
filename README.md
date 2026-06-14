# Drone Imagery Analysis: Aerial GCP Detection & Classification

A multi-task deep learning pipeline for automated Ground Control Point (GCP) detection and shape classification in aerial drone imagery.

## Overview

This project predicts:

* Ground Control Point (GCP) center coordinates `(x, y)`
* Marker shape classification:

  * Cross
  * Square
  * L-Shaped

The solution uses a shared EfficientNet-B4 backbone with separate localization and classification heads trained jointly.

## Architecture

### Backbone

* EfficientNet-B4 (ImageNet pretrained via timm)

### Multi-Task Heads

#### Keypoint Localization Head

* 1×1 convolutional heatmap predictor
* Differentiable Soft-Argmax
* Outputs normalized coordinates `(x, y)` in `[0,1]`

#### Shape Classification Head

* Global Average Pooling
* Fully connected classifier
* Outputs probabilities for:

  * Cross
  * Square
  * L-Shaped

## Training Configuration

| Parameter         | Value             |
| ----------------- | ----------------- |
| Backbone          | EfficientNet-B4   |
| Image Size        | 640 × 640         |
| Epochs            | 50                |
| Batch Size        | 4                 |
| Optimizer         | AdamW             |
| Learning Rate     | 1e-4              |
| Weight Decay      | 1e-4              |
| Scheduler         | CosineAnnealingLR |
| Gradient Clipping | 5.0               |
| Validation Split  | 20%               |

### Loss Function

Total Loss:

Loss = Keypoint Loss + 0.5 × Classification Loss

#### Keypoint Loss

* SmoothL1Loss (β = 0.01)

#### Classification Loss

* CrossEntropyLoss
* Label smoothing = 0.1

## Data Augmentation

Training augmentations:

* Horizontal Flip
* Vertical Flip
* RandomRotate90
* ShiftScaleRotate
* RandomBrightnessContrast
* HueSaturationValue
* Gaussian Blur
* Motion Blur
* Gaussian Noise

All geometric transforms are applied consistently to images and keypoints.

## Dataset

The dataset contains aerial imagery collected from multiple mining and surveying environments.

Annotations include:

* GCP center coordinates
* Marker shape labels

Classes:

* Cross
* Square
* L-Shaped

## Training Results

Best Model:

* Epoch: 41

Validation Metrics:

| Metric   | Score |
| -------- | ----- |
| Macro F1 | 1.000 |
| PCK@10   | 0.010 |
| PCK@25   | 0.050 |
| PCK@50   | 0.150 |

Checkpoint:

checkpoints/best_model.pth

## Inference

Generate predictions:

```bash
python src/inference.py \
    --test_dir data/test_dataset \
    --checkpoint checkpoints/best_model.pth \
    --output predictions.json
```

With Test-Time Augmentation:

```bash
python src/inference.py \
    --test_dir data/test_dataset \
    --checkpoint checkpoints/best_model.pth \
    --output predictions.json \
    --tta
```

## Output Format

```json
{
  "project1/survey1/2/DJI_0431.JPG": {
    "mark": {
      "x": 1024.5,
      "y": 850.2
    },
    "verified_shape": "L-Shaped"
  }
}
```

## Project Structure

```text
drone/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── eda.py
├── checkpoints/
│   └── best_model.pth
├── predictions.json
├── requirements.txt
└── README.md
```

## Environment

Training performed using:

* Python 3.12
* PyTorch
* timm
* OpenCV
* Albumentations
* Kaggle GPU Environment

## Repository

GitHub Repository:

https://github.com/leahmarymathew/drone
