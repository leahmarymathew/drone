import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """Convert [B, 1, H, W] heatmap to normalised (x, y) coords via differentiable soft-argmax."""
    B, _, H, W = heatmap.shape
    heat = F.softmax(heatmap.view(B, -1), dim=-1).view(B, 1, H, W)

    grid_x = torch.linspace(0, 1, W, device=heatmap.device).view(1, 1, 1, W)
    grid_y = torch.linspace(0, 1, H, device=heatmap.device).view(1, 1, H, 1)

    x = (heat * grid_x).sum(dim=(2, 3)).squeeze(1)  # [B]
    y = (heat * grid_y).sum(dim=(2, 3)).squeeze(1)  # [B]
    return torch.stack([x, y], dim=1)               # [B, 2]


class GCPNet(nn.Module):
    """
    Multi-task network: shared EfficientNet-B4 backbone with two heads —
    one for keypoint regression (normalised x,y via soft-argmax) and one for shape classification.
    """

    def __init__(
        self,
        backbone: str = 'efficientnet_b4',
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        # No global pool — spatial feature maps kept for keypoint head
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=''
        )
        feat_dim = self.backbone.num_features

        # Classification branch: GAP → linear
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_neck = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_rate),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_rate / 2),
            nn.Linear(256, num_classes),
        )

        # Keypoint branch: 1×1 conv heatmap + soft-argmax (preserves spatial info)
        self.kp_head = nn.Sequential(
            nn.Conv2d(feat_dim, 64, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)                          # [B, feat_dim, H, W]

        pooled = self.cls_pool(feats).flatten(1)          # [B, feat_dim]
        cls_logits = self.cls_head(self.cls_neck(pooled))

        heatmap = self.kp_head(feats)                     # [B, 1, H, W]
        kp = soft_argmax_2d(heatmap)                      # [B, 2] in [0, 1]

        return kp, cls_logits
