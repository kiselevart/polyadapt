"""Pretrained R3D-18 baselines for ablation.

r3d_frozen   — backbone frozen, only classification head trained (linear probe).
r3d_finetune — full fine-tune: all backbone params + head trained.
"""

import torch
import torch.nn as nn
import torchvision


class R3DBaseline(nn.Module):

    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        bb = torchvision.models.video.r3d_18(weights="DEFAULT")

        if freeze_backbone:
            for p in bb.parameters():
                p.requires_grad_(False)

        self.stem   = bb.stem
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.avgpool = bb.avgpool

        self.fc = nn.Linear(512, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))
